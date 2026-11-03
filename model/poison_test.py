"""第 53 轮：L6（复合摘要的 UB 与毒值必须依赖输入毒值）的外部证据。

毒值本身不可观测，但“模型说有定义”可以检验：调用方在 MLIR 里真正制造毒值（shli 移位量 = 位宽 ⇒ 毒值，LangRef 与 LLVM 一致），
把它作为实参传给复合 op f（单独的 func.func，经 func.call 调用；O3 下 LLVM 会内联并利用毒值折叠）。
模型对调用方的推导只经 f 的摘要（摘要实例化），不看 f 的体。模型给出确定值 ⇒ JIT 在 O0 和 O3 下都必须得到这个值。
反方向（模型说毒值/UB）是精化自由度，不可测，跳过（与 V5 相同）。

P1 外部对照：随机复合 f（arith 整数 op、标志、cmpi + select，i8–i64），随机毒化 f 的一部分实参；模型确定值 ⇒ JIT 一致。
   另报“毒化实参而模型仍给确定值”的输入数（毒值被 select 吸收或未流到结果）——这才是 L6 真正被检验的部分，必须 > 0。
P2 消融灵敏度：把 f 的摘要换成不依赖输入毒值的版本（core_test X1 的 v3 式：输入毒值标志代入假），同一批程序上应出现反例。
用法：PYTHONPATH=model python3 model/poison_test.py [程序数]
"""
import sys, random, copy, collections, ctypes, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
import core as C, loop_test as LT, verdict as V
from core import blk, node
from values_test import attr_text, OLD, NEW, FLAGS_OF

INS = [('a', 'i64'), ('b', 'i64'), ('c', 'i64')]
PRED = list(range(10))

def gen_f(rng, T):
    w = int(T[1:]); k = [0]; ops = []; xs = ['x', 'y', 'z']
    def fresh(): k[0] += 1; return f'f{k[0]}'
    def cst():
        v = fresh(); ops.append(C.cst(v, rng.choice([0, 1, 2, 3, w - 1, w, -1, 7, 1 << (w - 1)]), T)); return v
    pick = lambda: rng.choice(xs[-3:]) if rng.random() < 0.6 else rng.choice(xs)
    for _ in range(rng.randint(2, 6)):
        r = rng.random(); v = fresh()
        if r < 0.35:                                                 # 吸收毒值的形状：select
            c = fresh(); ops.append(node(c, 'arith.cmpi', ins=[pick(), cst() if rng.random() < 0.5 else pick()], outs=[(c, 'i1')],
                                         props={'predicate': rng.choice(PRED)}))
            ops.append(node(v, 'arith.select', ins=[c, pick(), pick()], outs=[(v, T)]))
        else:
            op = rng.choice(OLD + NEW)
            fl = {f: True for f in FLAGS_OF.get(op, ()) if rng.random() < 0.3}
            ops.append(C.op2(v, op, pick(), cst() if rng.random() < 0.5 else pick(), T, **fl))
        xs.append(v)
    return [blk('entry', ops + [C.ret('fr', xs[-1])], args=[('x', T), ('y', T), ('z', T)])]

def gen_main(rng, T, fname, mask):
    w = int(T[1:]); ops = []; args = []
    for i, a in enumerate(('a', 'b', 'c')):
        v = a
        if T != 'i64': v = f't{i}'; ops.append(node(v, 'arith.trunci', ins=[a], outs=[(v, T)]))
        if mask[i]:                                                  # 制造毒值：x << w
            s = f's{i}'; p = f'p{i}'
            ops += [C.cst(s, w, T), C.op2(p, 'arith.shli', v, s, T)]; v = p
        args.append(v)
    ops.append(node('k', fname, ins=args, outs=[('k', T)]))
    r = 'k'
    if T != 'i64': ops.append(node('e', 'arith.extsi', ins=['k'], outs=[('e', 'i64')])); r = 'e'
    return [blk('entry', ops + [C.ret('mr', r)], args=INS)]

def reach(fb):
    """f 的哪些形参在数据流上可达返回值。"""
    dep = {a: {a} for a in ('x', 'y', 'z')}
    for n in fb[0]['ops']:
        d = set().union(*[dep.get(x, set()) for x in n['ins']]) if n['ins'] else set()
        if n['op'] == 'func.return': return d
        for o, _ in n['outs']: dep[o] = d
    return set()

def emit_fn(name, blocks, ins, rty, callee=None, iface=False):
    ty = dict(ins); out = []
    for n in blocks[0]['ops']:
        for o, t in n['outs']: ty[o] = t
        if n['op'] == 'func.return': out.append(f'  return %{n["ins"][0]} : {rty}'); continue
        if callee and n['op'] == callee[0]:
            out.append(f'  %{n["outs"][0][0]} = func.call @{callee[1]}(' + ', '.join('%' + x for x in n['ins']) + ') : ('
                       + ', '.join(ty[x] for x in n['ins']) + f') -> {n["outs"][0][1]}'); continue
        sig = '(' + ', '.join(ty[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
        out.append(f'  %{n["outs"][0][0]} = "{n["op"]}"(' + ', '.join('%' + x for x in n['ins']) + f') {attr_text(n)} : {sig}')
    at = ' attributes {llvm.emit_c_interface}' if iface else ''
    return f'func.func @{name}(' + ', '.join(f'%{a}: {t}' for a, t in ins) + f') -> {rty}{at} {{\n' + '\n'.join(out) + '\n}'

PIPE = 'builtin.module(convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'
def jit(src, inputs, opt):
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse(PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=opt); res = []
        for xs in inputs:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); res.append(r[0] % (1 << 64))
        return res

def v3_of(f, name):
    """不依赖输入毒值的摘要（消融）：输入毒值标志一律代入假。"""
    g = copy.copy(f); g.name = name
    nz = [(p, C.F_) for _, p in f.holes]
    g.ubf = z3.substitute(f.ubf, nz); g.outv = [(v, z3.substitute(p, nz)) for v, p in f.outv]
    C.reg(g); return g

def run(n, seed=53):
    rng = random.Random(seed); st = collections.Counter(); ex = []
    for t in range(n):
        T = rng.choice(['i8', 'i16', 'i32', 'i64'])
        fb = gen_f(rng, T); fn = f'PF{t}'
        try: f = C.derive(fn, [('x', T), ('y', T), ('z', T)], [('q', T)], fb)
        except C.ValueUnknown: st['跳过：f 值层未知'] += 1; continue
        v3_of(f, fn + '_v3')
        mask = [0, 0, 0]
        while not any(mask): mask = [int(rng.random() < 0.5) for _ in range(3)]
        mb = gen_main(rng, T, fn, mask)
        try:
            G = C.derive(f'PM{t}', INS, [('q', 'i64')], mb)
            G3 = C.derive(f'PM{t}_v3', INS, [('q', 'i64')], [dict(mb[0], ops=[dict(o, op=fn + '_v3') if o['op'] == fn else o for o in mb[0]['ops']])])
        except C.ValueUnknown: st['跳过：main 值层未知'] += 1; continue
        src = emit_fn(fn, fb, [('x', T), ('y', T), ('z', T)], T) + '\n' + emit_fn('main', mb, INS, 'i64', callee=(fn, fn), iface=True)
        ins = [[rng.choice([0, 1, 2, 3, 5, 7, 100, 255, -1, -128, 1 << 31, (1 << 63) - 1, -(1 << 63), rng.randrange(-1 << 63, 1 << 63)])
                for _ in range(3)] for _ in range(4)]
        live = any(mask[i] and p in reach(fb) for i, p in enumerate(('x', 'y', 'z')))
        md = [LT.model_eval(G, xs) for xs in ins]; m3 = [LT.model_eval(G3, xs) for xs in ins]
        st['程序'] += 1
        for opt in (0, 3):
            try: outs = jit(src, ins, opt)
            except Exception as e: st['跳过：JIT 失败'] += 1; ex.append(('jit', str(e)[:200], src)); break
            for xs, mv, m3v, o in zip(ins, md, m3, outs):
                st['输入'] += 1
                if mv[0] == 'val':
                    st['模型确定值（实参含毒值）'] += 1
                    st['确定值且毒值实参数据流可达结果'] += live
                    if mv[1] == o: st['一致'] += 1
                    else:
                        st['反例'] += 1
                        if len(ex) < 3: ex.append((f'O{opt}', xs, mv[1], o, src))
                else: st[f'模型 {mv[0]}'] += 1
                if m3v[0] == 'val' and m3v[1] != o: st[f'消融反例/O{opt}'] += 1
    return st, ex

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    st, ex = run(n)
    print(dict(sorted(st.items())))
    for e in ex: print('  ✗', e)
    V.check('P1 毒值实参的确定值 vs JIT', st['程序'] >= n * 0.8 and st['反例'] == 0 and st['确定值且毒值实参数据流可达结果'] >= 50,
            f"程序 {st['程序']}，输入 {st['输入']}（O0+O3）：模型确定值 {st['模型确定值（实参含毒值）']} 中一致 {st['一致']}、反例 {st['反例']}（其中毒值实参在数据流上可达结果、被 select 等吸收的 {st['确定值且毒值实参数据流可达结果']}）；模型毒值 {st['模型 poison']}、UB {st['模型 UB']}、未定 {st['模型 ?']}")
    V.check('P2 消融灵敏度', st['消融反例/O3'] > 0,
            f"摘要不依赖输入毒值（v3 式）：反例 O0 {st['消融反例/O0']}、O3 {st['消融反例/O3']}")
    V.finish()
