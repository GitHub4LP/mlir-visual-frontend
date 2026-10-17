"""值层语义的外部审计（第 21 轮，spec/laws.md §3）。真值只取 MLIR 自己：LangRef 定义的执行结果（JIT）与 MLIR 利用毒值标志的变换。
  V1 无标志程序 vs JIT：LangRef 规定无 nsw/nuw 的 addi/subi/muli 按 2^n 取模（index 的 n 为目标位宽），不产生毒值。
     模型在这类程序上给出“毒值”即弱于 MLIR（导入的程序被改了语义）；给出不同的值即错。
  V3 定向翻译确认：x op k（带 nsw/nuw）再与常量比较——int-range-optimizations 正是在这种形状上利用溢出标志折叠比较。
  V2 翻译确认：随机 i8/index 函数（带 nsw/nuw）交给 MLIR 的 canonicalize + int-range-optimizations（后者利用溢出标志与 index 位宽
     推区间并折叠），读回后判定 refines(原程序, 结果)。不是精化 ⇒ 模型的毒值条件比 MLIR 窄（模型把 MLIR 视为毒值的值当成确定值）。
用法：PYTHONPATH=model python3 model/sem_test.py [V1 程序数] [V2 用例数]
"""
import random, sys, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
import core as C, loop_test as LT
from core import blk, node

def strip_flags(bs):
    for b in bs:
        for n in b['ops']:
            for k in ('nuw', 'nsw'): n.get('props', {}).pop(k, None)
            for rg in n.get('regions', []): strip_flags(rg)
    return bs

def V1(n, rng):
    st = dict(程序=0, 输入=0, 一致=0, 值不一致=0, 模型毒值=0, 模型UB=0, 未定=0)
    for t in range(n):
        g = strip_flags(LT.gen(rng, t)); G = C.derive(f'V1_{t}', LT.INS, [('q', 'i64')], g)
        st['程序'] += 1
        ins = [[rng.choice([0, 1, 2, 5, -1, 1 << 40, (1 << 62), (1 << 63) - 1, -(1 << 63), rng.randrange(-1 << 63, 1 << 63)]) for _ in range(3)] for _ in range(4)]
        outs = LT.jit(LT.emit(g), ins)
        for xs, o in zip(ins, outs):
            st['输入'] += 1; r = LT.model_eval(G, xs)
            if r[0] == 'val':
                if r[1] == o: st['一致'] += 1
                else: st['值不一致'] += 1
            elif r[0] == 'poison':
                st['模型毒值'] += 1
                if st['模型毒值'] <= 1: print('  ✗ 无标志程序被模型判为毒值，MLIR 执行得', o, '输入', xs, '\n', LT.emit(g))
            elif r[0] == 'UB': st['模型UB'] += 1               # divui 除以 0：LangRef 定义为 UB，JIT 的结果不作数
            else: st['未定'] += 1
    return st

# ───────────────────────── V2
FLAGS = [(), ('nsw',), ('nuw',), ('nsw', 'nuw')]
EDGE = {'i8': [0, 1, 2, 100, 127, -128, -1, 64], 'index': [0, 1, 2, -1, (1 << 62), (1 << 63) - 1, -(1 << 63), 1 << 31, (1 << 31) - 1]}
def gen(rng, t, T):
    k = [0]; pool = {T: ['a', 'b'], 'i1': ['c']}; ops = []
    def fresh(): k[0] += 1; return f'q{t}_{k[0]}'
    for _ in range(rng.randint(3, 8)):
        v = fresh(); r = rng.random(); xs = pool[T]
        pick = lambda: rng.choice(xs[-3:] if rng.random() < 0.6 else xs)
        if r < 0.2: ops.append(C.cst(v, rng.choice(EDGE[T]), T)); xs.append(v)
        elif r < 0.65:
            op = rng.choice(['arith.addi', 'arith.subi', 'arith.muli', 'arith.addi', 'arith.andi', 'arith.ori', 'arith.divui'])
            pr = {f: True for f in rng.choice(FLAGS)} if op in ('arith.addi', 'arith.subi', 'arith.muli') else {}
            ops.append(C.op2(v, op, pick(), pick(), T, **pr)); xs.append(v)
        elif r < 0.85: ops.append(C.op2(v, 'arith.cmpi', pick(), pick(), 'i1', predicate=rng.randrange(10))); pool['i1'].append(v)
        else: ops.append(node(v, 'arith.select', ins=[rng.choice(pool['i1']), pick(), pick()], outs=[(v, T)])); xs.append(v)
    last = rng.choice([pool[T][-1], pool['i1'][-1]])
    if last in pool['i1']:
        e = fresh(); ops.append(node(e, 'arith.select', ins=[last, pool[T][-1], 'a'], outs=[(e, T)])); last = e
    return [blk('entry', ops + [C.ret(f'ret{t}', last)], args=[('a', T), ('b', T), ('c', 'i1')])]

def emit(blocks, T):
    nm = {'a': '%arg0', 'b': '%arg1', 'c': '%arg2'}; ty = {'a': T, 'b': T, 'c': 'i1'}
    v = lambda x: nm.setdefault(x, '%' + x); out = []
    for n in blocks[0]['ops']:
        for o, t in n['outs']: ty[o] = t
        op, pr = n['op'], n.get('props', {})
        if op == 'func.return': out.append(f'  return {v(n["ins"][0])} : {T}'); continue
        at = ''
        if op == 'arith.constant': at = f'<{{value = {pr["value"]} : {pr["type"]}}}>'
        elif op == 'arith.cmpi': at = f'<{{predicate = {pr["predicate"]} : i64}}>'
        else:
            fl = [f for f in ('nsw', 'nuw') if pr.get(f)]
            if fl: at = f'<{{overflowFlags = #arith.overflow<{", ".join(fl)}>}}>'
        sig = '(' + ', '.join(ty[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
        out.append(f'  {v(n["outs"][0][0])} = "{op}"(' + ', '.join(v(x) for x in n['ins']) + f') {at} : {sig}')
    return f'func.func @f(%arg0: {T}, %arg1: {T}, %arg2: i1) -> {T} {{\n' + '\n'.join(out) + '\n}'

KNOWN = {'arith.constant', 'arith.addi', 'arith.subi', 'arith.muli', 'arith.divui', 'arith.andi', 'arith.ori', 'arith.xori',
         'arith.cmpi', 'arith.select', 'func.return'}
def read_back(func, T):
    names = {}; cnt = [0]; ops = []
    for i, a in enumerate(func.regions[0].blocks[0].arguments): names[a.get_name()] = 'abc'[i]
    def vid(val):
        k = val.get_name()
        if k not in names: cnt[0] += 1; names[k] = f'm{cnt[0]}'
        return names[k]
    for op in func.regions[0].blocks[0].operations:
        o = op.operation; name = o.name
        if name not in KNOWN: raise KeyError(name)
        outs = [(vid(r), str(r.type)) for r in o.results]; ins = [vid(x) for x in o.operands]; pr = {}
        if name == 'arith.constant': pr = {'value': ir.IntegerAttr(o.attributes['value']).value, 'type': str(o.results[0].type)}
        elif name == 'arith.cmpi': pr = {'predicate': ir.IntegerAttr(o.attributes['predicate']).value}
        elif 'overflowFlags' in o.attributes:
            s = str(o.attributes['overflowFlags']); pr = {f: True for f in ('nsw', 'nuw') if f in s}
        ops.append(node(f'n{cnt[0]}_{len(ops)}', name, ins=ins, outs=outs, props=pr)); cnt[0] += 1
    return [blk('entry', ops, args=[('a', T), ('b', T), ('c', 'i1')])]

def gen3(rng, t, T):
    k, bnd = rng.choice(EDGE[T] + [3, -3, 50]), rng.choice(EDGE[T] + [-28, 28, -100])
    op = rng.choice(['arith.addi', 'arith.subi', 'arith.muli']); fl = rng.choice(FLAGS[1:])
    ops = [C.cst('k', k, T), C.op2('s', op, 'a', 'k', T, **{f: True for f in fl}), C.cst('bd', bnd, T),
           C.op2('c1', 'arith.cmpi', 's', 'bd', 'i1', predicate=rng.randrange(10)),
           node('o', 'arith.select', ins=['c1', 'a', 'b'], outs=[('o', T)]), C.ret(f'ret{t}', 'o')]
    return [blk('entry', ops, args=[('a', T), ('b', T), ('c', 'i1')])]

PIPE = 'builtin.module(func.func(canonicalize,int-range-optimizations,canonicalize))'
def V2(n, rng, g_=None):
    st = dict(用例=0, 精化=0, 反例=0, 求解器未定=0, 含未建模op=0, 有变化=0)
    for t in range(n):
        T = rng.choice(['i8', 'index']); g = (g_ or gen)(rng, t, T); ins = [('a', T), ('b', T), ('c', 'i1')]
        G = C.derive(f'V2_{t}', ins, [('q', T)], g)
        with ir.Context():
            src = emit(g, T); m = ir.Module.parse(src); assert m.operation.verify()
            PassManager.parse(PIPE).run(m.operation)
            try: h = read_back(m.body.operations[0], T)
            except KeyError: st['含未建模op'] += 1; continue
            txt = str(m)
        st['用例'] += 1; st['有变化'] += txt.count('\n') < src.count('\n')
        H = C.derive(f'V2h_{t}', ins, [('q', T)], h)
        u0 = C.STATS['unknown']; ok = C.refines(G, H)
        if ok: st['精化'] += 1
        elif C.STATS['unknown'] > u0: st['求解器未定'] += 1
        else:
            st['反例'] += 1
            if st['反例'] <= 2: print('  ✗ MLIR 的结果不是模型意义下的精化（模型的毒值条件窄于 MLIR）\n', src, '\n  →\n', txt)
    return st

if __name__ == '__main__':
    v3 = 'v3' in sys.argv
    if v3: sys.argv.remove('v3')
    if 'old' in sys.argv: sys.argv.remove('old'); C.IXOVF, C.NSW = 'poison', False      # 对照：第 6–20 轮的语义
    n1 = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n2 = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    print(f'── 语义：index 溢出 = {C.IXOVF}，nsw = {"建模" if getattr(C, "NSW", False) else "未建模"}')
    if not v3: print('── V1 无标志程序 vs MLIR 执行:', V1(n1, random.Random(7)))
    print('── V3 定向（溢出标志 + 常量比较）的翻译确认:', V2(n2, random.Random(9), gen3))
    if v3: sys.exit()
    print('── V2 MLIR canonicalize + int-range-optimizations 的翻译确认:', V2(n2, random.Random(8)))
