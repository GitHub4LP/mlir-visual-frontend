"""L11 外部化（第 30 轮）：用户 ADT 的抽象语义 + 1:N 降级，对照同一提交 MLIR 的 JIT。
模型的结论：降级程序在抽象函数下精化抽象程序（defs.refines_lowered，此前只有自证）。
外部检验：随机程序（defs_test.gen_opt：option 的构造/判别/投影、scf.if 返回 option、以 is_some 为条件）
  1 按模型降级（D.lower_blocks）并内联成纯 arith/scf/ub 程序，交给 MLIR 降级 + JIT；
  2 穷举降级输入 x̂ = (a, b, tag, val)，共 1024 个；
  3 对模型判定抽象程序在 abs(x̂) 上有定义（无 UB、无毒值）的输入，要求 abs(JIT(x̂)) = Sem_A(abs x̂)。
  Sem_A 直接在抽象层求值（ADT 是 z3 Datatype），不经过降级，所以这检验的是“抽象语义 + 降级”整体，而不是降级的自洽。
灵敏度：把 option.is_some 的降级换成判 tag==0 的错误实现，JIT 必须抓到差异。
用法：PYTHONPATH=model python3 model/adtjit_test.py [程序数]
"""
import random, sys, re, ctypes, itertools, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
import core as C, defs as D, defs_test as DT
from core import blk, node, I3

def emit(blocks, rtys):
    nm = {}
    def v(x):
        if x not in nm: nm[x] = '%v' + str(len(nm)) + '_' + re.sub(r'[^A-Za-z0-9_]', '_', x)
        return nm[x]
    ty = {}
    for x, t in blocks[0]['args']: ty[x] = t
    def ops(bs, ind):
        out = []
        for n in bs[0]['ops']:
            for o, t in n['outs']: ty[o] = t
            res = ', '.join(v(o) for o, _ in n['outs']); pr = n.get('props', {}); op = n['op']
            tys = lambda xs: ', '.join(ty[x] for x in xs)
            if op == 'func.return': out.append(ind + 'return ' + ', '.join(v(x) for x in n['ins']) + ' : ' + tys(n['ins'])); continue
            if op == 'scf.yield': out.append(ind + 'scf.yield ' + ', '.join(v(x) for x in n['ins']) + ' : ' + tys(n['ins'])); continue
            if op == 'ub.poison': out.append(ind + f'{res} = ub.poison : {n["outs"][0][1]}'); continue
            if op == 'scf.if':
                out.append(ind + f'{res} = scf.if {v(n["ins"][0])} -> (' + ', '.join(t for _, t in n['outs']) + ') {')
                out += ops(n['regions'][0], ind + '  '); out.append(ind + '} else {'); out += ops(n['regions'][1], ind + '  '); out.append(ind + '}')
                continue
            if op == 'arith.constant':
                t = pr.get('type', 'i3'); at = f'<{{value = {pr["value"] % (2 if t == "i1" else 8)} : {t}}}>'
            elif op == 'arith.cmpi': at = f'<{{predicate = {pr["predicate"]} : i64}}>'
            else:
                assert not pr, (op, pr); at = ''
            sig = '(' + tys(n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
            out.append(ind + f'{res} = "{op}"(' + ', '.join(v(x) for x in n['ins']) + f') {at} : {sig}')
        return out
    args = ', '.join(f'{v(x)}: {t}' for x, t in blocks[0]['args'])
    return f'func.func @f({args}) -> (' + ', '.join(rtys) + ') {\n' + '\n'.join(ops(blocks, '  ')) + '\n}'

PIPE = 'builtin.module(convert-scf-to-cf,convert-ub-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'
ALL = list(itertools.product(range(8), range(8), range(2), range(8)))     # (a, b, tag, val)

def jit(blocks, rtys):
    """f(i3, i3, i1, i3) -> rtys，外包 main(4×i64) -> i64；结果按 8 位一段打包。"""
    W = {'i1': 1, 'i3': 3}
    call_ty = '(i3, i3, i1, i3) -> (' + ', '.join(rtys) + ')'
    rs = ', '.join(f'%r{i}' for i in range(len(rtys)))
    body = ['%a = arith.trunci %x0 : i64 to i3', '%b = arith.trunci %x1 : i64 to i3', '%t = arith.trunci %x2 : i64 to i1', '%u = arith.trunci %x3 : i64 to i3',
            f'{rs} = func.call @f(%a, %b, %t, %u) : {call_ty}', '%acc0 = arith.constant 0 : i64']
    for i, t in enumerate(rtys):
        body += [f'%e{i} = arith.extui %r{i} : {t} to i64', f'%s{i} = arith.constant {8 * i} : i64', f'%h{i} = arith.shli %e{i}, %s{i} : i64',
                 f'%acc{i + 1} = arith.ori %acc{i}, %h{i} : i64']
    body.append(f'return %acc{len(rtys)} : i64')
    wrap = ('func.func @main(%x0: i64, %x1: i64, %x2: i64, %x3: i64) -> i64 attributes {llvm.emit_c_interface} {\n  ' + '\n  '.join(body) + '\n}')
    src = emit(blocks, rtys) + '\n' + wrap
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify(), src
        PassManager.parse(PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=0); out = []
        for xs in ALL:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r)
            out.append([(r[0] >> (8 * i)) & ((1 << W[t]) - 1) for i, t in enumerate(rtys)])
        return out

def abs_eval(A, opt, rty, xs):
    """在抽象层求 Sem_A(abs x̂)：返回 ('skip', 原因) 或 ('val', z3 值)。"""
    a, b, t, u = xs
    ov = opt.ctors[1](z3.BitVecVal(u, 3)) if t else opt.ctors[0]()
    pairs = []
    for (h, hp), x in zip(A.holes, [z3.BitVecVal(a, 3), z3.BitVecVal(b, 3), ov]):
        pairs += [(h, x), (hp, z3.BoolVal(False))]
    sub = lambda f: z3.simplify(z3.substitute(f, pairs))
    if not all(z3.is_true(sub(g)) for _, g in A.guards): return 'skip', '守卫'
    if not z3.is_false(sub(A.ubf)): return 'skip', 'UB'
    v, p = A.outv[0]
    if not z3.is_false(sub(p)): return 'skip', '毒值'
    return 'val', sub(v)

def agree(val, jout, opt, rty):
    if rty == I3: return z3.is_true(z3.simplify(val == z3.BitVecVal(jout[0], 3)))
    t, u = jout
    lv = opt.ctors[1](z3.BitVecVal(u, 3)) if t else opt.ctors[0]()
    return z3.is_true(z3.simplify(val == lv))

def run(n, seed=30, mutate=False):
    opt = D.define_adt('option', [('none', []), ('some', [('val', I3)])])
    saved = dict(opt.lower_ops)
    if mutate:
        C.derive('bad_is_some_j', [('v0', 'i1'), ('v1', I3)], [('r', 'i1')],
                 [blk('entry', [C.cst('k', 0, 'i1'), node('eq', 'arith.cmpi', ins=['v0', 'k'], outs=[('eq', 'i1')], props={'predicate': 0}),
                                C.ret('rr', 'eq')], args=[('v0', 'i1'), ('v1', I3)])])
        opt.lower_ops['option.is_some'] = 'bad_is_some_j'
    st = dict(程序=0, 输入=0, 抽象有定义=0, 一致=0, 不符=0, 抽象UB=0, 抽象毒值=0, 返回option=0)
    rng = random.Random(seed)
    try:
        for t in range(n):
            ins, ops, (rv, rty) = DT.gen_opt(rng, 30000 + t)
            prog = [blk('entry', ops + [C.ret('r', rv)], args=ins)]
            A = C.derive(f'AJ{t}{"m" if mutate else ""}', ins, [('q', rty)], prog)
            low = C.inline_all(D.lower_blocks(prog))
            rtys = opt.lowered if rty == 'option' else [rty]
            js = jit(low, rtys); st['程序'] += 1; st['返回option'] += rty == 'option'
            bad = 0
            for xs, jo in zip(ALL, js):
                st['输入'] += 1; k, val = abs_eval(A, opt, rty, xs)
                if k == 'skip':
                    st['抽象UB'] += val == 'UB'; st['抽象毒值'] += val == '毒值'; continue
                st['抽象有定义'] += 1
                if agree(val, jo, opt, rty): st['一致'] += 1
                else: st['不符'] += 1; bad += 1
            if bad and not mutate: print(f'  ✗ 程序 {t}: {bad} 个输入 abs(JIT) ≠ Sem_A\n', emit(low, rtys))
    finally:
        opt.lower_ops = saved
    return st

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print('── A11 L11 抽象语义 + 降级 vs JIT:', run(n))
    m = run(n, mutate=True)
    print('── A11 变异（is_some 降级判 tag==0）不符输入:', m['不符'], '/', m['抽象有定义'])
