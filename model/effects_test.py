"""L7 外部化（第 30 轮）：效应组合（compose_eff）对“无效应”的结论 vs MLIR 自己的判断。
真值：同一提交 MLIR 的 canonicalize。把同一段体包进结果不用的 scf.execute_region，MLIR 依据 ODS 的 MemoryEffects
（RecursiveMemoryEffects、alloc 的结果效应）决定能否整体删除（wouldOpBeTriviallyDead / SimplifyDeadAlloc）。
  可靠方向：模型 Eff = ∅ ⇒ MLIR 删除（否则模型把 MLIR 认为有效应的体判成纯的，可被前端用来做 DCE/CSE）。
  完备方向：MLIR 删除而模型 Eff ≠ ∅ ⇒ 模型保守（只计数、分类）。
穷举语法：局部分配 m 上的 释放∈{无, 同块, 分支内} × store∈{0,1} × load∈{0,1}；参数 p 上的 动作∈{无, store, load, free}。
用法：PYTHONPATH=model python3 model/effects_test.py
"""
import itertools, collections
import mlir.ir as ir
from mlir.passmanager import PassManager
import core as C
from core import blk, node, I3

def model_case(fr, st, ld, pa, tag):
    ops = [node('m', 'memref.alloc', outs=[('m', 'memref')]), C.cst('v', 1)]
    if st: ops.append(node('s', 'memref.store', ins=['v', 'm']))
    if ld: ops.append(node('l', 'memref.load', ins=['m'], outs=[('l', I3)]))
    if pa == 'store': ops.append(node('ps', 'memref.store', ins=['v', 'p']))
    if pa == 'load': ops.append(node('pl', 'memref.load', ins=['p'], outs=[('pl', I3)]))
    if pa == 'free': ops.append(node('pf', 'memref.dealloc', ins=['p']))
    if fr == 'same': ops.append(node('d', 'memref.dealloc', ins=['m']))
    if fr == 'if':
        ops.append(node('if', 'scf.if', ins=['b'], outs=[], regions=[[blk('t', [node('d', 'memref.dealloc', ins=['m']), node('y', 'scf.yield', ins=[])])],
                                                                     [blk('e', [node('y2', 'scf.yield', ins=[])])]]))
    args = [('p', 'memref'), ('b', 'i1')]
    return C.derive(f'eff_{tag}', args, [('q', I3)], [blk('entry', ops + [C.cst('z', 0), C.ret('r', 'z')], args=args)])

def mlir_case(fr, st, ld, pa):
    T = 'memref<4xi3>'; b = ['%m = memref.alloc() : ' + T, '%v = arith.constant 1 : i3', '%i = arith.constant 0 : index']
    if st: b.append(f'memref.store %v, %m[%i] : {T}')
    if ld: b.append(f'%l = memref.load %m[%i] : {T}')
    if pa == 'store': b.append(f'memref.store %v, %p[%i] : {T}')
    if pa == 'load': b.append(f'%pl = memref.load %p[%i] : {T}')
    if pa == 'free': b.append(f'memref.dealloc %p : {T}')
    if fr == 'same': b.append(f'memref.dealloc %m : {T}')
    if fr == 'if': b.append(f'scf.if %b {{ memref.dealloc %m : {T} }}')
    body = '\n    '.join(b + ['%c = arith.constant 0 : i3', 'scf.yield %c : i3'])
    return (f'func.func @f(%p: {T}, %b: i1) -> i3 {{\n  %r = scf.execute_region -> i3 {{\n    {body}\n  }}\n'
            f'  %z = arith.constant 0 : i3\n  return %z : i3\n}}')

def mlir_removed(src):
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse('builtin.module(func.func(canonicalize))').run(m.operation)
        t = str(m)
    return 'execute_region' not in t and 'memref.' not in t

def run():
    st = collections.Counter(); ex = collections.defaultdict(list)
    for k, (fr, s, l, pa) in enumerate(itertools.product(['none', 'same', 'if'], [0, 1], [0, 1], ['none', 'store', 'load', 'free'])):
        S = model_case(fr, s, l, pa, k); pure = S.eff == frozenset(); rm = mlir_removed(mlir_case(fr, s, l, pa))
        st['用例'] += 1; st['模型纯'] += pure; st['MLIR 删除'] += rm
        key = '一致' if pure == rm else ('不可靠：模型纯、MLIR 保留' if pure else '保守：MLIR 删除、模型有效应')
        st[key] += 1
        if pure != rm: ex[key].append((fr, s, l, pa, sorted(S.eff) if S.eff != C.TOP else 'TOP'))
        if S.eff_diag: st['带泄漏诊断'] += 1
    return st, ex

def mutation():
    """灵敏度：把参数上的 Free 从效应里删掉，检验必须报出不可靠。"""
    o = C.compose_eff
    def bad(effs, b):
        e, d = o(effs, b); return (e if e == C.TOP else frozenset(x for x in e if x[0] != 'Free')), d
    C.compose_eff = bad
    try: st, _ = run()
    finally: C.compose_eff = o
    return st['不可靠：模型纯、MLIR 保留']

if __name__ == '__main__':
    st, ex = run()
    print('── X7 L7 效应组合 vs MLIR canonicalize:', dict(st))
    for k, v in ex.items():
        print('    ', k, len(v)); [print('        ', x) for x in v[:6]]
    print('── X7 变异（丢掉参数上的 Free）抓出:', mutation())
