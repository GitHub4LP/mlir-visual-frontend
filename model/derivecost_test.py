"""第 48 轮：推导代价回归。

第 47 轮发现：tensor op 把上一个 tensor 的整项嵌进新 z3 Lambda，嵌套深度随链长线性增长，z3 抽象绑定变量按深度重复遍历
共享子项 ⇒ 推导时间随链长指数增长（直线 12 个 insert_slice 13 s、14 个 > 25 s；语料一个 10 趟循环 > 20 分钟）。
第 48 轮在项层做 β/投影化简（core._proj/_sel）。判定（fork 子进程，限时 90 s，只防挂起）：
  D1 直线链：N = 8、40 个 insert_slice，在限时内推导出事实；
  D2 循环展开：语料 tensor_dim_of_iter_arg_insertslice 的同构循环，展开 10 / 64 趟（关闭 KTUNROLL），在限时内推导出事实；
  D3 If 合流：scf.if 两支各自产生 tensor 后再 insert_slice，推导出事实。
语义不变性由 tensorjit（T/T2/T3）与 valimport（A2/A3）的 JIT 对照负责。
"""
import sys, z3
import mlir.ir as ir
import core as C, irimport as I, verdict as V
from compcorpus_test import _funcs
from valimport_test import derive_imported

def _unused_lam_depth(e):   # 第 48 轮试过的“Lambda 嵌套深度”度量：对旧实现也给 1，不灵敏，弃用（保留作记录）
    memo = {}
    def d(x):
        k = x.get_id()
        if k in memo: return memo[k]
        if z3.is_quantifier(x): r = (1 if x.is_lambda() else 0) + d(x.body())
        elif z3.is_app(x): r = max([d(c) for c in x.children()], default=0)
        else: r = 0
        memo[k] = r; return r
    return d(e)

def derive_src(src, tag):
    m = ir.Module.parse(src, ir.Context()); root = I.import_module(m)
    fn = next(iter(_funcs(root)))
    return derive_imported(tag, fn)

def chain(N):
    body, cur = [], '%t'
    for k in range(N):
        body.append(f'  %r{k} = tensor.insert_slice %s into {cur}[{k % 2}, 0] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<?x?xi32>'); cur = f'%r{k}'
    return 'func.func @f(%t: tensor<?x?xi32>, %s: tensor<2x2xi32>) -> tensor<?x?xi32> {\n' + '\n'.join(body) + f'\n  return {cur} : tensor<?x?xi32>\n}}'

def loop(T):
    return f'''func.func @g(%t: tensor<?x?xi32>, %t2: tensor<10x10xi32>) -> tensor<?x?xi32> {{
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cT = arith.constant {T} : index
  %0 = scf.for %i = %c0 to %cT step %c1 iter_args(%a = %t) -> (tensor<?x?xi32>) {{
    %2 = tensor.insert_slice %t2 into %a[0, 0] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<?x?xi32>
    %3 = tensor.insert_slice %t2 into %2[1, 1] [10, 10] [1, 1] : tensor<10x10xi32> into tensor<?x?xi32>
    scf.yield %3 : tensor<?x?xi32>
  }}
  return %0 : tensor<?x?xi32>
}}'''

IFSRC = '''func.func @h(%t: tensor<?x?xi32>, %s: tensor<2x2xi32>, %c: i1) -> tensor<?x?xi32> {
  %r = scf.if %c -> (tensor<?x?xi32>) {
    %a = tensor.insert_slice %s into %t[0, 0] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<?x?xi32>
    %b = tensor.insert_slice %s into %a[1, 0] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<?x?xi32>
    scf.yield %b : tensor<?x?xi32>
  } else {
    %a = tensor.insert_slice %s into %t[0, 1] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<?x?xi32>
    scf.yield %a : tensor<?x?xi32>
  }
  %z = tensor.insert_slice %s into %r[0, 0] [2, 2] [1, 1] : tensor<2x2xi32> into tensor<?x?xi32>
  return %z : tensor<?x?xi32>
}'''

def timed(src, tag, kt=None, limit=90):
    """在 fork 子进程里推导，返回 (秒, 结果) 或 ('超时', None)。限时只是为了不挂起；判据是“链长 40 / 64 趟能在限时内完成”，
    旧实现（第 47 轮前）直线 14 个 insert_slice 已 > 25 s，10 趟循环 > 20 分钟，指数与多项式之间相差若干数量级，计时判据稳健。"""
    import multiprocessing as mp, time
    def child(q):
        if kt is not None: C.KTUNROLL = kt
        t = time.time(); S, why = derive_src(src, tag); q.put((round(time.time() - t, 2), '事实' if S is not None else why[:60]))
    ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=child, args=(q,)); p.start()
    try: r = q.get(timeout=limit)
    except Exception: r = ('超时', None)
    p.kill(); p.join(); return r

if __name__ == '__main__':
    LIM = 90
    o1 = {f'N={N}': timed(chain(N), f'dc_chain{N}') for N in (8, 40)}
    print('── D1 直线链（秒, 结果）:', o1)
    V.check('D1 直线链', all(r[1] == '事实' for r in o1.values()), str(o1))
    o2 = {f'trip={T}': timed(loop(T), f'dc_loop{T}', kt=10 ** 9) for T in (10, 64)}
    print('── D2 循环展开（KTUNROLL 关闭）:', o2)
    V.check('D2 循环展开', all(r[1] == '事实' for r in o2.values()), str(o2))
    o3 = timed(IFSRC, 'dc_if'); print('── D3 If 合流:', o3)
    V.check('D3 If 合流', o3[1] == '事实', str(o3))
    V.finish()
