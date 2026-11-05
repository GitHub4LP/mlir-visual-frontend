"""第 77 轮：区域类型流（closure §9.17）。

区域 op 的块参数与结果类型不再取图上写定的值，而是沿控制流的“入口边”由已知类型定义：
  - RegionBranchOpInterface 的边（9bae84b0 SCF.cpp getSuccessorRegions/getEntrySuccessorOperands）：
    父 → 区域入口块参数、区域终结子 → 父结果。每个后继输入取一条确定的前驱边作为定义，其余边由 nf 规则作为约束检查。
  - LinalgOp 的体：第 k 个块参数 = 第 k 个操作数的元素类型（memref/有秩 tensor）或其本身（verifyStructuredOpInterface 的 bbArg 规则）。
MLIR 要求这些等式成立，所以任何被 MLIR 接受的类型指派都满足它们：用它们定义内部类型不丢掉任何可接受的端口类型（最大性延伸到区域），
而判定层不变（nf 规则照旧检查全部边），可靠性不受影响。odscomp.derive（符号）与 composite.instantiate（具体）共用这张表，保证 Φ ≡ 逐点。
写不进表的情况（区域/块个数不符、没有对应终结子）返回 None：保持写定类型。

源的写法：('opd', i) 父的第 i 个操作数；('term', ri, k) 区域 ri 入口块终结子的第 k 个操作数；('elt', src) 取元素类型。
"""
ENABLED = True

def elt(t):
    if t is None: return None
    if t[0] == 'memref' or (t[0] == 'tensor' and t[1] is not None): return t[2]
    return t

def _term(n, ri, name=None):
    rs = n.get('regions') or []
    if ri >= len(rs) or len(rs[ri]) != 1 or not rs[ri][0]['ops']: return None
    t = rs[ri][0]['ops'][-1]
    return t if name is None or t['op'] == name else None

def flow(n):
    """→ (pre, post)：pre[ri] = [(k, 源)] 走区域 ri 之前给入口块第 k 个参数；post = [(k, 源)] 走完区域后给第 k 个结果。None = 不建模。"""
    if not ENABLED: return None
    op, rs = n['op'], n.get('regions') or []
    nargs = lambda ri: len(rs[ri][0]['args']) if ri < len(rs) and len(rs[ri]) == 1 else None
    nres = len(n['outs'])
    if op == 'scf.for':                                   # 父 → 体：(lb, inits)；父 → 父（零次迭代）：inits → results
        if len(rs) != 1 or nargs(0) is None: return None
        ni = len(n['ins']) - 3
        if nargs(0) != 1 + ni or nres != ni: return None
        return ({0: [(0, ('opd', 0))] + [(1 + k, ('opd', 3 + k)) for k in range(ni)]}, [(k, ('opd', 3 + k)) for k in range(ni)])
    if op == 'scf.if':                                    # InferTypeOpInterface：results = then 的 yield
        y = _term(n, 0, 'scf.yield')
        if y is None or len(y['ins']) != nres: return None
        return ({}, [(k, ('term', 0, k)) for k in range(nres)])
    if op == 'scf.while':                                 # 父 → before：inits；condition → after、父：condition 的值
        c = _term(n, 0, 'scf.condition')
        if len(rs) != 2 or c is None or nargs(0) != len(n['ins']) or nargs(1) is None: return None
        nc = len(c['ins']) - 1
        if nargs(1) != nc or nres != nc: return None
        return ({0: [(k, ('opd', k)) for k in range(len(n['ins']))], 1: [(k, ('term', 0, 1 + k)) for k in range(nc)]},
                [(k, ('term', 0, 1 + k)) for k in range(nres)])
    if op in ('scf.execute_region', 'scf.index_switch'):  # 区域 → 父：取第一个区域入口块的 yield
        y = _term(n, 0, 'scf.yield')
        if y is None or len(y['ins']) != nres: return None
        return ({}, [(k, ('term', 0, k)) for k in range(nres)])
    if len(rs) == 1 and nargs(0) is not None and nargs(0) <= len(n['ins']) and _is_linalg(op):
        return ({0: [(k, ('elt', ('opd', k))) for k in range(nargs(0))]}, [])
    return None

def _is_linalg(op):
    if not op.startswith('linalg.'): return False
    import nfrules as NF
    return NF._is_linalg(op)

def resolve(src, opd, term):
    """opd(i)、term(ri, k) 给出类型或表达式；elt 用 mk_elt 包装（具体：函数；符号：表达式）。"""
    if src[0] == 'opd': return opd(src[1])
    if src[0] == 'term': return term(src[1], src[2])
    raise ValueError(src)
