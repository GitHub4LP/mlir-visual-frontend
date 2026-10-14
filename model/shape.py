"""推导原型 v1：类型结构用合一（复用 solver.py），维度/整数用 Presburger（isl）。
模型：每个维度都是运行期 extent 变量（≥0），静态类型中的 ? 只是投影；
复合 op 的 Φ = ∃内部变量. ∧约束，用 isl 做投影；静态维度从 Φ 反推（Φ ⊨ e=c）。
"""
import re, islpy as isl
from solver import V, C, Store, solve

class G:
    def __init__(s): s.st, s.cons = Store(), []
    def ext(s, h='e'):                       # 新鲜 extent
        v = V('dim', h); s.c('{0} >= 0', v); return v
    def c(s, fmt, *vs): s.cons.append((fmt, vs))
    def tensor(s, rank, elt=None, h='e'):
        return C('rtensor', *[s.ext(h) for _ in range(rank)], elt or V('type', 'T'))
    def dims(s, t):
        t = s.st.find(t); return t.args[:-1]

# ---- op 语义（只依赖 Schema 事实；来源见注释）
def extract_slice(g, src, off, size, stride=None):
    """TensorOps.cpp ExtractSliceOp::verify + verifyInBoundsSlice（ViewLikeInterface.cpp）。
    MLIR 只在静态时检查：off < dim、off+(size-1)*stride < dim。负 offset、stride≥1 未被检查，
    这里按语义要求加入（标为 [语义]）。"""
    r = len(off); stride = stride or [1] * r
    s = g.tensor(r); g.st.unify(src, s, 'extract_slice.src')
    T = g.st.find(s).args[-1]
    for d, o, n, k in zip(g.dims(s), off, size, stride):
        g.c('{0} >= 0', o); g.c('{0} >= 0', n)                     # [语义]
        g.c('{0} < {1}', o, d)                                      # MLIR 原样（即使 size=0）
        if isinstance(k, int): g.c(f'{{0}} + ({{1}} - 1) * {k} < {{2}}', o, n, d)
        else: g.c('{0} >= 1', k); g.c('{0} + {1} - 1 < {2}', o, n, d)  # 非线性 → 只保留可靠弱化（S）
    return C('rtensor', *size, T)

def pad(g, src, low, high):
    """TensorOps.td：result dim = low + dim + high。MLIR 未规定 low/high 的符号，这里取 ≥0 [语义]。"""
    r = len(low); s = g.tensor(r); g.st.unify(src, s, 'pad.src')
    out = g.tensor(r)
    for d, l, h, e in zip(g.dims(s), low, high, g.dims(out)):
        g.c('{0} >= 0', l); g.c('{0} >= 0', h); g.c('{0} = {1} + {2} + {3}', e, l, d, h)
    g.st.unify(g.st.find(out).args[-1], g.st.find(s).args[-1], 'pad.elt')
    return out

def linalg_from_maps(maps):
    """从 indexing_maps（数据）导入形状约束：operand_i 的第 j 维 = loop[map_i(j)]，属于 E。"""
    parsed = [re.match(r'\((.*)\)\s*->\s*\((.*)\)', m).groups() for m in maps]
    loops = [x.strip() for x in parsed[0][0].split(',')]
    def op(g, *operands):
        L = {l: g.ext(l) for l in loops}
        for t, (_, res) in zip(operands, parsed):
            exprs = [x.strip() for x in res.split(',')] if res.strip() else []
            g.st.unify(t, C('rtensor', *[L[x] for x in exprs], V('type', 'T')), 'linalg.operand')
        return operands[-1]                     # DPS：结果类型 = outs 类型
    return op
matmul = linalg_from_maps(['(m,n,k)->(m,k)', '(m,n,k)->(k,n)', '(m,n,k)->(m,n)'])

# ---- 推导：∃ 投影 + 静态投影
def derive(build, inputs, ints=()):
    """build(g, *边界项) -> 输出项；inputs: [(名字, rank)]；ints: 边界整数参数（offset/size 等）。"""
    g = G()
    bi = {n: g.tensor(r, h=n) for n, r in inputs}
    bp = {n: V('dim', n) for n in ints}
    out = build(g, *bi.values(), *bp.values()); solve(g.st)
    name = {}
    def nm(v):
        v = g.st.find(v)
        if isinstance(v, int): return str(v)
        return name.setdefault(v, f'{v.hint}{len(name)}' if v.hint not in bp else v.hint)
    for p, v in bp.items(): name[g.st.find(v)] = p
    bnd = []                                     # 边界变量 = 边界项中的 extent + 整数参数
    for t in [*bi.values(), out]:
        for d in g.dims(t):
            if not isinstance(g.st.find(d), int) and nm(d) not in bnd: bnd.append(nm(d))
    for p in bp: bnd.append(p) if p not in bnd else None
    body = [f.format(*map(nm, vs)) for f, vs in g.cons]
    allv = list(dict.fromkeys(bnd + sorted({x for b in body for x in re.findall(r'[A-Za-z_]\w*', b)})))
    S = isl.Set('{ [%s] : %s }' % (', '.join(allv), ' and '.join(body)))
    S = S.project_out(isl.dim_type.set, len(bnd), len(allv) - len(bnd))   # ∃ 消去
    for i, v in enumerate(bnd): S = S.set_dim_name(isl.dim_type.set, i, v)
    S = S.coalesce().remove_redundancies()
    sig = {n: [nm(d) for d in g.dims(t)] for n, t in bi.items()}
    sig['→'] = [nm(d) for d in g.dims(out)]
    return S, sig, bnd

def fixed(S, bnd, v):                            # Φ ⊨ v = c ?
    i = bnd.index(v); P = S.project_out(isl.dim_type.set, i + 1, len(bnd) - i - 1).project_out(isl.dim_type.set, 0, i)
    if P.is_empty(): return 'UNSAT'
    return P.sample_point().get_coordinate_val(isl.dim_type.set, 0).to_python() if P.is_singleton() else '?'

def report(title, S, sig, bnd):
    print(f'■ {title}')
    print('  签名:', ', '.join(f'{k}:[{"×".join(v)}]' for k, v in sig.items()))
    print('  Φ   :', 'UNSAT（静态错误）' if S.is_empty() else S)
    if not S.is_empty():
        print('  静态投影:', {v: fixed(S, bnd, v) for v in sig['→'] if v in bnd})

if __name__ == '__main__':
    # 1 slice ∘ slice，并验证"融合为单个 slice"的改写合法：Φ_comp ⊆ Φ_fused[o=o1+o2]
    S, sig, bnd = derive(lambda g, x, o1, n1, o2, n2:
        extract_slice(g, extract_slice(g, x, [o1], [n1]), [o2], [n2]), [('x', 1)], ['o1', 'n1', 'o2', 'n2'])
    report('slice∘slice', S, sig, bnd)
    X = sig['x'][0]
    fused = isl.Set('{ [%s] : %s >= 0 and o1+o2 >= 0 and n2 >= 0 and o1+o2 < %s and o1+o2+n2-1 < %s }'
                    % (', '.join(bnd), X, X, X))      # 单个 slice x[o1+o2 : +n2] 的前提
    print('  融合改写合法 (Φ_comp ⊆ Φ_fused):', S.is_subset(fused))
    # 2 pad ∘ crop：MLIR 对动态输入只给 tensor<?>；这里推出 结果 extent = 输入 extent
    S, sig, bnd = derive(lambda g, x, l, h: extract_slice(g, pad(g, x, [l], [h]), [l], [g.st.find(x).args[0]]),
                         [('x', 1)], ['l', 'h'])
    report('crop(pad(x,l,h), l, |x|)', S, sig, bnd)
    print('  ⊨ SameOperandsAndResultShape:', sig['x'] == sig['→'])
    # 3 越界：静态 tensor<8> 上 slice[6:+4] → UNSAT；动态 tensor<?> 上 → 前提 x≥10（MLIR 两者都放行后者）
    S, sig, bnd = derive(lambda g, x: (g.st.unify(g.st.find(x).args[0], 8, 'static'),
                         extract_slice(g, x, [6], [4]))[1], [('x', 1)])
    report('static tensor<8>[6:+4]', S, sig, bnd)
    S, sig, bnd = derive(lambda g, x: extract_slice(g, x, [6], [4]), [('x', 1)])
    report('dynamic tensor<?>[6:+4]', S, sig, bnd)
    # 4 matmul 从 indexing_maps 导入，再接 slice：维度关系穿过 ∃ 保留
    S, sig, bnd = derive(lambda g, A, B, C0, r: extract_slice(g, matmul(g, A, B, C0), [0, 0], [r, r]),
                         [('A', 2), ('B', 2), ('C0', 2)], ['r'])
    report('slice(matmul(A,B,C0), r×r)', S, sig, bnd)
