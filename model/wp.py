"""推导原型 v2：三谓词 + 正确的量词 + 控制层（scf.if / scf.for）。
  Sem：结果关系（运行期值、extent 之间）   Def：无 UB 条件   Wf：MLIR verifier 静态条件
复合 op 的边界前提（最弱前提）：
  Pre(b) = ¬∃r. Sem(b,r) ∧ ¬Def(b,r)     —— 对运行期内部量取 ∀（不是 ∃！）
  Post(b,o) = ∃r. Sem ∧ Def
类型/静态变量仍然用 ∃（由合一负责，见 solver.py）。只有整数层放在这里。
"""
import islpy as isl, itertools
_n = itertools.count()

class K:
    def __init__(s): s.vars, s.sem, s.dfn = [], [], []
    def v(s, h):
        x = f'{h}_{next(_n)}'; s.vars.append(x); return x
    def tensor(s, h, rank): return [s.v(h) for _ in range(rank)]

# ---- 原语（每个都给出 Sem / Def；Wf 在 wf_lemma 中按 op 单独检查）
def extract_slice(k, src, off, size, stride=1):
    out = []
    for d, o, n in zip(src, off, size):
        k.dfn += [f'{o} >= 0', f'{n} >= 0', f'{o} + ({n} - 1) * {stride} < {d} or {n} = 0', f'{o} <= {d}']
        e = k.v('s'); k.sem.append(f'{e} = {n}'); out.append(e)
    return out
def pad(k, src, low, high):
    out = []
    for d, l, h in zip(src, low, high):
        k.dfn += [f'{l} >= 0', f'{h} >= 0']
        e = k.v('p'); k.sem.append(f'{e} = {l} + {d} + {h}'); out.append(e)
    return out
def elementwise(k, a, b):                      # arith.addf 在张量上：SameOperandsAndResultType
    k.dfn += [f'{x} = {y}' for x, y in zip(a, b)]  # 静态时由 Wf 拒绝；动态时属于 UB → 进入 Def
    out = k.tensor('w', len(a)); k.sem += [f'{o} = {x}' for o, x in zip(out, a)]; return out
def subi(k, a, b): r = k.v('i'); k.sem.append(f'{r} = {a} - {b}'); return r
def cmpi_slt(k, a, b): c = k.v('c'); k.sem.append(f'({c} = 1 and {a} < {b}) or ({c} = 0 and {a} >= {b})'); return c
def opaque_bool(k): c = k.v('c'); k.sem.append(f'0 <= {c} <= 1'); return c   # 依赖张量内容的条件
def scf_if(k, c, then_fn, else_fn):
    """RegionBranch：两支分别建模，约束按 c 守卫；结果 extent 为两支的析取。"""
    outs = []
    for flag, fn in ((1, then_fn), (0, else_fn)):
        sub = K(); sub.vars = k.vars; r = fn(sub)
        g = lambda x: f'({c} != {flag} or ({x}))'
        k.sem += [g(x) for x in sub.sem]; k.dfn += [g(x) for x in sub.dfn]; outs.append(r)
    res = k.tensor('if', len(outs[0]))
    k.sem += [f'(({c} = 1 and {r} = {a}) or ({c} = 0 and {r} = {b}))' for r, a, b in zip(res, *outs)]
    return res

def S(vars_, cons):
    return isl.Set('{ [%s] : %s }' % (', '.join(vars_), ' and '.join(f'({c})' for c in cons) or 'true'))

def derive(build, inputs, ints=()):
    k = K(); bi = {n: k.tensor(n, r) for n, r in inputs}; bp = {n: k.v(n) for n in ints}
    out = build(k, *bi.values(), *bp.values())
    bnd = [x for t in bi.values() for x in t] + list(bp.values())
    inner = [x for x in k.vars if x not in bnd and x not in out]
    order = bnd + out + inner
    nonneg = [f'{x} >= 0' for t in bi.values() for x in t]
    sem = S(order, k.sem + nonneg); dfn = S(order, k.dfn or ['true'])
    proj = lambda X, keep: X.project_out(isl.dim_type.set, keep, len(order) - keep)
    bad = proj(sem.subtract(dfn), len(bnd))                          # ∃r. Sem ∧ ¬Def
    pre = proj(S(order, nonneg), len(bnd)).subtract(bad).coalesce()  # ¬bad（在合法输入域内）
    post = proj(sem.intersect(dfn), len(bnd) + len(out)).coalesce()
    return pre, post

def wf_lemma():
    """逐 op 的导入期义务：Def ⇒ Wf（Wf 取 MLIR verifyInBoundsSlice 在各种静态组合下的检查）。"""
    d, o, n = 'd', 'o', 'n'
    Def = S([d, o, n], [f'{o} >= 0', f'{n} >= 0', f'{o} + {n} - 1 < {d} or {n} = 0', f'{o} <= {d}', f'{d} >= 0'])
    cases = {'d,o,n 全静态': [f'{o} < {d}', f'{o} + {n} - 1 < {d}'],
             'd,o 静态，n 动态': [f'{o} < {d}']}
    for name, wf in cases.items():
        cex = Def.subtract(S([d, o, n], wf))
        print(f'  Def ⇒ Wf [{name}]:', cex.is_empty() or f'✗ 反例 {cex.sample_point()}')
    canon = Def.intersect(S([d, o, n], [f'{n} >= 1']))
    print('  加规范化（n=0 → tensor.empty）后 [全静态]:', canon.subtract(S([d, o, n], cases['d,o,n 全静态'])).is_empty())

def loop(body, r=1):
    """scf.for 的 C 层推导。body 是复合 op（carried extent e → f），t 为循环次数。
    Pre(t,x) = ¬∃k<t, f. R^k(x,f) ∧ ¬BodyPre(f)；Post(t,x,y) = R^t(x,y)。R^k 由 isl power 计算，并给出 exact 标志。"""
    pre_b, post_b = derive(body, [('e', r)])
    R = isl.Map(_as_map(post_b, r)).intersect_domain(pre_b.reset_space(isl.Map(_as_map(post_b, r)).domain().get_space()))
    P, exact = R.power()                                         # { [k] -> [[e] -> [f]] : k ≥ 1 }
    W = P.wrap().flatten()                                       # [k, e.., f..]
    ident = isl.Set('{ [k, %s, %s] : k = 0 and %s }' % (', '.join(f'e{i}' for i in range(r)), ', '.join(f'f{i}' for i in range(r)),
                    ' and '.join(f'e{i} = f{i} and e{i} >= 0' for i in range(r))))
    W = W.union(ident.reset_space(W.get_space()))
    A = W.insert_dims(isl.dim_type.set, 0, 1)                   # [t, k, e.., f..]
    K_ = isl.Set('{ [t, k, %s] : 0 <= k < t }' % ', '.join(f'z{i}' for i in range(2 * r))).reset_space(A.get_space())
    notpre = pre_b.complement().insert_dims(isl.dim_type.set, 0, 2 + r).reset_space(A.get_space())
    bad = A.intersect(K_).intersect(notpre).project_out(isl.dim_type.set, 2 + r, r).project_out(isl.dim_type.set, 1, 1)
    dom = isl.Set('{ [t, %s] : t >= 0 and %s }' % (', '.join(f'x{i}' for i in range(r)), ' and '.join(f'x{i} >= 0' for i in range(r))))
    pre = dom.subtract(bad.reset_space(dom.get_space())).coalesce()
    post = A.intersect(isl.Set('{ [t, k, %s] : k = t }' % ', '.join(f'z{i}' for i in range(2 * r))).reset_space(A.get_space())) \
            .project_out(isl.dim_type.set, 1, 1).intersect(pre.add_dims(isl.dim_type.set, r).reset_space(
            A.project_out(isl.dim_type.set, 1, 1).get_space())).coalesce()
    return pre, post, exact

# ---- isl 小工具：按位置重组维度（避免名字冲突）
def _as_map(st, r):
    s = str(st); body = s[s.index(':') + 1:] if ':' in s else ' }'
    head = s[s.index('['):s.index(']') + 1]
    vs = [x.strip() for x in head[1:-1].split(',')]
    return '{ [%s] -> [%s] :%s' % (', '.join(vs[:r]), ', '.join(vs[r:]), body)
def _rename(st, names):
    s = str(st); head = s[s.index('['):s.index(']') + 1]
    vs = [x.strip() for x in head[1:-1].split(',')]
    for a, b in zip(vs, names): s = s.replace(a, b)
    return s
def _lift(st, n, front=True):
    sp = st.get_space()
    if front: return st.insert_dims(isl.dim_type.set, 0, n)
    return st.add_dims(isl.dim_type.set, n)
def _union(a, b): return a.union(b.reset_space(a.get_space()))
def _align(a, b): return a.reset_space(b.get_space())

if __name__ == '__main__':
    print('■ 逐 op 引理：extract_slice 的 Def ⇒ Wf'); wf_lemma()

    pre, post = derive(lambda k, x, l, h: extract_slice(k, pad(k, x, [l], [h]), [l], x), [('x', 1)], ['l', 'h'])
    print('■ crop(pad(x,l,h))  Pre:', pre, '\n  Post:', post)

    pre, _ = derive(lambda k, x, y: elementwise(k, extract_slice(k, x, ['1'], [subi(k, x[0], '1')]), y),
                    [('x', 1), ('y', 1)])
    print('■ addf(x[1:], y)  Pre:', pre)

    # scf.if：条件由边界整数决定 vs 由张量内容决定（必须对两支都成立）
    def br(k, x, n, cond):
        c = cond(k, n)
        return scf_if(k, c, lambda s: extract_slice(s, x, ['0'], [n]), lambda s: extract_slice(s, x, ['0'], ['4']))
    pre, _ = derive(lambda k, x, n: br(k, x, n, lambda k, n: cmpi_slt(k, n, '4')), [('x', 1)], ['n'])
    print('■ if (n<4) x[0:+n] else x[0:+4]   Pre:', pre)
    pre, _ = derive(lambda k, x, n: br(k, x, n, lambda k, n: opaque_bool(k)), [('x', 1)], ['n'])
    print('■ if (data) x[0:+n] else x[0:+4]  Pre:', pre)
    # 对照：若错误地对运行期条件取 ∃（v1 的做法），会得到不可靠的弱前提
    k = K(); x = k.tensor('x', 1); n = k.v('n'); c = opaque_bool(k)
    scf_if(k, c, lambda s: extract_slice(s, x, ['0'], [n]), lambda s: extract_slice(s, x, ['0'], ['4']))
    allv = x + [n] + [v for v in k.vars if v not in x + [n]]
    wrong = S(allv, k.sem + k.dfn + [f'{x[0]} >= 0']).project_out(isl.dim_type.set, 2, len(allv) - 2).coalesce()
    print('  ✗ 若用 ∃ 消去:', wrong)

    # scf.for：每次迭代 x ← x[1:]，共 t 次；以及 extent 不变的 body（elementwise 自加）
    pre, post, ex = loop(lambda k, e: extract_slice(k, e, ['1'], [subi(k, e[0], '1')]))
    print('■ for t 次: x ← x[1:]      Pre(t,x):', pre, '\n  Post(t,x,y):', post, ' exact =', ex)
    pre, post, ex = loop(lambda k, e: elementwise(k, e, e))
    print('■ for t 次: x ← x + x      Pre(t,x):', pre, '\n  Post(t,x,y):', post, ' exact =', ex)
    pre, post, ex = loop(lambda k, e: extract_slice(k, pad(k, e, ['0'], ['1']), ['0'], [subi(k, e[0], '-1')]))
    print('■ for t 次: x ← pad(x,0,1)  Post(t,x,y):', post, ' exact =', ex)
