"""统一核心 v3 的整数层与效应层：
  每个值 v 带一个毒值条件 P_v；每个 op 给出 Sem、UB（立即 UB）、poison（产生毒值的条件）、对毒值敏感的输入、效应。
  复合 op：
    Pre(b)       = ¬∃r. Sem ∧ UB            UB 包含“毒值流入敏感输入”
    MayPoison_o  = ∃r. Sem ∧ Pre ∧ P_o      （对外报告：输出何时可能是毒值；这相当于推导出 nuw 一类的标志）
    Speculatable ⇔ Pre ≡ 合法输入域；Pure ⇔ Speculatable ∧ 效应 = ∅
  效应：并集；目标重定位到边界端口；局部的 alloc…free 互相抵消；未建模的效应为 ⊤。
"""
import islpy as isl, itertools
_n = itertools.count()

class Val:
    def __init__(s, var, poison='false'): s.var, s.poison = var, poison

class K:
    def __init__(s): s.vars, s.sem, s.ub, s.dom, s.eff = [], [], [], [], []
    def v(s, h, lo=None, hi=None):
        x = f'{h}_{next(_n)}'; s.vars.append(x)
        if lo is not None: s.dom.append(f'{lo} <= {x} <= {hi}')
        return x
    def sink(s, val): s.ub.append(val.poison)            # 毒值流入敏感输入 ⇒ UB

W = 255                                                  # i8，无符号视角
def addi(k, a, b, nuw=False):                            # ArithOps.td：带 nuw 时溢出的结果为毒值
    r = k.v('add', 0, W)
    k.sem.append(f'{r} = ({a.var} + {b.var}) mod {W + 1}')
    own = f'{a.var} + {b.var} > {W}' if nuw else 'false'
    return Val(r, f'({a.poison}) or ({b.poison}) or ({own})')
def divui(k, a, b):                                      # ArithOps.cpp：X/0 ⇒ UB；除数为毒值同样是 UB
    r = k.v('div', 0, W); k.sink(b)
    k.ub.append(f'{b.var} = 0')
    return Val(r, a.poison)                              # 商的精确语义为非线性，此处不需要
def extract_slice(k, d, o, n):                           # tensor.extract_slice 是 Pure：越界视为毒值，不是 UB
    r = k.v('s'); k.sem.append(f'{r} = {n.var}')
    return Val(r, f'({d.poison}) or ({o.poison}) or ({n.poison}) or {o.var} < 0 or {n.var} < 0 or {o.var} + {n.var} > {d.var}')
def extract_elem(k, t, i):                               # tensor.extract：越界或读到毒值 ⇒ UB
    k.ub.append(f'{i.var} < 0 or {i.var} >= {t.var}'); k.sink(t); k.sink(i)
    return Val(k.v('el', 0, W))

def S(vs, cs): return isl.Set('{ [%s] : %s }' % (', '.join(vs), ' and '.join(f'({c})' for c in cs) or 'true'))

def derive(build, ins):
    k = K(); bv = [Val(k.v(n, lo, hi)) for n, lo, hi in ins]
    outs = build(k, *bv); outs = outs if isinstance(outs, tuple) else (outs,)
    bnd = [b.var for b in bv]; rest = [x for x in k.vars if x not in bnd]
    order = bnd + rest; nb = len(bnd)
    dom = S(order, k.dom).project_out(isl.dim_type.set, nb, len(rest))
    sem = S(order, k.sem + k.dom)
    bad = sem.intersect(S(order, [' or '.join(f'({u})' for u in k.ub) or 'false'])).project_out(isl.dim_type.set, nb, len(rest))
    pre = dom.subtract(bad).coalesce()
    may = [sem.intersect(S(order, [o.poison])).project_out(isl.dim_type.set, nb, len(rest)).intersect(pre).coalesce() for o in outs]
    return pre, dom, may

def facts(name, build, ins):
    pre, dom, may = derive(build, ins)
    spec = pre.is_equal(dom)
    print(f'■ {name}\n  Pre: {pre}\n  Speculatable: {spec}' + ('' if spec else '（条件可推测：仅在 Pre 下）'))
    for i, m in enumerate(may): print(f'  out{i} 可能为毒值: {"永不" if m.is_empty() else m}')

# ---- 效应层
TOP = 'Unknown'
def compose_effects(nodes, boundary_of):
    """nodes: [(op, [(kind, resource, target)] 或 TOP)]；boundary_of: 值 → 边界端口，或 None（局部值）。
    规则：在复合 op 内部新分配的缓冲，返回之前外部不可能持有它的别名，所以对它的 Read/Write 外部不可见；
          它被局部释放且不逃逸时，Allocate 与 Free 互相抵消；逃逸时，Allocate 重定位到输出端口。"""
    fresh = {t for _, e in nodes if e != TOP for k, _, t in e if k == 'Allocate'}
    freed = {t for _, e in nodes if e != TOP for k, _, t in e if k == 'Free'}
    out = set()
    for op, effs in nodes:
        if effs == TOP: return TOP
        for kind, res, tgt in effs:
            if tgt in fresh:
                if kind in ('Read', 'Write'): continue
                if boundary_of.get(tgt) is None and tgt in freed: continue
            out.add((kind, res, boundary_of.get(tgt) or tgt))
    return out

if __name__ == '__main__':
    U8 = (0, W)
    facts('a / (a +nuw b)', lambda k, a, b: divui(k, a, addi(k, a, b, nuw=True)), [('a', *U8), ('b', *U8)])
    facts('a / (a + b)（回绕加法）', lambda k, a, b: divui(k, a, addi(k, a, b)), [('a', *U8), ('b', *U8)])
    facts('a +nuw b（直接返回）', lambda k, a, b: addi(k, a, b, nuw=True), [('a', *U8), ('b', *U8)])
    facts('slice(x,0,n)', lambda k, x, n: extract_slice(k, x, Val('0'), n), [('x', 0, 10**6), ('n', 0, 10**6)])
    facts('extract(slice(x,0,n), 0)', lambda k, x, n: extract_elem(k, extract_slice(k, x, Val('0'), n), Val('0')),
          [('x', 0, 10**6), ('n', 0, 10**6)])

    print('■ 效应组合')
    tmp = [('alloc', [('Allocate', 'Default', 't')]), ('copy', [('Read', 'Default', 'x'), ('Write', 'Default', 't')]),
           ('compute', [('Read', 'Default', 't'), ('Write', 'Default', 'y')]), ('dealloc', [('Free', 'Default', 't')])]
    print('  临时缓冲（局部 alloc/free）:', compose_effects(tmp, {'x': 'in0', 'y': 'in1', 't': None}))
    esc = tmp[:3]
    print('  缓冲逃逸为结果:', compose_effects(esc, {'x': 'in0', 'y': 'in1', 't': 'out0'}))
    print('  分配了但未释放、未逃逸（泄漏）:', compose_effects(esc, {'x': 'in0', 'y': 'in1', 't': None}))
    print('  含无接口 op:', compose_effects(tmp + [('ext.call', TOP)], {'x': 'in0', 'y': 'in1', 't': None}))
