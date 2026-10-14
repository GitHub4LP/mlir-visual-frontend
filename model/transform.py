"""变换层 v0：pattern、fold、终止性，以及对 MLIR canonicalize 的翻译确认。

  pattern = ⟨变量及其类型, 参数, 左边模板, 前提, 右边构造⟩
  定义期检查（一次，对全部参数符号地）：前提 ⇒ refines(左, 右)；有效应时降为 D（值层精化不足以说明删改效应）
  局部 → 整体：精化是前同余（替换子项为其精化，整个程序被精化）—— 随机宿主检验
  fold 由 Sem 派生：输入全为常量、不 UB、结果不是毒值 ⇒ 折叠为常量。所以任何 op（包括复合 op）都有 fold
  终止性：度量 Σ_节点 树大小；每个 pattern 右边的非变量部分更小，且每个变量出现次数不增 ⇒ 每步严格下降
检验见 model/transform_test.py。
"""
import itertools, z3
from dataclasses import dataclass, field
import core as C
from core import blk, node, I3

@dataclass
class Pattern:
    name: str
    vars: dict                # 变量 → 类型
    params: list              # 常量/谓词参数名（值是 i3 位模式或谓词编号）
    lhs: tuple                # 模板：变量名 | ('const', 参数, 类型) | (op, {参数: 值或 '$参数'}, [子模板])
    rhs: object               # (变量 → 值 id, 参数 → 值, 取新名) -> (新节点, 结果值)
    pre: object = None        # 参数 → z3 Bool
    out: str = I3

# ───────────────────────── 定义期检查
TY = [I3]          # 检查时的位宽：pattern 按 i3 书写，check(pt, 'i32') 把 i3 换成 i32（Alive 式的逐位宽有界验证）
def ty(t): return TY[0] if t == I3 else t
def _build_lhs(t, env, P, ops, fresh):
    if isinstance(t, str): return env[t]
    if t[0] == 'const':
        v = fresh(); ops.append(C.cst(v, P[t[1]], ty(t[2]))); return v
    op, pr, kids = t
    ins = [_build_lhs(k, env, P, ops, fresh) for k in kids]
    v = fresh(); props = {k: (P[x[1:]] if isinstance(x, str) and x.startswith('$') else x) for k, x in pr.items()}
    oty = 'i1' if op == 'arith.cmpi' else TY[0]
    ops.append(node(v, op, ins=ins, outs=[(v, oty)], props=props)); return v

def _fresh(pfx):
    c = itertools.count(); return lambda: f'{pfx}{next(c)}'

def param_domain(p, P):
    """常量参数取 i3 位模式；谓词参数（k…）取 0..9（由 predicate 的种类定义给出，见 defs.py）；标志参数（f…）是布尔。"""
    if p.startswith('f'): return z3.BoolVal(True)
    if not p.startswith('k'): return z3.BoolVal(True)
    return z3.And(P[p] >= 0, P[p] <= 9)

def check(pt, width=I3):
    """返回 True / False / None（None = 有效应，降为 D）。"""
    TY[0] = width
    try: return _check(pt)
    finally: TY[0] = I3
def _check(pt):
    # 常量参数用该位宽的位向量（避免 ℤ↔位向量混合，那会让 z3 在 i8 以上卡住）；谓词参数用 ℤ；标志参数用布尔
    mk = lambda p: z3.Bool(f'{pt.name}.{p}') if p.startswith('f') else z3.Int(f'{pt.name}.{p}') if p.startswith('k') else z3.BitVec(f'{pt.name}.{p}', C.W(TY[0]))
    P = {p: mk(p) for p in pt.params}
    ins = [(v, ty(t)) for v, t in pt.vars.items()]; env = {v: v for v in pt.vars}
    lops = []; lr = _build_lhs(pt.lhs, env, P, lops, _fresh('l'))
    rops, rr = pt.rhs(env, P, _fresh('r'))
    L = C.derive(f'{pt.name}.L', ins, [('q', ty(pt.out))], [blk('entry', lops + [C.ret('lr', lr)], args=ins)])
    R = C.derive(f'{pt.name}.R', ins, [('q', ty(pt.out))], [blk('entry', rops + [C.ret('rr', rr)], args=ins)])
    pre = z3.And([param_domain(p, P) for p in pt.params] + ([pt.pre(P)] if pt.pre else []))
    return C.refines(L, R, pre)

# ───────────────────────── 匹配与改写
def defs_of(blocks, acc=None):
    acc = {} if acc is None else acc
    for b in blocks:
        for n in b['ops']:
            for o, _ in n['outs']: acc[o] = n
            for r in n['regions']: defs_of(r, acc)
    return acc

def match(t, v, D, env, P):
    if isinstance(t, str):
        if t in env: return env[t] == v
        env[t] = v; return True
    n = D.get(v)
    if n is None: return False
    if t[0] == 'const':
        if n['op'] != 'arith.constant': return False
        v = n['props']['value'] % (1 << C.W(n['props'].get('type', I3)))
        if t[1] in P and P[t[1]] != v: return False          # 同名参数须取同一个值
        P[t[1]] = v; return True
    op, pr, kids = t
    if n['op'] != op or len(n['ins']) != len(kids): return False
    for k, x in pr.items():
        have = n.get('props', {}).get(k, False if k == 'nuw' else None)
        if isinstance(x, str) and x.startswith('$'): P[x[1:]] = have
        elif have != x: return False
    return all(match(k, a, D, env, P) for k, a in zip(kids, n['ins']))

def sites(pt, blocks):
    """全部匹配点：(根节点 id, 变量绑定, 参数绑定)，前提已按具体参数求值。"""
    D = defs_of(blocks); out = []
    for v, n in D.items():
        env, P = {}, {}
        if n['outs'] and v == n['outs'][0][0] and match(pt.lhs, v, D, env, P):
            if pt.pre is None or z3.is_true(z3.simplify(pt.pre({k: z3.BoolVal(x) if isinstance(x, bool) else z3.IntVal(x) for k, x in P.items()}))):
                out.append((n['id'], env, P))
    return out

def rewrite(blocks, pt, site, tag):
    """把根节点换成右边的节点，根的结果的所有使用改名为右边的结果。内部节点保留（若已无使用，交给 DCE）。"""
    rid, env, P = site
    rops, rr = pt.rhs(env, P, _fresh(f'{tag}_'))
    root_out = [None]
    def rw(bs):
        res = []
        for b in bs:
            ops = []
            for n in b['ops']:
                if n['id'] == rid: root_out[0] = n['outs'][0][0]; ops += rops; continue
                ops.append(dict(n, regions=[rw(r) for r in n['regions']]))
            res.append(dict(b, ops=ops))
        return res
    nb = rw(blocks)
    def ren(bs):
        return [dict(b, ops=[dict(n, ins=[rr if x == root_out[0] else x for x in n['ins']], regions=[ren(r) for r in n['regions']])
                             for n in b['ops']]) for b in bs]
    return ren(nb)

def dce(blocks):
    """删除结果无使用、且派生为“无效应”的节点（不要求不 UB：删掉 UB 是精化）。"""
    while True:
        used = set()
        def u(bs):
            for b in bs:
                for n in b['ops']: used.update(n['ins']); [u(r) for r in n['regions']]
        u(blocks); changed = [False]
        def d(bs):
            out = []
            for b in bs:
                ops = []
                for n in b['ops']:
                    sch = C.REG.get(n['op'])
                    dead = n['outs'] and all(o not in used for o, _ in n['outs']) and sch is not None and not sch.eff and not n['regions']
                    if dead: changed[0] = True; continue
                    ops.append(dict(n, regions=[d(r) for r in n['regions']]))
                out.append(dict(b, ops=ops))
            return out
        blocks = d(blocks)
        if not changed[0]: return blocks

# ───────────────────────── 终止性
def tree_size(t):
    """(非常量 op 个数, 溢出标志个数)，按字典序比较。"""
    if isinstance(t, str) or t[0] == 'const': return (0, 0)
    ks = [tree_size(k) for k in t[2]]
    return (1 + sum(a for a, _ in ks), int(t[1].get('nuw') is True) + sum(b for _, b in ks))
def var_occ(t, acc=None):
    acc = {} if acc is None else acc
    if isinstance(t, str): acc[t] = acc.get(t, 0) + 1
    elif t[0] != 'const':
        for k in t[2]: var_occ(k, acc)
    return acc
def rhs_shape(pt):
    """右边构造成的树（用符号参数），用于静态度量比较。"""
    P = {p: (z3.Bool if p.startswith('f') else z3.Int)(p) for p in pt.params}
    ops, r = pt.rhs({v: v for v in pt.vars}, P, _fresh('s'))
    D = {n['outs'][0][0]: n for n in ops if n['outs']}
    def tr(v):
        if v in pt.vars: return v
        n = D[v]
        if n['op'] == 'arith.constant': return ('const', '_', '')
        return (n['op'], {'nuw': n.get('props', {}).get('nuw', False)}, [tr(x) for x in n['ins']])
    return tr(r)
def decreasing(pt):
    r = rhs_shape(pt); lo, ro = var_occ(pt.lhs), var_occ(r)
    return tree_size(r) < tree_size(pt.lhs) and all(ro.get(v, 0) <= lo.get(v, 0) for v in ro)

def measure(blocks):
    """(Σ_节点 展开树的非常量 op 个数, Σ_节点 展开树的标志个数)，字典序；在 DAG 上记忆化计算。"""
    D = defs_of(blocks); memo = {}
    def ts(v):
        if v not in D: return (0, 0)
        if v in memo: return memo[v]
        n = D[v]
        if n['op'] == 'arith.constant': memo[v] = (0, 0); return memo[v]
        ks = [ts(x) for x in n['ins']]
        memo[v] = (1 + sum(a for a, _ in ks), int(n.get('props', {}).get('nuw') is True) + sum(b for _, b in ks))
        return memo[v]
    xs = [ts(v) for v in D]
    return (sum(a for a, _ in xs), sum(b for _, b in xs))

# ───────────────────────── fold 由 Sem 派生
def fold(op, consts, props=None):
    """所有输入为常量时：一定不 UB 且结果一定不是毒值 ⇒ 常量；一定是毒值 ⇒ 'poison'（折叠为 ub.poison）；
    可能 UB ⇒ 不折叠（None）。MLIR 把毒值结果折叠成回绕值，那也是精化，但丢掉了“这里是毒值”这一事实。"""
    sch = C.REG[op]
    args = [(C.bv(c, t), C.F_) for c, (_, t) in zip(consts, sch.ins)]
    outs, ub = C.instantiate(sch, props or {}, args) if C.is_comp(sch) else sch.fn(props or {}, args)
    if not C.prove(z3.Not(ub)): return None
    if C.prove(outs[0][1]): return 'poison'
    if not C.prove(z3.Not(outs[0][1])): return None
    return z3.simplify(outs[0][0]).as_long()

# ───────────────────────── 规则库（arith 的一部分规范化，以及几条错误规则做对照）
def K(i, v, t=None): return C.cst(i, v, ty(t or I3))
def R(i, op, a, b, **p): return C.op2(i, op, a, b, **p)
PATTERNS = [
    Pattern('add0', {'x': I3}, ['c', 'f'], ('arith.addi', {'nuw': '$f'}, ['x', ('const', 'c', I3)]),
            lambda e, P, f: ([], e['x']), pre=lambda P: P['c'] == 0),
    Pattern('sub_self', {'x': I3}, ['f'], ('arith.subi', {'nuw': '$f'}, ['x', 'x']),
            lambda e, P, f: ((lambda z: ([K(z, 0)], z))(f()))),
    Pattern('sel_same', {'c': 'i1', 'x': I3}, [], ('arith.select', {}, ['c', 'x', 'x']), lambda e, P, f: ([], e['x'])),
    Pattern('div1', {'x': I3}, ['c'], ('arith.divui', {}, ['x', ('const', 'c', I3)]),
            lambda e, P, f: ([], e['x']), pre=lambda P: P['c'] == 1),
    Pattern('div_self', {'x': I3}, [], ('arith.divui', {}, ['x', 'x']),
            lambda e, P, f: ((lambda z: ([K(z, 1)], z))(f()))),
    Pattern('cmp_self_t', {'x': I3}, ['k'], ('arith.cmpi', {'predicate': '$k'}, ['x', 'x']),
            lambda e, P, f: ((lambda z: ([K(z, 1, 'i1')], z))(f())), pre=lambda P: z3.Or([P['k'] == i for i in (0, 3, 5, 7, 9)]), out='i1'),
    Pattern('cmp_self_f', {'x': I3}, ['k'], ('arith.cmpi', {'predicate': '$k'}, ['x', 'x']),
            lambda e, P, f: ((lambda z: ([K(z, 0, 'i1')], z))(f())), pre=lambda P: z3.Or([P['k'] == i for i in (1, 2, 4, 6, 8)]), out='i1'),
    Pattern('add_add', {'x': I3}, ['c1', 'c2'],
            ('arith.addi', {'nuw': False}, [('arith.addi', {'nuw': False}, ['x', ('const', 'c1', I3)]), ('const', 'c2', I3)]),
            lambda e, P, f: (lambda k, s: ([K(k, P['c1'] + P['c2']), R(s, 'arith.addi', e['x'], k)], s))(f(), f())),
    Pattern('add_add_nuw', {'x': I3}, ['c1', 'c2'],
            ('arith.addi', {'nuw': True}, [('arith.addi', {'nuw': True}, ['x', ('const', 'c1', I3)]), ('const', 'c2', I3)]),
            lambda e, P, f: (lambda k, s: ([K(k, P['c1'] + P['c2']), R(s, 'arith.addi', e['x'], k, nuw=True)], s))(f(), f())),
    Pattern('add_add_mixed', {'x': I3}, ['c1', 'c2'],     # 标志取交集：内层 nuw、外层回绕 ⇒ 合并后不带标志
            ('arith.addi', {'nuw': False}, [('arith.addi', {'nuw': True}, ['x', ('const', 'c1', I3)]), ('const', 'c2', I3)]),
            lambda e, P, f: (lambda k, s: ([K(k, P['c1'] + P['c2']), R(s, 'arith.addi', e['x'], k)], s))(f(), f())),
]
EXTRA = [   # 有效、度量下降，但丢信息（之后的 nuw 推理用不上了），不放进规范化集合
    Pattern('drop_nuw', {'x': I3, 'y': I3}, [], ('arith.addi', {'nuw': True}, ['x', 'y']),
            lambda e, P, f: (lambda s: ([R(s, 'arith.addi', e['x'], e['y'])], s))(f())),
]
WRONG = [   # 对照：看起来合理，实际无效
    Pattern('✗add_add_mixed', {'x': I3}, ['c1', 'c2'],     # 内层 nuw、外层回绕，合并后保留 nuw
            ('arith.addi', {'nuw': False}, [('arith.addi', {'nuw': True}, ['x', ('const', 'c1', I3)]), ('const', 'c2', I3)]),
            lambda e, P, f: (lambda k, s: ([K(k, P['c1'] + P['c2']), R(s, 'arith.addi', e['x'], k, nuw=True)], s))(f(), f())),
    Pattern('✗add_nuw', {'x': I3, 'y': I3}, [], ('arith.addi', {'nuw': False}, ['x', 'y']),     # 添加 nuw
            lambda e, P, f: (lambda s: ([R(s, 'arith.addi', e['x'], e['y'], nuw=True)], s))(f())),
    Pattern('✗sel_true', {'c': 'i1', 'x': I3, 'y': I3}, [], ('arith.select', {}, ['c', 'x', 'y']),   # 选择恒取真分支
            lambda e, P, f: ([], e['x'])),
    Pattern('✗mul_div', {'x': I3}, ['c'],                    # (x * c) / c → x（回绕乘法下不成立）
            ('arith.divui', {}, [('arith.muli', {'nuw': False}, ['x', ('const', 'c', I3)]), ('const', 'c', I3)]),
            lambda e, P, f: ([], e['x']), pre=lambda P: P['c'] != 0),
    Pattern('✗sub_self_dup', {'x': I3}, [], ('arith.subi', {'nuw': False}, ['x', 'x']),   # 正确，但不减小度量（复制变量）
            lambda e, P, f: (lambda a, s: ([R(a, 'arith.addi', e['x'], e['x']), R(s, 'arith.subi', a, a)], s))(f(), f())),
]
