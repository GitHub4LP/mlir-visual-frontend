"""统一核心 v4：一种 IR、一种 Schema、一次推导。

IR：structure.py 的 module/region/block/node 表示，是唯一的图表示（v3 的扁平 DAG 已废弃）。
Schema v4 = ⟨签名; 值层（每个输出的值与毒值标志、UB，都是“输入值 × 输入毒值标志”的函数）; Eff; Struct; 信任等级; Impl⟩
derive(体) 一次遍历同时给出 值层 / 效应层 / 结构层 的事实；结果仍是 Schema，可再作为节点使用。
值层是多 sort 的：iN 用 z3 位向量（任意位宽；i3 便于穷举预言机，见 core_test.py）；index 用数学整数 ℤ，
溢出出目标位宽 IW 的有符号范围即前端 UB（L17：MLIR 的回绕是它的精化；ℤ 上的线性部分可交给 Presburger）。
scf.for：静态小循环精确展开；其余用“按结构命名的未解释函数”做摘要（L18，确定性 ⇒ 嵌套 ≡ 内联仍成立），
可用声明的不变式收窄，不变式在定义期做归纳检查，通过后作为公理进入摘要（L19）。见 spec/loops.md。
运行：PYTHONPATH=model python3 model/core.py
"""
import re, itertools, z3
from dataclasses import dataclass, field
import structure as ST
from structure import blk, node

M = 8; MAX = M - 1
TOP = 'Unknown'                       # 未建模效应
_n = itertools.count()
def fv(h): return f'{re.sub(r"[^A-Za-z0-9]", "_", h)}_{next(_n)}'
IW = 64                               # 目标的 index 位宽（前端 UB 的界）
KUNROLL = 64                          # 静态循环精确展开的上限
def is_int(t): return isinstance(t, str) and re.fullmatch(r'i\d+', t) is not None
def hi(t): return (1 << W(t)) - 1

class ValueUnknown(Exception): pass

def W(t): return int(t[1:]) if is_int(t) else None
SORTS = {}                            # 用户类型 → z3 sort（由 defs.py 的类型定义登记）
def zsort(t):
    if is_int(t): return z3.BitVecSort(W(t))
    if t == 'index': return z3.IntSort()
    return SORTS.get(t)
def zb(x): return z3.BoolVal(x) if isinstance(x, bool) else x               # 参数可以是具体值或符号（复合 op 的参数）
def zbv(x, t='i3'):
    if t == 'index': return z3.IntVal(sx(x % (1 << IW), IW)) if isinstance(x, int) else x
    if isinstance(x, int): return z3.BitVecVal(x, W(t))
    return x if z3.is_bv(x) and x.size() == W(t) else z3.Int2BV(x, W(t))
def sx(v, w): return v - (1 << w) if v >= 1 << (w - 1) else v
def is_ix(x): return z3.is_int(x)
LO, HI = -(1 << (IW - 1)), 1 << (IW - 1)
def U(x): return z3.If(x < 0, x + (1 << IW), x)                             # index 的无符号解释
def S(x): return z3.If(x >= HI, x - (1 << IW), x)
def ovf(r): return z3.Or(r < LO, r >= HI)
def inrange(x): return z3.And(x >= LO, x < HI) if x is not None and z3.is_int(x) else T_   # index 值的良构：总在 IW 位有符号范围内
def outty(p, d='i3'): return (p.get('__out') or [p.get('type', d)])[0]
@dataclass
class Schema:
    name: str
    ins: list                         # [(端口, 类型)]
    outs: list
    fn: object = None                 # 值层：fn(props, [(值, 毒值)]) -> ([(值, 毒值)], UB)，全部是 z3 项
    eff: object = frozenset()         # {(kind, resource, 端口)} 或 TOP
    exact: bool = True
    trust: dict = field(default_factory=dict)
    value_known: bool = True
    # 复合 op
    body: list = None
    mode: str = 'transparent'
    summary: object = None            # 结构摘要（structure.StructSummary）
    struct_diag: list = field(default_factory=list)
    eff_diag: list = field(default_factory=list)
    holes: list = None                # 摘要里代表输入端口的占位项 [(值, 毒值)]
    outv: list = None                 # 摘要：每个输出的 (值, 毒值)
    ubf: object = None                # 摘要：UB 条件
    params: list = field(default_factory=list)   # 复合 op 的参数 [(名字, 'int'|'bool')]，摘要对它们是符号的
    pconsts: dict = field(default_factory=dict)
    ax: object = None                 # 摘要附带的公理（循环不变式在出口成立等），查询时作为假设

F_, T_ = z3.BoolVal(False), z3.BoolVal(True)
def bv(x, t='i3'): return zbv(x, t)
def anyp(a): return z3.Or([p for _, p in a]) if a else F_

# ───────────────────────── 原生 op（语义为人工覆盖层，结构取自 structure.STRUCT）
REG = {}
def reg(s): REG[s.name] = s; return s
def nat(name, ins, outs, fn, eff=frozenset()):
    return reg(Schema(name, ins, outs, fn, frozenset(eff), trust={'值层/效应': 'authored', '结构': 'ods'}))
I3 = 'i3'
def _ixub(r, px, py): return z3.And(z3.Not(px), z3.Not(py), ovf(r))      # index 溢出 ⇒ 前端 UB（L17）
def _add(p, a):
    (x, px), (y, py) = a
    nuw = zb(p.get('nuw', False))
    if is_ix(x): r = x + y; return [(r, z3.Or(px, py, z3.And(nuw, U(x) + U(y) >= 1 << IW)))], _ixub(r, px, py)
    return [(x + y, z3.Or(px, py, z3.And(nuw, z3.Not(z3.BVAddNoOverflow(x, y, False)))))], F_
def _sub(p, a):
    (x, px), (y, py) = a
    nuw = zb(p.get('nuw', False))
    if is_ix(x): r = x - y; return [(r, z3.Or(px, py, z3.And(nuw, U(x) < U(y))))], _ixub(r, px, py)
    return [(x - y, z3.Or(px, py, z3.And(nuw, z3.ULT(x, y))))], F_
def _div(p, a):
    (x, px), (y, py) = a
    if is_ix(x): return [(S(U(x) / U(y)), px)], z3.Or(py, y == 0)
    return [(z3.UDiv(x, y), px)], z3.Or(py, y == 0)               # 除数为 0 或毒值 ⇒ UB
PRED = ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge']   # arith::CmpIPredicate 的编号
def _cmp(p, a):
    (x, px), (y, py) = a
    if is_ix(x): rel = [x == y, x != y, x < y, x <= y, x > y, x >= y, U(x) < U(y), U(x) <= U(y), U(x) > U(y), U(x) >= U(y)]
    else: rel = [x == y, x != y, x < y, x <= y, x > y, x >= y, z3.ULT(x, y), z3.ULE(x, y), z3.UGT(x, y), z3.UGE(x, y)]
    k = p['predicate']
    c = rel[k] if isinstance(k, int) else z3.Or([z3.And(k == i, r) for i, r in enumerate(rel)])
    return [(z3.If(c, bv(1, 'i1'), bv(0, 'i1')), z3.Or(px, py))], F_
def _sel(p, a):
    (c, pc), (x, px), (y, py) = a
    return [(z3.If(c == 1, x, y), z3.Or(pc, z3.If(c == 1, px, py)))], F_
_pz = itertools.count()
def _poison(p, a):
    return [(z3.FreshConst(zsort(outty(p)), 'poison'), T_)], F_
nat('arith.constant', [], [('r', I3)], lambda p, a: ([(zbv(p['value'], p.get('type') or outty(p)), F_)], F_))
nat('arith.addi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _add)
nat('arith.subi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _sub)
nat('arith.divui', [('lhs', I3), ('rhs', I3)], [('r', I3)], _div)
nat('arith.cmpi', [('lhs', I3), ('rhs', I3)], [('r', 'i1')], _cmp)
nat('arith.select', [('c', 'i1'), ('a', I3), ('b', I3)], [('r', I3)], _sel)
nat('ub.poison', [], [('r', I3)], _poison)
def _bin(f, ovfc=None):
    def g(p, a):
        (x, px), (y, py) = a
        if is_ix(x):                                              # index 上的位运算：经 IW 位向量
            if ovfc is None: return [(z3.BV2Int(f(z3.Int2BV(x, IW), z3.Int2BV(y, IW)), True), z3.Or(px, py))], F_
            r = x * y                                             # muli
            return [(r, z3.Or(px, py, z3.And(zb(p.get('nuw', False)), U(x) * U(y) >= 1 << IW)))], _ixub(r, px, py)
        return [(f(x, y), z3.Or(px, py, z3.And(zb(p.get('nuw', False)), ovfc(x, y)) if ovfc else F_))], F_
    return g
nat('arith.muli', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x * y, lambda x, y: z3.Not(z3.BVMulNoOverflow(x, y, False))))
nat('arith.andi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x & y))
nat('arith.ori', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x | y))
nat('arith.xori', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x ^ y))
def _cast(kind):
    def g(p, a):
        (x, px), t = a[0], outty(p)
        if kind == 'trunci': return [(z3.Extract(W(t) - 1, 0, x), px)], F_
        if kind in ('extui', 'extsi'): return [((z3.ZeroExt if kind == 'extui' else z3.SignExt)(W(t) - x.size(), x), px)], F_
        if t == 'index':                                                              # iN → index：有/无符号扩展
            r = z3.BV2Int(x, kind == 'index_cast'); ORIG[r.get_id()] = (r, x, kind == 'index_cast'); return [(r, px)], F_
        o = ORIG.get(x.get_id())                # index → iN 直接作用在 iN → index 上：在位向量里化简（ℤ↔位向量混合是 z3 的弱点）
        if o is not None and o[0].eq(x):        # 按项本身比较（AST id 会被回收复用；表里留着引用也防止回收）
            _, y, sg = o; w = W(t)
            r = z3.Extract(w - 1, 0, y) if w <= y.size() else (z3.SignExt if sg else z3.ZeroExt)(w - y.size(), y)
            return [(r, px)], F_
        return [(z3.Int2BV(x, W(t)), px)], F_                                         # index → iN：截断
    return g
for k in ('trunci', 'extui', 'extsi', 'index_cast', 'index_castui'):
    nat(f'arith.{k}', [('in', 'T')], [('r', 'T2')], _cast(k))
ORIG = {}
_ch = itertools.count()
def _load(p, a): return [(z3.FreshConst(zsort(outty(p)), f'load{next(_ch)}'), F_)], F_   # 读到的内容：非确定（自由变量）
nat('memref.alloc', [], [('m', 'memref')], lambda p, a: ([(None, F_)], F_), eff={('Allocate', 'Default', 'm')})
nat('memref.dealloc', [('m', 'memref')], [], lambda p, a: ([], F_), eff={('Free', 'Default', 'm')})
nat('memref.load', [('m', 'memref')], [('r', I3)], _load, eff={('Read', 'Default', 'm')})
nat('memref.store', [('v', I3), ('m', 'memref')], [], lambda p, a: ([], F_), eff={('Write', 'Default', 'm')})

def instantiate(sch, props, args):
    """复合 op 的摘要实例化 = 项代换（嵌套 ≡ 内联的根据）。"""
    sub = _subst(sch, props, args)
    return [(sub(v) if v is not None else None, sub(p)) for v, p in sch.outv], sub(sch.ubf)
def inst_ax(sch, props, args): return _subst(sch, props, args)(sch.ax if sch.ax is not None else T_)
def _subst(sch, props, args):
    pairs = [(h, a) for hs, as_ in zip(sch.holes, args) for h, a in zip(hs, as_) if h is not None and a is not None]
    for nm, kind in sch.params:
        if nm in props:
            v = props[nm]
            pairs.append((sch.pconsts[nm], z3.IntVal(v) if isinstance(v, int) and not isinstance(v, bool) else zb(v)))
    return lambda t: z3.substitute(t, pairs) if pairs else t
def is_comp(sch): return sch.body is not None

# ───────────────────────── 推导：一次遍历
class Ctx:
    def __init__(s, params=None): s.env, s.ub, s.eff, s.exact, s.params, s.ax, s.diag = {}, F_, [], True, params or {}, [], []
    def props(s, n):
        p = {k: s.params[v[1:]] if isinstance(v, str) and v.startswith('$') else v for k, v in n.get('props', {}).items()}
        p['__out'] = [t for _, t in n['outs']]
        return p
    def walk(s, blocks, g, path):
        if len(blocks) != 1: raise ValueUnknown('多块 CFG 的值层尚未建模')
        for n in blocks[0]['ops']:
            op = n['op']
            if op in ('scf.yield', 'func.return'): return [s.env[v] for v in n['ins']]
            if op == 'scf.if':
                c, pc = s.env[n['ins'][0]]
                s.ub = z3.Or(s.ub, z3.And(g, pc))                       # 以毒值为条件分支 ⇒ UB
                t = s.walk(n['regions'][0], z3.And(g, c == 1), path + ((n['id'], 1),))
                e = s.walk(n['regions'][1], z3.And(g, c == 0), path + ((n['id'], 0),))
                for (o, _), (tv, tp), (ev, ep) in zip(n['outs'], t, e):
                    s.env[o] = (z3.If(c == 1, tv, ev), z3.Or(pc, z3.If(c == 1, tp, ep)))
                continue
            if op == 'scf.for': s.loop(n, g, path); continue
            sch = REG.get(op)
            if sch is None or not sch.value_known: raise ValueUnknown(op)
            args = [s.env.get(v, (None, F_)) for v in n['ins']]
            pr = s.props(n)
            if is_comp(sch) and not sch.exact and sch.mode == 'transparent' and sch.body is not None:
                outs, ub = s.specialize(sch, pr, args, g, path + ((n['id'], 'call'),))   # 透明模式：摘要不精确时在调用点特化（L18）
            elif is_comp(sch): outs, ub = instantiate(sch, pr, args); s.ax.append(inst_ax(sch, pr, args))
            else: outs, ub = sch.fn(pr, args)
            s.ub = z3.Or(s.ub, z3.And(g, ub))                           # 分支里的 UB 只在该路径上算数
            for (o, _), r in zip(n['outs'], outs): s.env[o] = r
            vids = {p: v for (p, _), v in zip(sch.ins, n['ins'])} | {p: o for (p, _), (o, _) in zip(sch.outs, n['outs'])}
            if sch.eff == TOP: s.eff.append(TOP)
            else: s.eff += [(k, r, vids[p], path, n['id']) for k, r, p in sch.eff]
            s.exact &= sch.exact
        raise ValueUnknown('块没有终结子')

    def specialize(s, sch, pr, args, g, path):
        """在调用点按实参重新推导体（独立的 Ctx：体内命名不会与宿主冲突）。UB 只在本路径上算数，由调用方以 g 守卫。"""
        sub = Ctx({nm: (z3.IntVal(pr[nm]) if isinstance(pr.get(nm), int) and not isinstance(pr.get(nm), bool) else zb(pr[nm])) if nm in pr else sch.pconsts[nm]
                   for nm, _ in sch.params})
        for (v, _), a in zip(sch.body[0]['args'], args): sub.env[v] = a
        ys = sub.walk(sch.body, T_, path)
        s.ax += sub.ax; s.diag += sub.diag; s.exact &= sub.exact
        return ys, sub.ub

    def loop(s, n, g, path):
        """scf.for。步长 ≤ 0 或边界是毒值 ⇒ UB。静态且趟数 ≤ KUNROLL ⇒ 精确展开；否则摘要（L18、L19）。"""
        (lb, plb), (ub, pub), (st, pst) = [s.env[v] for v in n['ins'][:3]]
        inits = [s.env[v] for v in n['ins'][3:]]
        s.ub = z3.Or(s.ub, z3.And(g, z3.Or(plb, pub, pst, st <= 0)))
        body = n['regions'][0]; iv, *its = [a for a, _ in body[0]['args']]
        L, Ub, St = (z3.simplify(x) for x in (lb, ub, st))
        if all(z3.is_int_value(x) for x in (L, Ub, St)) and St.as_long() > 0:
            l, u, k = L.as_long(), Ub.as_long(), St.as_long(); trip = max(0, -(-(u - l) // k))
            if trip <= KUNROLL:
                cur = inits
                for j in range(trip):
                    s.env[iv] = (z3.IntVal(l + j * k), F_)
                    for a, c in zip(its, cur): s.env[a] = c
                    cur = s.walk(body, g, path + ((n['id'], j),))
                for (o, _), c in zip(n['outs'], cur): s.env[o] = c
                return
        # 摘要：体在符号状态上走一遍（收集效应、体的 UB、一步转移），不计入外层 UB
        caps = free_vals(body, set([iv] + its))
        siv = z3.Int(fv('iv')); sits = [(z3.FreshConst(zsort(t), 'it'), z3.FreshBool('itp')) for (_, t) in body[0]['args'][1:]]
        s.env[iv] = (siv, F_)
        for a, c in zip(its, sits): s.env[a] = c
        ub0, ne = s.ub, len(s.eff); s.ub = F_
        ys = s.walk(body, T_, path + ((n['id'], '*'),))
        bub, s.ub = s.ub, ub0
        pure = len(s.eff) == ne
        args = [lb, ub, st] + [x for v, p in inits for x in (v, p)] + [x for c in caps for x in s.env.get(c, (None, F_)) if x is not None]
        key = 'loop_' + canon(n)
        nonempty = lb < ub
        def F(nm, rng):
            if pure:
                return z3.Function(f'{key}.{nm}', *[a.sort() for a in args], rng)(*args) if args else z3.Const(f'{key}.{nm}', rng)
            s.exact = False; return z3.FreshConst(rng, f'{key}.{nm}')     # 体有效应：结果依赖内存，非确定
        outs = [(z3.If(nonempty, F(f'v{j}', iv0.sort()), iv0), z3.If(nonempty, F(f'p{j}', z3.BoolSort()), p0))
                for j, (iv0, p0) in enumerate(inits)]
        fub = F('ub', z3.BoolSort())
        s.ax += [inrange(v) for v, _ in outs]
        s.exact = False
        inv = n.get('props', {}).get('inv')
        if inv is not None:                      # inv(iv, [(值, 毒值)], {'lb','ub','st'}) -> z3 Bool
            B = {'lb': lb, 'ub': ub, 'st': st}
            pre_body = z3.And(lb <= siv, siv < ub, st > 0, inrange(siv), *[inrange(v) for v, _ in sits], inv(siv, sits, B))
            ok_pres = prove(z3.Implies(z3.And(z3.And(s.ax) if s.ax else T_, pre_body), inv(siv + st, ys, B)))
            ok_ub = ok_pres and prove(z3.Implies(z3.And(z3.And(s.ax) if s.ax else T_, pre_body), z3.Not(bub)))
            init = z3.And(st > 0, nonempty, inv(lb, inits, B))
            if ok_pres:
                final = lb + ((ub - lb + st - 1) / st) * st
                s.ax.append(z3.Implies(init, inv(final, outs, B)))
                if ok_ub: s.ax.append(z3.Implies(init, z3.Not(fub)))
                if not prove(z3.Implies(z3.And(g, st > 0, nonempty), inv(lb, inits, B))):
                    s.diag.append(f"{n['id']}: 不变式在入口不一定成立（仍可靠，但出口事实只在满足入口条件时可用）")
            else: s.diag.append(f"{n['id']}: 不变式不是归纳的，忽略")
        s.ub = z3.Or(s.ub, z3.And(g, z3.Not(z3.Or(plb, pub, pst)), st > 0, nonempty, fub))
        for (o, _), c in zip(n['outs'], outs): s.env[o] = c

def free_vals(blocks, bound):
    """区域里使用、但不在区域内定义的值（按首次出现的顺序）。"""
    out, defd = [], set(bound)
    def sc(bs):
        for b in bs:
            defd.update(a for a, _ in b['args'])
            for n in b['ops']:
                for v in n['ins']:
                    if v not in defd and v not in out: out.append(v)
                for r in n['regions']: sc(r)
                defd.update(o for o, _ in n['outs'])
    sc(blocks); return out
def canon(n):
    """与命名无关的结构键：输入按位置、捕获按首次出现、内部定义按出现顺序编号；不变式不参与（不改变语义）。"""
    import hashlib
    body = n['regions'][0]; nm = {}
    for i, v in enumerate(n['ins']): nm.setdefault(('in', i), f'i{i}')
    caps = free_vals(body, set(a for a, _ in body[0]['args']))
    for j, c in enumerate(caps): nm[c] = f'c{j}'
    cnt = itertools.count()
    def d(v): nm.setdefault(v, f'd{next(cnt)}'); return nm[v]
    def ser(bs):
        out = []
        for b in bs:
            out.append(('blk', tuple((d(a), t) for a, t in b['args'])))
            for x in b['ops']:
                pr = tuple(sorted((k, str(v)) for k, v in x.get('props', {}).items() if k != 'inv'))
                ins = tuple(nm.get(v, '?') for v in x['ins'])
                out.append((x['op'], pr, ins, tuple(d(o) + ':' + t for o, t in x['outs']), tuple(tuple(ser(r)) for r in x['regions'])))
        return out
    return hashlib.sha1(repr((len(n['ins']), tuple(t for _, t in n['outs']), ser(body))).encode()).hexdigest()[:10]

PROVE_MS = 20000
STATS = {'unknown': 0}
def prove(f):
    """有效 ⇔ ¬f 不可满足。求解器超时或 unknown ⇒ 不算证明（可靠方向），并计数。"""
    sv = z3.Solver(); sv.set(timeout=PROVE_MS); sv.add(z3.Not(f)); r = sv.check()
    if r == z3.unknown: STATS['unknown'] += 1
    return r == z3.unsat

def compose_eff(effs, boundary):
    """效应组合（v4：区域感知）。effs: [(kind, res, 值, 区域路径, 节点)]；boundary: 值 → 边界端口。
    新缓冲在返回前无外部别名 ⇒ 对它的 Read/Write 不可见；Allocate 与 Free 只在**同一块**内才抵消，
    只在某一分支里释放 ⇒ 其余路径泄漏（v3 的“局部 alloc+free 抵消”不看控制流，是不可靠的）。"""
    if TOP in effs: return TOP, []
    alloc = {t: p for k, _, t, p, _ in effs if k == 'Allocate'}
    frees = {}
    for k, _, t, p, _ in effs:
        if k == 'Free': frees.setdefault(t, []).append(p)
    out, diag = set(), []
    for k, res, t, p, nid in effs:
        if t in alloc:
            if k in ('Read', 'Write', 'Free'): continue
            if t in boundary: out.add(('Allocate', res, boundary[t])); continue
            if alloc[t] in frees.get(t, []): continue
            diag.append(f'{nid}: {t} ' + ('只在部分路径上释放，其余路径泄漏' if t in frees else '分配后既不释放也不逃逸（泄漏）'))
            continue
        out.add((k, res, boundary.get(t, t)))
    return frozenset(out), diag

def derive(name, ins, outs, blocks, mode='transparent', params=()):
    """ins/outs: [(端口, 类型)]；blocks: 实现体（入口块参数 = 输入端口）。返回 Schema 并登记。"""
    sch = Schema(name, ins, outs, trust={'*': 'derived'}, body=blocks, mode=mode, params=list(params))
    sch.pconsts = {nm: (z3.Int if k == 'int' else z3.Bool)(f'{name}.${nm}') for nm, k in params}
    # 结构层：把体包成 func 检查 R1–R8，并求结构摘要
    ft = '(' + ', '.join(t for _, t in ins) + ') -> (' + ', '.join(t for _, t in outs) + ')'
    sch.struct_diag = list(ST.check(ST.M(ST.F(name, blocks, ftype=ft))))
    sch.summary = ST.derive_struct(blocks)
    c = Ctx(sch.pconsts)
    holes = []
    for (p, ty), (v, _) in zip(ins, blocks[0]['args']):
        so = zsort(ty)
        h = (z3.Const(f'{name}.{p}', so) if so is not None else None, z3.Bool(f'{name}.Z.{p}'))
        c.env[v] = h; holes.append(h)
        if ty == 'index': c.ax.append(inrange(h[0]))
    try: ys = c.walk(blocks, T_, ())
    except ValueUnknown:
        sch.value_known, sch.exact = False, False
        reg(sch); return sch
    sch.holes, sch.outv, sch.ubf = holes, [(z3.simplify(v) if v is not None else None, z3.simplify(p)) for v, p in ys], z3.simplify(c.ub)
    sch.exact = c.exact; sch.ax = z3.simplify(z3.And(c.ax)) if c.ax else T_; sch.loop_diag = c.diag
    sch.env = {v: c.env[v] for v in _toplevel_vals(blocks) if v in c.env}     # 顶层值的项：补全查询用
    boundary = {v: p for (p, _), (v, _) in zip(ins, blocks[0]['args'])} | {v: p for (p, _), v in zip(outs, _term(blocks))}
    sch.eff, sch.eff_diag = compose_eff(c.eff, boundary)
    sch.eff = sch.eff if sch.eff == TOP else frozenset((k, r, p) for k, r, p in sch.eff)
    reg(sch); return sch

def _toplevel_vals(blocks): return [v for v, _ in blocks[0]['args']] + [o for n in blocks[0]['ops'] for o, _ in n['outs']]
def _term(blocks): return blocks[0]['ops'][-1]['ins'] if blocks[0]['ops'] else []
def pretty(f):
    g = z3.Goal(); g.add(f); r = z3.Tactic('ctx-solver-simplify')(g)[0]
    return str(z3.And(*r) if len(r) != 1 else r[0]) if len(r) else 'True'

def no_poison(s):
    return z3.And([z3.Not(p) for _, p in s.holes]) if s.holes else T_
def user_pre(s):
    """面向用户的前提：输入都不是毒值时不触发 UB。"""
    return z3.simplify(z3.And(no_poison(s), z3.Not(s.ubf)))
def may_poison(s, j=0): return z3.simplify(z3.And(z3.Not(s.ubf), s.outv[j][1]))

def as_comp(s):
    """原生 op 包成单节点复合 op，以便用同一套查询（caps、refines）。"""
    if s.body is not None: return s
    outs = [(f'o{i}', t) for i, (_, t) in enumerate(s.outs)]
    return derive(f'wrap[{s.name}]', s.ins, s.outs,
                  [blk('entry', [node('n', s.name, ins=[p for p, _ in s.ins], outs=outs), ret('r', *[o for o, _ in outs])], args=s.ins)])

def caps(s):
    s = as_comp(s)
    if not s.value_known: return {'Speculatable': None, 'Pure': None}
    spec = prove(z3.Implies(s.ax, z3.Not(s.ubf)))   # 毒值输入也不得触发 UB（推测执行时输入可能是毒值）
    return {'Speculatable': spec, 'Speculatable|非毒值输入': prove(z3.Implies(z3.And(s.ax, no_poison(s)), z3.Not(s.ubf))),
            'Pure': spec and s.eff != TOP and not s.eff}

def refines(src, tgt, assume=True):
    """tgt 精化 src（把 src 改写成 tgt 有效）：src 不 UB ⇒ tgt 不 UB；src 输出不是毒值 ⇒ tgt 输出相同且不是毒值。
    只覆盖值层：任一方有效应时返回 None（值层精化不足以说明删改效应是否合法，须声明）。assume：参数的前提。"""
    if src.eff or tgt.eff: return None
    pairs = [(ht, hs) for xs, xt in zip(src.holes, tgt.holes) for hs, ht in zip(xs, xt) if hs is not None]
    sb = lambda t: z3.substitute(t, pairs) if pairs else t
    conds = [z3.Not(sb(tgt.ubf))]
    for (vs, ps), (vt, pt) in zip(src.outv, tgt.outv):
        conds.append(z3.Implies(z3.Not(ps), z3.And(z3.Not(sb(pt)), *([sb(vt) == vs] if vs is not None else []))))
    return prove(z3.Implies(z3.And(assume, src.ax, sb(tgt.ax), z3.Not(src.ubf)), z3.And(conds)))

def equivalent(a, b):
    """两个 Schema 的值层是否相同（按端口位置对齐）：UB 相同；不 UB 时毒值相同；不是毒值时值相同。"""
    pairs = [(hb, ha) for xa, xb in zip(a.holes, b.holes) for ha, hb in zip(xa, xb) if ha is not None]
    sb = lambda t: z3.substitute(t, pairs)
    conds = [a.ubf == sb(b.ubf)]
    for (va, pa), (vb, pb) in zip(a.outv, b.outv):
        conds.append(z3.Implies(z3.Not(a.ubf), pa == sb(pb)))
        if va is not None: conds.append(z3.Implies(z3.And(z3.Not(a.ubf), z3.Not(pa)), va == sb(vb)))
    return prove(z3.Implies(z3.And(a.ax, sb(b.ax)), z3.And(conds)))

# ───────────────────────── 透明内联（全内联，作为“嵌套 ≡ 内联”的对照）
def inline_all(blocks, env=None, pfx='', penv=None):
    env = dict(env or {}); out = []; penv = penv or {}
    sp = lambda pr: {k: penv.get(v[1:], v) if isinstance(v, str) and v.startswith('$') else v for k, v in pr.items()}
    for b in blocks:
        ops = []
        for v, _ in b['args']: env.setdefault(v, pfx + v)
        args = [(env[v], t) for v, t in b['args']]
        for n in b['ops']:
            sch = REG.get(n['op'])
            if sch is not None and sch.body is not None and sch.mode == 'transparent':
                inner = dict(zip([v for v, _ in sch.body[0]['args']], [env.get(v, v) for v in n['ins']]))
                p2 = f"{pfx}{n['id']}."
                ib = inline_all([dict(sch.body[0], args=[])], inner, p2, sp(n.get('props', {})))[0]
                term = ib['ops'][-1]; ops += ib['ops'][:-1]
                for (o, _), y in zip(n['outs'], term['ins']): env[o] = y
                continue
            m = dict(n, id=pfx + n['id'], ins=[env.get(v, v) for v in n['ins']], props=sp(n.get('props', {})))
            for o, _ in n['outs']: env[o] = pfx + o
            m['outs'] = [(env[o], t) for o, t in n['outs']]
            m['regions'] = [inline_all(r, env, pfx, penv) for r in n['regions']]
            ops.append(m)
        out.append(dict(b, args=args, ops=ops))
    return out

# ───────────────────────── 编辑操作：提取为复合 op（带契约的图变换）
def extract(name, block, lo, hi_, types, mode='transparent'):
    """把 block['ops'][lo:hi_] 提取为复合 op。捕获的值提升为输入端口（L3），后面还会用到的值成为输出端口。
    契约：提取前后，宿主的 Pre 与 MayPoison 相等（core_test.py 用 z3 判定等价）。"""
    seg = block['ops'][lo:hi_]
    defined, used = {}, []
    def scan(ops):
        for n in ops:
            for v in n['ins']:
                if v not in defined and v not in used: used.append(v)
            for v, t in n['outs']: defined[v] = t
            for r in n['regions']:
                for b in r:
                    for v, t in b['args']: defined[v] = t
                    scan(b['ops'])
    scan(seg)
    later = set()
    def uses(ops):
        for n in ops:
            later.update(n['ins'])
            for r in n['regions']:
                for b in r: uses(b['ops'])
    uses(block['ops'][hi_:])
    caps_ = [v for v in used if v not in defined]
    outs = [v for v in defined if v in later and any(v == o for n in seg for o, _ in n['outs'])]
    pm = {v: f'{name}_in{i}' for i, v in enumerate(caps_)}
    body_ops = [ren(n, pm) for n in seg] + [node(f'{name}_ret', 'func.return', ins=[pm.get(v, v) for v in outs])]
    body = [blk('entry', body_ops, args=[(pm[v], types[v]) for v in caps_])]
    s = derive(name, [(pm[v], types[v]) for v in caps_], [(f'o{i}', types[v]) for i, v in enumerate(outs)], body, mode)
    call = node(f'{name}_call', name, ins=caps_, outs=[(v, types[v]) for v in outs])
    return s, dict(block, ops=block['ops'][:lo] + [call] + block['ops'][hi_:])
def ren(n, pm):
    return dict(n, ins=[pm.get(v, v) for v in n['ins']],
                regions=[[dict(b, ops=[ren(x, pm) for x in b['ops']]) for b in r] for r in n['regions']])

# ───────────────────────── 小工具
def cst(i, v, t='i3'): return node(i, 'arith.constant', outs=[(i, t)], props={'value': v, 'type': t})
def op2(i, op, a, b, t='i3', **p): return node(i, op, ins=[a, b], outs=[(i, t)], props=p)
def ret(i, *vs): return node(i, 'func.return', ins=list(vs))

if __name__ == '__main__':
    f = derive('f', [('x', I3), ('y', I3)], [('q', I3)],
               [blk('entry', [op2('d', 'arith.divui', 'x', 'y'), ret('r', 'd')], args=[('x', I3), ('y', I3)])])
    print('■ f(x,y) = x / y   Pre:', pretty(user_pre(f)), ' caps:', caps(f))
    g = derive('g', [('a', I3), ('b', I3)], [('q', I3)],
               [blk('entry', [cst('z', 0), op2('c', 'arith.cmpi', 'z', 'b', 'i1', predicate=6),
                              node('if', 'scf.if', ins=['c'], outs=[('r', I3)], regions=[
                                  [blk('t', [op2('d', 'arith.divui', 'a', 'b'), node('y1', 'scf.yield', ins=['d'])])],
                                  [blk('e', [node('y2', 'scf.yield', ins=['z'])])]]),
                              ret('rr', 'r')], args=[('a', I3), ('b', I3)])])
    print('■ g(a,b) = 0<b ? a/b : 0   Pre:', pretty(user_pre(g)), ' caps:', caps(g), ' 结构违例:', g.struct_diag)
