"""统一核心 v4：一种 IR、一种 Schema、一次推导。

IR：structure.py 的 module/region/block/node 表示，是唯一的图表示（v3 的扁平 DAG 已废弃）。
Schema v4 = ⟨签名; 值层（每个输出的值与毒值标志、UB，都是“输入值 × 输入毒值标志”的函数）; Eff; Struct; 信任等级; Impl⟩
derive(体) 一次遍历同时给出 值层 / 效应层 / 结构层 的事实；结果仍是 Schema，可再作为节点使用。
值层用 z3 QF_BV（v3.1 的决定：op 确定 ⇒ wp 只做代换，精确、无量词）。i3 位宽便于穷举预言机（core_test.py）。
下标 / extent / 循环的 Presburger 层（wp.py、shape.py）尚未并入，见 spec/coverage.md。
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
def is_int(t): return t in ('i3', 'i1')
def hi(t): return 1 if t == 'i1' else MAX

class ValueUnknown(Exception): pass

def W(t): return 1 if t == 'i1' else 3
SORTS = {}                            # 用户类型 → z3 sort（由 defs.py 的类型定义登记）
def zsort(t):
    if t in ('i3', 'i1'): return z3.BitVecSort(W(t))
    return SORTS.get(t)
def zb(x): return z3.BoolVal(x) if isinstance(x, bool) else x               # 参数可以是具体值或符号（复合 op 的参数）
def zbv(x, t='i3'): return z3.BitVecVal(x, W(t)) if isinstance(x, int) else z3.Int2BV(x, W(t))
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

F_, T_ = z3.BoolVal(False), z3.BoolVal(True)
def bv(x, t='i3'): return z3.BitVecVal(x, W(t))
def anyp(a): return z3.Or([p for _, p in a]) if a else F_

# ───────────────────────── 原生 op（语义为人工覆盖层，结构取自 structure.STRUCT）
REG = {}
def reg(s): REG[s.name] = s; return s
def nat(name, ins, outs, fn, eff=frozenset()):
    return reg(Schema(name, ins, outs, fn, frozenset(eff), trust={'值层/效应': 'authored', '结构': 'ods'}))
I3 = 'i3'
def _add(p, a):
    (x, px), (y, py) = a
    return [(x + y, z3.Or(px, py, z3.And(zb(p.get('nuw', False)), z3.Not(z3.BVAddNoOverflow(x, y, False)))))], F_
def _sub(p, a):
    (x, px), (y, py) = a
    return [(x - y, z3.Or(px, py, z3.And(zb(p.get('nuw', False)), z3.ULT(x, y))))], F_
def _div(p, a):
    (x, px), (y, py) = a
    return [(z3.UDiv(x, y), px)], z3.Or(py, y == 0)               # 除数为 0 或毒值 ⇒ UB
PRED = ['eq', 'ne', 'slt', 'sle', 'sgt', 'sge', 'ult', 'ule', 'ugt', 'uge']   # arith::CmpIPredicate 的编号
def _cmp(p, a):
    (x, px), (y, py) = a
    rel = [x == y, x != y, x < y, x <= y, x > y, x >= y, z3.ULT(x, y), z3.ULE(x, y), z3.UGT(x, y), z3.UGE(x, y)]
    k = p['predicate']
    c = rel[k] if isinstance(k, int) else z3.Or([z3.And(k == i, r) for i, r in enumerate(rel)])
    return [(z3.If(c, bv(1, 'i1'), bv(0, 'i1')), z3.Or(px, py))], F_
def _sel(p, a):
    (c, pc), (x, px), (y, py) = a
    return [(z3.If(c == 1, x, y), z3.Or(pc, z3.If(c == 1, px, py)))], F_
_pz = itertools.count()
def _poison(p, a):
    t = p.get('type', 'i3')
    return [(z3.FreshConst(zsort(t), 'poison'), T_)], F_
nat('arith.constant', [], [('r', I3)], lambda p, a: ([(zbv(p['value'], p.get('type', 'i3')), F_)], F_))
nat('arith.addi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _add)
nat('arith.subi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _sub)
nat('arith.divui', [('lhs', I3), ('rhs', I3)], [('r', I3)], _div)
nat('arith.cmpi', [('lhs', I3), ('rhs', I3)], [('r', 'i1')], _cmp)
nat('arith.select', [('c', 'i1'), ('a', I3), ('b', I3)], [('r', I3)], _sel)
nat('ub.poison', [], [('r', I3)], _poison)
def _bin(f, ovf=None):
    def g(p, a):
        (x, px), (y, py) = a
        return [(f(x, y), z3.Or(px, py, z3.And(zb(p.get('nuw', False)), ovf(x, y)) if ovf else F_))], F_
    return g
nat('arith.muli', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x * y, lambda x, y: z3.Not(z3.BVMulNoOverflow(x, y, False))))
nat('arith.andi', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x & y))
nat('arith.ori', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x | y))
nat('arith.xori', [('lhs', I3), ('rhs', I3)], [('r', I3)], _bin(lambda x, y: x ^ y))
nat('arith.extui', [('in', 'i1')], [('r', I3)], lambda p, a: ([(z3.ZeroExt(W(p.get('type', 'i3')) - a[0][0].size(), a[0][0]), a[0][1])], F_))
_ch = itertools.count()
def _load(p, a): return [(z3.BitVec(f'load{next(_ch)}', 3), F_)], F_        # 读到的内容：非确定（自由变量）
nat('memref.alloc', [], [('m', 'memref')], lambda p, a: ([(None, F_)], F_), eff={('Allocate', 'Default', 'm')})
nat('memref.dealloc', [('m', 'memref')], [], lambda p, a: ([], F_), eff={('Free', 'Default', 'm')})
nat('memref.load', [('m', 'memref')], [('r', I3)], _load, eff={('Read', 'Default', 'm')})
nat('memref.store', [('v', I3), ('m', 'memref')], [], lambda p, a: ([], F_), eff={('Write', 'Default', 'm')})

def instantiate(sch, props, args):
    """复合 op 的摘要实例化 = 项代换（嵌套 ≡ 内联的根据）。"""
    pairs = [(h, a) for hs, as_ in zip(sch.holes, args) for h, a in zip(hs, as_) if h is not None]
    for nm, kind in sch.params:
        if nm in props:
            v = props[nm]
            pairs.append((sch.pconsts[nm], z3.IntVal(v) if isinstance(v, int) and not isinstance(v, bool) else zb(v)))
    sub = lambda t: z3.substitute(t, pairs) if pairs else t
    return [(sub(v) if v is not None else None, sub(p)) for v, p in sch.outv], sub(sch.ubf)
def is_comp(sch): return sch.body is not None

# ───────────────────────── 推导：一次遍历
class Ctx:
    def __init__(s, params=None): s.env, s.ub, s.eff, s.exact, s.params = {}, F_, [], True, params or {}
    def props(s, n):
        return {k: s.params[v[1:]] if isinstance(v, str) and v.startswith('$') else v for k, v in n.get('props', {}).items()}
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
            sch = REG.get(op)
            if sch is None or not sch.value_known: raise ValueUnknown(op)
            args = [s.env.get(v, (None, F_)) for v in n['ins']]
            pr = s.props(n)
            outs, ub = instantiate(sch, pr, args) if is_comp(sch) else sch.fn(pr, args)
            s.ub = z3.Or(s.ub, z3.And(g, ub))                           # 分支里的 UB 只在该路径上算数
            for (o, _), r in zip(n['outs'], outs): s.env[o] = r
            vids = {p: v for (p, _), v in zip(sch.ins, n['ins'])} | {p: o for (p, _), (o, _) in zip(sch.outs, n['outs'])}
            if sch.eff == TOP: s.eff.append(TOP)
            else: s.eff += [(k, r, vids[p], path, n['id']) for k, r, p in sch.eff]
            s.exact &= sch.exact
        raise ValueUnknown('块没有终结子')

def prove(f):
    sv = z3.Solver(); sv.add(z3.Not(f)); return sv.check() == z3.unsat

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
    try: ys = c.walk(blocks, T_, ())
    except ValueUnknown:
        sch.value_known, sch.exact = False, False
        reg(sch); return sch
    sch.holes, sch.outv, sch.ubf = holes, [(z3.simplify(v) if v is not None else None, z3.simplify(p)) for v, p in ys], z3.simplify(c.ub)
    sch.exact = c.exact
    boundary = {v: p for (p, _), (v, _) in zip(ins, blocks[0]['args'])} | {v: p for (p, _), v in zip(outs, _term(blocks))}
    sch.eff, sch.eff_diag = compose_eff(c.eff, boundary)
    sch.eff = sch.eff if sch.eff == TOP else frozenset((k, r, p) for k, r, p in sch.eff)
    reg(sch); return sch

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
    spec = prove(z3.Not(s.ubf))                     # 毒值输入也不得触发 UB（推测执行时输入可能是毒值）
    return {'Speculatable': spec, 'Speculatable|非毒值输入': prove(z3.Implies(no_poison(s), z3.Not(s.ubf))),
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
    return prove(z3.Implies(z3.And(assume, z3.Not(src.ubf)), z3.And(conds)))

def equivalent(a, b):
    """两个 Schema 的值层是否相同（按端口位置对齐）：UB 相同；不 UB 时毒值相同；不是毒值时值相同。"""
    pairs = [(hb, ha) for xa, xb in zip(a.holes, b.holes) for ha, hb in zip(xa, xb) if ha is not None]
    sb = lambda t: z3.substitute(t, pairs)
    conds = [a.ubf == sb(b.ubf)]
    for (va, pa), (vb, pb) in zip(a.outv, b.outv):
        conds.append(z3.Implies(z3.Not(a.ubf), pa == sb(pb)))
        if va is not None: conds.append(z3.Implies(z3.And(z3.Not(a.ubf), z3.Not(pa)), va == sb(vb)))
    return prove(z3.And(conds))

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
