"""定义层 v0：种类 = ⟨参数, 约束, 语义⟩。attribute、type、property、ADT、interface 都由它定义，原生的与用户的同构。

  §1 静态项：attribute 与 type 统一为带种类的项，按**指称**存储（'3 : i2' 与 '-1 : i2' 是同一个项）
  §2 op 参数（property）与元数据（可丢弃 attribute）分开；参数有种类、默认值、约束，并且只有参数能进入语义
  §3 复合 op 的参数：约束与默认值由实现体推导（∃ 不需要：参数直接代入，约束取合取）
  §4 用户类型（ADT）：由定义生成 op（构造、判别、投影）与降级（1:N 类型转换），降级正确性 = 抽象函数下的精化
  §5 interface：声明（方法 + 定律）与实现表分开；实现分 derived / checked / declared 三档；定律写成定向精化
检验见 model/defs_test.py。
"""
import itertools, z3
from dataclasses import dataclass, field
import core as C
from core import blk, node, I3

# ───────────────────────── §1 静态项
@dataclass(frozen=True)
class Term:
    kind: str
    args: tuple
    def __str__(s): return KINDS[s.kind].show(*s.args)

@dataclass
class Kind:
    name: str
    cat: str                  # 'type' | 'attr'
    params: list              # [(名字, 参数种类)]；参数种类：'int' 'str' 'sym' 'type' 'attr' ('list', x) ('enum', [...])
    check: object = None      # (*args) -> [违例]
    show: object = None
    trust: str = 'builtin'    # builtin（MLIR 内建）| ods（方言导入）| user
KINDS = {}
def kind(*a, **kw): k = Kind(*a, **kw); KINDS[k.name] = k; return k

def _sort_ok(srt, v):
    if srt == 'int': return isinstance(v, int) and not isinstance(v, bool)
    if srt in ('str', 'sym'): return isinstance(v, str)
    if srt in ('type', 'attr'): return isinstance(v, Term) and KINDS[v.kind].cat == srt
    if srt[0] == 'list': return isinstance(v, tuple) and all(_sort_ok(srt[1], x) for x in v)
    if srt[0] == 'enum': return v in srt[1]
    return False

def verify(t):
    """项的良构：参数种类 + 种类约束，递归到子项。"""
    k = KINDS.get(t.kind)
    if k is None: return [f'未知种类 {t.kind}']
    if len(t.args) != len(k.params): return [f'{t.kind} 需要 {len(k.params)} 个参数']
    errs = [f'{t.kind}.{n} 的种类应为 {s}' for (n, s), v in zip(k.params, t.args) if not _sort_ok(s, v)]
    if errs: return errs
    for v in t.args:
        for x in (v if isinstance(v, tuple) else (v,)):
            if isinstance(x, Term): errs += verify(x)
    return errs + (k.check(*t.args) if k.check else [])

SIGN = ['signless', 'signed', 'unsigned']
kind('IntegerType', 'type', [('width', 'int'), ('sign', ('enum', SIGN))],
     check=lambda w, s: [] if 1 <= w <= 16777215 else ['位宽须在 1..16777215'],
     show=lambda w, s: {'signless': 'i', 'signed': 'si', 'unsigned': 'ui'}[s] + str(w))
kind('IndexType', 'type', [], show=lambda: 'index')
def iT(w, s='signless'): return Term('IntegerType', (w, s))
INDEX = Term('IndexType', ())
def width(ty): return 64 if ty.kind == 'IndexType' else ty.args[0]

def _ia_show(ty, bits):
    w = width(ty); sg = 'signless' if ty.kind == 'IndexType' else ty.args[1]
    if ty == iT(1): return 'true' if bits else 'false'
    v = bits if sg == 'unsigned' else (bits - (1 << w) if bits >> (w - 1) else bits)
    return str(v) if ty == iT(64) else f'{v} : {ty}'        # MLIR 打印时省略 i64
kind('IntegerAttr', 'attr', [('type', 'type'), ('bits', 'int')],
     check=lambda ty, b: ([] if ty.kind in ('IntegerType', 'IndexType') else ['类型须为整数或 index']) +
                         ([] if 0 <= b < (1 << width(ty)) else ['位模式超出位宽']),
     show=_ia_show)
def int_lit(v, ty):
    """字面量 → 项（指称是位模式）。范围规则与 MLIR 解析器一致（defs_test D1 差分）：
    signless 接受有符号与无符号两种读法；signed / unsigned 只接受各自的范围。"""
    w = width(ty); sg = 'signless' if ty.kind == 'IndexType' else ty.args[1]
    lo, hi = {'signless': (-(1 << (w - 1)), (1 << w) - 1), 'signed': (-(1 << (w - 1)), (1 << (w - 1)) - 1),
              'unsigned': (0, (1 << w) - 1)}[sg]
    if not lo <= v <= hi: return None
    return Term('IntegerAttr', (ty, v % (1 << w)))
kind('StringAttr', 'attr', [('value', 'str')], show=lambda v: f'"{v}"')
kind('ArrayAttr', 'attr', [('elems', ('list', 'attr'))], show=lambda es: '[' + ', '.join(map(str, es)) + ']')
kind('TypeAttr', 'attr', [('type', 'type')], show=lambda t: str(t))
kind('SymbolRefAttr', 'attr', [('root', 'sym'), ('nested', ('list', 'sym'))],
     show=lambda r, ns: '@' + '::@'.join((r,) + ns))
kind('UnitAttr', 'attr', [], show=lambda: 'unit')
kind('OverflowFlagsAttr', 'attr', [('flags', ('list', ('enum', ['nsw', 'nuw'])))], trust='ods',
     check=lambda fl: [] if list(fl) == sorted(set(fl), key=['nsw', 'nuw'].index) else ['标志须去重并按 nsw, nuw 排序'],
     show=lambda fl: '#arith.overflow<' + (', '.join(fl) or 'none') + '>')
def OF(*fl): return Term('OverflowFlagsAttr', (tuple(fl),))

# ───────────────────────── §2 op 参数与元数据
@dataclass
class Param:
    name: str
    accept: object            # (项, 节点) -> ([静态违例], 值域公式)；值域公式对符号参数是 z3 项
    denote: object            # 项 -> {core 参数: 指称}
    keys: tuple = ()          # 该参数在 core 里对应的键（符号参数原样传 '$名字'）
    default: Term = None
    required: bool = False

def _value_of(t): return t.args[1] if isinstance(t, Term) else t     # 符号参数直接是 z3 项
def _acc_const(t, n):
    if not (isinstance(t, Term) and t.kind == 'IntegerAttr'): return ['value 须为 IntegerAttr'], True
    return ([] if str(t.args[0]) == n['outs'][0][1] else ['value 的类型须等于结果类型']), True
def _acc_pred(t, n):
    if not (isinstance(t, Term) and t.kind == 'IntegerAttr' and t.args[0] == iT(64)): return ['predicate 须为 i64 IntegerAttr'], True
    v = _value_of(t)
    return [], (z3.And(v >= 0, v <= 9) if z3.is_expr(v) else 0 <= v <= 9)
def _acc_of(t, n):
    if z3.is_expr(t): return [], True                              # 符号的 nuw 标志
    return ([] if isinstance(t, Term) and t.kind == 'OverflowFlagsAttr' else ['overflowFlags 须为 #arith.overflow']), True
def _den_of(t):
    if isinstance(t, Term): return {'nuw': 'nuw' in t.args[0], 'nsw': 'nsw' in t.args[0]}   # 第 21 轮：此前丢掉 nsw（语义不可靠）
    return {'nuw': t}                                              # 符号：z3 Bool（复合 op 参数只开放 nuw）
OPSIG = {
    'arith.constant': [Param('value', _acc_const, lambda t: {'value': _value_of(t)}, ('value',), required=True)],
    'arith.cmpi': [Param('predicate', _acc_pred, lambda t: {'predicate': _value_of(t)}, ('predicate',), required=True)],
    'arith.addi': [Param('overflowFlags', _acc_of, _den_of, ('nuw',), default=OF())],
    'arith.subi': [Param('overflowFlags', _acc_of, _den_of, ('nuw',), default=OF())],
    'arith.divui': [], 'arith.select': [], 'ub.poison': [],
}
CONSERVATIVE = {'P1': '未知参数名：MLIR 静默丢弃，模型拒绝', 'P2': '元数据与参数同名：MLIR 会把元数据提升为参数（改变语义），模型拒绝'}

def verify_node(n, sym_ok=None):
    """返回 (违例, 值域公式)。sym_ok: 复合 op 的符号参数名 → 声明的项模板（值是 z3 变量）。"""
    sig = {p.name: p for p in OPSIG.get(n['op'], [])}
    errs, rng = [], []
    for k in n.get('props', {}):
        if k not in sig: errs.append(('P1', f'{n["id"]}: 未知参数 {k}'))
    for k in n.get('attrs', {}):
        if k in sig: errs.append(('P2', f'{n["id"]}: 元数据 {k} 与参数同名'))
    for p in sig.values():
        t = n.get('props', {}).get(p.name, p.default)
        if t is None:
            if p.required: errs.append(('P3', f'{n["id"]}: 缺少必需参数 {p.name}'))
            continue
        if isinstance(t, str) and t.startswith('$'):
            if not sym_ok or t[1:] not in sym_ok: errs.append(('P5', f'{n["id"]}: 参数 {t} 未声明')); continue
            t = sym_ok[t[1:]]
        elif isinstance(t, Term):
            errs += [('P4', f'{n["id"]}: {e}') for e in verify(t)]
        e, r = p.accept(t, n); errs += [('P4', f'{n["id"]}: {x}') for x in e]
        if r is False: errs.append(('P4', f'{n["id"]}: {p.name} 的值超出允许范围'))
        rng.append(r)
    return errs, rng

def to_core(blocks, sym=None):
    """语法层（参数是项）→ core（参数是指称）。符号参数保留为 '$名字'，由 core 在推导时绑定。"""
    out = []
    for b in blocks:
        ops = []
        for n in b['ops']:
            pr = {}
            for p in OPSIG.get(n['op'], []):
                t = n.get('props', {}).get(p.name, p.default)
                if t is None: continue
                if isinstance(t, str): pr.update({k: t for k in p.keys})
                else: pr.update(p.denote(t))
            if n['op'] in ('arith.constant', 'ub.poison'): pr['type'] = n['outs'][0][1]
            ops.append(dict(n, props=pr, regions=[to_core(r, sym) for r in n['regions']]))
        out.append(dict(b, ops=ops))
    return out

# ───────────────────────── §3 复合 op 的参数
def derive_params(name, params, blocks):
    """params: [(名字, 项模板)]，模板里值的位置放 None。返回 (静态违例, 值域约束)。
    规则：参数代入体内每个使用点；静态部分（种类、类型）必须逐点一致，值域约束取合取（精确，E）。"""
    sym, errs, rng = {}, [], []
    for nm, tpl in params:
        if tpl.kind == 'IntegerAttr': sym[nm] = Term(tpl.kind, tpl.args[:-1] + (z3.Int(f'{name}.${nm}'),))
        else: sym[nm] = z3.Bool(f'{name}.${nm}')
    def walk(bs):
        for b in bs:
            for n in b['ops']:
                e, r = verify_node(n, sym); errs.extend(e); rng.extend(x for x in r if z3.is_expr(x))
                for rg in n['regions']: walk(rg)
    walk(blocks)
    return errs, z3.simplify(z3.And(rng)) if rng else z3.BoolVal(True)

# ───────────────────────── §4 用户类型（ADT）
ADTS = {}
@dataclass
class ADT:
    name: str
    variants: list            # [(构造名, [(字段, 标量类型)])]，至多 2 个变体（判别位用 i1）
    sort: object = None
    ctors: list = None
    lowered: list = None      # 降级后的标量类型序列
    layout: dict = None       # (变体, 字段) → 降级序列中的位置
    trust: str = 'user'

def define_adt(name, variants):
    """登记 z3 sort，生成 op（构造、判别、投影），生成降级（类型转换 + 每个 op 的降级实现）。"""
    assert len(variants) <= 2
    # 第 28 轮（L54）：z3 按名字合并同名 Datatype——重定义会改写旧 sort 的构造子定义域，旧项随之失效。同名同构造幂等，否则拒绝。
    if name in ADTS:
        if ADTS[name].variants == variants: return ADTS[name]
        raise ValueError(f'类型 {name} 已定义为 {ADTS[name].variants}，不能重定义')
    if C.is_int(name) or name == 'index' or C.tparse(name) is not None or name in C.SORTS:
        raise ValueError(f'类型名 {name} 与内建/已登记类型冲突')
    dt = z3.Datatype(name)
    for cn, fs in variants: dt.declare(cn, *[(f'{name}_{cn}_{f}', C.zsort(t)) for f, t in fs])
    a = ADT(name, variants, dt.create()); ADTS[name] = a; C.SORTS[name] = a.sort
    a.ctors = [a.sort.constructor(k) for k in range(len(variants))]
    tag = len(variants) > 1
    a.lowered = (['i1'] if tag else []) + [t for _, fs in variants for _, t in fs]
    pos = 1 if tag else 0; a.layout = {}
    for k, (cn, fs) in enumerate(variants):
        for f, t in fs: a.layout[(k, f)] = pos; pos += 1
    def gen(opname, ins, outs, fn):
        C.reg(C.Schema(opname, ins, outs, fn, frozenset(), trust={'*': f'由类型 {name} 的定义生成'}))
    for k, (cn, fs) in enumerate(variants):
        ctor = a.ctors[k]
        gen(f'{name}.{cn}', fs, [('r', name)],
            lambda p, args, ctor=ctor: ([(ctor(*[v for v, _ in args]), C.anyp(args))], C.F_))
        if tag:
            rec = a.sort.recognizer(k)
            gen(f'{name}.is_{cn}', [('v', name)], [('r', 'i1')],
                lambda p, args, rec=rec: ([(z3.If(rec(args[0][0]), C.bv(1, 'i1'), C.bv(0, 'i1')), args[0][1])], C.F_))
        for j, (f, t) in enumerate(fs):
            acc = a.sort.accessor(k, j); rec = a.sort.recognizer(k)
            gen(f'{name}.{cn}.{f}', [('v', name)], [('r', t)],      # 变体不对 ⇒ 毒值（不是 UB）
                lambda p, args, acc=acc, rec=rec: ([(acc(args[0][0]), z3.Or(args[0][1], z3.Not(rec(args[0][0])) if tag else C.F_))], C.F_))
    _gen_lowering(a)
    return a

def abs_fn(a, vals):
    """抽象函数：降级后的 [(值, 毒值)] → (ADT 值, 毒值)。多对一：none 的载荷是什么都行。"""
    tag = len(a.variants) > 1
    def mk(k): return a.ctors[k](*[vals[a.layout[(k, f)]][0] for f, _ in a.variants[k][1]])
    def pz(k): return z3.Or([vals[a.layout[(k, f)]][1] for f, _ in a.variants[k][1]]) if a.variants[k][1] else C.F_
    if not tag: return mk(0), pz(0)
    t, pt = vals[0]
    return z3.If(t == 1, mk(1), mk(0)), z3.Or(pt, z3.If(t == 1, pz(1), pz(0)))

def _gen_lowering(a):
    """每个生成的 op 都有一个由 ADT 定义机械生成的降级实现（复合 op，只用标量 op）。"""
    tag = len(a.variants) > 1; L = a.lowered; a.lower_ops = {}
    def comp(opname, ins, outs, ops, rets):
        nm = f'lower[{opname}]'
        C.derive(nm, ins, outs, [blk('entry', ops + [C.ret(nm + '.ret', *rets)], args=ins)]); a.lower_ops[opname] = nm
    vin = [(f'v{i}', t) for i, t in enumerate(L)]
    for k, (cn, fs) in enumerate(a.variants):
        ops, rets = [], []
        if tag: ops.append(C.cst('tag', k, 'i1')); rets.append('tag')
        for k2, (cn2, fs2) in enumerate(a.variants):
            for f, t in fs2:
                if k2 == k: rets.append(f)
                else: ops.append(node(f'pz_{cn2}_{f}', 'ub.poison', outs=[(f'pz_{cn2}_{f}', t)], props={'type': t})); rets.append(f'pz_{cn2}_{f}')
        comp(f'{a.name}.{cn}', fs, [(f'o{i}', t) for i, t in enumerate(L)], ops, rets)
        if tag:
            comp(f'{a.name}.is_{cn}', vin, [('r', 'i1')],
                 [C.cst('k', k, 'i1'), node('eq', 'arith.cmpi', ins=['v0', 'k'], outs=[('eq', 'i1')], props={'predicate': 0})], ['eq'])
        for f, t in fs:
            src = f'v{a.layout[(k, f)]}'
            if not tag: comp(f'{a.name}.{cn}.{f}', vin, [('r', t)], [], [src]); continue
            comp(f'{a.name}.{cn}.{f}', vin, [('r', t)],
                 [C.cst('k', k, 'i1'), node('eq', 'arith.cmpi', ins=['v0', 'k'], outs=[('eq', 'i1')], props={'predicate': 0}),
                  node('pz', 'ub.poison', outs=[('pz', t)], props={'type': t}),
                  node('sel', 'arith.select', ins=['eq', src, 'pz'], outs=[('sel', t)])], ['sel'])

def refines_lowered(A, Lw, types_in, types_out):
    """数据精化：∀ 降级输入 x̂。¬UB_A(abs x̂) ⇒ ¬UB_L(x̂) ∧ (¬P_A ⇒ ¬P_L ∧ abs(out_L) = out_A)。"""
    pairs, k = [], 0
    for (hv, hp), t in zip(A.holes, types_in):
        if t in ADTS:
            n = len(ADTS[t].lowered); v, p = abs_fn(ADTS[t], Lw.holes[k:k + n]); k += n
        else: v, p = Lw.holes[k]; k += 1
        if hv is not None: pairs.append((hv, v))
        pairs.append((hp, p))
    sub = lambda e: z3.substitute(e, pairs) if pairs else e
    conds, k = [], 0
    for (av, ap), t in zip(A.outv, types_out):
        if t in ADTS:
            n = len(ADTS[t].lowered); lv, lp = abs_fn(ADTS[t], Lw.outv[k:k + n]); k += n
        else: (lv, lp), k = Lw.outv[k], k + 1
        conds.append(z3.Implies(z3.Not(sub(ap)), z3.And(z3.Not(lp), lv == sub(av))))
    return C.prove(z3.Implies(z3.Not(sub(A.ubf)), z3.And([z3.Not(Lw.ubf)] + conds)))

def check_adt_lowering(a):
    """定义期检查一次：每个生成的 op 与它的降级实现满足数据精化。"""
    res = {}
    for opname, low in a.lower_ops.items():
        A, Lw = C.REG[opname], C.REG[low]
        hole = C.derive(f'abs[{opname}]', A.ins, A.outs,
                        [blk('entry', [node('x', opname, ins=[p for p, _ in A.ins], outs=[(f'o{i}', t) for i, (_, t) in enumerate(A.outs)]),
                                       C.ret('r', *[f'o{i}' for i in range(len(A.outs))])], args=A.ins)])
        res[opname] = refines_lowered(hole, Lw, [t for _, t in A.ins], [t for _, t in A.outs])
    return res

def lower_blocks(blocks):
    """程序级 1:N 类型转换：ADT 值 → 若干标量值；生成的 op → 其降级实现；scf.if 的结果与 yield 一并展开。"""
    vm = {}
    def ty_list(t): return ADTS[t].lowered if t in ADTS else [t]
    def exp(v, t): return [f'{v}#{i}' for i in range(len(ty_list(t)))] if t in ADTS else [v]
    def lb(bs):
        out = []
        for b in bs:
            args = []
            for v, t in b['args']: vm[v] = exp(v, t); args += list(zip(vm[v], ty_list(t)))
            ops = []
            for n in b['ops']:
                ins = [x for v in n['ins'] for x in vm.get(v, [v])]
                outs = []
                for o, t in n['outs']: vm[o] = exp(o, t); outs += list(zip(vm[o], ty_list(t)))
                a = next((A for A in ADTS.values() if n['op'] in A.lower_ops), None)
                op = a.lower_ops[n['op']] if a else n['op']
                ops.append(dict(n, op=op, ins=ins, outs=outs, regions=[lb(r) for r in n['regions']]))
            out.append(dict(b, args=args, ops=ops))
        return out
    return lb(blocks)

# ───────────────────────── §5 interface
@dataclass
class Interface:
    name: str
    methods: dict             # 方法名 → (输入个数, 输出个数)，类型都是 Self
    laws: list                # [(定律名, 左边的体构造, 右边的体构造, 输入个数)]，含义：左 → 右 的改写有效
    trust: str = 'user'
IFACES, IMPLS = {}, {}        # IMPLS[(接口, 类型)] = (方法 → op 名, 档位)；与类型定义分开 ⇒ 第三方可以给原生类型加实现

def implement(iface, ty, methods, trust='checked'):
    I = IFACES[iface]; rep = {}
    for m, op in methods.items():
        s = C.REG[op]; ni, no = I.methods[m]
        if [t for _, t in s.ins] != [ty] * ni or [t for _, t in s.outs] != [ty] * no:
            raise TypeError(f'{op} 的签名不是 {I.methods[m]} 个 {ty}')
    if trust == 'checked':
        for law, lhs, rhs, k in I.laws:
            ins = [(f'x{i}', ty) for i in range(k)]
            def mk(nm, f):
                ops, r = f(methods, ty)
                return C.derive(nm, ins, [('q', ty)], [blk('entry', ops + [C.ret(nm + '.r', r)], args=ins)])
            L, R = mk(f'{iface}<{ty}>.{law}.L', lhs), mk(f'{iface}<{ty}>.{law}.R', rhs)
            rep[law] = (C.refines(L, R), C.refines(R, L))      # (左→右 有效, 右→左 有效)
    IMPLS[(iface, ty)] = (methods, trust, rep)
    return rep

def law_ok(iface, ty, law):
    """改写引擎能用某条定律（左 → 右）的前提：实现是 checked，且该方向已被证明。declared 的实现不能用于推导。"""
    m = IMPLS.get((iface, ty))
    return bool(m) and m[1] == 'checked' and m[2][law][0]

def call(i, op, ins, ty): return node(i, op, ins=ins, outs=[(i, ty)])
# 定律的两边：(方法表, 类型) -> (体内 op, 结果值)
IFACES['Monoid'] = Interface('Monoid', {'unit': (0, 1), 'combine': (2, 1)}, [
    ('assoc', lambda m, t: ([call('l1', m['combine'], ['x0', 'x1'], t), call('q', m['combine'], ['l1', 'x2'], t)], 'q'),
              lambda m, t: ([call('r1', m['combine'], ['x1', 'x2'], t), call('q', m['combine'], ['x0', 'r1'], t)], 'q'), 3),
    ('left_id', lambda m, t: ([call('u', m['unit'], [], t), call('q', m['combine'], ['u', 'x0'], t)], 'q'),
                lambda m, t: ([], 'x0'), 1),
    ('right_id', lambda m, t: ([call('u', m['unit'], [], t), call('q', m['combine'], ['x0', 'u'], t)], 'q'),
                 lambda m, t: ([], 'x0'), 1),
])

def monomorphize(name, generic_body, iface, ty, ins, outs):
    """泛型复合 op（体里用 '接口.方法' 作 op 名）在类型 ty 处实例化：查实现表替换方法，再推导。"""
    if (iface, ty) not in IMPLS: raise TypeError(f'{ty} 没有实现 {iface}')
    meth = IMPLS[(iface, ty)][0]
    def sub(bs):
        return [dict(b, args=[(v, ty if t == 'Self' else t) for v, t in b['args']],
                     ops=[dict(n, op=meth.get(n['op'].split('.', 1)[1], n['op']) if n['op'].startswith(iface + '.') else n['op'],
                               outs=[(o, ty if t == 'Self' else t) for o, t in n['outs']], regions=[sub(r) for r in n['regions']])
                          for n in b['ops']]) for b in bs]
    return C.derive(f'{name}<{ty}>', [(p, ty if t == 'Self' else t) for p, t in ins],
                    [(p, ty if t == 'Self' else t) for p, t in outs], sub(generic_body))
