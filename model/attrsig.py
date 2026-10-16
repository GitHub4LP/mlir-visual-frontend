"""第 13 轮：属性层 —— op 的属性参数与属性约束从 ODS 导入；属性值的文法与查询交给 MLIR 自己的属性解析器。

教训（第 12 轮 corpus.md）：手写的文法与假设在真实 IR 前全错，导入的全对。所以这里不手写属性文法：
  属性值 = MLIR Attribute 对象（ir.Attribute.parse），谓词原子是对它的查询；
  属性参数集合 = ODS arguments 里的属性项（名字、必需/可选/缺省/Unit、约束 Pred 树）；
  方言属性的 isa（::mlir::arith::FastMathFlagsAttr）→ 由 ODS 的 AttrDef 记录（cppNamespace::cppClassName → attrName）判定；
  接口原子（TypedAttr、DeviceMappingAttrInterface…）MLIR Python 绑定无法查询 ⇒ 记为上近似（恒真），计入导入账。

对前端的产物（指导性）：每个属性参数的 **取值域** —— 必需性、缺省值、枚举取值、整数宽度与范围、数组元素约束 ——
即属性编辑器（下拉、数字框、开关）与补全的依据；违反时给出的诊断与 MLIR 的 “attribute 'x' failed to satisfy constraint” 对应。
"""
import re, functools
import odsjson as O

UPPER = 'upper'      # 上近似原子的标记
STATS = {}

def _ctx():
    import typesig as TS
    if not TS._CTX:
        import mlir.ir as ir
        TS._CTX.append(ir.Context()); TS._CTX[0].allow_unregistered_dialects = True
    return TS._CTX[0]

@functools.lru_cache(None)
def parse(txt):
    import mlir.ir as ir
    with _ctx(): return ir.Attribute.parse(txt)

@functools.lru_cache(None)
def attrdefs():
    """C++ 类名 → 方言属性名（'::mlir::arith::FastMathFlagsAttr' → 'arith.fastmath'），取自全部方言的 AttrDef 记录。"""
    out = {}
    for dia in O.DIALECTS:
        for k, r in O.records(dia).items():
            if isinstance(r, dict) and 'AttrDef' in r.get('!superclasses', []) and r.get('cppClassName') and r.get('attrName'):
                out[O.norm(f"{r['cppNamespace']}::{r['cppClassName']}")] = r['attrName']
    return out

def _A(cls):
    import mlir.ir as ir
    C = getattr(ir, cls)
    return lambda a: C.isinstance(a)

def _int(a):
    import mlir.ir as ir
    return ir.IntegerAttr(a) if ir.IntegerAttr.isinstance(a) else None

def _ity(a, pred):
    i = _int(a)
    return i is not None and pred(i.type)

def _sl(w):
    import mlir.ir as ir
    return lambda t: ir.IntegerType.isinstance(t) and ir.IntegerType(t).is_signless and ir.IntegerType(t).width == w

def _apint(a):
    i = _int(a)
    if i is None: return None
    import mlir.ir as ir
    w = ir.IntegerType(i.type).width if ir.IntegerType.isinstance(i.type) else 64
    v = i.value
    return v, w

@functools.lru_cache(None)
def specialized_enums():
    """genSpecializedAttr 的 I32/I64EnumAttr：C++ 类（如 ::mlir::arith::CmpFPredicateAttr）是带取值检查的 IntegerAttr。
    → (位宽, 合法取值集合 或 None, 位枚举的合法位掩码 或 None)。"""
    out = {}
    for dia in O.DIALECTS:
        d = O.records(dia)
        for k, r in d.items():
            if isinstance(r, dict) and 'EnumAttrInfo' in r.get('!superclasses', []) and r.get('genSpecializedAttr') and r.get('specializedAttrClassName'):
                vals = None if r.get('validBits') is not None else {d[O.ref(e)]['value'] for e in r.get('enumerants', []) if O.ref(e) in d}
                out[O.norm(f"{r['cppNamespace']}::{r['specializedAttrClassName']}")] = (r['bitwidth'], vals, r.get('validBits'))
    return out

def _dialect_isa(cpp):
    """方言属性类的 isa。在导入时解析（解析不了 ⇒ Unknown，进导入账），不在求值时才发现。"""
    nm = attrdefs().get(cpp)
    if nm is not None:
        dia, _, mn = nm.partition('.')
        def f(a):
            s = str(a)                                # MLIR 有两种打印：#arith.fastmath<…> 与 #index<cmp_predicate …>、#shard<axisarray[[0]]>
            if s == '#' + nm or s.startswith('#' + nm + '<'): return True
            m = re.match(r'#([\w$]+)<([\w$.]+)', s)    # 第 18 轮：助记符是关键字，按“最长标识符”取，不假设其后是空格或 '>'（shard 误拒 6 例）
            return bool(m) and m.group(1) == dia and m.group(2) == mn
        return f
    se = specialized_enums().get(cpp)
    if se is not None:
        w, vals, bits = se
        def g(a):
            i = _int(a)
            if i is None or not _sl(w)(i.type): return False
            v = i.value % (1 << w)
            return (v & ~bits) == 0 if bits is not None else v in vals
        return g
    raise O.Unknown(('attrclass', cpp))

def _typed(a):
    import mlir.ir as ir
    return ir.StringAttr.isinstance(a) or str(a.type) != 'none'

def _typeof_attr(a, pred):
    import mlir.ir as ir
    return ir.TypeAttr.isinstance(a) and pred(ir.TypeAttr(a).value)

ATOM_RE = [
    (r'\(::llvm::isa<::mlir::(IntegerAttr|ArrayAttr|StringAttr|UnitAttr|BoolAttr|FlatSymbolRefAttr|TypeAttr|AffineMapAttr|DenseI64ArrayAttr|DenseI32ArrayAttr|DenseIntElementsAttr)>\(\$_self\)\)',
     lambda c: _A({'DenseIntElementsAttr': 'DenseIntElementsAttr'}.get(c, c))),
    (r'\(::llvm::isa<::mlir::DictionaryAttr>\(\$_self\)\)', lambda: _A('DictAttr')),
    (r'\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getType\(\)\.isSignlessInteger\((\d+)\)\)', lambda w: lambda a: _ity(a, _sl(int(w)))),
    (r'\(::llvm::isa<::mlir::IndexType>\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getType\(\)\)\)',
     lambda: lambda a: _ity(a, lambda t: __import__('mlir.ir', fromlist=['x']).IndexType.isinstance(t))),
    (r'\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getValue\(\)\.isStrictlyPositive\(\)\)', lambda: lambda a: (_apint(a) or (0, 0))[0] > 0),
    (r'\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getValue\(\)\.isPowerOf2\(\)\)',
     lambda: lambda a: (lambda p: p is not None and (lambda u: u > 0 and u & (u - 1) == 0)(p[0] % (1 << p[1]) if p[1] else 0))(_apint(a))),
    (r'\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getInt\(\)>=(-?\d+)\)', lambda n: lambda a: _int(a).value >= int(n)),
    (r'\(::llvm::cast<::mlir::IntegerAttr>\(\$_self\)\.getInt\(\)<=(-?\d+)\)', lambda n: lambda a: _int(a).value <= int(n)),
    (r'\(::llvm::isa<::mlir::FunctionType>\(::llvm::cast<::mlir::TypeAttr>\(\$_self\)\.getValue\(\)\)\)',
     lambda: lambda a: _typeof_attr(a, lambda t: __import__('mlir.ir', fromlist=['x']).FunctionType.isinstance(t))),
    (r'\(::llvm::isa<::mlir::MemRefType>\(::llvm::cast<::mlir::TypeAttr>\(\$_self\)\.getValue\(\)\)\)',
     lambda: lambda a: _typeof_attr(a, lambda t: __import__('mlir.ir', fromlist=['x']).MemRefType.isinstance(t))),
    (r'\(::llvm::cast<::mlir::DenseIntElementsAttr>\(\$_self\)\.getType\(\)\.getElementType\(\)\.isIndex\(\)\)',
     lambda: lambda a: 'xindex>' in str(__import__('mlir.ir', fromlist=['x']).DenseIntElementsAttr(a).type)),
    (r'\(true\)', lambda: lambda a: True),
    (r'\(::llvm::isa<(::mlir::\w+::\w+Attr)>\(\$_self\)\)', lambda c: _dialect_isa(c)),
    # 接口原子：Python 绑定没有接口查询。TypedAttr 与 PoisonAttrInterface 按 MLIR 的实现登记（cpred 级登记，经 A2 差分审计）；
    # 设备映射接口的实现者在未导入的方言（gpu 等）里 ⇒ 上近似
    (r'\(::llvm::isa<::mlir::TypedAttr>\(\$_self\)\)', lambda: lambda a: _typed(a)),
    (r'\(::llvm::isa<::mlir::ub::PoisonAttrInterface>\(\$_self\)\)', lambda: lambda a: str(a).startswith('#ub.poison')),
    (r'\(::llvm::isa<::mlir::(DeviceMappingAttrInterface|DeviceMaskingAttrInterface)>\(\$_self\)\)', lambda c: UPPER),
]
# 类型谓词作用在 TypeAttr 的值上（SubstLeaves '$_self' → getValue()）：借用类型层的原子，类型由 MLIR 文本转 core 类型项
def _type_of(a):
    import mlir.ir as ir, typesig as TS
    return TS.parse(str(ir.TypeAttr(a).value)) if ir.TypeAttr.isinstance(a) else None

class _Atoms(dict):
    def __init__(self): super().__init__(); self.upper = set()
    def get(self, k, default=None):
        for rx, mk in ATOM_RE:
            m = re.fullmatch(rx, k)
            if m:
                f = mk(*m.groups())
                if f == UPPER: self.upper.add(k); return lambda a: True
                return f
        t = O.ATOM.get(k)                    # 类型原子：$_self 已被换成类型
        return t
    def __contains__(self, k): return self.get(k) is not None
    def __getitem__(self, k):
        v = self.get(k)
        if v is None: raise KeyError(k)
        return v

ATOMS = _Atoms()
XFORM = {O.norm('::llvm::cast<::mlir::TypeAttr>($_self).getValue()'): _type_of}

def _interp(d, name):
    """odsjson.interp 加上属性数组的 Concat 形状（all_of(ArrayAttr, [&](Attribute attr){ return attr && (…); })）。"""
    if name is None or name not in d: raise O.Unknown(('pred-ref', name))      # 第 18 轮：新方言里有非 def 引用的谓词子项 ⇒ 进导入账，不崩
    r = d[name]; sc = r['!superclasses']
    if 'Concat' in sc and 'ArrayAttr' in r.get('prefix', ''):
        if O.norm(r['prefix']) != O.norm('::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>($_self), [&](::mlir::Attribute attr) { return attr && ('):
            raise O.Unknown(('concat', r['prefix']))
        c = _interp(d, O.ref((r.get('children') or [r.get('child')])[0]))
        import mlir.ir as ir
        return lambda env: ir.ArrayAttr.isinstance(env['$_self']) and all(c({**env, '$_self': e, 'attr': e}) for e in ir.ArrayAttr(env['$_self']))
    if 'SubstLeaves' in sc and r['replacement'].strip() == 'attr':
        c = _interp(d, O.ref((r.get('children') or [r.get('child')])[0])); return lambda env: c({**env, '$_self': env['attr']})
    if any(k in sc for k in ('And', 'Or', 'Neg')):
        kids = [_interp(d, O.ref(x)) for x in (r.get('children') or [r.get('child')])]
        if 'And' in sc: return lambda env: all(k(env) for k in kids)
        if 'Or' in sc: return lambda env: any(k(env) for k in kids)
        return lambda env: not kids[0](env)
    if 'SubstLeaves' in sc:
        k = O.norm(r['replacement'].strip())
        if r['pattern'] != '$_self' or k not in XFORM: raise O.Unknown(('subst', r['replacement'].strip()))
        c = _interp(d, O.ref((r.get('children') or [r.get('child')])[0])); x = XFORM[k]
        return lambda env: (lambda v: v is not None and c({**env, '$_self': v}))(x(env['$_self']))
    return O.interp(d, name, ATOMS, {})

def _unwrap(d, con):
    """OptionalAttr / DefaultValuedAttr / DefaultValuedOptionalAttr / UnitAttr → (必需性, 缺省值文本, 约束记录名)。"""
    r = d[con]; sc = r.get('!superclasses', [])
    if con == 'UnitAttr' or 'UnitAttr' in sc: return 'unit', None, con
    if 'DefaultValuedOptionalAttr' in sc: return 'default', r.get('defaultValue'), O.ref(r['baseAttr']) or con
    if 'DefaultValuedAttr' in sc: return 'default', r.get('defaultValue'), O.ref(r['baseAttr']) or con
    if 'OptionalAttr' in sc: return 'optional', None, O.ref(r['baseAttr']) or con
    return 'required', None, con

def _domain(d, con):
    """取值域（给前端编辑器）：枚举 → 取值列表；整数 → 宽度；数组 → 元素约束摘要。"""
    r = d.get(con, {})
    ens = r.get('enumerants') or (d.get(O.ref(r.get('enum')) or '', {}) or {}).get('enumerants')
    if ens:
        names = [d[O.ref(e)].get('str') or d[O.ref(e)].get('symbol') for e in ens if O.ref(e) in d]
        return {'enum': names, 'bitenum': 'BitEnumAttr' in str(d.get(O.ref(r.get('enum')) or '', {}).get('!superclasses', [])) or 'BitEnum' in str(r.get('!superclasses'))}
    return {'summary': r.get('summary')}

@functools.lru_cache(None)
def attr_args(opname):
    """ODS 中 op 的属性参数：[dict(name, need, default, cons, pred, upper, unknown, domain)]；未导入的 op ⇒ None。"""
    dia = opname.split('.')[0]
    if dia not in O.DIALECTS or opname not in O.ops(dia): return None
    d = O.records(dia); r = d[O.ops(dia)[opname]]; out = []
    for con, nm in O.dag_args(r['arguments']):
        rc = d.get(con, {}); sc = rc.get('!superclasses', [])
        if not ('Attr' in sc or 'AttrConstraint' in sc or 'Property' in sc or 'PropConstraint' in sc): continue
        need, dflt, base = _unwrap(d, con)
        ent = dict(name=nm, need=need, default=dflt, cons=base, pred=None, unknown=None, upper=False, domain=_domain(d, base),
                   prop='Property' in sc or 'PropConstraint' in sc)
        p = O.ref(d[base].get('predicate')) if base in d else None
        if ent['prop'] and p is None: ent['unknown'] = ('property-no-pred', con)
        elif p is None: ent['pred'] = lambda env: True
        else:
            before = set(ATOMS.upper)
            try: ent['pred'] = _interp(d, p)
            except O.Unknown as e: ent['unknown'] = e.args[0]
            ent['upper'] = ATOMS.upper != before or bool(_has_upper(d, p))
        out.append(ent)
    return out

def _has_upper(d, p):
    s = set(); O.leaves(d, p, s)
    return any(ATOMS.get(O.norm(t)) is not None and O.norm(t) in ATOMS.upper for k, t in s if k == 'atom')

def check(opname, attrs):
    """attrs: 名字 → MLIR 属性文本。返回 [(属性名, 消息)]：缺必需属性、违反约束。未导入 op ⇒ []。
    额外的属性名不算错（MLIR 允许可丢弃属性）。"""
    aa = attr_args(opname)
    if aa is None: return []
    out = []
    for e in aa:
        if e['name'] not in attrs:
            if e['need'] == 'required': out.append((e['name'], f"缺必需属性 '{e['name']}'"))
            continue
        if e['pred'] is None: continue
        try: a = parse(attrs[e['name']])
        except Exception: out.append((e['name'], '属性文本无法解析')); continue
        try: ok = e['pred']({'$_self': a})
        except Exception as ex:                 # 原子在此值上无定义 ⇒ 不下结论（上近似），但记账
            STATS['求值无定义'] = STATS.get('求值无定义', 0) + 1; ok = True
        if not ok: out.append((e['name'], f"'{e['name']}' 不满足 {e['cons']}"))
    return out

def ledger():
    """导入账：属性参数总数、精确（全部原子已知）/上近似（含接口原子）/未知。"""
    import collections
    c = collections.Counter(); unk = collections.Counter()
    for dia in O.DIALECTS:
        for op in O.ops(dia):
            for e in attr_args(op) or []:
                k = 'unknown' if e['unknown'] else 'upper' if e['upper'] else 'exact'
                c[k] += 1
                if e['unknown']: unk[str(e['unknown'])[:90]] += 1
    return c, unk


# ── core 参数 → ODS 属性（第 13 轮：唯一的翻译处；oracle.emit_core 与 core.typecheck 共用）
class NotMLIR(LookupError): pass        # core 参数在 MLIR 这个 op 上没有对应属性 ⇒ 该实例不可降级到 MLIR
class Symbolic(LookupError): pass       # 符号参数（模式变量、复合 op 的 $参数）⇒ 不是具体实例，不下结论

def from_core(n):
    """core 节点的参数 → {ODS 属性名: MLIR 属性文本}。已导入的节点（props['_attrs']）原样返回。"""
    p = dict(n.get('props', {}))
    if '_attrs' in p: return dict(p['_attrs'])
    aa = attr_args(n['op']); names = {e['name'] for e in aa} if aa is not None else set()
    out = {}
    def own(a):
        if a not in names: raise NotMLIR(f"core 参数在 MLIR {n['op']} 上没有对应属性 '{a}'")
    for k, v in p.items():
        if (isinstance(v, str) and v.startswith('$')) or not isinstance(v, (int, bool, str, float)): raise Symbolic(k)   # $参数、z3 模式变量
    if 'value' in p:
        v = p.pop('value')
        if not (isinstance(v, (int, bool)) or re.fullmatch(r'-?\d+|true|false', str(v))): raise Symbolic('value')
        own('value'); out['value'] = f"{str(v).lower() if isinstance(v, bool) else v} : {p.get('type') or n['outs'][0][1]}"
    p.pop('type', None)
    fl = [k for k in ('nsw', 'nuw') if p.pop(k, False) is True]
    if fl: own('overflowFlags'); out['overflowFlags'] = f"#arith.overflow<{', '.join(fl)}>"
    if 'predicate' in p and isinstance(p['predicate'], int): own('predicate'); out['predicate'] = f"{p.pop('predicate')} : i64"
    if 'exact' in p:
        if p.pop('exact') is True: own('isExact'); out['isExact'] = 'unit'
    if p: raise NotMLIR(f'未登记的 core 参数 {sorted(p)}')
    return out

def rel_check(opname, attrs, ins, outs):
    """ODS 中引用属性的 op 级关系（目前全部 12 方言只有 arith.constant 的 AllTypesMatch<[value, result]>）。
    ins/outs：core 类型项（typesig.parse）。"""
    dia = opname.split('.')[0]
    if dia not in O.DIALECTS or opname not in O.ops(dia): return []
    d = O.records(dia); r = d[O.ops(dia)[opname]]
    names_in = [nm for con, nm in O.dag_args(r['arguments']) if 'TypeConstraint' in d.get(con, {}).get('!superclasses', [])]
    names_out = [nm for con, nm in O.dag_args(r['results'])]
    an = {e['name'] for e in attr_args(opname)}
    def ty(nm):
        if nm in an:
            if nm not in attrs: return None
            import typesig as TS
            a = parse(attrs[nm]); t = str(a.type)
            return None if t == 'none' else TS.parse(t)
        if nm in names_out and len(names_out) == len(outs): return outs[names_out.index(nm)]
        if nm in names_in and len(names_in) == len(ins): return ins[names_in.index(nm)]
        return None
    out = []
    for t in r.get('traits', []):
        tr = d.get(O.ref(t), {})
        if 'AllTypesMatch' in tr.get('!superclasses', []) and any(v in an for v in tr.get('values', [])):
            ts = [ty(v) for v in tr['values']]
            if None not in ts and len(set(ts)) > 1: out.append(f"{{{', '.join(tr['values'])}}} 类型不同")
    return out

def node_check(n, ins, outs):
    """core.typecheck 的属性层：返回诊断列表（终判，与自动签名同级：A1/A2 证明不误拒）。
    已导入节点：必需性 + 约束 + 关系。core 形式的节点：翻译后的属性做约束 + 关系；必需性不查（core 的简写由降级补齐，如切片的 static_*）。"""
    if attr_args(n['op']) is None: return []
    try: at = from_core(n)
    except Symbolic: return []
    except NotMLIR as e: return [str(e)]
    try:
        imported = '_attrs' in n.get('props', {})
        d = [m for k, m in check(n['op'], at) if imported or not m.startswith('缺必需属性')]
        return d + rel_check(n['op'], at, ins, outs)
    except Exception as e:
        STATS['属性层异常'] = STATS.get('属性层异常', 0) + 1; return []
