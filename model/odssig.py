"""由 tblgen JSON 自动构造 op 的类型签名，并给每个 op 记一笔导入账（第 9 轮）。
签名条目分级：
  ods      端口约束（Pred 树 + 内建原子）；有结构字段的 trait（AllTypesMatch、TypesMatchWith、AllElementTypesMatch、
           AllRanksMatch、AllShapesMatch、ShapedTypeMatchesElementCountAndTypes）；已知 C++ 实现的 native trait；tblgen 自动生成的类型推断
  cpred    没有结构字段的 PredOpTrait：按文本模式登记（OPPRED_RE），可审计
  declared C++ verify()/verifyRegions()、带 verify 的接口、C++ inferReturnTypes：只能声明；这里只记账，不实现
账目里的“未知”（unknown）就是导入缺口：某个条目既不能自动导入、也没有登记。
"""
import re, itertools, functools
import odsjson as O
from typesig import shaped, ranked, elt, compat_shape, _elementwise, i1same, encoding

def _base(d, c):
    """Variadic/Optional/VariadicOfVariadic → (基约束记录, 多重性)"""
    sc = d[c]['!superclasses']
    if 'VariadicOfVariadic' in sc: return O.ref(d[c]['baseType']), 'varvar'
    if 'Variadic' in sc: return O.ref(d[c]['baseType']), 'variadic'
    if 'Optional' in sc: return O.ref(d[c]['baseType']), 'optional'
    return c, 'one'

NATIVE_OK = {   # native trait → 关系（C++ 实现见 mlir/lib/IR/Operation.cpp 的 OpTrait::impl::verify*）
    'SameOperandsAndResultType': 'samec_all', 'SameTypeOperands': 'same_ops', 'SameOperandsShape': 'shape_ops',
    'SameOperandsAndResultShape': 'shape_all', 'SameOperandsElementType': 'elt_ops', 'SameOperandsAndResultElementType': 'elt_all',
    'Elementwise': 'elementwise'}
NATIVE_NOVERIFY = {'AlwaysSpeculatableImplTrait', 'RecursivelySpeculatableImplTrait', 'IsCommutative', 'Scalarizable', 'Vectorizable',
                   'Tensorizable', 'IsIdempotent', 'ConstantLike', 'HasRecursiveMemoryEffects', 'MemRefsNormalizable', 'InferTypeOpAdaptor'}
STRUCTURAL = re.compile(r'IsTerminator|HasParent<|SingleBlock|NoTerminator|IsIsolatedFromAbove|AffineScope|AutomaticAllocationScope|'
                        r'SymbolTable|NoRegionArguments|HasOnlyGraphRegion|ReturnLike|AttrSizedOperandSegments|AttrSizedResultSegments|HasParallelRegion|DistinctObjectsTrait')
# 无结构字段的 PredOpTrait：按“模板”登记（cpred 级，人工审计）。模板 = 规范化文本中把 $名 依次换成 $1 $2 … 后的样子。
_TD = '((::llvm::isa<::mlir::TensorType>($1.getType())?::llvm::cast<::mlir::TensorType>($1.getType()).getShape():::llvm::ArrayRef<int64_t>{}))'
_TD2 = _TD.replace('$1', '$2')
OPPRED_T = {
    f'({_TD}=={_TD2}&&{_TD2}=={_TD})': 'tdims',                                   # “输入与输出的 tensor 维相同”（arith 转换类 op）
}
OPPRED_RE = [   # 位置式的（$_op.getResult(i) / getOperand(j)）
    (r'\(getElementTypeOrSelf\(\$_op\.getResult\((\d+)\)\)==getElementTypeOrSelf\(\$_op\.getOperand\((\d+)\)\)\)',
     lambda r, o: ('elt_eq', [('res', int(r)), ('opd', int(o))])),
]
def _template(txt):
    names = list(dict.fromkeys(re.findall(r'\$(?!_)(\w+)', txt)))
    t = txt
    for i, n in enumerate(names): t = re.sub(r'\$' + n + r'\b', f'${i + 1}', t)
    return t, names
XF_OP = {   # TypesMatchWith 的 transformer（对 $_self）
    O.norm('$_self'): lambda t: t, O.norm('::llvm::cast<Type>($_self)'): lambda t: t,
    O.norm('::getI1SameShape($_self)'): i1same,
    **{O.norm(f'::llvm::cast<{k}>($_self).getElementType()'): (lambda t: t[2] if shaped(t) else None)
       for k in ('MemRefType', 'TensorType', 'RankedTensorType', 'ShapedType', 'VectorType', '::mlir::ShapedType')},
    O.norm('::llvm::cast<ComplexType>($_self).getElementType()'): lambda t: t[1] if t[0] == 'complex' else None,
    O.norm('SmallVector<Type, 2>(::llvm::cast<RankedTensorType>($_self).getNumElements(), ::llvm::cast<RankedTensorType>($_self).getElementType())'):
        lambda t: [t[2]] * _numel(t) if ranked(t) and t[0] == 'tensor' and None not in t[1] else None,
}
def _numel(t):
    n = 1
    for x in t[1]: n *= x
    return n

def _traits(d, t, out):
    tr = d[O.ref(t)]
    if 'TraitList' in tr['!superclasses']:
        for u in tr['traits']: _traits(d, u, out)
    else: out.append((O.ref(t), tr))
    return out

def _iface(d, trait):
    idx = d.setdefault('__iface__', {})
    if not idx:
        for n in d['!instanceof'].get('OpInterface', []):
            r = d[n]
            if not r.get('!anonymous') and r.get('trait'): idx[r['trait']] = r
    return idx.get(trait)

def _tp(dialect, b, S):
    """端口类型谓词。第 18 轮：含未登记原子（方言自有类型，如 !gpu.async.token、!transform.any_op）的约束 ⇒ 记入 unknown，谓词取 ⊤（上近似：
    不能因此拒绝；L29 要求的方向）。此前该处直接抛出，导致整个方言导入失败。"""
    try: return O.type_pred(dialect, b, O.ATOM, O.XFORM)
    except O.Unknown as e:
        txt = str(e.args[0])
        # 第 19 轮：isa<…Interface> 是开放世界——实现者可由外部模型在运行时附加（如 builtin tensor 经扩展实现 bufferization::TensorLikeType），
        # ODS 里看不到全部实现者 ⇒ 只能正向接受，不能据此拒绝（L43）。
        S['unknown'].append(('开放接口 isa: ' if 'Interface>' in txt else '端口类型原子: ') + txt[:70]); return lambda t: True

@functools.lru_cache(None)
def sig(dialect, opname):
    d = O.records(dialect); r = d[O.ops(dialect)[opname]]
    S = {'op': opname, 'ins': [], 'outs': [], 'attrs': [], 'rel': [], 'unknown': [], 'structural': [], 'declared': [],
         'regions': len(O.dag_args(r['regions'])), 'successors': len(O.dag_args(r['successors']))}
    unwrap = lambda c: O.ref(d[c]['constraint']) if c in d and 'OpVariable' in d[c]['!superclasses'] else c   # Arg<…>/Res<…> 包装
    for c, nm in O.dag_args(r['arguments']):
        c = unwrap(c); sc = d[c]['!superclasses'] if c in d else []
        if 'TypeConstraint' in sc:
            b, m = _base(d, c); S['ins'].append((nm, _tp(dialect, b, S), m, b))
        else:
            req = not (any(k in sc for k in ('OptionalAttr', 'DefaultValuedAttr', 'DefaultValuedOptionalAttr', 'UnitAttr', 'UnitProp'))
                       or c in ('UnitAttr', 'UnitProp') or d.get(c, {}).get('isOptional') == 1)       # UnitAttr 缺省即 false
            S['attrs'].append((nm, c, sc[-1] if sc else '?', req))
    for c, nm in O.dag_args(r['results']):
        b, m = _base(d, unwrap(c)); S['outs'].append((nm, _tp(dialect, b, S), m, b))
    if r['hasVerifier']: S['declared'].append('C++ verify()')
    if r['hasRegionVerifier']: S['declared'].append('C++ verifyRegions()')
    infer_iface = False
    for name, tr in [x for t in r['traits'] for x in _traits(d, t, [])]:
        sc = tr['!superclasses']; kind = sc[-1]
        if 'PredOpTrait' in sc:
            if kind in ('AllTypesMatch', 'AllElementTypesMatch', 'AllRanksMatch', 'AllShapesMatch'):
                S['rel'].append(({'AllTypesMatch': 'same', 'AllElementTypesMatch': 'elt_eq', 'AllRanksMatch': 'rank_eq', 'AllShapesMatch': 'shape_eq'}[kind],
                                 [('name', v) for v in tr['values']], 'ods'))
            elif kind in ('TypesMatchWith', 'OptionalTypesMatchWith'):
                k = O.norm(tr['transformer'])
                if k in XF_OP: S['rel'].append(('match', [('name', tr['lhs']), ('name', tr['rhs'])], 'ods', XF_OP[k], kind == 'OptionalTypesMatchWith'))
                else: S['unknown'].append(f'{kind} 变换: {" ".join(tr["transformer"].split())[:60]}')
            elif kind == 'ShapedTypeMatchesElementCountAndTypes':
                S['rel'].append(('count_types', [('name', tr['shaped']), ('name', tr['elements'])], 'ods'))
            elif kind == 'BooleanConditionOrMatchingShape':
                S['rel'].append(('cond_shape', [('name', 'condition'), ('name', 'result')], 'cpred'))   # 无字段，按类名登记（arith.select）
            else:
                s = set(); O.leaves(d, O.ref(tr['predicate']), s); txt = [x[1] for x in s if x[0] == 'atom']
                hit = None
                if len(txt) == 1:
                    t, names = _template(O.norm(txt[0]))
                    if t in OPPRED_T: hit = (OPPRED_T[t], names)
                    for rx, mk in OPPRED_RE:
                        m = re.fullmatch(rx, O.norm(txt[0]))
                        if m: hit = mk(*m.groups())
                if hit: S['rel'].append((hit[0], [x if isinstance(x, tuple) else ('name', x) for x in hit[1]], 'cpred'))
                else: S['unknown'].append(f'PredOpTrait: {tr.get("summary", "")[:60]}')
        elif 'NativeOpTrait' in sc or 'ParamNativeOpTrait' in sc or 'GenInternalOpTrait' in sc:
            t = tr.get('trait') or name
            if t in NATIVE_OK: S['rel'].append((NATIVE_OK[t], [], 'ods'))
            elif t in NATIVE_NOVERIFY: pass
            elif STRUCTURAL.search(t): S['structural'].append(t)
            else: S['unknown'].append(f'native trait {t}')
        elif 'OpInterfaceTrait' in sc or 'InterfaceTrait' in sc or 'DeclareOpInterfaceMethods' in sc or 'MemoryEffects' in sc:
            t = tr.get('trait', '')
            tr = _iface(d, t) or tr          # DeclareOpInterfaceMethods<X> 的匿名记录不带 X 的 verify 字段（第 9 轮发现的导入缺陷）
            if t.startswith('InferTypeOpInterface'): infer_iface = True
            if tr.get('verify'): S['declared'].append(f'接口 verify: {t}')
    # tblgen 自动推断：每个结果都能由“可构造类型 / SameOperandsAndResultType / AllTypesMatch / TypesMatchWith”得到时，生成 inferReturnTypes，
    # 而 InferTypeOpInterface 的 verifyTrait 要求实际结果类型 == 推断结果
    S['infer'] = _ods_infer(d, r, S)
    S['infer_cpp'] = infer_iface and S['infer'] is None
    if S['infer_cpp']: S['declared'].append('C++ inferReturnTypes')
    return S

def _ods_infer(d, r, S):
    if not S['outs'] or any(m != 'one' for _, _, m, _ in S['outs']): return None
    ins = [n for n, _, m, _ in S['ins']]
    rule = {}
    for nm, _, _, b in S['outs']:
        bc = d[b].get('builderCall') if b in d else None
        if bc: rule[nm] = ('build', bc)
    if any(x[0] == 'samec_all' for x in S['rel']) and S['ins'] and S['ins'][0][2] == 'one':
        for nm, *_ in S['outs']: rule.setdefault(nm, ('copy', ins[0]))
    changed = True
    while changed:
        changed = False
        for x in S['rel']:
            if x[0] == 'same':
                names = [v for _, v in x[1]]; src = next((v for v in names if v in ins or v in rule), None)
                for v in names:
                    if v not in ins and v not in rule and src and v in [o for o, *_ in S['outs']]: rule[v] = ('copy', src); changed = True
            if x[0] == 'match' and not x[4]:
                a, b = x[1][0][1], x[1][1][1]
                if b not in rule and b in [o for o, *_ in S['outs']] and (a in ins or a in rule): rule[b] = ('xf', a, x[3]); changed = True
    if all(o in rule for o, *_ in S['outs']): return rule
    return None

BUILD = {'$_builder.getIndexType()': ('index',), '$_builder.getI1Type()': ('i', 1, ''), '$_builder.getType<::mlir::shape::SizeType>()': ('shape.size',),
         '$_builder.getType<::mlir::shape::ShapeType>()': ('shape.shape',), '$_builder.getType<::mlir::shape::WitnessType>()': ('shape.witness',),
         '$_builder.getI64Type()': ('i', 64, ''), '$_builder.getI32Type()': ('i', 32, ''), '$_builder.getF32Type()': ('f', 'f32')}

def check(S, ins, outs, levels=('ods', 'cpred')):
    """ins/outs：按端口分组的类型列表（[[t…] 每组]）。返回违例列表。"""
    bad = []
    env = {}
    partial = outs is None                       # 结果端口组无法绑定：只检查操作数一侧（仍是上近似）
    if partial: outs = []
    for (nm, p, m, _), ts in zip(S['ins'] + (S['outs'] if not partial else []), ins + outs):
        env[nm] = ts
        if m == 'one' and len(ts) != 1 or m == 'optional' and len(ts) > 1: bad.append(f'{nm}: 个数'); continue
        if not all(p(t) for t in ts): bad.append(f'{nm}: 端口约束')
    opd = [t for ts in ins for t in ts]; res = [t for ts in outs for t in ts]
    def get(x):
        if x[0] == 'name': return env.get(x[1], [])
        return [(opd if x[0] == 'opd' else res)[x[1]]] if x[1] < len(opd if x[0] == 'opd' else res) else []
    E = lambda t: t[2] if shaped(t) else t
    for x in S['rel']:
        if x[2] not in levels: continue
        k = x[0]
        if partial and (not x[1] and k not in ('same_ops', 'shape_ops', 'elt_ops') or
                        any(a[0] == 'res' or a[0] == 'name' and a[1] not in [n for n, *_ in S['ins']] for a in x[1])): continue
        groups = [get(a) for a in x[1]]; ts = [t for g in groups for t in g]
        if k == 'samec_all':
            al = opd + res; ref = (res or opd)[:1]                # 第 24 轮：MLIR 的 verifySameOperandsAndResultType 逐个对照 result(0)，不是两两
            ok = all(E(a) == E(r) and compat_shape(a, r) and a[0] == r[0] for r in ref for a in al)
            ok = ok and len({encoding(t) for t in al if t[0] == 'tensor'}) <= 1      # 第 22 轮：verifySameOperandsAndResultType 要求编码相同
        elif k == 'same_ops': ok = all(a == opd[0] for a in opd)
        elif k == 'shape_ops': ok = all(compat_shape(a, b) for a, b in itertools.combinations(opd, 2))
        elif k == 'shape_all': ok = all(compat_shape(a, b) for a, b in itertools.combinations(opd + res, 2))
        elif k == 'elt_ops': ok = all(E(a) == E(opd[0]) for a in opd)
        elif k == 'elt_all': ok = all(E(a) == E((opd + res)[0]) for a in opd + res)
        elif k == 'elementwise': ok = _elementwise(opd, res)
        elif k == 'same': ok = all(t == ts[0] for t in ts)
        elif k == 'elt_eq': ok = all(E(t) == E(ts[0]) for t in ts)
        elif k == 'rank_eq': ok = all(ranked(t) for t in ts) and len({len(t[1]) for t in ts}) <= 1
        elif k == 'shape_eq': ok = all(ranked(t) for t in ts) and len({t[1] for t in ts}) <= 1
        elif k == 'tdims':
            sh = lambda t: t[1] if t[0] == 'tensor' and t[1] is not None else (None if t[0] == 'tensor' else ())
            ok = all(sh(t) is not None for t in ts) and len({sh(t) for t in ts}) <= 1
        elif k == 'match':
            a, b = groups
            if x[4] and (not a or not b): continue
            if not a: continue
            want = x[3](a[0]); ok = want is not None and (b == want if isinstance(want, list) else b == [want])
        elif k == 'count_types':
            sh, el = groups; ok = bool(sh) and ranked(sh[0]) and None not in sh[0][1] and len(el) == _numel(sh[0]) and all(t == sh[0][2] for t in el)
        elif k == 'cond_shape':
            c, rr = groups; ok = bool(c) and bool(rr) and (c[0] == ('i', 1, '') or (shaped(c[0]) and shaped(rr[0]) and ranked(c[0]) and ranked(rr[0]) and c[0][1] == rr[0][1]))
        else: ok = True
        if not ok: bad.append(f'{k}({",".join(str(a[1]) for a in x[1])})')
    if S['infer'] and 'ods' in levels and not partial:
        for nm, rl in S['infer'].items():
            if rl[0] == 'build': want = BUILD.get(rl[1].strip())
            elif rl[0] == 'copy': want = (env.get(rl[1]) or [None])[0]
            else:
                src = env.get(rl[1]) or []
                want = rl[2](src[0]) if src else None
            if want is None and rl[0] == 'build': continue          # 未登记的 builderCall：不检查
            if env.get(nm) != [want] and not (isinstance(want, list) and env.get(nm) == want): bad.append(f'推断: {nm}')
    return bad

def ledger(dialect, opname):
    """导入账：auto（ODS 自动、无缺口）/ cpred（用到登记的模式）/ declared（有 C++ 校验）/ unknown（有未覆盖条目）"""
    S = sig(dialect, opname)
    cls = 'unknown' if S['unknown'] else 'declared' if S['declared'] else 'cpred' if any(x[2] == 'cpred' for x in S['rel']) else 'auto'
    return cls, S

# ── 供 core 使用：按 op 名查签名；把扁平的操作数/结果表绑定到端口组 ──────────────────
@functools.lru_cache(None)
def lookup(op):
    dia = op.split('.')[0]
    if dia not in O.DIALECTS or op not in O.ops(dia): return None
    return ledger(dia, op)

def bind(S, flat, ports, seg=None):
    """flat：类型列表；ports：S['ins'] 或 S['outs']。只有一个非 one 组时可唯一绑定；否则需 seg（各组个数）。不可绑定 ⇒ None。"""
    if seg is not None:
        if len(seg) != len(ports) or sum(seg) != len(flat): return None
        out, i = [], 0
        for k in seg: out.append(flat[i:i + k]); i += k
        return out
    multi = [j for j, p in enumerate(ports) if p[2] != 'one']
    fixed = len(ports) - len(multi)
    if len(multi) > 1: return None
    if not multi: return [[t] for t in flat] if len(flat) == fixed else None
    k = len(flat) - fixed
    if k < 0 or (ports[multi[0]][2] == 'optional' and k > 1): return None
    j = multi[0]; return [[t] for t in flat[:j]] + [flat[j:j + k]] + [[t] for t in flat[j + k:]]


def region_cons(opname):
    """ODS 的区域约束 → [(名字, 下界块数, 上界块数或 None, 可变个数)]。没有导入该 op ⇒ None。
    SizedRegion<n>：恰 n 块；MaxSizedRegion<n>：至多 n 块（可为空）；AnyRegion：任意（可为空）；VariadicRegion<R>：R 可重复。"""
    import re as _re
    dia = opname.split('.')[0]
    if dia not in O.DIALECTS: return None
    d = O.records(dia); ops = O.ops(dia)
    if opname not in ops: return None
    out = []
    for c, nm in O.dag_args(d[ops[opname]]['regions']):
        r = d.get(c, {}); sc = r.get('!superclasses', []); sm = r.get('summary', '') or ''
        var = 'VariadicRegion' in sc
        m = _re.search(r'with (at most )?(\d+) blocks?', sm)
        if m and not m.group(1) and 'MaxSizedRegion' not in sc: lo, hi = int(m.group(2)), int(m.group(2))
        elif m: lo, hi = 0, int(m.group(2))
        else: lo, hi = 0, None
        out.append((nm, lo, hi, var))
    return out
