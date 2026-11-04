"""第 66 轮：把 C++ verify / 接口 verify 中的类型规则规范化为约束原语（用户 2026-11-04：“这就是我说的超越，就是要建模”）。

规范化原子与 ODS 关系同构：('nf', 引用端口组, 'nf', 呈现文本, 判定函数)，挂在 odssig 签名的 S['rel'] 上，
所以 core.typecheck（逐点）、odscomp.derive（复合 op 的 Φ）、projcomp（单端口范围）都直接使用，内置 op 与复合 op 同一套机制。
一个 op 的规则只有在 nf_test 中与 MLIR verifier 精确一致（误拒 0、误收 0）后才登记进 REG；
登记后它覆盖的 declared 条目（C++ verify、接口 verify）从签名残差中移出，记入 S['nf_covered']。

原语（只依赖类型项，见 typesig 的表示）：
  under(t, 容器集, 元素种类集)  MLIR getUnderlyingType：t 是标量或允许的容器，且元素属于允许的种类 ⇒ 元素类型；否则 None
  width(e)                      getIntOrFloatBitWidth（整数位宽 / 浮点 FloatType::getWidth）
  same_enc(a, b)                hasSameEncoding：两者都是有秩 tensor 时编码相同
  no_unranked_tensor(t)         MLIR 在此崩溃 ⇒ 规范化为拒绝（见 _valid）
  compat(a, b)                  verifyCompatibleShape(Type, Type)：同为非 shaped，或都 shaped 且（任一无秩，或秩相同且静态维相等）
  signless(e) / is_index(e)
依据：MLIR 9bae84b0 inventory/cpp/Arith_IR_ArithOps.cpp（verifyExtOp/verifyTruncateOp/checkWidthChangeCast/checkIntFloatCast/
areIndexCastCompatible/BitcastOp::areCastCompatible）与 CastOpInterface 的 verifyCastInterfaceOp（至少一个结果 + areCastCompatible）。
"""
import functools
import typesig as TS

@functools.lru_cache(None)
def _fwidth(name):
    import mlir.ir as ir
    with ir.Context():
        return ir.FloatType(ir.Type.parse(name)).width

def kind(e):
    return {'i': 'int', 'f': 'float', 'index': 'index'}.get(e[0])

def width(e):
    if e is None: return None
    if e[0] == 'i': return e[1]
    if e[0] == 'f': return _fwidth(e[1])
    return None

def under(t, conts, kinds):
    if TS.shaped(t) and t[0] not in conts: return None
    e = TS.elt(t)
    return e if kind(e) in kinds else None

def same_enc(a, b):
    if not (a[0] == b[0] == 'tensor' and a[1] is not None and b[1] is not None): return True
    return TS.encoding(a) == TS.encoding(b)

def compat(a, b):
    sa, sb = TS.shaped(a), TS.shaped(b)
    if not sa: return not sb
    if not sb: return False
    if a[1] is None or b[1] is None: return True
    return len(a[1]) == len(b[1]) and all(x is None or y is None or x == y for x, y in zip(a[1], b[1]))

def signless(e): return e is not None and e[0] == 'i' and e[2] == ''
def is_index(e): return e is not None and e[0] == 'index'

VT, VTM = ('vector', 'tensor'), ('vector', 'tensor', 'memref')

# ───────────────────────── arith 转换族
def no_unranked_tensor(t): return not (t[0] == 'tensor' and t[1] is None)
# MLIR 9bae84b0 的 arith 转换族在任一端是无秩 tensor 时断言崩溃（cast<RankedTensorType>，nf_test 试跑发现）：
# MLIR 不能产出合法 IR ⇒ 规范化为拒绝（前端不得生成），这是对 MLIR 的补全而非复述。
def _valid(a, b): return no_unranked_tensor(a) and no_unranked_tensor(b) and same_enc(a, b) and compat(a, b)

def _width_change(kinds, cmp, sym):
    def f(a, b):
        if not _valid(a, b): return False
        s, d = under(a, VT, kinds), under(b, VT, kinds)
        return s is not None and d is not None and cmp(width(d), width(s))
    return f, f'ranked?(in,out) ∧ compat(in,out) ∧ enc(in)=enc(out) ∧ in,out∈{{标量,vector,tensor}}⟨{"/".join(kinds)}⟩ ∧ width(elt(out)) {sym} width(elt(in))'

def _int_float(src, dst):
    def f(a, b): return _valid(a, b) and under(a, VT, src) is not None and under(b, VT, dst) is not None
    return f, f'ranked?(in,out) ∧ compat(in,out) ∧ enc(in)=enc(out) ∧ in∈{{标量,vector,tensor}}⟨{"/".join(src)}⟩ ∧ out∈{{标量,vector,tensor}}⟨{"/".join(dst)}⟩'

def _index_cast():
    def f(a, b):
        if not _valid(a, b): return False
        s, d = under(a, VTM, ('int', 'index')), under(b, VTM, ('int', 'index'))
        return s is not None and d is not None and (is_index(s) and signless(d) or signless(s) and is_index(d))
    return f, 'ranked?(in,out) ∧ compat(in,out) ∧ enc(in)=enc(out) ∧ in,out∈{标量,vector,tensor,memref} ∧ {elt(in),elt(out)} = {index, 无符号性整数}'

def _bitcast():
    def f(a, b):
        if not _valid(a, b): return False
        s, d = under(a, VTM, ('int', 'float')), under(b, VTM, ('int', 'float'))
        return s is not None and d is not None and width(s) == width(d)
    return f, 'ranked?(in,out) ∧ compat(in,out) ∧ enc(in)=enc(out) ∧ in,out∈{标量,vector,tensor,memref}⟨int/float⟩ ∧ width(elt(in)) = width(elt(out))'

import operator as _op
CAST_COVERS = {'C++ verify()', '接口 verify: CastOpInterface::Trait'}
_CAST = {
    'arith.extui': _width_change(('int',), _op.gt, '>'), 'arith.extsi': _width_change(('int',), _op.gt, '>'),
    'arith.trunci': _width_change(('int',), _op.lt, '<'), 'arith.extf': _width_change(('float',), _op.gt, '>'),
    'arith.truncf': _width_change(('float',), _op.lt, '<'),
    'arith.uitofp': _int_float(('int',), ('float',)), 'arith.sitofp': _int_float(('int',), ('float',)),
    'arith.fptoui': _int_float(('float',), ('int',)), 'arith.fptosi': _int_float(('float',), ('int',)),
    'arith.index_cast': _index_cast(), 'arith.index_castui': _index_cast(), 'arith.bitcast': _bitcast(),
}

def _cast_rules(S):
    fn, txt = _CAST[S['op']]
    i, o = S['ins'][0][0], S['outs'][0][0]
    return [('nf', [('name', i), ('name', o)], 'nf', txt, lambda A, B, fn=fn: len(A) == 1 and len(B) == 1 and fn(A[0], B[0]), False)]

# ───────────────────────── 第 67 轮：tensor/memref/vector 的一批（规则可读取 op 的属性：attrs = {属性名: MLIR 属性文本}）
import re as _re
def ints(txt):
    """array<i64: 1, 0> / [1, 0] → [1, 0]；None ⇒ None。"""
    if txt is None: return None
    t = txt.strip()
    body = (t.split(':', 1)[1] if ':' in t else '') if t.startswith('array<') else t     # array<i64> 是空数组（i64 不是元素）
    return [int(x) for x in _re.findall(r'-?\d+', body)]
def rank(t): return len(t[1]) if TS.shaped(t) and t[1] is not None else None
def numel(t):
    n = 1
    for d in t[1]: n *= d
    return n
def nscalable(t): return sum(TS.scalable(t))

def _r(refs, txt, fn, attrs=False):
    return ('nf', [('name', r) for r in refs], 'nf', txt, fn, attrs)

def _dim(S):          # ShapedDimOpInterface：恰一个结果且为 index（ODS 结果约束 Index 已蕴含）
    return [_r(['result'], 'result = index', lambda R: len(R) == 1 and R[0] == ('index',))]
def _load(S):         # LoadOp::verify：#indices = rank(memref)
    return [_r(['memref', 'indices'], '#indices = rank(memref)', lambda M, I: len(M) == 1 and rank(M[0]) is not None and len(I) == rank(M[0]))]
def _store(S):        # StoreOp::verify：#操作数 = 2 + rank(memref)
    return [_r(['memref', 'indices'], '#indices = rank(memref)', lambda M, I: len(M) == 1 and rank(M[0]) is not None and len(I) == rank(M[0]))]
def _tcast(S):        # tensor.CastOp::areCastCompatible：同元素类型 ∧ verifyCompatibleShape（不比较编码）
    return [_r(['source', 'dest'], 'elt(in) = elt(out) ∧ compat(in,out)',
               lambda A, B: len(A) == len(B) == 1 and A[0][0] == B[0][0] == 'tensor' and A[0][2] == B[0][2] and compat(A[0], B[0]))]
def broadcastable(s, d):
    """vector::isBroadcastableTo(s, d)。"""
    if d[0] != 'vector': return False
    if not TS.shaped(s) and s[0] in ('i', 'f', 'index') and s == d[2]: return True
    if s[0] != 'vector': return False
    sr, dr = len(s[1]), len(d[1])
    if sr > dr: return False
    lead = dr - sr; ss, ds = TS.scalable(s), TS.scalable(d)
    for k in range(sr):
        a, b, fa, fb = s[1][k], d[1][lead + k], ss[k], ds[lead + k]
        if a != 1 and a != b: return False
        if (a == 1 and fa and b != 1) or (fa != fb and (a != 1 or fa)): return False
    return True
def _bcast(S):
    return [_r(['source', 'vector'], 'broadcastable(source, vector)：标量=elt(vector)，或尾部各维相等或源为 1（可伸缩标志按 isBroadcastableTo）',
               lambda A, B: len(A) == len(B) == 1 and broadcastable(A[0], B[0]))]
def _shape_cast(S):
    return [_r(['source', 'result'], 'elt 相同 ∧ numel 相同 ∧ 可伸缩维个数相同',
               lambda A, B: len(A) == len(B) == 1 and A[0][2] == B[0][2] and numel(A[0]) == numel(B[0]) and nscalable(A[0]) == nscalable(B[0]))]
def _transpose(S):
    def f(A, B, attrs):
        p = ints(attrs.get('permutation'))
        if p is None or len(A) != 1 or len(B) != 1: return False
        a, b = A[0], B[0]; r = len(b[1])
        if len(a[1]) != r or len(p) != r or sorted(p) != list(range(r)): return False
        return all(b[1][i] == a[1][p[i]] for i in range(r))
    return [_r(['vector', 'result'], 'rank 相同 ∧ permutation 是 0..rank-1 的排列 ∧ dim(result,i) = dim(vector,perm[i])', f, attrs=True)]

# ───────────────────────── 第 68 轮：切片族（tensor.extract_slice / insert_slice）
DYN = -(1 << 63)                       # ShapedType::kDynamic
def offsets_sizes_strides_ok(rank_, nO, nS, nT, attrs):
    """OffsetSizeAndStrideOpInterface（verifyOffsetSizeAndStrideOp，maxRanks = {rank, rank, rank}）：
    三个静态数组长度都为 rank；各自的动态操作数个数 = 数组中 kDynamic 的个数；静态 offset/size 非负。返回 (o, s, t) 或 None。"""
    o, sz, st = (ints(attrs.get(k)) for k in ('static_offsets', 'static_sizes', 'static_strides'))
    if None in (o, sz, st) or rank_ is None: return None
    if (nS, nT, nO) != (sz.count(DYN), st.count(DYN), o.count(DYN)): return None
    if not (len(o) == len(sz) == len(st) == rank_): return None
    if any(x < 0 and x != DYN for x in o + sz): return None
    return o, sz, st
def in_bounds(shape, o, sz, st):
    """verifyInBoundsSlice：静态源维上，offset < dim，且 offset + (size-1)*stride < dim（size/stride 静态时）。"""
    for i, d in enumerate(shape):
        if d is None or o[i] == DYN: continue
        if o[i] >= d: return False
        if sz[i] == DYN or st[i] == DYN: continue
        if o[i] + (sz[i] - 1) * st[i] >= d: return False
    return True
def rank_reduced(exp_dims, exp_elt, cand):
    """isRankReducedType(expected, cand)：类型相同即成立；否则 rank(cand) ≤ rank(expected)，
    computeRankReductionMask（matchDynamic=false：? 只与 ? 匹配，未匹配的维必须是静态 1），且元素类型相同。不比较编码。"""
    if cand[1] is None: return False
    red = list(cand[1]); j = 0
    if len(red) > len(exp_dims): return False
    for d in exp_dims:
        if j < len(red) and d == red[j]: j += 1; continue
        if d != 1: return False
    return j == len(red) and cand[2] == exp_elt
def _dims_of(sz): return tuple(None if x == DYN else x for x in sz)
def _extract_slice(S):
    def f(Src, O, Z, T, R, attrs):
        if len(Src) != 1 or len(R) != 1: return False
        src, res = Src[0], R[0]
        ost = offsets_sizes_strides_ok(rank(src), len(O), len(Z), len(T), attrs)
        if ost is None: return False
        o, sz, st = ost
        return rank_reduced(_dims_of(sz), src[2], res) and in_bounds(src[1], o, sz, st)
    return [_r(['source', 'offsets', 'sizes', 'strides', 'result'],
               'static_* 长度 = rank(source)，动态操作数个数 = kDynamic 个数，静态 offset/size ≥ 0；'
               'result 是 tensor<static_sizes × elt(source)> 的降秩（只能去掉静态 1 维，elt 相同，不比较编码）；静态维上不越界', f, attrs=True)]
def _insert_slice(S):
    def f(Src, D, O, Z, T, R, attrs):
        if len(Src) != 1 or len(D) != 1 or len(R) != 1: return False
        src, dst = Src[0], D[0]
        ost = offsets_sizes_strides_ok(rank(dst), len(O), len(Z), len(T), attrs)
        if ost is None: return False
        o, sz, st = ost
        return R[0] == dst and rank_reduced(_dims_of(sz), dst[2], src) and in_bounds(dst[1], o, sz, st)
    return [_r(['source', 'dest', 'offsets', 'sizes', 'strides', 'result'],
               'static_* 长度 = rank(dest)，动态个数相符，静态 offset/size ≥ 0；source 是 tensor<static_sizes × elt(dest)> 的降秩；'
               '静态维上不越界；result = dest（DestinationStyle）', f, attrs=True)]

# ───────────────────────── 第 68 轮：vector.transfer_read / transfer_write
def affine_map(txt):
    """affine_map<(d0, d1)[s0] -> (d1, 0)> → (维数, 符号数, 结果表)；结果项为 ('d', k) / ('c', v) / ('e', 文本)。"""
    if txt is None: return None
    m = _re.fullmatch(r'\s*(?:affine_map<)?\s*\(([^)]*)\)\s*(?:\[([^\]]*)\])?\s*->\s*\((.*)\)\s*>?\s*', txt)
    if not m: return None
    dims = [x.strip() for x in m.group(1).split(',') if x.strip()]
    syms = [x.strip() for x in (m.group(2) or '').split(',') if x.strip()]
    res = []
    for e in [x.strip() for x in m.group(3).split(',')] if m.group(3).strip() else []:
        if e in dims: res.append(('d', dims.index(e)))
        elif _re.fullmatch(r'-?\d+', e): res.append(('c', int(e)))
        else: res.append(('e', e))
    return len(dims), len(syms), res
def bool_array(txt):
    if txt is None: return None
    return _re.findall(r'true|false', txt)
def bits(t):
    """DataLayout 默认（无 DLTI）下的 getTypeSizeInBits；不支持的类型返回 None（MLIR 报 missing data layout，致命）。"""
    if t[0] in ('i', 'f'): return width(t)
    if t == ('index',): return 64
    if t[0] == 'opaque' and _re.fullmatch(r'!llvm\.ptr(<\d+>)?', t[1]): return 64     # LLVMPointerType 无 DLTI 时 64 位（各地址空间）
    if t[0] == 'opaque': raise NFUnknown(t[1])                                           # 其他方言类型的 DataLayoutTypeInterface：未建模 ⇒ 不判
    if t[0] == 'complex':
        w = bits(t[1]) if len(t) > 1 and isinstance(t[1], tuple) else None
        if w is None: return None
        b = -(-w // 8); al = 1
        while al < b: al *= 2
        ia = al * 8
        return -(-w // ia) * ia + w
    return None
class NFUnknown(Exception):
    """规则在该输入上未建模（方言自定义类型等）：odssig.check 不据此拒绝（上近似），见 closure §9.8 边界。"""
def valid_vector_elt(t):
    """VectorType::isValidElementType：int/index/float 或实现 VectorElementTypeInterface 的类型（llvm.ptr 等）。"""
    if t[0] in ('i', 'f') or t == ('index',): return True
    if t[0] == 'opaque' and _re.fullmatch(r'!llvm\.ptr(<\d+>)?', t[1]): return True
    if t[0] == 'opaque': raise NFUnknown(t[1])
    return False
def transfer_common(Sh, V, M, nidx, attrs):
    """verifyTransferOp + 索引个数 + verifyPermutationMap。返回 (ok, 置换表)。"""
    pm, inb = affine_map(attrs.get('permutation_map')), bool_array(attrs.get('in_bounds'))
    if pm is None or inb is None: return False, None
    nd, ns, res = pm
    if Sh[0] not in ('memref', 'tensor') or Sh[1] is None: return False, None          # 无秩 tensor：getRank 断言（崩溃）⇒ 拒绝
    if nidx != len(Sh[1]): return False, None
    if V[0] != 'vector': return False, None
    vr = len(V[1]); e = Sh[2]
    if e[0] == 'vector':
        if vr == 0 or not e[1]: return False, None                                    # getShape().back() 于 0 维：断言
        bs, bv = bits(e[2]), bits(V[2])
        if bs is None or bv is None or (bv * V[1][-1]) % (bs * e[1][-1]): return False, None
        if len(e[1]) > vr or len(res) != vr - len(e[1]) or M: return False, None
    else:
        bs, bv = bits(e), bits(V[2])
        if bs is None or bv is None or (bv * (1 if vr == 0 else V[1][-1])) % bs: return False, None
        if len(res) != vr: return False, None
    if ns != 0 or nd != len(Sh[1]): return False, None
    # verifyPermutationMap（掩码推断要求它成立，否则断言）：每个结果是维或常数 0，维不重复
    seen = set()
    for k, v in res:
        if k == 'e' or (k == 'c' and v != 0) or (k == 'd' and v in seen): return False, None
        if k == 'd': seen.add(v)
    if M:
        used = sorted(seen); pos = {v: i for i, (k, v) in enumerate(res) if k == 'd'}
        sc = TS.scalable(V)
        shape = [V[1][pos[d]] for d in used] or [1]
        scal = tuple(sc[pos[d]] for d in used) or (False,)
        m = M[0]
        if not (m[0] == 'vector' and m[2][0] == 'i' and m[2][1] == 1 and list(m[1]) == shape and tuple(TS.scalable(m)) == tuple(scal if used else (False,))): return False, None
    if len(res) != len(inb): return False, None
    return True, res
def _transfer_read(S):
    def f(B, I, P, M, R, attrs):
        if len(B) != 1 or len(P) != 1 or len(R) != 1: return False
        ok, _ = transfer_common(B[0], R[0], M, len(I), attrs)
        if not ok: return False
        e, p = B[0][2], P[0]
        return p == e if e[0] == 'vector' else (p == e and valid_vector_elt(p))
    return [_r(['base', 'indices', 'padding', 'mask', 'vector'],
               '#indices = rank(base)；base 为 memref 或有秩 tensor；最内维位宽整除；permutation_map 输入维 = rank(base)、无符号、结果数 = rank(vector)（元素为 vector 时减其秩）、'
               '结果为不重复的维或 0；mask = 由 permutation_map 推出的 i1 vector；|in_bounds| = 结果数；padding = elt(base)', f, attrs=True)]
def _transfer_write(S):
    def f(V, B, I, M, R, attrs):
        if len(V) != 1 or len(B) != 1: return False
        ok, res = transfer_common(B[0], V[0], M, len(I), attrs)
        if not ok or any(k == 'c' for k, _ in res): return False                       # hasBroadcastDim
        b = B[0]                                                                       # DestinationStyle：tensor base ⇔ 有结果且类型相同
        return (len(R) == 1 and R[0] == b) if b[0] == 'tensor' else len(R) == 0
    return [_r(['valueToStore', 'base', 'indices', 'mask', 'result'],
               '同 transfer_read 的公共部分（无 padding）；permutation_map 不含常数 0（无广播维）；base 为 tensor 时 result = base，为 memref 时无结果', f, attrs=True)]

# 候选规则（待 nf_test 证实）与已登记规则分开：REG 只放经检验的 op
CANDIDATES = {op: (CAST_COVERS, _cast_rules) for op in _CAST}
CANDIDATES.update({
    'tensor.dim': ({'接口 verify: ShapedDimOpInterface::Trait'}, _dim), 'memref.load': ({'C++ verify()'}, _load),
    'memref.store': ({'C++ verify()'}, _store), 'tensor.cast': ({'接口 verify: CastOpInterface::Trait'}, _tcast),
    'vector.broadcast': ({'C++ verify()'}, _bcast), 'vector.shape_cast': ({'C++ verify()'}, _shape_cast),
    'vector.transpose': ({'C++ verify()'}, _transpose),
    'tensor.extract_slice': ({'C++ verify()', '接口 verify: OffsetSizeAndStrideOpInterface::Trait'}, _extract_slice),
    'tensor.insert_slice': ({'C++ verify()', '接口 verify: OffsetSizeAndStrideOpInterface::Trait', '接口 verify: DestinationStyleOpInterface::Trait'}, _insert_slice),
    'vector.transfer_read': ({'C++ verify()', '接口 verify: DestinationStyleOpInterface::Trait'}, _transfer_read),
    'vector.transfer_write': ({'C++ verify()', '接口 verify: DestinationStyleOpInterface::Trait'}, _transfer_write)})
REG = dict(CANDIDATES)      # 第 66 轮起：nf_test N1 全 PASS 的 op 才留在这里（closure §9.6/§9.7）；未通过的 op 须从 CANDIDATES 删去或修正
ENABLED = True

# ───────────────────────── 第 69 轮：接口级规则（一条规则覆盖实现该接口的所有 op；op 特有的参数从 ODS 记录读出）
def _op_record(op):
    import odsjson as O
    dia, nm = op.split('.', 1)
    try: d = O.records(dia)
    except Exception: return None
    for n in d['!instanceof']['Op']:
        r = d[n]
        if r.get('opName') == nm and d[O.ref(r['opDialect'])]['name'] == dia: return r
    return None
def dps_init_group(S):
    """getDpsInitsMutable() { return get<Name>Mutable(); } ⇒ 端口组名（ODS 访问器名 = 驼峰化的端口名）。读不出返回 None（该 op 不套用接口规则）。"""
    r = _op_record(S['op'])
    txt = (r or {}).get('extraClassDeclaration') or ''
    m = _re.search(r'getDpsInitsMutable\(\)\s*\{\s*return\s+get(\w+)Mutable\(\);\s*\}', txt)
    if not m: return None
    acc = m.group(1)
    for nm, *_ in S['ins']:
        if ''.join(w[:1].upper() + w[1:] for w in nm.split('_')) == acc: return nm
    return None
def _dps(S):
    """verifyDestinationStyleOpInterface：每个 init 是 tensor 或 memref（含无秩）；tensor 结果个数 = tensor init 个数；
    第 i 个 init 若为 tensor，则第 i 个结果存在且类型与之相同（getTiedOpResult 按 init 下标取结果；越界为断言，按拒绝）。"""
    g = dps_init_group(S)
    if g is None: return None
    outs = [nm for nm, *_ in S['outs']]
    def f(I, *R):
        res = [t for grp in R for t in grp]
        if any(t[0] not in ('tensor', 'memref') for t in I): return False
        tin = [i for i, t in enumerate(I) if t[0] == 'tensor']
        if sum(t[0] == 'tensor' for t in res) != len(tin): return False
        return all(i < len(res) and res[i] == I[i] for i in tin)
    return [_r([g] + outs, f'DPS：{g} 每项为 tensor/memref；tensor 结果个数 = tensor {g} 个数；第 i 个 tensor {g} = 第 i 个结果', f)]
IFACE = {'接口 verify: DestinationStyleOpInterface::Trait': _dps}

def apply(S, reg=None):
    """给签名加上规范化原子，并把覆盖的 declared 条目移出。先按 op（REG），再按接口（IFACE）。"""
    reg = REG if reg is None else reg
    if not ENABLED: return S
    S = dict(S); S['rel'] = list(S['rel']); S.setdefault('nf_covered', [])
    if S['op'] in reg:
        covers, mk = reg[S['op']]
        S['rel'] += mk(S)
        S['nf_covered'] = S['nf_covered'] + [d for d in S['declared'] if d in covers]
        S['declared'] = [d for d in S['declared'] if d not in covers]
    if reg is REG:
        for d in [d for d in S['declared'] if d in IFACE]:
            rs = IFACE[d](S)
            if rs is None: continue
            S['rel'] += rs; S['nf_covered'].append(d); S['declared'] = [x for x in S['declared'] if x != d]
    return S

# ───────────────────────── 第 70 轮：LinalgOp::Trait（verifyStructuredOpInterface + IndexingMapOpInterface::verifyImpl），closure §9.10
# 依据：Linalg/IR/LinalgInterfaces.cpp verifyStructuredOpInterface、Interfaces/IndexingMapOpInterface.cpp verifyImpl、
#       IR/AffineMap.cpp inversePermutation/concatAffineMaps、tools/mlir-linalg-ods-gen/mlir-linalg-ods-yaml-gen.cpp（命名 op 的 getIndexingMaps）。
# 映射来源：linalg.generic ← 属性 indexing_maps / iterator_types；opdsl 命名 op ← LinalgNamedStructuredOps.yaml（符号按 index 属性或默认值绑定）。
# 手写 getIndexingMaps 的 9 个 op（matmul 族、contract、elementwise、transpose、broadcast、reduce、map）暂不套用，条目留在残差。
import os as _os
_YAML = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'inventory', 'LinalgNamedStructuredOps.yaml')

@functools.lru_cache(None)
def linalg_yaml():
    import yaml
    class L(yaml.SafeLoader): pass
    def _any(loader, suffix, node):
        if isinstance(node, yaml.MappingNode): return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode): return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)
    L.add_multi_constructor('!', _any)
    out = {}
    for doc in yaml.load_all(open(_YAML), Loader=L):
        if not doc: continue
        so = doc['structured_op']; nm = doc['metadata']['name']
        out[nm] = so
    return out

def _dense_i64(txt, n):
    """index 属性文本（dense<[2, 2]> : tensor<2xi64> 或 dense<1> : tensor<2xi64>）→ 整数表；读不出 None。"""
    if txt is None: return None
    m = _re.match(r'\s*dense<\s*(\[[^\]]*\]|-?\d+)\s*>', txt)
    if not m: return None
    v = m.group(1)
    if v.startswith('['): xs = [int(x) for x in _re.findall(r'-?\d+', v)]
    else: xs = [int(v)] * n
    return xs if len(xs) == n else None

def _yaml_provider(nm):
    """命名 op：返回 f(操作数类型表, attrs) → (映射表, 循环数)；不可套用（映射用到非属性符号等）返回 None。"""
    import affine as AF
    so = linalg_yaml().get(nm)
    if so is None or 'static_indexing_maps' not in (so.get('indexing_maps') or {}): return None
    maps_txt = so['indexing_maps']['static_indexing_maps']
    rankpoly = not so.get('iterator_types')
    nloops_fixed = len(so.get('iterator_types') or [])
    parsed = [AF.parse_map(t) for t in maps_txt]
    binds = []                                        # (属性名, 符号位置, 属性下标, 默认值, 长度)
    for a in so['args']:
        if a['kind'] != 'index_attr': continue
        _, _, rs = AF.parse_map(a['index_attr_map'])
        for idx, r in enumerate(rs):
            ss = AF.uses_syms(r)
            if r[0] == 'lin' and len(r[1]) == 1 and not r[2] and list(r[1].values()) == [1] and len(ss) == 1:
                binds.append((a['name'], ss.pop(), idx, a['default_indices'][idx], len(rs)))
    bound = {b[1] for b in binds}
    if not rankpoly and any(AF.uses_syms(e) - bound for _, _, es in parsed for e in es): return None
    nops = sum(a['kind'] in ('input_tensor', 'output_tensor', 'scalar') for a in so['args'])
    def prov(ops, attrs):
        vals = {}
        for name, pos, idx, dflt, n in binds:
            if name in attrs:
                xs = _dense_i64(attrs[name], n)
                if xs is None: raise NFUnknown(f'{name}={attrs[name]}')
                vals[pos] = xs[idx]
            else: vals[pos] = dflt
        memo = attrs.get('linalg.memoized_indexing_maps')
        if memo is not None:                                           # getIndexingMaps 先读缓存属性（MLIR 自己在 verify 时写入；带着它的 IR 按缓存的映射判定）
            try: return [AF.parse_map(x) for x in _split_maps(memo)], nloops_fixed
            except AF.AffineError as e: raise NFUnknown(str(e))
        maps = [(nd, 0, [AF.subst_syms(e, vals) for e in es]) for nd, _, es in parsed]
        return maps, nloops_fixed
    return prov, rankpoly, nops

class _Crash(Exception):
    """MLIR 在此输入上断言失败（崩溃）；规范化为拒绝。"""

def _lrank(t):
    """LinalgOp getRank：VectorType 是元素类型，取 0；tensor/memref 取秩（无秩断言 ⇒ 崩溃）；其余 0。"""
    if t[0] in ('tensor', 'memref'):
        if t[1] is None: raise _Crash()
        return len(t[1])
    return 0
def _lshape(t):
    if t[0] in ('tensor', 'memref'):
        if t[1] is None: raise _Crash()
        return [(-1 if d is None else d) for d in t[1]]           # -1 代表 kDynamic
    return []

def structured_ok(ops, maps, nloops_iter, bbargs):
    """verifyStructuredOpInterface 在操作数类型 ops、映射 maps（(维数, 符号数, 表达式表)）、迭代器个数、区域入口块参数类型下是否通过。"""
    import affine as AF
    is_t = lambda t: t[0] == 'tensor'; is_m = lambda t: t[0] == 'memref'
    if ops and not ((not any(map(is_m, ops)) and any(map(is_t, ops))) or (not any(map(is_t, ops)) and any(map(is_m, ops)))): return False
    # IndexingMapOpInterface::verifyImpl
    if len(maps) != len(ops): return False
    if any(ns for _, ns, _ in maps): return False                  # 有符号：inversePermutation 断言（崩溃）或“unexpected symbols”，均拒绝
    nd = max([m[0] for m in maps], default=0)                      # concatAffineMaps：维数取最大
    allres = [e for _, _, es in maps for e in es]
    shapes = [_lshape(t) for t in ops]
    flat = [d for s in shapes for d in s]
    if nd == 0 and not allres: ranges = []                         # 空映射：inversePermutation 原样返回
    else:
        pos = [None] * nd
        for i, e in enumerate(allres):
            d = AF.is_dim(e)
            if d is not None and pos[d] is None: pos[d] = i
        if any(p is None for p in pos): return False               # 不可逆
        if len(flat) != len(allres): raise _Crash()                # compose 的值个数与映射结果数不符（秩不符时在后面拒绝，但先求 ranges 会断言）
        ranges = [flat[p] for p in pos]
    for (mnd, _, es), s in zip(maps, shapes):
        if mnd != len(ranges): return False
        if len(es) != len(s): return False
    if all(r != -1 for r in ranges):
        lo = [0] * len(ranges); hi = [r - 1 for r in ranges]
        for (_, _, es), s in zip(maps, shapes):
            for k, e in enumerate(es):
                if s[k] == -1 or s[k] == 0: continue
                a, b = AF.evaluate(e, lo), AF.evaluate(e, hi)
                if min(a, b) < 0: return False
                inf = max(a, b) + 1
                if AF.is_dim(e) is not None:
                    if inf != s[k]: return False
                elif inf > s[k]: return False
    # verifyStructuredOpInterface 其余部分
    if any(m[0] != nloops_iter for m in maps): return False
    if bbargs is None: return False                                # 没有区域或区域不是恰一个块
    if len(bbargs) != len(ops): return False
    for t, a in zip(ops, bbargs):
        el = t[2] if (t[0] == 'memref' or (t[0] == 'tensor' and t[1] is not None)) else t
        if el != a: return False
    return True

def _linalg_iface(S):
    import affine as AF
    nm = S['op'].split('.', 1)[1]
    names = [n for n, *_ in S['ins']]
    if 'inputs' not in names or 'outputs' not in names: return None
    outs = [n for n, *_ in S['outs']]
    if S['op'] == 'linalg.generic':
        def f(I, O, *R_at):
            attrs = R_at[-1]; ops = list(I) + list(O)
            if any(t[0] == 'opaque' for t in ops): raise NFUnknown('方言类型')
            try:
                maps = [AF.parse_map(x) for x in _split_maps(attrs.get('indexing_maps'))]
                nit = len(_re.findall(r'iterator_type<', attrs.get('iterator_types') or ''))
                return structured_ok(ops, maps, nit, _bbargs(attrs))
            except _Crash: return False
            except AF.AffineError as e: raise NFUnknown(str(e))
        return [_r(['inputs', 'outputs'] + outs, 'LinalgOp：纯 tensor/纯 buffer；#indexing_maps = #操作数、可逆、无符号、维数 = 循环数 = #iterator_types、结果数 = 秩；静态形状与循环范围一致；区域入口块参数 = 各操作数的元素类型', f, attrs=True)]
    p = _yaml_provider(nm)
    if p is None: return None
    prov, rankpoly, nops = p
    def g(I, O, *R_at):
        attrs = R_at[-1]; ops = list(I) + list(O)
        if any(t[0] == 'opaque' for t in ops): raise NFUnknown('方言类型')
        try:
            if rankpoly:
                if not O: return False                                     # getDpsInitOperand(0) 越界：断言
                n = _lrank(O[0])
                maps = [(n, 0, [] if _lrank(t) == 0 else [AF._lin({('d', i): 1}) for i in range(n)]) for t in ops]
                return structured_ok(ops, maps, n, _bbargs(attrs))
            maps, nit = prov(ops, attrs)
            return structured_ok(ops, maps, nit, _bbargs(attrs))
        except _Crash: return False
    return [_r(['inputs', 'outputs'] + outs, f'LinalgOp（{nm}，映射取自 opdsl yaml）：同 linalg.generic 的结构化检查', g, attrs=True)]

def _split_maps(txt):
    if txt is None: raise NFUnknown('缺 indexing_maps')
    return _re.findall(r'affine_map<.*?>(?=\s*[,\]])', txt)

def _bbargs(attrs):
    """区域 0 入口块参数的类型（由 nf_attrs 放入）；区域无块为 None；没有区域信息时未建模。"""
    if '\x00bbargs' not in attrs: raise NFUnknown('无区域信息')
    v = attrs['\x00bbargs']
    return None if v is None else [TS.parse(t) for t in v]

def nf_attrs(n, at):
    """给 nf 规则用的属性表：在 from_core 的结果上加区域 0 入口块参数类型（只给 nf 规则看，不作为 MLIR 属性）。"""
    if at is None: return None
    rs = n.get('regions') or []
    if not rs: return at
    at = dict(at)
    r0 = rs[0]
    at['\x00bbargs'] = None if len(r0) != 1 else [t for _, t in r0[0].get('args', [])]
    return at

IFACE['接口 verify: LinalgOp::Trait'] = _linalg_iface
