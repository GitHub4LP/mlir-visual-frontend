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

# 候选规则（待 nf_test 证实）与已登记规则分开：REG 只放经检验的 op
CANDIDATES = {op: (CAST_COVERS, _cast_rules) for op in _CAST}
CANDIDATES.update({
    'tensor.dim': ({'接口 verify: ShapedDimOpInterface::Trait'}, _dim), 'memref.load': ({'C++ verify()'}, _load),
    'memref.store': ({'C++ verify()'}, _store), 'tensor.cast': ({'接口 verify: CastOpInterface::Trait'}, _tcast),
    'vector.broadcast': ({'C++ verify()'}, _bcast), 'vector.shape_cast': ({'C++ verify()'}, _shape_cast),
    'vector.transpose': ({'C++ verify()'}, _transpose)})
REG = dict(CANDIDATES)      # 第 66 轮起：nf_test N1 全 PASS 的 op 才留在这里（closure §9.6/§9.7）；未通过的 op 须从 CANDIDATES 删去或修正
ENABLED = True

def apply(S, reg=None):
    """给签名加上规范化原子，并把覆盖的 declared 条目移出。"""
    reg = REG if reg is None else reg
    if not ENABLED or S['op'] not in reg: return S
    covers, mk = reg[S['op']]
    S = dict(S); S['rel'] = list(S['rel']) + mk(S)
    S['nf_covered'] = [d for d in S['declared'] if d in covers]
    S['declared'] = [d for d in S['declared'] if d not in covers]
    return S
