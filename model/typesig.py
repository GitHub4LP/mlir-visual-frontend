"""类型层 v0（第 7 轮）：op 签名 = 端口约束 + 端口间关系，按来源分级；检查、推断、连接与补全查询。
来源（信任等级）：
  ods       端口约束、可自动导入的 trait（SameOperandsAndResultType、SameTypeOperands、AllTypesMatch、SameOperandsAndResultShape、
            TypesMatchWith 中已知的变换子）——从 .td 直接导入，不手写
  cpred     .td 中的 PredOpTrait/CPred：C++ 片段，按项目规则登记为原子组合（人工改写，可审计）
  declared  只存在于 C++ verify() 的条件：声明，无法从 .td 得到
类型项：('i', 宽, 符号'' | 's' | 'u') | ('index',) | ('f', 名) | (容器 'tensor'|'vector'|'memref', 维元组或 None(无秩), 元素)；维 None = ?
"""
import re, itertools, time
import ods_import as O

# ───────────────────────── 类型项
def parse(s):
    s = s.strip()
    m = re.fullmatch(r'([su]?)i(\d+)', s)
    if m: return ('i', int(m.group(2)), m.group(1))
    if s == 'index': return ('index',)
    if s in ('f16', 'bf16', 'f32', 'f64'): return ('f', s)
    m = re.fullmatch(r'complex<(.*)>', s)
    if m: return ('complex', parse(m.group(1)))
    m = re.fullmatch(r'!shape\.(\w+)', s)
    if m: return ('shape.' + m.group(1),)
    m = re.fullmatch(r'(tensor|vector|memref)<(.*)>', s)
    if m:
        k, body = m.groups()
        if body.startswith('*x'): return (k, None, parse(body[2:]))
        mm = re.fullmatch(r'((?:(?:\d+|\?)x)*)(.+)', body)
        dims = tuple(None if d == '?' else int(d) for d in mm.group(1).split('x')[:-1])
        return (k, dims, parse(mm.group(2)))
    raise ValueError(s)
def show(t):
    if t[0] == 'i': return f'{t[2]}i{t[1]}'
    if t[0] == 'index': return 'index'
    if t[0] == 'f': return t[1]
    if t[0] == 'complex': return f'complex<{show(t[1])}>'
    if t[0].startswith('shape.'): return '!' + t[0]
    if t[1] is None: return f'{t[0]}<*x{show(t[2])}>'
    return f'{t[0]}<' + ''.join(('?' if d is None else str(d)) + 'x' for d in t[1]) + show(t[2]) + '>'
def shaped(t): return t[0] in ('tensor', 'vector', 'memref')
def elt(t): return t[2] if shaped(t) else t
def ranked(t): return shaped(t) and t[1] is not None

# ───────────────────────── 端口约束（ODS 名 → 谓词），从 CommonTypeConstraints/ArithOps 的定义逐条对应
def _sl_int(t, nz=True): return t[0] == 'i' and t[2] == '' and (t[1] > 0 or not nz)
def _container_of(p): return lambda t: (t[0] == 'tensor' or (t[0] == 'vector' and t[1])) and p(t[2])   # 值语义容器：tensor（含无秩）、vector
def _or(*ps): return lambda t: any(p(t) for p in ps)
_SIOI = lambda t: _sl_int(t) or t[0] == 'index'
PORT = {
    'AnyType': lambda t: True,
    'Index': lambda t: t[0] == 'index',
    'Arith_SignlessIntegerOrIndexLike': _or(_SIOI, _container_of(_SIOI)),
    'BoolLike': _or(lambda t: t == ('i', 1, ''), _container_of(lambda t: t == ('i', 1, ''))),
    'FloatLike': _or(lambda t: t[0] == 'f', _container_of(lambda t: t[0] == 'f')),
    'SignlessFixedWidthIntegerLike': _or(_sl_int, lambda t: t[0] == 'vector' and bool(t[1]) and _sl_int(t[2]), lambda t: t[0] == 'tensor' and _sl_int(t[2])),
    'IndexCastTypeConstraint': _or(_SIOI, _container_of(_SIOI), lambda t: t[0] == 'memref' and t[1] is not None and _SIOI(t[2])),
    'AnyRankedTensor': lambda t: t[0] == 'tensor' and t[1] is not None,
    'AnyTensor': lambda t: t[0] == 'tensor',
    'AnyNon0RankedOrUnrankedTensor': lambda t: t[0] == 'tensor' and (t[1] is None or len(t[1]) > 0),
    'AnyStaticShapeTensor': lambda t: t[0] == 'tensor' and t[1] is not None and None not in t[1],
}
# vector<…> 的 0 秩：MLIR 允许 vector<i8>，这里的类型池不生成它

# ───────────────────────── 端口间关系
def compat_shape(a, b):
    """mlir::verifyCompatibleShape：要么都不是 shaped，要么都是；无秩与任何秩相容；秩相等时逐维相容（? 与任何维相容）。"""
    if shaped(a) != shaped(b): return False
    if not shaped(a) or a[1] is None or b[1] is None: return True
    return len(a[1]) == len(b[1]) and all(x is None or y is None or x == y for x, y in zip(a[1], b[1]))
def _elementwise(ins, outs):
    """OpTrait::impl::verifyElementwise：可映射类型 = vector、tensor（含无秩）；有秩与无秩 tensor 是不同的基类型。"""
    mp = lambda t: t[0] in ('tensor', 'vector')
    base = lambda t: (t[0], t[1] is None)
    mi, mo = [t for t in ins if mp(t)], [t for t in outs if mp(t)]
    if not mi and not mo: return True
    if not mi or not mo or len(mo) != len(outs): return False
    ts = mi + mo
    return all(base(t) == base(ts[0]) for t in ts) and all(compat_shape(a, b) for a, b in itertools.combinations(ts, 2))
def i1same(t): return (t[0], t[1], ('i', 1, '')) if shaped(t) else ('i', 1, '')
XF = {'$_self': lambda t: [t], 'getElementType': lambda t: [t[2]] if shaped(t) else None,
      'getI1SameShape': lambda t: [i1same(t)],
      'getNumElements': lambda t: [t[2]] * _numel(t) if ranked(t) and None not in t[1] else None}
def _numel(t):
    n = 1
    for d in t[1]: n *= d
    return n
def _xf(code):
    """TypesMatchWith 的变换子：识别已知形式；未知 ⇒ None（该关系降为未导入）。"""
    for k in ('getNumElements', 'getI1SameShape', 'getElementType'):
        if k in code: return k
    return '$_self' if code.strip() == '$_self' else None

def _args(s):
    out, d, cur = [], 0, ''
    for c in s:
        if c in '<([': d += 1
        elif c in '>)]': d -= 1
        if c == ',' and d == 0: out.append(cur.strip()); cur = ''
        else: cur += c
    if cur.strip(): out.append(cur.strip())
    return out

# ───────────────────────── ODS 导入
def import_sig(src, name):
    H = O._headers(src); chain, cur = [], name
    while cur in H: chain.append(H[cur]); cur = H[cur]['parent']
    r = O.import_op(src, name, O.KNOWN)
    tsub = {}                           # 模板参数替换：沿继承链找到声明了 From/To 等形参的类，取子类传给它的实参
    for h in chain:
        cm = re.search(r'^class\s+' + h['parent'] + r'\s*<(.*?)>\s*:', src, re.M | re.S)
        am = re.match(r'\s*\w+\s*<(.*)>', h['head'], re.S)
        if cm and am:
            formals = [f.split()[-1].split('=')[0].strip() for f in _args(cm.group(1))]
            for f, a in zip(formals, _args(am.group(1))): tsub.setdefault(f, tsub.get(a, a))
    def port(c):
        c = tsub.get(c, c)
        mult = 'one'
        mm = re.fullmatch(r'(Variadic|Optional)<(.*)>', c)
        if mm: mult, c = ('var' if mm.group(1) == 'Variadic' else 'opt'), mm.group(2)
        return c, mult
    ins = [(n, *port(c)) for n, c in (r['args'] or []) if not re.search(r'Attr\b|Attr<|AttrInterface', c)]
    outs = [(n, *port(c)) for n, c in (r['results'] or [])]
    heads = ' '.join(h['head'] for h in chain)
    rel = []
    allp = [n for n, *_ in ins] + [n for n, *_ in outs]
    if 'SameOperandsAndResultType' in r['traits']: rel.append(('samec', allp, 'ods'))   # MLIR 的实现：元素类型相同 ∧ 形状相容（不是类型相等）
    if 'SameOperandsAndResultType' in r['traits'] and ins and len(outs) == 1:
        # ODS 为“结果类型全可由 trait 推出”的 op 自动生成 InferTypeOpInterface，verify 时要求结果 == 推断（取第一个操作数），是相等而不只是相容
        rel.append(('match', ins[0][0], outs[0][0], '$_self', 'ods'))
    if re.search(r'\bSameTypeOperands\b', heads): rel.append(('same', [n for n, *_ in ins], 'ods'))
    if 'ElementwiseMappable' in heads: rel.append(('elementwise', allp, 'ods'))     # 原生 trait：验证器在模型里实现一次，按名导入
    if 'SameOperandsAndResultShape' in r['traits']: rel.append(('shape', allp, 'ods'))
    for m in re.finditer(r'AllTypesMatch<\[(.*?)\]>', heads): rel.append(('same', re.findall(r'"(\w+)"', m.group(1)), 'ods'))
    for m in re.finditer(r'TypesMatchWith<\s*"[^"]*"\s*,\s*"(\w+)"\s*,\s*"(\w+)"\s*,\s*((?:"[^"]*"\s*)+)', heads):
        code = ''.join(re.findall(r'"([^"]*)"', m.group(3))); k = _xf(code)
        rel.append(('match', m.group(1), m.group(2), k, 'ods') if k else ('unimported', m.group(0)[:60], 'ods'))
    return dict(name=name, ins=ins, outs=outs, rel=rel, traits=r['traits'])

# 登记的 CPred（.td 里是 C++ 片段）与声明（只在 C++ verify() 里）
def _tdims(a, b):
    """SameInputOutputTensorDims：tensor 的 getShape 逐维相同。MLIR 对无秩 tensor 调 getShape 会断言崩溃（verifier 的缺陷），
    前端保守地拒绝：降低出去的程序会让 MLIR 崩溃。"""
    if any(t[0] == 'tensor' and t[1] is None for t in (a, b)): return False
    sh = lambda t: tuple(t[1]) if t[0] == 'tensor' else ()
    return sh(a) == sh(b)
def _cond_or_shape(c, r):              # AllShapesMatch 是 getShape 逐项相等；无秩时 MLIR 同样会崩溃，保守拒绝
    return c == ('i', 1, '') or (shaped(c) and shaped(r) and c[1] == r[1] and (c[1] is not None or c[0] == r[0] == 'tensor'))
def _width(rel, name):
    f = lambda a, b: elt(a)[0] == 'i' and elt(b)[0] == 'i' and rel(elt(b)[1], elt(a)[1])
    f.__name__ = name; return f
def _ixcast(a, b):
    ea, eb = elt(a), elt(b)
    return (ea[0] == 'index') != (eb[0] == 'index') and all(e[0] in ('index', 'i') for e in (ea, eb))
def _tcast(a, b): return a[0] == b[0] == 'tensor' and a[2] == b[2] and compat_shape(a, b)
def _dyn_count(res, dyn): return ranked(res) and sum(d is None for d in res[1]) == len(dyn)
def _rank_eq(t, idx): return ranked(t) and len(t[1]) == len(idx)
def _slice_res(src, offs, szs, strs, res):
    return ranked(src) and len(offs) == len(szs) == len(strs) == len(src[1]) and ranked(res) and res[2] == src[2] \
        and len(res[1]) == len(src[1]) and all(d is None for d in res[1])
def _islice(src, dst, offs, szs, strs):
    return ranked(src) and ranked(dst) and len(offs) == len(szs) == len(strs) == len(dst[1]) == len(src[1]) \
        and src[2] == dst[2] and all(d is None for d in src[1])
EXTRA = {
    'Arith_TruncIOp': [('pred', ['in', 'out'], _tdims, 'cpred'), ('pred', ['in', 'out'], _width(lambda o, i: o < i, 'narrower'), 'declared')],
    'Arith_ExtUIOp': [('pred', ['in', 'out'], _tdims, 'cpred'), ('pred', ['in', 'out'], _width(lambda o, i: o > i, 'wider'), 'declared')],
    'Arith_ExtSIOp': [('pred', ['in', 'out'], _tdims, 'cpred'), ('pred', ['in', 'out'], _width(lambda o, i: o > i, 'wider'), 'declared')],
    'Arith_IndexCastOp': [('pred', ['in', 'out'], _tdims, 'cpred'), ('pred', ['in', 'out'], _ixcast, 'declared')],
    'Arith_IndexCastUIOp': [('pred', ['in', 'out'], _tdims, 'cpred'), ('pred', ['in', 'out'], _ixcast, 'declared')],
    'SelectOp': [('pred', ['condition', 'result'], _cond_or_shape, 'cpred')],
    'Tensor_ExtractOp': [('predv', ['tensor', 'indices'], _rank_eq, 'declared')],
    'Tensor_InsertOp': [('predv', ['dest', 'indices'], _rank_eq, 'declared')],
    'Tensor_EmptyOp': [('predv', ['result', 'dynamicSizes'], _dyn_count, 'declared')],
    'Tensor_SplatOp': [('predv', ['aggregate', 'dynamicSizes'], _dyn_count, 'declared')],
    'Tensor_CastOp': [('pred', ['source', 'dest'], _tcast, 'declared')],
    'Tensor_ExtractSliceOp': [('predv', ['source', 'offsets', 'sizes', 'strides', 'result'], _slice_res, 'declared')],
    'Tensor_InsertSliceOp': [('predv', ['source', 'dest', 'offsets', 'sizes', 'strides'], _islice, 'declared')],
}
LIB_OPS = {'inventory/Arith_ArithOps.td': {'arith.addi': 'Arith_AddIOp', 'arith.subi': 'Arith_SubIOp', 'arith.muli': 'Arith_MulIOp',
               'arith.divui': 'Arith_DivUIOp', 'arith.andi': 'Arith_AndIOp', 'arith.xori': 'Arith_XOrIOp', 'arith.cmpi': 'Arith_CmpIOp',
               'arith.select': 'SelectOp', 'arith.trunci': 'Arith_TruncIOp', 'arith.extui': 'Arith_ExtUIOp', 'arith.extsi': 'Arith_ExtSIOp',
               'arith.index_cast': 'Arith_IndexCastOp', 'arith.index_castui': 'Arith_IndexCastUIOp', 'arith.addf': 'Arith_AddFOp'},
           'inventory/Tensor_TensorOps.td': {'tensor.dim': 'Tensor_DimOp', 'tensor.extract': 'Tensor_ExtractOp', 'tensor.insert': 'Tensor_InsertOp',
               'tensor.empty': 'Tensor_EmptyOp', 'tensor.splat': 'Tensor_SplatOp', 'tensor.from_elements': 'Tensor_FromElementsOp',
               'tensor.cast': 'Tensor_CastOp', 'tensor.extract_slice': 'Tensor_ExtractSliceOp', 'tensor.insert_slice': 'Tensor_InsertSliceOp'}}
SIGS = {}
def load():
    if SIGS: return SIGS
    for path, ops in LIB_OPS.items():
        src = open(path).read()
        for op, d in ops.items():
            s = import_sig(src, d); s['rel'] += EXTRA.get(d, []); s['op'] = op
            unknown = [c for _, c, _ in s['ins'] + s['outs'] if c not in PORT]
            s['unknown_ports'] = unknown; SIGS[op] = s
    return SIGS

# ───────────────────────── 检查
LEVELS = ('ods', 'cpred', 'declared')
def bind(sig, ins, outs):
    """按端口组把操作数/结果绑定到端口名。变长端口组：没有段属性时只允许一个变长组（与 MLIR 相同）；有段属性时按 seg 给出长度。"""
    def b(ports, vals, seg=None):
        env, k = {}, 0
        nvar = sum(m != 'one' for _, _, m in ports)
        fixed = sum(m == 'one' for _, _, m in ports)
        for j, (n, _, m) in enumerate(ports):
            if m == 'one':
                if k >= len(vals): return None
                env[n] = [vals[k]]; k += 1
            else:
                cnt = seg[j] if seg else len(vals) - fixed
                if cnt < 0 or (m == 'opt' and cnt > 1): return None
                env[n] = vals[k:k + cnt]; k += cnt
        return env if k == len(vals) else None
    return b(sig['ins'], ins, sig.get('seg')), b(sig['outs'], outs)

def items(sig, levels=LEVELS):
    """签名的全部条目：('port', 端口, 约束) 与关系。复合 op 的推导签名（compsig.py）按条目逐一代换。"""
    return [('port', n, c) for n, c, _ in sig['ins'] + sig['outs'] if c in PORT] + [r for r in sig['rel'] if r[-1] in levels]

def ok(sig, r, env):
    """一个条目在 env（端口名 → 类型列表）下是否成立；端口类型未定义（None）⇒ 不成立。"""
    if any(t is None for ts in env.values() for t in ts): return False
    if r[0] == 'port': return all(PORT[r[2]](t) for t in env[r[1]])
    if r[0] == 'same':
        ts = [t for p in r[1] for t in env.get(p, [])]; return all(t == ts[0] for t in ts)
    if r[0] == 'samec':
        ts = [t for p in r[1] for t in env.get(p, [])]
        return all(elt(a) == elt(b) and compat_shape(a, b) and a[0] == b[0] for a, b in itertools.combinations(ts, 2))
    if r[0] == 'elementwise':
        return _elementwise([t for n, *_ in sig['ins'] for t in env[n]], [t for n, *_ in sig['outs'] for t in env[n]])
    if r[0] == 'shape':
        ts = [t for p in r[1] for t in env.get(p, [])]; return all(compat_shape(a, b) for a, b in itertools.combinations(ts, 2))
    if r[0] == 'match':
        _, a, b, k, _ = r
        if not env.get(a): return True
        want = XF[k](env[a][0]); return want is not None and list(env[b]) == want
    if r[0] == 'pred': return bool(r[2](*[env[p][0] for p in r[1]]))
    if r[0] == 'predv':
        var = {nn for nn, _, m in sig['ins'] + sig['outs'] if m != 'one'}
        return bool(r[2](*[env[p] if p in var else env[p][0] for p in r[1]]))
    return True

def label(r):
    if r[0] == 'port': return ('ods', f'{r[1]}: 不满足 {r[2]}')
    if r[0] in ('pred', 'predv'): return (r[-1], f'{r[1]} 不满足 {r[2].__name__}')
    if r[0] == 'match': return (r[-1], f'{r[2]} 须为 {r[3]}({r[1]})')
    return (r[-1], {'same': '类型须相同', 'samec': '元素类型须相同且形状相容', 'shape': '形状须相容',
                    'elementwise': 'Elementwise：非标量端口须同为一种容器且形状相容'}.get(r[0], r[0]) + f': {r[1]}')

COMPS = {}          # 复合 op 的推导签名（compsig.py 注册），与原生签名走同一个 check
def check(op, ins, outs, levels=LEVELS, seg=None):
    """返回违例列表 [(来源, 说明)]；空表 = 接受。ins/outs：类型项列表。"""
    if op in COMPS: return COMPS[op].check(ins, outs, levels)
    sig = dict(load()[op]); sig['seg'] = seg
    ei, eo = bind(sig, ins, outs)
    if ei is None or eo is None: return [('ods', '操作数/结果个数不符')]
    env = {**ei, **eo}
    return [label(r) for r in items(sig, levels) if not ok(sig, r, env)]

def infer(op, ins, seg=None):
    """由操作数推出结果类型；推不出（结果类型是用户的选择，如 trunci 的目标宽度）⇒ None。"""
    sig = load()[op]
    if len(sig['outs']) != 1 or sig['outs'][0][2] != 'one': return None
    rn = sig['outs'][0][0]
    ei, _ = bind(dict(sig, seg=seg), ins, [('index',)])
    if ei is None: return None
    for r in sig['rel']:
        if r[0] in ('same', 'samec') and rn in r[1]:
            for p in r[1]:
                if p != rn and ei.get(p): return ei[p][0]
        if r[0] == 'match' and r[2] == rn and ei.get(r[1]): return XF[r[3]](ei[r[1]][0])[0]
    c = sig['outs'][0][1]
    return ('index',) if c == 'Index' else None

# ───────────────────────── 编辑期查询
def connectable(op, port, t):
    """值 t 能否接到 op 的端口 port（只看端口约束，关系要等其余端口确定）。纯函数，微秒级。"""
    s = load()[op]
    for n, c, _ in s['ins']:
        if n == port: return PORT.get(c, lambda _: True)(t)
    return False
def candidates(t):
    """补全：哪些 op 的哪个输入端口接受 t。"""
    return [(op, n) for op, s in load().items() for n, c, _ in s['ins'] if PORT.get(c, lambda _: True)(t)]

if __name__ == '__main__':
    for op, s in load().items():
        print(op, [(n, c, m) for n, c, m in s['ins']], '->', [(n, c) for n, c, _ in s['outs']], [r[0] + ':' + r[-1] for r in s['rel']])
