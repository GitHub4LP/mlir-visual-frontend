"""第 66 轮：规范化规则（nfrules）对 MLIR verifier 的检验。

方法：规则只依赖少数类型特征（容器种类、秩与维、编码、元素种类、位宽、符号性）。按这些特征的等价类构造类型池，
对每个候选 op 枚举端口类型组合（第 67 轮起每个 op 有自己的生成器 GEN，含属性取值），比较 模型（ODS 签名 ∧ 规范化原子）与 MLIR verifier 的判定。
组合：同容器下元素两两全组合（容器 15 种（含布局、内存空间、两种编码、可伸缩维） × 元素 13² 种），加上容器两两全组合 × 4 个代表元素对。

阈值（运行前写定，事后不改）：
  N1 每个候选 op：误拒（模型拒、MLIR 收）0，误收（模型收、MLIR 拒）0；MLIR 接受 ≥ 1 且拒绝 ≥ 1（非空检验）；
     MLIR 无法解析的组合（类型本身非法）不计，但比较数 ≥ 生成数的 80%。
     （试跑 arith.extsi 后改，事后改动如实记录：原阈值“MLIR 崩溃 0”。MLIR 断言崩溃 = 不能产出合法 IR，
      按拒绝计入比较，模型须拒绝；崩溃数单列报告。）
  N0 消融（只报告）：去掉规范化原子（只用 ODS 签名）时的误收数，说明规则确有作用。
真实 IR 上的检验沿用 odscomp_test S3（误拒 0；残差为空时误收 0）与 projcomp P1–P5，阈值不变。
用法：PYTHONPATH=model python3 model/nf_test.py [op 名…]
"""
import re, random, sys, itertools, collections, multiprocessing as mp

CONTS = ['{}', 'vector<4x{}>', 'vector<2x{}>', 'vector<[4]x{}>', 'tensor<4x{}>', 'tensor<?x{}>', 'tensor<*x{}>',
         'tensor<4x{}, "enc">', 'tensor<2x4x{}>', 'memref<4x{}>', 'memref<*x{}>',
         'memref<4x{}, strided<[2]>>', 'memref<4x{}, 1>', 'tensor<4x{}, "enc2">', 'vector<2x4x{}>']
ELTS = ['i1', 'i8', 'i16', 'i32', 'i64', 'si32', 'index', 'f16', 'bf16', 'f32', 'f64', 'f8E4M3FN', 'tf32']
REP = [('i8', 'i32'), ('i32', 'i8'), ('f16', 'f32'), ('i32', 'f32')]

def pairs():
    for c in CONTS:
        for a, b in itertools.product(ELTS, ELTS): yield c.format(a), c.format(b)
    for c1, c2 in itertools.product(CONTS, CONTS):
        if c1 == c2: continue
        for a, b in REP: yield c1.format(a), c2.format(b)

# 第 67 轮：每个 op 一个生成器，产出 (操作数类型表, 结果类型表, 属性 dict)
def _shape(dims, e):
    if dims is None: return e
    return 'vector<' + ''.join(f'{d}x' for d in dims) + e + '>'
VSHAPES = [(), (1,), (4,), ('[4]',), ('[1]',), (2, 4), (1, 4), (4, 1), (2, '[4]'), (3, 2, 4), (2, 1, 4), (4, 2)]
def _v(sh, e): return 'vector<' + ''.join(f'{d}x' for d in sh) + e + '>'

def gen_cast(op):
    for a, b in dict.fromkeys(pairs()): yield [a], [b], {}
def gen_dim(op):
    for c in CONTS + ['tensor<{}>', 'tensor<2x?x{}>']:
        for r in ('index', 'i64'): yield [c.format('f32'), 'index'], [r], {}
MREFS = ['memref<{}>', 'memref<4x{}>', 'memref<2x4x{}>', 'memref<?x4x{}>', 'memref<*x{}>', 'memref<4x{}, strided<[2]>>', 'memref<2x4x{}, 1>']
def gen_load(op):
    for m in MREFS:
        for k in range(4):
            for e in ('f32', 'i32'): yield [m.format('f32')] + ['index'] * k, [e], {}
def gen_store(op):
    for m in MREFS:
        for k in range(4):
            for e in ('f32', 'i32'): yield [e, m.format('f32')] + ['index'] * k, [], {}
def gen_bcast(op):
    srcs = [_v(sh, e) for sh in VSHAPES for e in ('f32', 'i32')] + ['f32', 'i32', 'index', 'f16']
    for a in srcs:
        for sh in VSHAPES: yield [a], [_v(sh, 'f32')], {}
def gen_shape_cast(op):
    for sa in VSHAPES:
        for sb in VSHAPES:
            for e in ('f32', 'i32'): yield [_v(sa, 'f32')], [_v(sb, e)], {}
PERMS = [[], [0], [1], [0, 1], [1, 0], [0, 0], [-1, 0], [0, 1, 2], [2, 0, 1], [1, 0, 2], [0, 2, 1], [2, 1, 0], [0, 1, 3]]
def gen_transpose(op):
    for sa in VSHAPES:
        for sb in VSHAPES:
            if len(sa) != len(sb) and (len(sa) + len(sb)) % 3: continue     # 秩不同的组合只取一部分（秩检查独立于其余特征）
            for p in PERMS:
                if abs(len(p) - len(sa)) > 1: continue
                yield [_v(sa, 'f32')], [_v(sb, 'f32')], {'permutation': 'array<i64' + (': ' + ', '.join(map(str, p)) if p else '') + '>'}
# 第 68 轮：切片族。源/目的形状 × 静态 offsets/sizes/strides（含 kDynamic、负数、越界、长度不符）× 结果/源类型（含降秩、编码、元素不同）
DYN = -(1 << 63)
def _t(dims, e='f32', enc=None):
    s = 'tensor<' + ''.join(('?' if d is None else str(d)) + 'x' for d in dims) + e + (f', "{enc}"' if enc else '') + '>'
    return s
SL_SHAPES = [(), (4,), (None,), (4, 6), (1, 6), (None, 6)]
def _sl_attrs(r):
    """每个秩给一组有代表性的 (offsets, sizes, strides)。"""
    opts = {0: [([], [], [])],
            1: [([0], [4], [1]), ([1], [2], [2]), ([DYN], [2], [1]), ([0], [DYN], [DYN]), ([3], [2], [1]), ([4], [1], [1]), ([0], [1], [1]), ([-1], [1], [1]), ([0], [-1], [1]), ([0, 0], [1, 1], [1, 1])],
            2: [([0, 0], [4, 6], [1, 1]), ([0, 0], [1, 6], [1, 1]), ([1, 2], [1, 3], [1, 1]), ([0, DYN], [DYN, 1], [1, DYN]), ([0, 4], [1, 3], [1, 1]), ([0, 0], [1, 1], [1, 1]), ([0, 0], [2, 3], [2, 2]), ([0], [4], [1])]}
    return opts.get(r, [])
def _arr(v): return 'array<i64' + (': ' + ', '.join(map(str, v)) if v else '') + '>'
def _cands(sz, e):
    """候选的切片类型：精确、降秩（去掉 1）、全动态、元素不同、带编码、秩过大。"""
    d = tuple(None if x == DYN else x for x in sz)
    out = {d, tuple(x for x in d if x != 1), tuple(None for _ in d), d + (1,)}
    if d: out |= {d[1:], d[:-1]}
    res = [_t(c, e) for c in out] + [_t(d, 'i32'), _t(d, e, 'enc')]
    return res
def gen_extract_slice(op):
    for sh in SL_SHAPES:
        for o, sz, st in _sl_attrs(len(sh)):
            at = {'static_offsets': _arr(o), 'static_sizes': _arr(sz), 'static_strides': _arr(st)}
            nd = [o.count(DYN), sz.count(DYN), st.count(DYN)]
            for extra in (0, 1):                      # 动态操作数个数正确 / 多一个
                cnt = [nd[0] + extra, nd[1], nd[2]]
                for res in _cands(sz, 'f32'):
                    yield [_t(sh)] + ['index'] * sum(cnt), [res], dict(at, operandSegmentSizes=f'array<i32: 1, {cnt[0]}, {cnt[1]}, {cnt[2]}>')
            yield [_t(sh, enc='enc')] + ['index'] * sum(nd), [_t(tuple(None if x == DYN else x for x in sz))], dict(at, operandSegmentSizes=f'array<i32: 1, {nd[0]}, {nd[1]}, {nd[2]}>')
def gen_insert_slice(op):
    for sh in SL_SHAPES:
        for o, sz, st in _sl_attrs(len(sh)):
            at = {'static_offsets': _arr(o), 'static_sizes': _arr(sz), 'static_strides': _arr(st)}
            nd = [o.count(DYN), sz.count(DYN), st.count(DYN)]
            for extra in (0, 1):
                cnt = [nd[0] + extra, nd[1], nd[2]]
                for src in _cands(sz, 'f32'):
                    for res in (_t(sh), _t(sh, 'i32')):
                        yield [src, _t(sh)] + ['index'] * sum(cnt), [res], dict(at, operandSegmentSizes=f'array<i32: 1, 1, {cnt[0]}, {cnt[1]}, {cnt[2]}>')
# 第 68 轮：transfer_read/write。源（memref/tensor/无秩/vector 元素/complex 元素/vector 本身）× 向量 × permutation_map × in_bounds × mask × 索引个数
TR_SRC = ['memref<4x8xf32>', 'tensor<4x8xf32>', 'memref<8xf32>', 'tensor<*xf32>', 'memref<4x8xi8>', 'memref<4xvector<2xf32>>', 'memref<4x8xcomplex<f32>>', 'memref<f32>', 'vector<4x8xf32>', 'memref<4x8x!llvm.ptr>']
TR_VEC = ['vector<4xf32>', 'vector<2x4xf32>', 'vector<[4]xf32>', 'vector<f32>', 'vector<2xf32>', 'vector<4xi8>', 'vector<4x2xf32>', 'vector<3xf32>', 'vector<8xi32>', 'vector<8x!llvm.ptr>']
TR_MAP = {0: ['affine_map<() -> ()>', 'affine_map<() -> (0)>'],
          1: ['affine_map<(d0) -> (d0)>', 'affine_map<(d0) -> (0)>', 'affine_map<(d0) -> ()>', 'affine_map<(d0) -> (d0, 0)>', 'affine_map<(d0)[s0] -> (d0)>', 'affine_map<(d0) -> (1)>'],
          2: ['affine_map<(d0, d1) -> (d1)>', 'affine_map<(d0, d1) -> (d0)>', 'affine_map<(d0, d1) -> (d0, d1)>', 'affine_map<(d0, d1) -> (d1, d0)>',
              'affine_map<(d0, d1) -> (0, d1)>', 'affine_map<(d0, d1) -> (d1, d1)>', 'affine_map<(d0, d1) -> (d0 + d1)>', 'affine_map<(d0, d1) -> ()>', 'affine_map<(d0) -> (d0)>']}
def _rank(t):
    m = re.match(r'(?:memref|tensor|vector)<((?:[0-9?]+x|\[[0-9]+\]x)*)', t)
    return None if '<*' in t else len(re.findall(r'x', m.group(1))) if m else 0
def _masks(vec, mp):
    """候选 mask：无、按规则推断的、形状错的、非可伸缩版本。"""
    out = [None, 'vector<4xi1>', 'vector<2x4xi1>', 'vector<4x2xi1>', 'vector<[4]xi1>', 'vector<1xi1>']
    return out
def gen_transfer(op):
    rng = random.Random(68)
    for src in TR_SRC:
        r = _rank(src) or 0
        for vi, vec in enumerate(TR_VEC):
            pert = vi < 2                                   # 独立缺陷（in_bounds 长度、索引个数、padding 类型）只在前两种向量上扰动
            for mp in TR_MAP.get(r, []) + (TR_MAP[1][:1] if r != 1 else []):
                n_res = mp.split('->')[1].count('d') + len(re.findall(r'[(, ]0[,)]', mp.split('->')[1]))
                for nb in ({n_res, n_res + 1} if pert else {n_res}):
                    inb = '[' + ', '.join(['false'] * nb) + ']'
                    for nidx in ({r, r + 1} if pert else {r}):
                        for mk in _masks(vec, mp):
                            if mk is not None and rng.random() < 0.6: continue
                            elt = re.search(r'(xcomplex<f32>|vector<2xf32>|f32|i8|!llvm.ptr)>$', src)
                            pad = {'xcomplex<f32>': 'complex<f32>', 'vector<2xf32>': 'vector<2xf32>', 'f32': 'f32', 'i8': 'i8', '!llvm.ptr': '!llvm.ptr'}.get(elt.group(1) if elt else 'f32', 'f32')
                            m_ = [mk] if mk else []
                            at = {'permutation_map': mp, 'in_bounds': inb}
                            if op == 'vector.transfer_read':
                                for pd in ({pad, 'f32', 'i8'} if pert else {pad}):
                                    yield [src] + ['index'] * nidx + [pd] + m_, [vec], dict(at, operandSegmentSizes=f'array<i32: 1, {nidx}, 1, {len(m_)}>')
                            else:
                                for res in ({(src,) if src.startswith('tensor') else (), (), ('tensor<4x8xf32>',)} if pert else {(src,) if src.startswith('tensor') else ()}):
                                    yield [vec, src] + ['index'] * nidx + m_, list(res), dict(at, operandSegmentSizes=f'array<i32: 1, 1, {nidx}, {len(m_)}>')
GEN = {'tensor.dim': gen_dim, 'memref.load': gen_load, 'memref.store': gen_store, 'tensor.cast': gen_cast,
       'vector.transfer_read': gen_transfer, 'vector.transfer_write': gen_transfer,
       'tensor.extract_slice': gen_extract_slice, 'tensor.insert_slice': gen_insert_slice,
       'vector.broadcast': gen_bcast, 'vector.shape_cast': gen_shape_cast, 'vector.transpose': gen_transpose}

def emit(op, ins, outs, attrs):
    args = ', '.join(f'%a{i}: {t}' for i, t in enumerate(ins))
    use = ', '.join(f'%a{i}' for i in range(len(ins)))
    pr = (' <{' + ', '.join(f'{k} = {v}' for k, v in attrs.items()) + '}>') if attrs else ''
    lhs = ', '.join(f'%r{i}' for i in range(len(outs)))
    lhs = lhs + ' = ' if outs else ''
    return (f'func.func @f({args}) {{\n  {lhs}"{op}"({use}){pr} : ({", ".join(ins)}) -> ({", ".join(outs)}) loc("n0")\n  return\n}}\n')

def run_op(op, q):
    import typesig as TS, odssig as G, oracle as OR, nfrules as N
    S = G.lookup(op)[1]; S0 = dict(S, rel=[x for x in S['rel'] if x[2] != 'nf'])
    st = collections.Counter(); ex = collections.defaultdict(list); seen = set()
    for ins, outs, attrs in GEN.get(op, gen_cast)(op):
        key = (tuple(ins), tuple(outs), tuple(sorted(attrs.items())))
        if key in seen: continue
        seen.add(key); st['生成'] += 1
        tr, diags = OR.get().verify(emit(op, ins, outs, attrs))
        if tr == 'parse': st['类型不可解析'] += 1; len(ex['不可解析']) < 2 and ex['不可解析'].append((ins, outs, diags[:1])); continue
        if tr == 'crash': st['MLIR 崩溃'] += 1; len(ex['崩溃']) < 2 and ex['崩溃'].append((ins, outs, attrs))
        try: ti, to = [TS.parse(t) for t in ins], [TS.parse(t) for t in outs]
        except Exception: st['模型不可解析'] += 1; continue
        seg = N.ints(attrs['operandSegmentSizes']) if 'operandSegmentSizes' in attrs else None
        gi, go = G.bind(S, ti, S['ins'], seg), G.bind(S, to, S['outs'])
        model = gi is not None and go is not None and not G.check(S, gi, go, attrs=attrs)
        ods = gi is not None and go is not None and not G.check(S0, gi, go, attrs=attrs)
        truth = tr == 'ok'
        st['比较'] += 1; st['MLIR 接受'] += truth
        if truth and not model: st['误拒'] += 1; len(ex['误拒']) < 4 and ex['误拒'].append((ins, outs, attrs))
        if model and not truth: st['误收'] += 1; len(ex['误收']) < 4 and ex['误收'].append((ins, outs, attrs, diags[0][1][:90] if diags else ''))
        if ods and not truth: st['消融误收'] += 1
    q.put((op, st, dict(ex)))

if __name__ == '__main__':
    import verdict as V, nfrules as N
    # 第 74 轮：N1 的网格只变异端口类型，不生成区域、indexing_maps 与必需属性；linalg op 在此网格上 MLIR 一律拒绝（接受 0，如“requires one region”），
    # 比较没有区分力，且 linalg.map 的结构性拒绝会被记成 nf 误收。linalg 由 nfiface_test N2a/N2b/N2d、packconst_test N2e、winograd_test N2f 检验。
    # 第 79 轮：同一理由推广到所有带区域的 op（第 76 轮起 scf.* 也进了候选：网格不生成区域，MLIR 以“expected 2 regions”拒绝，记成 nf 误收）。
    # 带区域的 op 由 scf_test N2g、regcomp_test R1/R2、regflow_test 检验。看到第 79 轮首跑结果后做的范围修正，如实记录。
    import odssig as G_
    def _has_regions(o):
        lk = G_.lookup(o); return bool(lk and lk[1].get('regions'))
    ops = sys.argv[1:] or sorted(o for o in N.CANDIDATES if not o.startswith('linalg.') and not _has_regions(o))
    q = mp.Queue(); ps = [mp.Process(target=run_op, args=(o, q)) for o in ops]
    for p in ps: p.start()
    res = {}
    for _ in ps:
        o, st, ex = q.get(); res[o] = (st, ex)
    for p in ps: p.join()
    ok = True; tot = collections.Counter()
    for o in ops:
        st, ex = res[o]; tot.update(st)
        good = (st['误拒'] == 0 and st['误收'] == 0 and st['MLIR 接受'] >= 1 and st['比较'] - st['MLIR 接受'] >= 1
                and st['比较'] >= 0.8 * st['生成'])
        ok &= good
        print(f"{o:22s} {'PASS' if good else 'FAIL'} 生成 {st['生成']} 比较 {st['比较']} 接受 {st['MLIR 接受']} 误拒 {st['误拒']} 误收 {st['误收']} "
              f"不可解析 {st['类型不可解析']} 崩溃 {st['MLIR 崩溃']} | 消融误收 {st['消融误收']}")
        for k, v in ex.items(): print('   ', k, v[:4])
    V.check('N1 nf≡MLIR', ok, f"op {len(ops)}，比较 {tot['比较']}（接受 {tot['MLIR 接受']}），误拒 {tot['误拒']}，误收 {tot['误收']}，崩溃 {tot['MLIR 崩溃']}；消融误收 {tot['消融误收']}")
    V.finish()
