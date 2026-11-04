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
import sys, itertools, collections, multiprocessing as mp

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
GEN = {'tensor.dim': gen_dim, 'memref.load': gen_load, 'memref.store': gen_store, 'tensor.cast': gen_cast,
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
        gi, go = G.bind(S, ti, S['ins']), G.bind(S, to, S['outs'])
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
    ops = sys.argv[1:] or sorted(N.CANDIDATES)
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
