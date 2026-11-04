"""第 66 轮：规范化规则（nfrules）对 MLIR verifier 的检验。

方法：规则只依赖少数类型特征（容器种类、秩与维、编码、元素种类、位宽、符号性）。按这些特征的等价类构造类型池，
对每个候选 op 枚举端口类型组合，比较 模型（ODS 签名 ∧ 规范化原子）与 MLIR verifier 的判定。
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

def run_op(op, q):
    import typesig as TS, odssig as G, oracle as OR, nfrules as N
    S = G.lookup(op)[1]; S0 = dict(S, rel=[x for x in S['rel'] if x[2] != 'nf'])
    st = collections.Counter(); ex = collections.defaultdict(list)
    for a, b in dict.fromkeys(pairs()):
        st['生成'] += 1
        text = f'func.func @f(%a: {a}) {{\n  %r = "{op}"(%a) : ({a}) -> {b} loc("n0")\n  return\n}}\n'
        tr, diags = OR.get().verify(text)
        if tr == 'parse': st['类型不可解析'] += 1; continue
        if tr == 'crash': st['MLIR 崩溃'] += 1; len(ex['崩溃']) < 2 and ex['崩溃'].append((a, b))
        try: ta, tb = TS.parse(a), TS.parse(b)
        except Exception: st['模型不可解析'] += 1; continue
        truth = tr == 'ok'; model = not G.check(S, [[ta]], [[tb]]); ods = not G.check(S0, [[ta]], [[tb]])
        st['比较'] += 1; st['MLIR 接受'] += truth
        if truth and not model: st['误拒'] += 1; len(ex['误拒']) < 4 and ex['误拒'].append((a, b))
        if model and not truth: st['误收'] += 1; len(ex['误收']) < 4 and ex['误收'].append((a, b, diags[0][1][:90] if diags else ''))
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
