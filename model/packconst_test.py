"""第 73 轮：linalg.pack/unpack 的值前提（常量 tile）检验（closure §9.13）。

语料里带动态常量 tile 的 pack/unpack 只有几例（nfiface_test 中 const 变异 24 例），不足以检验值前提，因此用合成网格补充。
判据（运行前写定）：
  N2e 完整模型（无预言机）≡ MLIR：误拒 0、误收 0；比较 ≥ 1000，其中含常量 tile ≥ 300，MLIR 接受 ≥ 100、拒绝 ≥ 100；工作进程崩溃 0。
网格：未打包端 1/2 维（各维取 6、8、?），inner_dims_pos 取合法或越界/重复，tile 取 静态 {2,4,0} / 常量 {0,1,2,3,4} / 端口，
outer_dims_perm 取 无 / [1,0]，打包端各维取 推出值 / ? / 推出值+1 / 常量值，pack 的 padding 取 无 / f32 / i32。
运行：PYTHONPATH=model python3 model/packconst_test.py [样本数] [种子]
"""
import sys, random, collections, multiprocessing as mp
DYN = -9223372036854775808

def gen(rng):
    op = rng.choice(['linalg.pack', 'linalg.unpack'])
    r = rng.choice([1, 2]); us = [rng.choice([6, 8, None]) for _ in range(r)]
    inner = rng.choice([[0]] if r == 1 else [[1], [0, 1], [1, 0], [0]])
    if rng.random() < 0.08: inner = rng.choice([[r], [0, 0], [0] * (r + 1)])
    tiles = []
    for _ in inner:
        k = rng.random()
        tiles.append(('s', rng.choice([2, 4, 4, 0 if rng.random() < 0.1 else 2])) if k < 0.3 else ('c', rng.choice([0, 1, 2, 3, 4, 4])) if k < 0.8 else ('p', None))
    outer = [1, 0] if r == 2 and rng.random() < 0.3 else []
    # 推出的打包形状（用“真实”tile 值，常量也算），再逐维扰动
    sh = list(us)
    for p, (k, v) in zip(inner, tiles):
        if p >= r or sh[p] is None: continue
        if k == 'p' or v == 0: sh[p] = None
        else: sh[p] = -((-sh[p]) // v)
    if outer: sh = [sh[i] for i in outer]
    inn = [v if k != 'p' else None for k, v in tiles]
    pk = sh + inn
    pk = [d if rng.random() < 0.6 else rng.choice([None, (d or 3) + 1, 4]) for d in pk]
    pk = [None if (d is not None and d < 0) else d for d in pk]
    pad = rng.choice([None, None, 'f32', 'i32']) if op == 'linalg.pack' else None
    t = lambda s: 'tensor<' + ''.join(('?' if d is None else str(d)) + 'x' for d in s) + 'f32>'
    ut, pt = t(us), t(pk)
    st = [v if k == 's' else DYN for k, v in tiles]
    nodes, ports, T, tin = [], ['%src', '%dst'], [ut if op == 'linalg.pack' else pt, pt if op == 'linalg.pack' else ut], []
    if pad: ports.append('%pad'); T.append(pad)
    for i, (k, v) in enumerate(tiles):
        if k == 'c':
            nodes.append({'id': f'arith.constant#c{i}', 'op': 'arith.constant', 'ins': [], 'outs': [(f'%c{i}', 'index')], 'succs': [], 'regions': [],
                          'props': {'_attrs': {'value': f'{v} : index'}, '_types': []}}); tin.append(f'%c{i}')
        elif k == 'p': ports.append(f'%t{i}'); T.append('index'); tin.append(f'%t{i}')
    at = {'inner_dims_pos': 'array<i64: ' + ', '.join(map(str, inner)) + '>', 'static_inner_tiles': 'array<i64: ' + ', '.join(map(str, st)) + '>'}
    if outer: at['outer_dims_perm'] = 'array<i64: 1, 0>'
    ins = ['%src', '%dst'] + (['%pad'] if pad else []) + tin
    if op == 'linalg.pack': at['operandSegmentSizes'] = f'array<i32: 1, 1, {1 if pad else 0}, {len(tin)}>'
    dt = T[1]
    tys = dict(zip(ports, T)) | {n['outs'][0][0]: 'index' for n in nodes}
    nodes.append({'id': f'{op}#0', 'op': op, 'ins': ins, 'outs': [('%r', dt)], 'succs': [], 'regions': [], 'props': {'_attrs': at, '_types': [tys[v] for v in ins]}})
    nodes.append({'id': 'func.return#0', 'op': 'func.return', 'ins': [], 'outs': [], 'succs': [], 'regions': [], 'props': {'_attrs': {}, '_types': []}})
    return nodes, ports, T, any(k == 'c' for k, _ in tiles)

def work(n, seed, q):
    import core as C, oracle as OR
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    for _ in range(n):
        nodes, ports, T, hasc = gen(rng)
        bs = [{'args': list(zip(ports, T)), 'ops': nodes}]
        sig = '(' + ', '.join(T) + ') -> ()'
        text = OR.emit_core(bs, sig)
        if text is None: st['不可发射'] += 1; continue
        tr, diags = OR.get().verify(text)
        if tr == 'parse': st['MLIR parse'] += 1; continue
        if tr == 'crash': st['MLIR 崩溃（按拒绝计）'] += 1
        ok = tr == 'ok'
        old = C.ORACLE_ON; C.ORACLE_ON = False
        try: rej = bool(C.typecheck(bs, sig))
        except Exception as e: st['模型异常'] += 1; ex['模型异常'].append(f'{type(e).__name__}: {e}'[:120]); continue
        finally: C.ORACLE_ON = old
        st['比较'] += 1; st['常量 tile'] += hasc; st['MLIR 接受'] += ok; st['MLIR 拒绝'] += not ok
        if ok and rej: st['误拒'] += 1; ex['误拒'].append(text[:400])
        if not ok and not rej: st['误收'] += 1; ex['误收'].append(text[:400] + ' || ' + ' '.join(d[1] for d in diags)[:150])
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1600; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 73
    ctx = mp.get_context('fork'); tot = collections.Counter(); exs = collections.defaultdict(list); W = 4
    qs = []
    for w in range(W):
        q = ctx.Queue(); p = ctx.Process(target=work, args=(n // W, seed * 100 + w, q)); p.start(); qs.append((p, q))
    for p, q in qs:
        try: st, ex = q.get(timeout=1100)
        except Exception: p.kill(); tot['工作进程崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:3]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:6]]
    V.check('N2e pack/unpack 常量 tile 完整模型≡MLIR', tot['比较'] >= 1000 and tot['常量 tile'] >= 300 and tot['MLIR 接受'] >= 100 and tot['MLIR 拒绝'] >= 100
            and tot['误拒'] == 0 and tot['误收'] == 0 and tot['工作进程崩溃'] == 0 and tot['模型异常'] == 0,
            f"比较 {tot['比较']}（常量 tile {tot['常量 tile']}，MLIR 接受 {tot['MLIR 接受']}、拒绝 {tot['MLIR 拒绝']}），误拒 {tot['误拒']}，误收 {tot['误收']}，MLIR 崩溃 {tot['MLIR 崩溃（按拒绝计）']}")
    V.finish()
