"""第 74 轮：linalg.softmax 与 winograd_{filter,input,output}_transform 的 nf 规则检验（closure §9.14）。

语料中这 4 个 op 的实例少，且多为合法的 F_4_3 静态形状；整除失败、F_2_5、动态维、left/right 单侧变换几乎没有，因此用合成网格补充。
判据（运行前写定）：
  N2f 完整模型（无预言机）≡ MLIR：误拒 0、误收 0；比较 ≥ 1500，每个 op 比较 ≥ 250 且 MLIR 接受 ≥ 30、拒绝 ≥ 30；工作进程崩溃 0，模型异常 0。
网格：fmr ∈ {F_2_3, F_4_3, F_2_5}；各维从小整数与 ? 中取，输出形状一半取按 MLIR 公式推出的值（再逐维扰动），一半随机；
softmax：tensor/memref，秩 1–3，元素类型 f32/f16，dimension ∈ [-1, 3]。
运行：PYTHONPATH=model python3 model/winograd_test.py [样本数] [种子]
"""
import sys, random, collections, multiprocessing as mp
FMR = [(2, 3), (4, 3), (2, 5)]

def T(k, s, e='f32'): return f'{k}<' + ''.join(('?' if d is None else str(d)) + 'x' for d in s) + e + '>'
def perturb(rng, s, pool):
    return [d if rng.random() < 0.75 else rng.choice(pool) for d in s]

def gen(rng):
    op = rng.choice(['softmax', 'filter', 'input', 'output'])
    D = [1, 2, 3, 4, 5, 6, 8, 10, None]
    at = {}
    if op == 'softmax':
        k = rng.choice(['tensor', 'memref']); r = rng.choice([1, 2, 3])
        si = [rng.choice(D) for _ in range(r)]
        so = perturb(rng, si, D) if rng.random() < 0.8 else [rng.choice(D) for _ in range(rng.choice([1, 2, 3]))]
        ei = 'f32'; eo = 'f32' if rng.random() < 0.85 else 'f16'
        ti, to = T(k, si, ei), T(k, so, eo)
        at['dimension'] = f'{rng.choice([-1, 0, 0, 1, 1, 2, 3])} : i64'
        return 'linalg.softmax', [ti, to], ([to] if k == 'tensor' else []), at
    fi = rng.randrange(3); m, r = FMR[fi]; a = m + r - 1; at['fmr'] = f'{fi} : i32'
    if op == 'filter':
        hw = [r, 1, None, 2, 3, 5]
        fs = [rng.choice(D), rng.choice(hw), rng.choice(hw), rng.choice(D)]
        if rng.random() < 0.5:
            o = [a if fs[1] == r else 1, a if fs[2] == r else 1, fs[3], fs[0]]
            o = perturb(rng, o, [None, a, 1, 6, 4])
        else: o = [rng.choice([a, 1, None, 6, 4]) for _ in range(2)] + [rng.choice(D) for _ in range(2)]
        ti, to = T('tensor', fs), T('tensor', o)
        return 'linalg.winograd_filter_transform', [ti, to], [to], at
    if op == 'input':
        hw = [None, 1, 4, 6, 8, 10, 3, 7]
        ins = [rng.choice(D), rng.choice(hw), rng.choice(hw), rng.choice(D)]
        if rng.random() < 0.6:
            left, right = rng.random() < 0.8, rng.random() < 0.8
            def ax(x, side):
                if x is None: return a, None
                return (a, int((x - (r - 1)) / m)) if side else (1, x)
            ah, th = ax(ins[1], left); aw, tw = ax(ins[2], right)
            o = perturb(rng, [ah, aw, th, tw, ins[0], ins[3]], [None, a, 1, 2, 0])
            o = [None if (d is not None and d < 0) else d for d in o]
        else: o = [rng.choice([a, 1, None]) for _ in range(2)] + [rng.choice([1, 2, 3, None]) for _ in range(2)] + [rng.choice(D) for _ in range(2)]
        ti, to = T('tensor', ins), T('tensor', o)
        return 'linalg.winograd_input_transform', [ti, to], [to], at
    v = [rng.choice([a, a, 1, None, 4, 6, 8]), rng.choice([a, a, 1, None, 4, 6, 8]), rng.choice([1, 2, 3, None]), rng.choice([1, 2, 3, None]), rng.choice(D), rng.choice(D)]
    if rng.random() < 0.6:
        def ax(al, t):
            if al is None or t is None: return None
            return (m if al != 1 else 1) * t
        o = perturb(rng, [v[4], ax(v[0], v[2]), ax(v[1], v[3]), v[5]], [None, 4, 8, 2])
    else: o = [rng.choice(D) for _ in range(4)]
    ti, to = T('tensor', v), T('tensor', o)
    return 'linalg.winograd_output_transform', [ti, to], [to], at

def work(n, seed, q):
    import core as C, oracle as OR
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    for _ in range(n):
        op, Ts, Rs, at = gen(rng)
        ports = ['%x', '%y']
        nodes = [{'id': f'{op}#0', 'op': op, 'ins': ports, 'outs': [(f'%r{i}', t) for i, t in enumerate(Rs)], 'succs': [], 'regions': [],
                  'props': {'_attrs': at, '_types': list(Ts)}},
                 {'id': 'func.return#0', 'op': 'func.return', 'ins': [], 'outs': [], 'succs': [], 'regions': [], 'props': {'_attrs': {}, '_types': []}}]
        bs = [{'args': list(zip(ports, Ts)), 'ops': nodes}]
        sig = '(' + ', '.join(Ts) + ') -> ()'
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
        st['比较'] += 1; st[f'{op}/比较'] += 1; st[f'{op}/MLIR 接受'] += ok; st[f'{op}/MLIR 拒绝'] += not ok
        if ok and rej: st['误拒'] += 1; ex['误拒'].append(text[-500:])
        if not ok and not rej: st['误收'] += 1; ex['误收'].append(text[-500:] + ' || ' + ' '.join(d[1] for d in diags)[:150])
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 74
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
    ops = ['linalg.softmax', 'linalg.winograd_filter_transform', 'linalg.winograd_input_transform', 'linalg.winograd_output_transform']
    per = all(tot[f'{o}/比较'] >= 250 and tot[f'{o}/MLIR 接受'] >= 30 and tot[f'{o}/MLIR 拒绝'] >= 30 for o in ops)
    V.check('N2f softmax/winograd 完整模型≡MLIR', tot['比较'] >= 1500 and per and tot['误拒'] == 0 and tot['误收'] == 0
            and tot['工作进程崩溃'] == 0 and tot['模型异常'] == 0,
            f"比较 {tot['比较']}（" + '，'.join(f"{o.split('.')[1]} {tot[o + '/比较']}/{tot[o + '/MLIR 接受']}/{tot[o + '/MLIR 拒绝']}" for o in ops)
            + f"，比较/接受/拒绝），误拒 {tot['误拒']}，误收 {tot['误收']}，MLIR 崩溃 {tot['MLIR 崩溃（按拒绝计）']}")
    V.finish()
