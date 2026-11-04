"""第 69 轮：接口级规范化规则的检验（closure §9.9）。

接口规则只覆盖 op 的一部分 declared 条目（op 自身的 C++ verify 等可能仍在残差中），因此不能要求“规则 ≡ MLIR 整体判定”。
判据（运行前写定）：在真实 IR 的单个 op 上变异 init 端口类型与结果类型，逐例比较规则 r 与 MLIR：
  N2a 误拒 0：MLIR 接受 ⇒ r 接受（规则可靠，可以有拒绝权）。
  N2b 漏判 0：MLIR 报出该接口的诊断 ⇒ r 拒绝（规则不弱于该接口的 verify）。
      （MLIR 拒绝但报的是别的诊断时，接口 verify 可能没有运行，不计。）
  N2c 非空：MLIR 接受 ≥ 1，接口诊断 ≥ 1，r 拒绝 ≥ 1；文件工作进程崩溃 0。
运行：PYTHONPATH=model python3 model/nfiface_test.py [语料文件上限] [起点]
语料：corpus/*/*.mlir 加 corpus_ext/Linalg（MLIR 9bae84b0 mlir/test/Dialect/Linalg；主语料不含 Linalg，单独存放以免改变既有基线）。
"""
import sys, re, glob, random, collections, multiprocessing as mp

IFACE_DIAG = {'DPS': re.compile(r'is a tensor or a memref|to be equal to the number of output tensors|to match type of corresponding result')}

def _variants(t):
    """一个类型的变异：tensor ↔ memref、换形状、无秩。"""
    m = re.fullmatch(r'(tensor|memref)<(.*)>', t)
    if not m: return []
    k, body = m.groups()
    if body.startswith('*x'): return [('memref' if k == 'tensor' else 'tensor') + '<' + body + '>']
    parts = body.split(',')[0]
    dims = re.match(r'((?:[0-9?]+x)*)(.*)', parts)
    d, e = dims.groups()
    other = 'memref' if k == 'tensor' else 'tensor'
    out = [f'{other}<{d}{e}>', f'{k}<*x{e}>' if k == 'tensor' else f'memref<*x{e}>', f'{k}<7x{d}{e}>']
    if e != 'f32': out.append(f'{k}<{d}f32>')
    return out

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odscomp as D, odssig as G, typesig as TS, structure as ST
    from composite import instantiate
    from compcorpus_test import _walk, _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    T_ = lambda t: True
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']
            tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
            for n in body:
                lk = G.lookup(n['op'])
                if not lk: continue
                S = lk[1]
                atoms = [x for x in S['rel'] if x[0] == 'nf' and str(x[3]).startswith('DPS')]
                if not atoms: continue
                x = atoms[0]
                ports = list(dict.fromkeys(n['ins']))
                if not ports: continue
                gname = x[1][0][1]
                try:
                    gi0 = G.bind(S, [TS.parse(tyof[v]) for v in n['ins']], S['ins'], D._seg(n))
                except Exception: continue
                if gi0 is None: continue
                names = [nm for nm, *_ in S['ins']]
                k0 = sum(len(g) for g in gi0[:names.index(gname)]); kn = len(gi0[names.index(gname)])
                init_vals = n['ins'][k0:k0 + kn]
                T0 = [tyof[v] for v in ports]
                cases = [('原类型', T0, None)]
                for v in init_vals:
                    j = ports.index(v)
                    for t in _variants(T0[j]): cases.append(('init', T0[:j] + [t] + T0[j + 1:], None))
                for k, (o, t) in enumerate(n['outs']):
                    for t2 in _variants(t): cases.append(('result', T0, (k, t2)))
                st['op 实例'] += 1
                for kind, T, rmut in cases:
                    bs = instantiate([n], ports, T)
                    if bs is None: st['不可实例化'] += 1; continue
                    node = bs[0]['ops'][0]
                    if rmut is not None:
                        k, t2 = rmut; node['outs'] = [(o, t2 if i == k else t) for i, (o, t) in enumerate(node['outs'])]
                    try:
                        ins_t = [TS.parse(t) for t in node['props']['_types']]; outs_t = [TS.parse(t) for _, t in node['outs']]
                        gi = G.bind(S, ins_t, S['ins'], D._seg(n)); go = G.bind(S, outs_t, S['outs'])
                    except Exception: st['类型不可解析'] += 1; continue
                    if gi is None or go is None: st['无法绑定'] += 1; continue
                    mini = dict(S, ins=[(nm, T_, mm, b) for nm, _, mm, b in S['ins']], outs=[(nm, T_, mm, b) for nm, _, mm, b in S['outs']], infer=None, rel=[x])
                    r_rej = bool(G.check(mini, gi, go))
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    if tr == 'crash': st['MLIR 崩溃'] += 1; continue
                    ok = tr == 'ok'; msg = ' '.join(d[1] for d in diags) if diags else ''
                    dps = bool(IFACE_DIAG['DPS'].search(msg))
                    st['比较'] += 1; st['MLIR 接受'] += ok; st['接口诊断'] += dps; st['规则拒绝'] += r_rej
                    st['变异/' + kind] += 1
                    if ok and r_rej: st['N2a 误拒'] += 1; note('误拒', f"{n['op']} {kind} {T} {rmut}")
                    if dps and not r_rej: st['N2b 漏判'] += 1; note('漏判', f"{n['op']} {kind} {T} {rmut} {msg[:90]}")
                    if not ok and not dps and r_rej: st['规则拒绝/MLIR 别的诊断'] += 1
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate((sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/*/*.mlir')))[start:lim], start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('N2a 接口规则无误拒', tot['比较'] > 0 and tot['N2a 误拒'] == 0 and tot['文件超时/崩溃'] == 0,
            f"比较 {tot['比较']}（op 实例 {tot['op 实例']}），MLIR 接受 {tot['MLIR 接受']}，误拒 {tot['N2a 误拒']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('N2b 接口诊断必拒', tot['接口诊断'] > 0 and tot['N2b 漏判'] == 0 and tot['MLIR 接受'] > 0 and tot['规则拒绝'] > 0,
            f"接口诊断 {tot['接口诊断']}，漏判 {tot['N2b 漏判']}，规则拒绝 {tot['规则拒绝']}（其中 MLIR 报别的诊断 {tot['规则拒绝/MLIR 别的诊断']}），MLIR 崩溃 {tot['MLIR 崩溃']}")
    V.finish()
