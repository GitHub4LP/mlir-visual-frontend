"""第 62 轮：复合 op 显式签名 Φ（model/odscomp.py）的外部检验。复合 op 取自 MLIR 测试集（compcorpus_test 的切片与变异）。

阈值（运行前写定，事后不改）：
  S1 化简 ≡ 原始：所有实例上 evaluate(simplify(Φ)) == evaluate(Φ)，不一致 0。
  S2 Φ ≡ 逐点判定（第 62 轮二跑后补：推导异常须为 0，否则切片被静默排除）：Φ(T) 与 composite.instantiate + core.typecheck（不开预言机）一致，不一致 0
     （instantiate 失败 ⇔ Φ 拒绝）。
  （第 64 轮）残差分为两类：定值残差指节点类型不含端口变量，校验结果是常量，在定义时对实现图校验一次（原类型被 MLIR 接受）即可消去；
  其余为残差。S3 只在原类型被接受后计数，所以“残差为空”指消去定值残差后为空。阈值不变。
  S3 对 MLIR：误拒（Φ 拒绝、MLIR 接受）0；误收（Φ 接受、MLIR 拒绝）只允许出现在 Unknown 残差非空的复合 op 上，残差为空时误收 0。
  S4 嵌套 ≡ 展开：把切片前半作为内层复合 op 嵌进外层，Φ_嵌套(T) == Φ_展开(T)，不一致 0，且实例数 > 0。
  S5（只报告，无阈值）：残差为空的复合 op 占比、化简前后原子数、呈现样例。
用法：PYTHONPATH=model python3 model/odscomp_test.py [每函数切片数] [每切片变异数] [文件数上限] [起始文件]
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import slices, mutate, _walk, _funcs

def run_file(path, q, kslice, kmut, seed):
    import mlir.ir as ir, irimport as I, core as C, oracle as OR, odscomp as D
    from composite import instantiate
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        pool = sorted({t for f in _walk(root) for t in [o[1] for o in f['outs']] if len(t) < 60})
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']
            cand = list(slices(body)); rng.shuffle(cand)
            tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
            for sl in cand[:kslice]:
                defined = {o for n in sl for o, _ in n['outs']}; ports, seen = [], set()
                for n in sl:
                    for v in n['ins']:
                        if v not in defined and v not in seen: seen.add(v); ports.append(v)
                if not ports: continue
                try: P = D.derive(sl, ports); Q = D.simplify(P)
                except Exception as e: st['推导异常'] += 1; note('推导异常', f'{[n["op"] for n in sl]} {type(e).__name__}: {str(e)[:90]}'); continue
                unk = bool(P.residual)
                st['复合 op'] += 1; st['复合 op/残差为空'] += not unk; st['复合 op/含定值残差'] += bool(P.ground); st['复合 op/仅定值残差'] += bool(P.ground) and not unk
                st['原子/化简前'] += len(P.atoms); st['原子/化简后'] += len(Q.atoms)
                if not unk and len(Q.atoms) and len(ex['呈现']) < 6: note('呈现', f'{[n["op"] for n in sl]} {D.render(Q, ports)}')
                for r in P.residual: st['残差/' + re.sub(r'^[\w.]+: ', '', r)[:40]] += 1
                # 嵌套：前半作内层
                nest = None
                if len(sl) >= 2:
                    h = len(sl) // 2; inn = sl[:h]; idef = [o for n in inn for o, _ in n['outs']]
                    iports, s2 = [], set()
                    for n in inn:
                        for v in n['ins']:
                            if v not in {o for x in inn for o, _ in x['outs']} and v not in s2: s2.add(v); iports.append(v)
                    if iports:
                        Pi = D.derive(inn, iports); D.set_outs(Pi, idef)
                        node = {'id': 'nest', 'op': 'comp.inner', 'ins': iports, 'outs': [(o, tyof[o]) for o in idef], 'props': {}, 'regions': [], 'succs': []}
                        nest = D.derive([node] + sl[h:], ports, inner={'comp.inner': Pi})
                T0 = [tyof[v] for v in ports]
                Ts = [('原类型', T0)] + [('变异', mutate(rng, T0, pool)) for _ in range(kmut)]
                for kind, T in Ts:
                    raw = D.evaluate(P, T)
                    if raw is None: st['类型文本不可解析'] += 1; continue
                    simp = D.evaluate(Q, T)
                    st['实例'] += 1; st['S1 不一致'] += simp != raw
                    if simp != raw: note('S1 不一致', f'{[n["op"] for n in sl]} {T} raw={raw}')
                    bs = instantiate(sl, ports, T); ft = '(' + ', '.join(T) + ') -> ()'
                    if bs is None: pt = False
                    else:
                        C.ORACLE_ON = False
                        try: pt = not C.typecheck(bs, ft)
                        except Exception: st['core 异常'] += 1; continue
                    if raw is D.UNK: st['点未定（nf 未建模）'] += 1
                    st['S2 不一致'] += raw is not D.UNK and pt != raw
                    if raw is not D.UNK and pt != raw: note('S2 不一致', f'{[n["op"] for n in sl]} {T} Φ={raw} 逐点={pt}')
                    if nest is not None:
                        nv = D.evaluate(nest, T); st['S4 实例'] += 1; st['S4 不一致'] += nv != raw
                        if nv != raw: note('S4 不一致', f'{[n["op"] for n in sl]} {T} 嵌套={nv} 展开={raw}')
                    if bs is None: continue
                    text = OR.emit_core(bs, ft)
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr in ('parse', 'crash'): st['MLIR ' + tr] += 1; continue
                    truth = tr == 'ok'
                    if kind == '原类型' and not truth: st['原类型独立后被拒（依赖上下文）'] += 1; break
                    st['S3 实例'] += 1; st['S3 MLIR 接受'] += truth
                    st['S3 误拒'] += truth and raw is False
                    if truth and raw is False: note('S3 误拒', f'{[n["op"] for n in sl]} {T}')
                    if raw is D.UNK: st['S3 点未定（交预言机）'] += 1; st['S3 点未定/MLIR 接受'] += truth
                    if raw is True and not truth:
                        st['S3 误收/残差非空（Unknown）' if unk else 'S3 误收/残差为空'] += 1
                        if not unk: note('S3 误收/残差为空', f'{[n["op"] for n in sl]} {T} {diags[0][1][:100] if diags else ""}')
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    kmut = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0          # 分段运行（后台单条命令上限 1200 s）
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[start:lim], start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, kslice, kmut, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('S1 化简≡原始', tot['文件超时/崩溃'] == 0 and tot['实例'] > 0 and tot['S1 不一致'] == 0, f"实例 {tot['实例']}，不一致 {tot['S1 不一致']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('S2 Φ≡逐点', tot['实例'] > 0 and tot['S2 不一致'] == 0 and tot['推导异常'] == 0, f"不一致 {tot['S2 不一致']}/{tot['实例']}，推导异常 {tot['推导异常']}")
    V.check('S3 对MLIR', tot['S3 实例'] > 0 and tot['S3 误拒'] == 0 and tot['S3 误收/残差为空'] == 0,
            f"实例 {tot['S3 实例']}（接受 {tot['S3 MLIR 接受']}），误拒 {tot['S3 误拒']}，误收：残差为空 {tot['S3 误收/残差为空']}，残差非空 {tot['S3 误收/残差非空（Unknown）']}；点未定 {tot['S3 点未定（交预言机）']}（MLIR 接受 {tot['S3 点未定/MLIR 接受']}）")
    V.check('S4 嵌套≡展开', tot['S4 实例'] > 0 and tot['S4 不一致'] == 0, f"实例 {tot['S4 实例']}，不一致 {tot['S4 不一致']}")
    print(f"S5 复合 op {tot['复合 op']}，残差为空 {tot['复合 op/残差为空']}（其中仅有定值残差 {tot['复合 op/仅定值残差']}，含定值残差 {tot['复合 op/含定值残差']}）；原子 {tot['原子/化简前']} → {tot['原子/化简后']}")
    V.finish()
