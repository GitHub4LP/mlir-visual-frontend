"""第 77 轮：区域类型流（regionflow）让最大性延伸到区域内（closure §9.17）。

对含区域的复合 op 切片（regcomp_test.rslices），比较区域类型流关闭（块参数/区域内结果取写定类型，第 75–76 轮）与打开时的 Φ：
判据（运行前写定）：
  F1 单调：Φ_旧(T) = 接受 ⇒ Φ_新(T) = 接受（不计点未定），违例 0。
  F2 增益真实：Φ_旧(T) 拒绝而 Φ_新(T) 接受、且残差为空的实例，MLIR 对新实例化全部接受（误收 0）；这类实例 ≥ 50。
  F3（只报告）：上述增益实例中，MLIR 对旧实例化（写定类型）拒绝的个数——说明旧签名确实不是最大的。
  文件崩溃 0、推导异常 0。
  （“接受”= Φ 为真且残差为空；Φ 为真而残差非空是 Unknown，不算接受。第 77 轮首跑把后者算作旧接受，F1 报 6 例违例，均为 linalg.reduce 残差非空，已按写定判据改正。）
F4（第 77 轮补充，在看到 F2 因随机变异命中太少而 FAIL 后写定，F2 原样保留为 FAIL）：
  一致替换元素类型（端口里所有 f32→f64、f16→f32、i32→i64、i64→i32 中的一种同时替换），
  Φ_旧 拒绝而 Φ_新 接受、残差为空的实例 ≥ 50，且 MLIR 对新实例化全部接受（误收 0）；
  同时 Φ_新 拒绝的实例里，MLIR 接受新实例化的个数 = 0（误拒 0）。
用法：PYTHONPATH=model python3 model/regflow_test.py [每函数切片数] [每切片变异数] [文件数上限] [起始文件]
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import mutate, _walk, _funcs
from regcomp_test import rslices, free_values, types_of, _nested

def run_file(path, q, kslice, kmut, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odscomp as D, regionflow as RF
    from composite import instantiate
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    def mlir_ok(bs, ft):
        if bs is None: return None
        text = OR.emit_core(bs, ft)
        if text is None: return None
        tr, _ = OR.get().verify(text)
        return None if tr == 'parse' else tr == 'ok'
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        pool = sorted({t for f in _walk(root) for t in [o[1] for o in f['outs']] if len(t) < 60})
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']; tyof = types_of(fn)
            cand = list(rslices(body)); rng.shuffle(cand)
            for sl in cand[:kslice]:
                ports = free_values(sl)
                if not ports or any(v not in tyof for v in ports): continue
                try:
                    RF.ENABLED = False; Po = D.derive(sl, ports)
                    RF.ENABLED = True; Pn = D.derive(sl, ports)
                except Exception as e:
                    RF.ENABLED = True; st['推导异常'] += 1; note('推导异常', f'{[n["op"] for n in sl]} {type(e).__name__}: {str(e)[:120]}'); continue
                st['复合 op'] += 1
                T0 = [tyof[v] for v in ports]
                for T in [T0] + [mutate(rng, T0, pool) for _ in range(kmut)]:
                    o, nw = D.evaluate(Po, T), D.evaluate(Pn, T)
                    if o is None or nw is None: continue
                    st['实例'] += 1
                    if o is D.UNK or nw is D.UNK: st['点未定'] += 1; continue
                    if o is True and Po.residual: st['旧为真但残差非空（Unknown）'] += 1
                    if o is True and not Po.residual and nw is not True: st['F1 违例'] += 1; note('F1 违例', f'{[n["op"] for n in sl]} {T}')
                    if o is False and nw is True:
                        st['增益'] += 1
                        if Pn.residual: st['增益/残差非空'] += 1; continue
                        ft = '(' + ', '.join(T) + ') -> ()'
                        okn = mlir_ok(instantiate(sl, ports, T), ft)
                        RF.ENABLED = False
                        try: oko = mlir_ok(instantiate(sl, ports, T), ft)
                        finally: RF.ENABLED = True
                        if okn is None: st['增益/MLIR 不可判'] += 1; continue
                        st['F2 增益（残差为空）'] += 1; st['F2 MLIR 接受'] += okn
                        if not okn: st['F2 误收'] += 1; note('F2 误收', f'{[n["op"] for n in sl]} {T}')
                        else: note('增益例', f'{[n["op"] for n in sl]} {T}')
                        st['F3 旧实例化 MLIR 拒绝'] += (oko is False)
                for a_, b_ in (('f32', 'f64'), ('f16', 'f32'), ('i32', 'i64'), ('i64', 'i32')):
                    if not any(re.search(r'\b' + a_ + r'\b', t) for t in T0): continue
                    T = [re.sub(r'\b' + a_ + r'\b', b_, t) for t in T0]
                    o, nw = D.evaluate(Po, T), D.evaluate(Pn, T)
                    if o is None or nw is None or o is D.UNK or nw is D.UNK or Pn.residual: continue
                    st['F4 实例'] += 1
                    ft = '(' + ', '.join(T) + ') -> ()'
                    okn = mlir_ok(instantiate(sl, ports, T), ft)
                    if okn is None: st['F4 MLIR 不可判'] += 1; continue
                    if nw is False and okn: st['F4 误拒'] += 1; note('F4 误拒', f'{[n["op"] for n in sl]} {T}')
                    if nw is True and not okn: st['F4 误收(全部)'] += 1; note('F4 误收', f'{[n["op"] for n in sl]} {T}')
                    if not (o is True and not Po.residual) and nw is True:
                        st['F4 增益'] += 1
                        if not okn: st['F4 误收'] += 1
                        else: note('F4 增益例', f'{[n["op"] for n in sl]} {T}')
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    kmut = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/Linalg/*.mlir'))
    for i, path in enumerate(files[start:lim], start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, kslice, kmut, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('F1 区域类型流单调', tot['实例'] > 0 and tot['F1 违例'] == 0 and tot['推导异常'] == 0 and tot['文件超时/崩溃'] == 0,
            f"实例 {tot['实例']}，违例 {tot['F1 违例']}，点未定 {tot['点未定']}，推导异常 {tot['推导异常']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('F2 区域增益为真', tot['F2 增益（残差为空）'] >= 50 and tot['F2 误收'] == 0,
            f"增益 {tot['增益']}（残差为空 {tot['F2 增益（残差为空）']}，MLIR 接受 {tot['F2 MLIR 接受']}，误收 {tot['F2 误收']}；残差非空 {tot['增益/残差非空']}，MLIR 不可判 {tot['增益/MLIR 不可判']}）")
    V.check('F4 一致替换元素类型的增益', tot['F4 增益'] >= 50 and tot['F4 误收'] == 0 and tot['F4 误拒'] == 0,
            f"实例 {tot['F4 实例']}，增益 {tot['F4 增益']}，增益误收 {tot['F4 误收']}，误拒 {tot['F4 误拒']}，Φ_新 真而 MLIR 拒 {tot['F4 误收(全部)']}，MLIR 不可判 {tot['F4 MLIR 不可判']}")
    print(f"F3 增益实例中旧实例化被 MLIR 拒绝 {tot['F3 旧实例化 MLIR 拒绝']}")
    V.finish()
