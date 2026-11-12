"""第 80 轮：C++ inferReturnTypes 用于复合 op 的正向推导（结果类型随端口变），closure §9.21。

只对含“等式语义”C++ 推断 op（nfrules 置 S['cppinfer']：推断完备且无 isCompatibleReturnTypes 覆盖）的切片，
比较正向推导关闭（这些 op 的结果取写定类型）与打开时的 Φ。切片：compcorpus_test.slices（无区域）与 regcomp_test.rslices（含区域）。
判据（运行前写定）：
  J1 单调：Φ_旧(T) 接受（为真且残差为空）⇒ Φ_新(T) 接受，违例 0；推导异常 0、文件崩溃 0。
  J2 增益真实：Φ_旧 不接受而 Φ_新 接受（残差为空）的实例，MLIR 对新实例化全部接受（误收 0）；这类实例合计 ≥ 20。
     变异：compcorpus_test.mutate 随机变异，加一致替换元素类型（f32→f64、f16→f32、i32→i64、i64→i32 之一同时替换全部端口）。
  J3 误拒 0：Φ_新 拒绝（不计点未定、残差为空）而 MLIR 接受新实例化的个数 = 0。
用法：PYTHONPATH=model python3 model/cppinfer_test.py [每函数切片数] [每切片变异数] [文件数上限] [起始文件]
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import mutate, _walk, _funcs, slices
from regcomp_test import rslices, free_values, types_of, _nested

def _has_cpp(sl):
    import odssig as G
    for n in sl:
        for m in [n] + list(_nested(n)):
            lk = G.lookup(m['op'])
            if lk and lk[1].get('cppinfer'): return True
    return False

def run_file(path, q, kslice, kmut, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odscomp as D, nfrules as NF
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
            seen = set(); cand = []
            for sl in list(slices(body)) + list(rslices(body)):
                k_ = tuple(id(n) for n in sl)
                if k_ in seen or not _has_cpp(sl): continue
                seen.add(k_); cand.append(sl)
            rng.shuffle(cand)
            for sl in cand[:kslice]:
                ports = free_values(sl)
                if not ports or any(v not in tyof for v in ports): continue
                try:
                    NF.CPPINFER_FWD = False; Po = D.derive(sl, ports)
                    NF.CPPINFER_FWD = True; Pn = D.derive(sl, ports)
                except Exception as e:
                    NF.CPPINFER_FWD = True; st['推导异常'] += 1; note('推导异常', f'{[n["op"] for n in sl]} {type(e).__name__}: {str(e)[:120]}'); continue
                st['复合 op'] += 1
                T0 = [tyof[v] for v in ports]
                Ts = [T0] + [mutate(rng, T0, pool) for _ in range(kmut)]
                for a_, b_ in (('f32', 'f64'), ('f16', 'f32'), ('i32', 'i64'), ('i64', 'i32')):
                    if any(re.search(r'\b' + a_ + r'\b', t) for t in T0): Ts.append([re.sub(r'\b' + a_ + r'\b', b_, t) for t in T0])
                for T in Ts:
                    o, nw = D.evaluate(Po, T), D.evaluate(Pn, T)
                    if o is None or nw is None: continue
                    st['实例'] += 1
                    if o is D.UNK or nw is D.UNK: st['点未定'] += 1; continue
                    acc_o = o is True and not Po.residual; acc_n = nw is True and not Pn.residual
                    if acc_o and not acc_n: st['J1 违例'] += 1; note('J1 违例', f'{[n["op"] for n in sl]} {T}')
                    if Pn.residual: st['残差非空'] += 1; continue
                    ft = '(' + ', '.join(T) + ') -> ()'
                    if not acc_o and acc_n:
                        okn = mlir_ok(instantiate(sl, ports, T), ft)
                        if okn is None: st['增益/MLIR 不可判'] += 1; continue
                        st['J2 增益'] += 1; st['J2 MLIR 接受'] += okn
                        if not okn: st['J2 误收'] += 1; note('J2 误收', f'{[n["op"] for n in sl]} {T}')
                        else: note('增益例', f'{[n["op"] for n in sl]} {T}')
                    elif nw is False:
                        okn = mlir_ok(instantiate(sl, ports, T), ft)
                        st['新拒绝'] += 1
                        if okn: st['J3 误拒'] += 1; note('J3 误拒', f'{[n["op"] for n in sl]} {T}')
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    kmut = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/Linalg/*.mlir'))
    for i, path in enumerate(files[start:lim], start):
        src = open(path).read()
        if not any(o in src for o in ('vector.shuffle', 'vector.to_elements', 'extract_strided_metadata', 'distinct_objects', 'bufferization.dealloc')): continue
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, kslice, kmut, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('J1 C++ 推断前推单调', tot['实例'] > 0 and tot['J1 违例'] == 0 and tot['推导异常'] == 0 and tot['文件超时/崩溃'] == 0,
            f"复合 op {tot['复合 op']}，实例 {tot['实例']}，违例 {tot['J1 违例']}，点未定 {tot['点未定']}，推导异常 {tot['推导异常']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('J2 C++ 推断增益为真', tot['J2 增益'] >= 20 and tot['J2 误收'] == 0,
            f"增益 {tot['J2 增益']}（MLIR 接受 {tot['J2 MLIR 接受']}，误收 {tot['J2 误收']}；MLIR 不可判 {tot['增益/MLIR 不可判']}）")
    V.check('J3 C++ 推断前推无误拒', tot['J3 误拒'] == 0, f"Φ_新 拒绝 {tot['新拒绝']}，其中 MLIR 接受 {tot['J3 误拒']}")
    V.finish()
