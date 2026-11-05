"""第 75 轮：含区域节点的复合 op 的显式签名 Φ_comp（closure §9.15）。

此前 odscomp_test/projcomp_test 的切片排除了所有带区域的 op（compcorpus_test.slices），Φ_comp 也不看区域内的 op。
本轮 odscomp.derive 把区域内的 op 并入 Φ_comp：区域内用到的外部值（捕获）作为端口；块参数与区域内结果取图上写定的类型；
子 op 读父 op 的 verify（linalg.yield / linalg.index）经父信息与父 op 上的 LinalgBody 原子规范化。
判据（运行前写定）：
  R1 Φ≡逐点（含区域）：切片至少含一个带区域的顶层 op；Φ(T) 与 composite.instantiate + core.typecheck（不开预言机）一致，不一致 0，推导异常 0，文件崩溃 0。
  R2 对MLIR（含区域）：实例 ≥ 1000，MLIR 接受 ≥ 200，误拒 0；残差为空（消去定值残差后）的误收 0；含 linalg 体（区域内有 linalg.yield）的实例 ≥ 30。
  R3（只报告）：复合 op 数、残差为空占比、残差来源前列。
用法：PYTHONPATH=model python3 model/regcomp_test.py [每函数切片数] [每切片变异数] [文件数上限] [起始文件]
语料：corpus/*/*.mlir 加 corpus_ext/Linalg。
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import mutate, _walk, _funcs

def _sym(n):
    return any(isinstance(v, str) and '@' in v for v in ((n.get('props') or {}).get('_attrs') or {}).values())
def _nested(n):
    for r in n.get('regions') or []:
        for b in r:
            for m in b['ops']: yield m; yield from _nested(m)
def rslices(body, kmax=3):
    ok = [n for n in body if not n.get('succs') and n['op'] not in ('func.return', 'func.call') and not _sym(n)
          and not any(_sym(m) or m['op'] in ('func.return', 'func.call') or m.get('succs') for m in _nested(n))]
    for i in range(len(ok)):
        for k in range(1, kmax + 1):
            sl = ok[i:i + k]
            if len(sl) < k: break
            idx = [body.index(n) for n in sl]
            if idx != list(range(idx[0], idx[0] + k)): break
            if any(n['regions'] for n in sl): yield sl
def free_values(sl):
    defined, ports, seen = set(), [], set()
    def use(v):
        if v not in defined and v not in seen: seen.add(v); ports.append(v)
    def walk(n):
        for v in n['ins']: use(v)
        for r in n.get('regions') or []:
            for b in r:
                for a, _ in b['args']: defined.add(a)
                for m in b['ops']:
                    walk(m)
                    for o, _ in m['outs']: defined.add(o)
    for n in sl:
        walk(n)
        for o, _ in n['outs']: defined.add(o)
    return ports
def types_of(fn):
    t = {}
    for x in _walk(fn):
        for r in x['regions']:
            for b in r:
                for a, ty in b['args']: t[a] = ty
        for o, ty in x['outs']: t[o] = ty
    return t

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
            body = fn['regions'][0][0]['ops']; tyof = types_of(fn)
            cand = list(rslices(body)); rng.shuffle(cand)
            for sl in cand[:kslice]:
                ports = free_values(sl)
                if not ports or any(v not in tyof for v in ports): continue
                lin = any(m['op'] == 'linalg.yield' for n in sl for m in _nested(n))
                try: P = D.derive(sl, ports); Q = D.simplify(P)
                except Exception as e: st['推导异常'] += 1; note('推导异常', f'{[n["op"] for n in sl]} {type(e).__name__}: {str(e)[:120]}'); continue
                unk = bool(P.residual)
                st['复合 op'] += 1; st['复合 op/残差为空'] += not unk; st['复合 op/含 linalg 体'] += lin
                for r in set(P.residual): st['残差/' + r[:50]] += 1
                T0 = [tyof[v] for v in ports]
                Ts = [('原类型', T0)] + [('变异', mutate(rng, T0, pool)) for _ in range(kmut)]
                for kind, T in Ts:
                    raw = D.evaluate(P, T)
                    if raw is None: st['类型文本不可解析'] += 1; continue
                    st['实例'] += 1; st['S1 不一致'] += D.evaluate(Q, T) != raw
                    bs = instantiate(sl, ports, T); ft = '(' + ', '.join(T) + ') -> ()'
                    if bs is None: pt = False
                    else:
                        C.ORACLE_ON = False
                        try: pt = not C.typecheck(bs, ft)
                        except Exception as e: st['core 异常'] += 1; note('core 异常', f'{type(e).__name__}: {str(e)[:100]}'); continue
                    if raw is D.UNK: st['点未定（nf 未建模）'] += 1
                    elif pt != raw: st['R1 不一致'] += 1; note('R1 不一致', f'{[n["op"] for n in sl]} {T} Φ={raw} 逐点={pt}')
                    if bs is None: continue
                    text = OR.emit_core(bs, ft)
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr in ('parse', 'crash'): st['MLIR ' + tr] += 1; continue
                    truth = tr == 'ok'
                    if kind == '原类型' and not truth: st['原类型独立后被拒（依赖上下文）'] += 1; note('原类型被拒', diags[0][1][:120] if diags else ''); break
                    st['R2 实例'] += 1; st['R2 MLIR 接受'] += truth; st['R2 含 linalg 体'] += lin
                    if truth and raw is False: st['R2 误拒'] += 1; note('R2 误拒', f'{[n["op"] for n in sl]} {T}')
                    if raw is D.UNK: st['R2 点未定'] += 1
                    if raw is True and not truth:
                        st['R2 误收/残差非空（Unknown）' if unk else 'R2 误收/残差为空'] += 1
                        if not unk: note('R2 误收/残差为空', f'{[n["op"] for n in sl]} {T} {diags[0][1][:120] if diags else ""}')
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
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items()) if not k.startswith('残差/')]
    print('── 残差来源前列'); [print(f'   {k}: {v}') for k, v in sorted(((k, v) for k, v in tot.items() if k.startswith('残差/')), key=lambda x: -x[1])[:15]]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('R1 Φ≡逐点（含区域）', tot['实例'] > 0 and tot['R1 不一致'] == 0 and tot['推导异常'] == 0 and tot['文件超时/崩溃'] == 0 and tot['S1 不一致'] == 0,
            f"实例 {tot['实例']}，不一致 {tot['R1 不一致']}，化简不一致 {tot['S1 不一致']}，推导异常 {tot['推导异常']}，文件崩溃 {tot['文件超时/崩溃']}，点未定 {tot['点未定（nf 未建模）']}")
    V.check('R2 对MLIR（含区域）', tot['R2 实例'] >= 1000 and tot['R2 MLIR 接受'] >= 200 and tot['R2 误拒'] == 0 and tot['R2 误收/残差为空'] == 0 and tot['R2 含 linalg 体'] >= 30,
            f"实例 {tot['R2 实例']}（接受 {tot['R2 MLIR 接受']}，含 linalg 体 {tot['R2 含 linalg 体']}），误拒 {tot['R2 误拒']}，误收：残差为空 {tot['R2 误收/残差为空']}，残差非空 {tot['R2 误收/残差非空（Unknown）']}；点未定 {tot['R2 点未定']}")
    print(f"R3 复合 op {tot['复合 op']}，残差为空 {tot['复合 op/残差为空']}，含 linalg 体 {tot['复合 op/含 linalg 体']}")
    V.finish()
