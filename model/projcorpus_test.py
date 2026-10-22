"""第 24 轮：L27 端口投影的外部检验（spec/closure.md §7）。复合 op 取自 MLIR 测试集的真实切片（与 compcorpus_test 同一抽取）。

前端动作：“把类型为 t 的线接到复合 op 的端口 p 上”。投影 composite.project 只回答 no（必然无补全）或 maybe。
  可靠性（硬判据）：project = no ⇒ 不存在其余端口的补全使 MLIR 接受。
     真值是 ∃ 量词，只能证伪：对每个 (p, t) 搜索补全——其余端口取 原类型 / t 本身 / 同文件类型池抽样，最多 K 个——
     交同一提交的 MLIR verifier。找到被接受的补全而投影说 no ⇒ 不可靠反例。
  能力：在“搜索找不到补全”的 (p, t) 上，投影拒绝了多少（不完备是允许的，L27；这里给出下界意义上的拒绝率）。
  （不再对照第 8 轮的 Φ_ts 投影：其节点检查器在真实切片上误拒 24%（第 22 轮），作为投影的承载层已被否定。）
  另测 project 在“其余端口已知（= 原类型）”时与完整判定（core.typecheck，不开预言机）的关系：前者是后者的必要条件。
用法：PYTHONPATH=model python3 model/projcorpus_test.py [每函数切片数] [每端口候选数] [补全搜索上限 K] [文件数上限]
"""
import sys, glob, random, collections, multiprocessing as mp
from compcorpus_test import slices, _walk, _funcs
from composite import instantiate, project

def run_file(path, q, kslice, kcand, K, seed):
    import mlir.ir as ir, irimport as I, core as C, oracle as OR, typesig as TS
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    cache = {}
    def truth(sl, ports, T):
        key = (id(sl), tuple(T))
        if key in cache: return cache[key]
        bs = instantiate(sl, ports, T); r = None
        if bs is not None:
            text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
            if text is not None:
                tr, _ = OR.get().verify(text)
                r = True if tr == 'ok' else False if tr == 'reject' else None
        else: r = False                                    # ODS 推断失败：该补全不可实例化
        cache[key] = r; return r
    for ch in I.split_chunks(src):
        if not ch.strip() or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        pool = sorted({t for f in _walk(root) for t in [o[1] for o in f['outs']] if len(t) < 60})
        if not pool: continue
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']
            tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
            cand = list(slices(body)); rng.shuffle(cand)
            for sl in cand[:kslice]:
                cache.clear()                              # 缓存按切片：id(sl) 会在切片被回收后复用（第一次全量运行的 6 个“反例”即此伪影）
                defined = {o for n in sl for o, _ in n['outs']}; ports, seen = [], set()
                for n in sl:
                    for v in n['ins']:
                        if v not in defined and v not in seen: seen.add(v); ports.append(v)
                if not ports: continue
                T0 = [tyof[v] for v in ports]
                if truth(sl, ports, T0) is not True: st['切片/原类型独立后不被接受（跳过）'] += 1; continue
                st['切片'] += 1; st['切片/多端口'] += len(ports) > 1
                for j, p in enumerate(ports):
                    cands = [T0[j]] + rng.sample(pool, min(kcand, len(pool)))
                    for t in dict.fromkeys(cands):
                        orig = t == T0[j]
                        v = project(sl, ports, p, t)[0]
                        # 补全搜索：其余端口 ∈ {原类型, t, 池抽样}
                        found = None; tried = 0
                        opts = [T0] + [[t if k != j and rng.random() < 0.5 else T0[k] for k in range(len(ports))]] + \
                               [[rng.choice(pool) if k != j else t for k in range(len(ports))] for _ in range(K)] + \
                               [[t] * len(ports)]
                        for T in opts:
                            T = list(T); T[j] = t
                            if tried >= K: break
                            tried += 1; r = truth(sl, ports, T)
                            if r is True: found = T; break
                        kind = '原类型' if orig else '候选'
                        st[f'{kind}/(端口,类型)'] += 1; st[f'{kind}/投影 no'] += v == 'no'
                        st[f'{kind}/找到补全'] += found is not None
                        if v == 'no' and found is not None:
                            st[f'{kind}/不可靠反例（no 但有补全）'] += 1; note('不可靠反例', f'{[n["op"] for n in sl]} 端口{j}={t} 补全 {found} {project(sl, ports, p, t)}')
                        if found is None: st[f'{kind}/未找到补全'] += 1; st[f'{kind}/未找到补全且投影 no'] += v == 'no'
                        # 其余端口固定为原类型：投影必须是完整判定的必要条件
                        if len(ports) > 1:
                            others = {q_: T0[k] for k, q_ in enumerate(ports) if k != j}
                            vf = project(sl, ports, p, t, others)[0]
                            T = list(T0); T[j] = t; bs = instantiate(sl, ports, T)
                            if bs is not None:
                                C.ORACLE_ON = False
                                try: d = C.typecheck(bs, '(' + ', '.join(T) + ') -> ()')
                                except Exception: d = None
                                if d is not None:
                                    st['固定其余/实例'] += 1; st['固定其余/投影 no'] += vf == 'no'; st['固定其余/Φ_ods 拒绝'] += bool(d)
                                    st['固定其余/投影 no 而 Φ_ods 接受（应为 0）'] += vf == 'no' and not d
                                    r = truth(sl, ports, T)
                                    if r is not None: st['固定其余/投影 no 而 MLIR 接受（应为 0）'] += vf == 'no' and r
    q.put((st, dict(ex)))

if __name__ == '__main__':
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    kcand = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    lim = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
        q = ctxm.Queue(); pr = ctxm.Process(target=run_file, args=(path, q, kslice, kcand, K, i)); pr.start()
        try: st, ex = q.get(timeout=600)
        except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; print('  ! 超时/崩溃', path, flush=True); continue
        pr.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
