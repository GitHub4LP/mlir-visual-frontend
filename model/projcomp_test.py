"""第 63 轮：复合 op 单端口最大类型范围 R_j（odscomp.project）的外部检验。只取 Unknown 残差为空的复合 op（第 62 轮 S3：其上 Φ ≡ MLIR）。

R_j(t) ≝ ∃ 其余端口. Φ。project 把精确相等的端口并成等价类；“精确”⇔ 没有原子把该类与类外端口相连，此时其余端口可保持原类型 T0：
  R_j(t) ⇔ Φ(T0[类 j ← t])，这一等式可以直接交 MLIR 判定。
三值范围 range3（第 63 轮首跑后加入；首跑只有“精确/上近似”二分，精确端口仅 29/477，原因是 SameOperandsAndResultType 等是兼容而非相等）：
  in = 找到见证赋值（其余端口取 {原类型, t}）；out = 类内必要条件不成立；unknown = 其余。
阈值（P1/P2 首跑前写定；P4/P5 在首跑后、全量运行前写定）：
  P1 精确端口 ≡ MLIR：对标“精确”的端口与候选 t，R_j(t) 与 MLIR 对 T0[类 j ← t] 内联结果的判定一致，不一致 0，且比较数 > 0。
  P2 上近似端口可靠：对标“上近似”的端口，R_j(t) 为假时，在类型池上穷举其余端口（组合数 ≤ 3000）找不到使 Φ 成立的补全，违例 0；
     超过组合上限的计入“未穷举”，不算通过证据。
  P4 in 可靠：range3 = in 的见证赋值交 MLIR，必须接受，不一致 0。
  P5 out 可靠：range3 = out 时，类型池上穷举其余端口（≤ 3000 组合）找不到补全，违例 0。
  P3（只报告）：in/out/unknown 占比；：标“精确”的端口占比；端口等价类合并数。
候选 t：同文件出现过的类型随机取 10 个，加上 T0[j]。
用法：PYTHONPATH=model python3 model/projcomp_test.py [每函数切片数] [文件数上限] [起始文件]
"""
import sys, re, glob, random, itertools, collections, multiprocessing as mp
from compcorpus_test import slices, _walk, _funcs

def run_file(path, q, kslice, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odscomp as D, typesig as TS
    from composite import instantiate
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    def mlir_ok(sl, ports, T):
        bs = instantiate(sl, ports, T)
        if bs is None: return False
        text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
        if text is None: return None
        tr, _ = OR.get().verify(text)
        return None if tr in ('parse', 'crash') else tr == 'ok'
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: root = I.import_module(ir.Module.parse(ch, ir.Context()))
        except Exception: continue
        pool = sorted({t for f in _walk(root) for t in [o[1] for o in f['outs']] if len(t) < 60})
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']
            tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
            cand = list(slices(body)); rng.shuffle(cand)
            for sl in cand[:kslice]:
                defined = {o for n in sl for o, _ in n['outs']}; ports, seen = [], set()
                for n in sl:
                    for v in n['ins']:
                        if v not in defined and v not in seen: seen.add(v); ports.append(v)
                if not ports: continue
                try: P = D.derive(sl, ports); Q = D.simplify(P)
                except Exception as e: st['推导异常'] += 1; note('推导异常', str(e)[:80]); continue
                if P.residual: continue
                T0 = [tyof[v] for v in ports]
                if D.evaluate(Q, T0) is not True or mlir_ok(sl, ports, T0) is not True: st['原类型不被接受（跳过）'] += 1; continue
                try: cls, rg = D.project(Q)
                except Exception as e: st['投影异常'] += 1; note('投影异常', str(e)[:80]); continue
                st['复合 op'] += 1; st['端口'] += len(ports); st['等价类合并'] += len(ports) - len(set(cls))
                cands = sorted(set(rng.sample(pool, min(10, len(pool))) + T0))
                for j in range(len(ports)):
                    exact = rg[j][1] is True; st['端口/精确' if exact else '端口/上近似'] += 1   # 第 85 轮：P1 只对原义“精确”；同值精确由 projexact_test Y1/Y2 检验
                    for t in rng.sample(cands, min(6, len(cands))) + [T0[j]]:
                        try: tp = TS.parse(t)
                        except Exception: continue
                        r = D.range_holds(Q, cls, rg, j, tp)
                        if r is D.UNK: st['点未定（nf 未建模）'] += 1
                        r3, w = D.range3(Q, j, t, T0); st['range3/' + r3] += 1
                        if r3 == 'in':
                            m = mlir_ok(sl, ports, w)
                            if m is None: st['P4 MLIR 不可判'] += 1
                            else:
                                st['P4 比较'] += 1; st['P4 不一致'] += m is not True
                                if m is not True: note('P4 不一致', f'{[n["op"] for n in sl]} 端口{j} {w}')
                        if r3 == 'unknown':     # 诊断（非判据）：未定点是见证搜索不足还是必要条件太弱
                            others = sorted({cls[i] for i in range(len(ports))} - {cls[j]})
                            ops = '+'.join(sorted({n['op'] for n in sl}))
                            if r is D.UNK: st['U/点未定'] += 1; st['U/点未定/' + ops] += 1
                            elif len(cands) ** len(others) > 3000: st['U/未穷举'] += 1; st['U/未穷举/' + ops] += 1
                            else:
                                hit = None
                                for combo in itertools.product(cands, repeat=len(others)):
                                    asg = dict(zip(others, combo)) | {cls[j]: t}
                                    v = D.evaluate(Q, [asg[cls[i]] for i in range(len(ports))])
                                    if v is True: hit = True; break
                                    if v is D.UNK: hit = hit or 'unk'
                                k = '搜索不足' if hit is True else '池内无补全' if hit is None else '池内仅未定'
                                st['U/' + k] += 1; st[f'U/{k}/' + ops] += 1; note('U/' + k, f'{ops} 端口{j} {t} T0={T0}')
                        if r3 == 'out':
                            others = sorted({cls[i] for i in range(len(ports))} - {cls[j]})
                            if len(cands) ** len(others) > 3000: st['P5 未穷举'] += 1
                            else:
                                st['P5 检查'] += 1
                                for combo in itertools.product(cands, repeat=len(others)):
                                    asg = dict(zip(others, combo)) | {cls[j]: t}
                                    if D.evaluate(Q, [asg[cls[i]] for i in range(len(ports))]):
                                        st['P5 违例'] += 1; note('P5 违例', f'{[n["op"] for n in sl]} 端口{j} {t}'); break
                        if exact and r is not D.UNK:
                            T = [t if cls[i] == cls[j] else T0[i] for i in range(len(ports))]
                            m = mlir_ok(sl, ports, T)
                            if m is None: st['P1 MLIR 不可判'] += 1; continue
                            st['P1 比较'] += 1; st['P1 接受'] += r
                            if r != m: st['P1 不一致'] += 1; note('P1 不一致', f'{[n["op"] for n in sl]} 端口{j} {t} R={r} MLIR={m} T0={T0}')
                        elif r is False:
                            others = sorted({cls[i] for i in range(len(ports))} - {cls[j]})
                            if len(cands) ** len(others) > 3000: st['P2 未穷举'] += 1; continue
                            st['P2 拒绝检查'] += 1
                            for combo in itertools.product(cands, repeat=len(others)):
                                asg = dict(zip(others, combo)) | {cls[j]: t}
                                if D.evaluate(Q, [asg[cls[i]] for i in range(len(ports))]):
                                    st['P2 违例'] += 1; note('P2 违例', f'{[n["op"] for n in sl]} 端口{j} {t}'); break
                        else: st['P2 上近似接受（不检）'] += 1
                if len(ex['呈现']) < 6: note('呈现', f'{[n["op"] for n in sl]} {D.render_ranges(Q, ports)}')
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[start:lim], start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, kslice, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('P1 精确端口≡MLIR', tot['文件超时/崩溃'] == 0 and tot['P1 比较'] > 0 and tot['P1 不一致'] == 0 and tot['推导异常'] == 0 and tot['投影异常'] == 0,
            f"比较 {tot['P1 比较']}（接受 {tot['P1 接受']}），不一致 {tot['P1 不一致']}，异常 {tot['推导异常']}+{tot['投影异常']}，文件崩溃 {tot['文件超时/崩溃']}，点未定 {tot['点未定（nf 未建模）']}")
    V.check('P2 上近似可靠', tot['P2 违例'] == 0, f"拒绝检查 {tot['P2 拒绝检查']}，违例 {tot['P2 违例']}，未穷举 {tot['P2 未穷举']}")
    V.check('P4 in可靠', tot['P4 比较'] > 0 and tot['P4 不一致'] == 0, f"比较 {tot['P4 比较']}，不一致 {tot['P4 不一致']}，MLIR 不可判 {tot['P4 MLIR 不可判']}")
    V.check('P5 out可靠', tot['P5 检查'] > 0 and tot['P5 违例'] == 0, f"检查 {tot['P5 检查']}，违例 {tot['P5 违例']}，未穷举 {tot['P5 未穷举']}")
    print(f"P3 range3：in {tot['range3/in']}，out {tot['range3/out']}，unknown {tot['range3/unknown']}")
    print(f"P3 复合 op {tot['复合 op']}，端口 {tot['端口']}：精确 {tot['端口/精确']}，上近似 {tot['端口/上近似']}；等价类合并 {tot['等价类合并']}")
    V.finish()
