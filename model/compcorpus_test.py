"""第 22 轮：组合封闭的外部检验——复合 op 从 MLIR 自己的测试集里抽取（不是我们的生成器）。

做法：对语料中 MLIR 接受的每个 func.func 的入口块，取连续的 1–4 个 op 作为“提取为复合 op”的实现体（edit 的 extract）：
  端口 = 体外定义、体内使用的值（按首次使用排序）；ODS 能推断结果类型的节点，结果写成推断（随端口类型变化），其余节点的结果是字面量。
  实例 T：端口类型的一个赋值。T0 = 原类型；另做若干变异（端口类型换成同文件出现过的类型 / 换元素类型），内部类型按 ODS 推断规则前推。
真值：把复合 op 在 T 上内联成独立的 func，交同一提交的 MLIR verifier（隔离子进程）。
被测：
  Φ_ods   自动签名 + 属性层（core.typecheck，不开预言机）——第 9 轮以来的有拒绝权的符号层，19 方言
  Φ_full  Φ_ods + 隔离单元预言机（core.typecheck，开预言机）——透明模式的判定，应 ≡ 真值
  Φ_ts    第 8 轮 compsig 的节点检查器（typesig 人工签名，23 个 op）——当前 L25–L27 的承载层
判据（L46，见 spec/closure.md）：Φ_ods 不误拒；全 auto 的复合 op 上 Φ_ods ≡ 真值（auto 精确在组合下封闭）；Φ_full ≡ 真值。
K3（IRDL，不透明身份）见 irdl_export.py。
用法：PYTHONPATH=model python3 model/compcorpus_test.py [每函数最多切片数] [每切片变异数] [文件数上限]
"""
import os, sys, re, glob, random, collections, multiprocessing as mp

def slices(body_ops, kmax=4):
    ops = [n for n in body_ops if not n['regions'] and not n.get('succs') and n['op'] not in ('func.return', 'func.call')
           and not any(isinstance(v, str) and '@' in v for v in (n['props'].get('_attrs') or {}).values())]
    for i in range(len(ops)):
        for k in range(1, kmax + 1):
            sl = ops[i:i + k]
            if len(sl) < k: break
            # 必须是体内的连续片段（中间没有被跳过的 op），否则提取会改变顺序
            idx = [body_ops.index(n) for n in sl]
            if idx != list(range(idx[0], idx[0] + k)): break
            yield sl

from composite import infer_types, instantiate

def mutate(rng, T, pool):
    T = list(T); j = rng.randrange(len(T)); r = rng.random()
    if r < 0.5 and pool: T[j] = rng.choice(pool)
    else:
        swaps = [('i32', 'i64'), ('i64', 'i32'), ('f32', 'f64'), ('f64', 'f32'), ('i1', 'i8'), ('index', 'i64'), ('i8', 'i16'), ('f16', 'f32')]
        a, b = rng.choice(swaps)
        T[j] = re.sub(rf'\b{a}\b', b, T[j]) if re.search(rf'\b{a}\b', T[j]) else rng.choice(pool or T)
    return T

def run_file(path, q, kslice, kmut, seed):
    import mlir.ir as ir, irimport as I, core as C, oracle as OR, typesig as TS, odssig as G
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read(); unreg = 'allow-unregistered-dialect' in src
    if unreg: q.put((st, dict(ex))); return                    # 未注册 op 的义务 MLIR 不查，不能作真值
    tsops = set(TS.load())
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        ctx = ir.Context()
        try: m = ir.Module.parse(ch, ctx)
        except Exception: continue
        try: root = I.import_module(m)
        except Exception: continue
        pool = sorted({t for f in _walk(root) for t in [o[1] for o in f['outs']] if len(t) < 60})
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']
            cand = list(slices(body)); rng.shuffle(cand)
            for sl in cand[:kslice]:
                defined = {o for n in sl for o, _ in n['outs']}; ports, seen = [], set()
                tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
                for n in sl:
                    for v in n['ins']:
                        if v not in defined and v not in seen: seen.add(v); ports.append(v)
                T0 = [tyof[v] for v in ports]
                kinds = [(G.lookup(n['op']) or ('无签名',))[0] for n in sl]
                auto = all(k == 'auto' for k in kinds)
                in_ts = all(n['op'] in tsops for n in sl)
                st['切片'] += 1; st['切片/全 auto'] += auto; st['切片/Φ_ts 可表示（23 op）'] += in_ts
                st['切片/含可推断节点'] += any(infer_types(n, [TS.parse(tyof[v]) for v in n['ins']]) not in (None, 'fail') for n in sl if _parses(TS, [tyof[v] for v in n['ins']]))
                Ts = [('原类型', T0)] + [('变异', mutate(rng, T0, pool)) for _ in range(kmut)] if ports else [('原类型', T0)]
                for kind, T in Ts:
                    bs = instantiate(sl, ports, T)
                    if bs is None: st[f'{kind}/推断失败（不可实例化）'] += 1; continue
                    ft = '(' + ', '.join(T) + ') -> ()'
                    text = OR.emit_core(bs, ft)
                    if text is None: st[f'{kind}/不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse':
                        st[f'{kind}/解析失败（类型文本）'] += 1; note('解析失败', diags[0][1][:100] if diags else ''); continue
                    if tr == 'crash': st[f'{kind}/MLIR 崩溃'] += 1; continue
                    truth = tr == 'ok'
                    if kind == '原类型' and not truth:
                        st['原类型/独立后被拒（依赖上下文）'] += 1; note('上下文依赖', f'{[n["op"] for n in sl]} {diags[0][1][:90] if diags else ""}'); break
                    C.ORACLE_ON = False
                    try: d_ods = C.typecheck(bs, ft)
                    except Exception as e: st[f'{kind}/core 异常'] += 1; note('core 异常', f'{type(e).__name__}: {str(e)[:90]}'); continue
                    C.ORACLE_ON = True; d_full = C.typecheck(bs, ft)
                    st[f'{kind}/实例'] += 1; st[f'{kind}/MLIR 接受'] += truth
                    po, pf = not d_ods, not d_full
                    st[f'{kind}/Φ_ods 误拒'] += truth and not po; st[f'{kind}/Φ_ods 漏判'] += (not truth) and po
                    if auto: st[f'{kind}/全auto 实例'] += 1; st[f'{kind}/全auto Φ_ods≠MLIR'] += po != truth
                    st[f'{kind}/Φ_full≠MLIR'] += pf != truth
                    if truth and not po: note('Φ_ods 误拒', f'{[n["op"] for n in sl]} {T} {d_ods[0][1][:100]}')
                    if auto and po != truth: note('全auto 分歧', f'{[n["op"] for n in sl]} {T} MLIR={truth} {diags[0][1][:100] if diags else ""}')
                    if pf != truth: note('Φ_full 分歧', f'{[n["op"] for n in sl]} {T} MLIR={truth}')
                    # K3 不透明身份：导出为 IRDL op
                    import irdl_export as X
                    nmx = f'c{abs(hash((path, id(sl)))) % 10**8}'
                    dtext, gaps = X.export(nmx, sl, ports)
                    R = [t for _, t in bs[0]['ops'][-2]['outs']]
                    xv, xm = X.get().verify(dtext, X.instance(nmx, T, R))
                    st[f'{kind}/IRDL {xv}'] += 1
                    if xv in ('ok', 'reject'):
                        xa = xv == 'ok'
                        st[f'{kind}/IRDL 误拒'] += truth and not xa; st[f'{kind}/IRDL 漏判'] += (not truth) and xa
                        st[f'{kind}/IRDL ≡ MLIR'] += xa == truth
                        st[f'{kind}/IRDL 无缺口导出'] += not gaps
                        st[f'{kind}/IRDL 无缺口导出且≢'] += (not gaps) and xa != truth
                        if truth and not xa: note('IRDL 误拒', f'{[n["op"] for n in sl]} {T} {xm[:150]}')
                        if (not truth) and xa:
                            for g in sorted(set(gaps)): st['IRDL 漏判归因/' + re.sub(r'^[\w.]+: ', '', g)[:40]] += 1
                    else: note(f'IRDL {xv}', f'{[n["op"] for n in sl]} {xm[:150]}\n{dtext}')
                    if in_ts:
                        tsbad = _ts_check(TS, bs)
                        st[f'{kind}/Φ_ts 实例'] += 1
                        st[f'{kind}/Φ_ts 误拒'] += truth and bool(tsbad); st[f'{kind}/Φ_ts 漏判'] += (not truth) and not tsbad
                        if truth and tsbad: note('Φ_ts 误拒', f'{[n["op"] for n in sl]} {T} {tsbad[0]}')
    q.put((st, dict(ex)))

def _parses(TS, ts):
    try: [TS.parse(t) for t in ts]; return True
    except Exception: return False

def _ts_check(TS, bs):
    tys = dict(bs[0]['args']); bad = []
    for n in bs[0]['ops']:
        for o, t in n['outs']: tys[o] = t
        if n['op'] == 'func.return': continue
        ins = [TS.parse(tys[v]) for v in n['ins']]; outs = [TS.parse(t) for _, t in n['outs']]
        seg = None
        a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
        if a:
            import irimport as I; seg = I._segs(a) or None
        try: bad += TS.check(n['op'], ins, outs, TS.LEVELS, seg)
        except Exception as e: bad.append(('异常', str(e)[:60]))
    return bad

def _walk(n):
    yield n
    for r in n['regions']:
        for b in r:
            for o in b['ops']: yield from _walk(o)
def _funcs(n):
    for x in _walk(n):
        if x['op'] == 'func.func' and x['regions'] and x['regions'][0]: yield x

if __name__ == '__main__':
    kslice = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    kmut = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 10 ** 9
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = sorted(glob.glob('corpus/*/*.mlir'))[:lim]
    for i, path in enumerate(files):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, kslice, kmut, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
