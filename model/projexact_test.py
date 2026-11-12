"""第 85 轮加 Y1–Y3（closure §9.25，同值见证精确）：Y3 = COPY_EXACT 关/开（A、B 均开）占比下降 ≥ 5 个百分点；
Y1/Y2 在开启状态下对 MLIR 检验（同 projcomp 的 instantiate + 预言机；MLIR 不可判另报）。
第 84 轮（closure §9.24）：X3 上近似端口类占比下降（增益为真）。
口径与盘点相同（work/r84/inv.py）：corpus 全部文件（排除 allow-unregistered-dialect），每函数随机 4 个切片（种子 84），
对每个复合 op 求 project，统计端口等价类中“上近似”的比例；新机制（ATTR_SPLIT、FIX_SINGLETON）关 / 开各算一次。
阈值（实现前写定）：X3 开启后占比比关闭时下降 ≥ 10 个百分点。X2（精确范围 ≡ MLIR）由 projcomp_test P1 承担，比较数 ≥ 1000。
用法：PYTHONPATH=model python3 model/projexact_test.py [文件数上限] [起始文件] [--y]
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import slices, _funcs

def run_file(path, q, on, copy=False):
    import mlir.ir as ir, irimport as I, odscomp as D
    D.ATTR_SPLIT = D.FIX_SINGLETON = on; D.COPY_EXACT = copy
    rng = random.Random(84); st = collections.Counter()
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put(st); return
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']; cand = list(slices(body)); rng.shuffle(cand)
            for sl in cand[:4]:
                defined = {o for n in sl for o, _ in n['outs']}; ports, seen = [], set()
                for n in sl:
                    for v in n['ins']:
                        if v not in defined and v not in seen: seen.add(v); ports.append(v)
                if not ports: continue
                try: P = D.derive(sl, ports); Q = D.simplify(P); cls, rg = D.project(Q)
                except Exception: st['推导异常'] += 1; continue
                st['复合 op'] += 1
                for c in set(cls): st['端口类'] += 1; st['端口类/上近似'] += not rg[c][1]
    q.put(st)

def run_y(path, q, kslice=4):
    import mlir.ir as ir, irimport as I, odscomp as D, oracle as OR, typesig as TS, itertools
    from composite import instantiate
    from compcorpus_test import _walk
    rng = random.Random(85); st = collections.Counter(); ex = collections.defaultdict(list)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, ex)); return
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
                try: Q = D.simplify(D.derive(sl, ports)); cls, rg = D.project(Q)
                except Exception: st['推导异常'] += 1; continue
                T0 = [tyof[v] for v in ports]
                if D.evaluate(Q, T0) is not True: continue
                cands = sorted(set(rng.sample(pool, min(10, len(pool))) + T0))
                for c in sorted(set(cls)):
                    if rg[c][1] != 'copy': continue
                    st['同值精确类'] += 1; K = Q._copyk[c]
                    for t in rng.sample(cands, min(6, len(cands))) + [T0[c]]:
                        try: tp = TS.parse(t)
                        except Exception: continue
                        r = D.range_holds(Q, cls, rg, c, tp)
                        if r is D.UNK: st['点未定'] += 1; continue
                        if r:
                            T = [t if cls[i] in K else T0[i] for i in range(len(ports))]
                            m = mlir_ok(sl, ports, T)
                            if m is None: st['Y1 MLIR 不可判'] += 1; continue
                            st['Y1 比较'] += 1
                            if m is not True:
                                st['Y1 不一致'] += 1
                                if len(ex['Y1']) < 4: ex['Y1'].append(f'{[n["op"] for n in sl]} 类{c} {T}')
                        else:
                            others = sorted(set(cls) - {c})
                            if len(cands) ** len(others) > 3000: st['Y2 未穷举'] += 1; continue
                            st['Y2 检查'] += 1
                            for combo in itertools.product(cands, repeat=len(others)):
                                asg = dict(zip(others, combo)) | {c: t}
                                if D.evaluate(Q, [asg[cls[i]] for i in range(len(ports))]) is True:
                                    st['Y2 违例'] += 1
                                    if len(ex['Y2']) < 4: ex['Y2'].append(f'{[n["op"] for n in sl]} 类{c} {t}')
                                    break
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    av = [a for a in sys.argv[1:] if not a.startswith('--')]; onlyy = '--only-y' in sys.argv
    lim = int(av[0]) if len(av) > 0 else 10 ** 9
    start = int(av[1]) if len(av) > 1 else 0
    ctxm = mp.get_context('fork'); res = {}
    files = sorted(glob.glob('corpus/**/*.mlir', recursive=True))[start:lim]
    for key, on, cp in (() if onlyy else (('关', False, False), ('开', True, False), ('同值开', True, True))):
        tot = collections.Counter()
        for path in files:
            q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, on, cp)); p.start()
            try: st = q.get(timeout=600)
            except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
            p.join(); tot.update(st)
        res[key] = tot; print(key, dict(tot))
    ytot = collections.Counter(); yex = collections.defaultdict(list)
    if '--y' in sys.argv or onlyy:
        for path in files:
            q = ctxm.Queue(); p = ctxm.Process(target=run_y, args=(path, q)); p.start()
            try: st, ex = q.get(timeout=600)
            except Exception: p.kill(); ytot['文件超时/崩溃'] += 1; continue
            p.join(); ytot.update(st)
            for k, v in ex.items(): yex[k] += v[:2]
        print('Y', dict(ytot)); [print('──', k, v[:6]) for k, v in yex.items()]
    rr = lambda k: res[k]['端口类/上近似'] / max(1, res[k]['端口类'])
    sh = lambda k: f"{res[k]['端口类/上近似']}/{res[k]['端口类']} = {rr(k)*100:.1f}%"
    ok = bool(res) and all(res[k]['推导异常'] == res['关']['推导异常'] and res[k]['文件超时/崩溃'] == 0 and res[k]['端口类'] == res['关']['端口类'] for k in res)
    if not onlyy: V.check('X3 上近似下降', ok and (rr('关') - rr('开')) * 100 >= 10, f"关 {sh('关')}，开 {sh('开')}，下降 {(rr('关')-rr('开'))*100:.1f} 个百分点")
    if not onlyy: V.check('Y3 上近似下降', ok and (rr('开') - rr('同值开')) * 100 >= 5, f"同值关 {sh('开')}，同值开 {sh('同值开')}，下降 {(rr('开')-rr('同值开'))*100:.1f} 个百分点")
    if '--y' in sys.argv or onlyy:
        V.check('Y1 同值见证可靠', ytot['文件超时/崩溃'] == 0 and ytot['Y1 比较'] >= 1000 and ytot['Y1 不一致'] == 0, f"比较 {ytot['Y1 比较']}，不一致 {ytot['Y1 不一致']}，MLIR 不可判 {ytot['Y1 MLIR 不可判']}，同值精确类 {ytot['同值精确类']}，点未定 {ytot['点未定']}，推导异常 {ytot['推导异常']}")
        V.check('Y2 拒绝可靠', ytot['Y2 违例'] == 0 and ytot['Y2 检查'] > 0, f"检查 {ytot['Y2 检查']}，违例 {ytot['Y2 违例']}，未穷举 {ytot['Y2 未穷举']}")
    V.finish()
