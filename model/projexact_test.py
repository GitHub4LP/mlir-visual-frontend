"""第 84 轮（closure §9.24）：X3 上近似端口类占比下降（增益为真）。
口径与盘点相同（work/r84/inv.py）：corpus 全部文件（排除 allow-unregistered-dialect），每函数随机 4 个切片（种子 84），
对每个复合 op 求 project，统计端口等价类中“上近似”的比例；新机制（ATTR_SPLIT、FIX_SINGLETON）关 / 开各算一次。
阈值（实现前写定）：X3 开启后占比比关闭时下降 ≥ 10 个百分点。X2（精确范围 ≡ MLIR）由 projcomp_test P1 承担，比较数 ≥ 1000。
用法：PYTHONPATH=model python3 model/projexact_test.py [文件数上限] [起始文件]
"""
import sys, re, glob, random, collections, multiprocessing as mp
from compcorpus_test import slices, _funcs

def run_file(path, q, on):
    import mlir.ir as ir, irimport as I, odscomp as D
    D.ATTR_SPLIT = D.FIX_SINGLETON = on
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

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ctxm = mp.get_context('fork'); res = {}
    for on in (False, True):
        tot = collections.Counter()
        for path in sorted(glob.glob('corpus/**/*.mlir', recursive=True))[start:lim]:
            q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, on)); p.start()
            try: st = q.get(timeout=600)
            except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
            p.join(); tot.update(st)
        res[on] = tot; print('开' if on else '关', dict(tot))
    r0 = res[False]['端口类/上近似'] / max(1, res[False]['端口类']); r1 = res[True]['端口类/上近似'] / max(1, res[True]['端口类'])
    ok = res[True]['推导异常'] == res[False]['推导异常'] and res[True]['文件超时/崩溃'] == 0 and res[True]['端口类'] == res[False]['端口类']
    V.check('X3 上近似下降', ok and (r0 - r1) * 100 >= 10, f"关 {res[False]['端口类/上近似']}/{res[False]['端口类']} = {r0*100:.1f}%，开 {res[True]['端口类/上近似']}/{res[True]['端口类']} = {r1*100:.1f}%，下降 {(r0-r1)*100:.1f} 个百分点")
    V.finish()
