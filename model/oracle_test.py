"""预言机层与符号层的分工检验（第 10 轮）。
  O1 判定 ≡ MLIR：随机直线图（11 个方言里可参与差分的 op，2–5 个节点），core.typecheck（符号层 + 预言机）与独立 fork 的 MLIR verifier 一致；
     只开符号层时：误拒必须为 0（L29 在图上成立），误收即预言机补上的部分
  O2 分工：多少图只靠符号层就有结论、多少要问预言机；预言机每次的耗时；缓存命中
  O3 诊断归位：单点改坏的图里，预言机诊断落在被改的那个节点上的比例
  O4 崩溃隔离：MLIR 断言崩溃的实例直接交给预言机，子进程重启，之后的查询结果不受影响
用法：PYTHONPATH=model python3 model/oracle_test.py [图数]
"""
import random, sys, time, collections
import core as C, odssig as G, odsjson as O, oracle as OR, typesig as TS
from structure import node, blk
from typesig_test import mlir_ok, CRASH
import odssig_test as OT

def ops_pool():
    out = []
    for dia in O.DIALECTS:
        for op in O.ops(dia):
            cls, S = G.ledger(dia, op)
            if OT.eligible(S): out.append((op, cls, S))
    return out

def graph(rng, POOL, OPS, k, good_bias=True):
    """直线图：每个节点按 odssig_test.draw 取端口类型；操作数优先复用已有的同类型值，否则新开函数参数。"""
    args, vals, ops, seq = [], [], [], []
    for i in range(k):
        op, cls, S = rng.choice(OPS)
        ins, outs = OT.draw(rng, S, POOL, [], 0 if good_bias and rng.random() < 0.95 else 1)
        flat = []
        for g in ins:
            for t in g:
                same = [v for v, tt in vals if tt == t]
                if same and rng.random() < 0.6: flat.append(rng.choice(same))
                else: a = f'a{len(args)}'; args.append((a, TS.show(t))); vals.append((a, t)); flat.append(a)
        res = []
        for g in outs:
            for t in g: v = f'n{i}_{len(res)}'; res.append((v, TS.show(t))); vals.append((v, t))
        props = {nm: 0 for nm, _, kind, req in S['attrs'] if req and kind == 'I64EnumAttr'}
        nd = node(f'n{i}', op, ins=flat, outs=res, props={'predicate': 0} if props else {})
        if any('AttrSizedOperandSegments' in t for t in S['structural']): nd['seg'] = [len(g) for g in ins]
        if any('AttrSizedResultSegments' in t for t in S['structural']): continue
        ops.append(nd); seq.append((op, cls))
    ops.append(node('ret', 'func.return'))
    return [blk('entry', ops, args=args)], seq

def O1(n, rng):
    import mlir.ir as ir
    ctx = ir.Context(); POOL = OT.pool(ctx); OPS = ops_pool()
    st = collections.Counter(); lat = []
    for _ in range(n):
        bs, seq = graph(rng, POOL, OPS, rng.choice([1, 2, 3, 4]))
        text = OR.emit_core(bs, '() -> ()')
        if text is None: st['不可发射'] += 1; continue
        nc = len(CRASH); truth = mlir_ok(ctx, text)
        if len(CRASH) > nc: st['MLIR 崩溃'] += 1
        C.ORACLE_ON = False; sym = not C.typecheck(bs, '() -> ()')
        C.ORACLE_ON = True; q0 = OR.get().stats['查询']; t0 = time.perf_counter()
        full = not C.typecheck(bs, '() -> ()')
        if OR.get().stats['查询'] > q0 and q0 > 0: lat.append(time.perf_counter() - t0)
        st['图'] += 1; st['MLIR 接受'] += truth
        st['符号+预言机 ' + ('一致' if full == truth else '误收' if full else '误拒')] += 1
        st['只符号层 ' + ('一致' if sym == truth else '误收' if sym else '误拒')] += 1
        st['全部 auto（无需预言机）'] += all(c == 'auto' for _, c in seq)
    st['预言机调用'] = OR.get().stats['查询']; lat.sort(); st['typecheck 含预言机，中位 µs'] = round(lat[len(lat) // 2] * 1e6) if lat else None
    st['发射缺陷'] = len(C.TC_STATS.get('发射缺陷', []))
    return st, POOL, OPS

def O3(n, rng, POOL, OPS):
    """符号层通过、MLIR 拒绝的图：预言机的诊断是否落到节点上，是否落在 declared/unknown 节点上（符号层管不到的那些）。"""
    st = collections.Counter(); dec = [x for x in OPS if x[1] != 'auto']
    while st['样本'] < n and st['尝试'] < 200 * n:
        st['尝试'] += 1
        bs, seq = graph(rng, POOL, dec + OPS, rng.choice([1, 2, 3]))
        C.ORACLE_ON = False; sym = C.typecheck(bs, '() -> ()'); C.ORACLE_ON = True
        if sym: continue
        d = C.typecheck(bs, '() -> ()')
        if not d: continue
        st['样本'] += 1
        ids = {nd['id']: nd for nd in bs[0]['ops']}
        st['诊断落在节点上'] += all(x in ids for x, _ in d)
        st['落在 declared/unknown 节点'] += all(x in ids and G.lookup(ids[x]['op']) and G.lookup(ids[x]['op'])[0] != 'auto' for x, _ in d)
        if st['样本'] <= 3: st[f'例{st["样本"]}'] = d[0][1][:90]
    return st

def O4():
    o = OR.Oracle()
    bad = 'func.func @f(%a: tensor<*xi32>) {\n  %r = "arith.trunci"(%a) : (tensor<*xi32>) -> (tensor<*xi8>) loc("n0")\n  return\n}'
    good = 'func.func @f(%a: i32) {\n  %r = "arith.trunci"(%a) : (i32) -> (i8) loc("n0")\n  return\n}'
    wrong = 'func.func @f(%a: i8) {\n  %r = "arith.trunci"(%a) : (i8) -> (i32) loc("n0")\n  return\n}'
    r = [o.verify(good)[0], o.verify(bad)[0], o.verify(good + ' ')[0], o.verify(wrong)]
    bs = [blk('entry', [node('n0', 'arith.trunci', ins=['a'], outs=[('r', 'tensor<*xi8>')]), node('ret', 'func.return')], args=[('a', 'tensor<*xi32>')])]
    return {'好→崩溃→好': r[:3], '重启': o.stats['重启'], '崩溃': o.stats['崩溃'], '崩溃后的拒绝诊断': r[3], '符号层先拒（不交给预言机）': C.typecheck(bs, '() -> ()')}

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(10)
    st, POOL, OPS = O1(n, rng)
    print('── O1/O2 判定与分工:', dict(st)); print('    缓存:', OR.get().stats)
    a, b = [], []
    for k in range(200):
        bs, _ = graph(rng, POOL, [x for x in OPS if x[1] != 'auto'], 3)
        t0 = time.perf_counter(); C.typecheck(bs, '() -> ()'); t1 = time.perf_counter(); C.typecheck(bs, '() -> ()'); t2 = time.perf_counter()
        a.append(t1 - t0); b.append(t2 - t1)
    a.sort(); b.sort(); print(f'    同一体再查一次（中位）：{a[100] * 1e6:.0f} µs → {b[100] * 1e6:.0f} µs（缓存命中）')
    print('── O3 诊断归位:', dict(O3(max(n // 5, 20), rng, POOL, OPS)))
    print('── O4 崩溃隔离:', O4())
    print('── core 计数:', {k: v for k, v in C.TC_STATS.items() if not k.endswith('例') and k != '发射缺陷'}); print('    人工签名误拒例:', C.TC_STATS.get('人工签名误拒例', [])[:3])
