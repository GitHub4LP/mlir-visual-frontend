"""预言机层与符号层的分工检验（第 10 轮）。
  O1 判定 ≡ MLIR：随机直线图（11 个方言里可参与差分的 op，2–5 个节点），core.typecheck（符号层 + 预言机）与独立 fork 的 MLIR verifier 一致；
     只开符号层时：误拒必须为 0（L29 在图上成立），误收即预言机补上的部分
  O2 分工：多少图只靠符号层就有结论、多少要问预言机；预言机每次的耗时；缓存命中
  O3 诊断归位：单点改坏的图里，预言机诊断落在被改的那个节点上的比例
  O4 崩溃隔离：MLIR 断言崩溃的实例直接交给预言机，子进程重启，之后的查询结果不受影响
  O5 区域、后继、多块（第 11 轮）：随机 scf.if / scf.for / cf.cond_br 图，带类型扰动。
     判定 = 结构层（structure.py）∧ 类型层（符号 + 预言机），与独立 fork 的 MLIR 比较；并统计不问预言机时各层漏掉什么
用法：PYTHONPATH=model python3 model/oracle_test.py [图数]
"""
import random, sys, time, collections, re
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

def rgraph(rng, depth=2, pmut=0.12):
    """随机结构化图。值都在 TY 中取类型；yield/后继实参以 pmut 的概率取错类型。"""
    TY = ['i8', 'i32', 'index', 'f32', 'tensor<4xi8>']
    k = [0]
    def nid(p): k[0] += 1; return f'{p}{k[0]}'
    def pick(vals, t, mut=True):
        same = [v for v, tt in vals if tt == t]
        if (mut and rng.random() < pmut) or not same: return rng.choice(vals)[0]
        return rng.choice(same)
    def seq(vals, d, n, term):
        ops, vals = [], list(vals)
        for _ in range(n):
            r = rng.random()
            if r < 0.45 or d == 0:
                ints = [(v, t) for v, t in vals if t in ('i8', 'i32', 'index')]
                if not ints: continue
                a, t = rng.choice(ints); b = pick(vals, t); o = nid('v')
                ops.append(node(o, rng.choice(['arith.addi', 'arith.muli', 'arith.xori']), ins=[a, b], outs=[(o, t)])); vals.append((o, t))
            elif r < 0.75:
                cs = [v for v, t in vals if t == 'i1']
                if not cs:
                    ints = [(v, t) for v, t in vals if t in ('i8', 'i32', 'index')]
                    if not ints: continue
                    a, t = rng.choice(ints); c = nid('c')
                    ops.append(node(c, 'arith.cmpi', ins=[a, pick(vals, t)], outs=[(c, 'i1')], props={'predicate': 2})); vals.append((c, 'i1')); cs = [c]
                rt = [rng.choice(TY) for _ in range(rng.choice([0, 1, 2]))]; o = nid('if')
                regs = []
                for br in 'te':
                    bops, _ = seq(vals, d - 1, rng.choice([0, 1, 2]), None)
                    inner = vals + [(x, t) for b in bops for x, t in b['outs']]
                    bops.append(node(nid('y'), 'scf.yield', ins=[pick(inner, t) for t in rt]))
                    regs.append([blk(nid(br), bops)])
                outs = [(f'{o}_{i}', t) for i, t in enumerate(rt)]
                cond = rng.choice(cs) if rng.random() > pmut else rng.choice(vals)[0]
                ops.append(node(o, 'scf.if', ins=[cond], outs=outs, regions=regs)); vals += outs
            else:
                idx = [v for v, t in vals if t == 'index']
                if not idx: continue
                its = [rng.choice(vals) for _ in range(rng.choice([0, 1, 2]))]; o = nid('for')
                iv = nid('iv'); bargs = [(iv, 'index')] + [(nid('it'), t if rng.random() > pmut / 2 else rng.choice(TY)) for _, t in its]
                bops, _ = seq(vals + bargs, d - 1, rng.choice([0, 1, 2]), None)
                inner = vals + bargs + [(x, t) for b in bops for x, t in b['outs']]
                bops.append(node(nid('y'), 'scf.yield', ins=[pick(inner, t) for _, t in its]))
                outs = [(f'{o}_{i}', t) for i, (_, t) in enumerate(its)]
                ops.append(node(o, 'scf.for', ins=[rng.choice(idx) if rng.random() > pmut / 2 else rng.choice(vals)[0] for _ in range(3)] + [v for v, _ in its], outs=outs,
                                regions=[[blk(nid('b'), bops, args=bargs)]])); vals += outs
        return ops, vals
    args = [('x0', 'i8'), ('x1', 'i32'), ('x2', 'index'), ('x3', 'f32'), ('x4', 'tensor<4xi8>')]
    ops, vals = seq(args, depth, rng.choice([2, 3, 4]), None)
    rt = rng.choice(TY)
    if rng.random() < 0.5:                               # 多块：cond_br 到两个带参数的块
        ints = [(v, t) for v, t in vals if t in ('i8', 'i32', 'index')]
        a, t = rng.choice(ints); c = nid('c')
        ops.append(node(c, 'arith.cmpi', ins=[a, a], outs=[(c, 'i1')], props={'predicate': 0}))
        bt = [rng.choice(TY) for _ in range(rng.choice([1, 2]))]
        L, R = nid('L'), nid('R')
        ops.append(node(nid('br'), 'cf.cond_br', ins=[c], succs=[(L, [pick(vals, t) for t in bt]), (R, [])]))
        la = [(nid('p'), t) for t in bt]
        lv = vals + la
        blocks = [blk('entry', ops, args=args),
                  blk(L, [node(nid('ret'), 'func.return', ins=[pick(lv, rt)])], args=la),
                  blk(R, [node(nid('ret'), 'func.return', ins=[pick(vals, rt)])])]
    else:
        ops.append(node(nid('ret'), 'func.return', ins=[pick(vals, rt)])); blocks = [blk('entry', ops, args=args)]
    return blocks, f'() -> ({rt})'

def O5(n, rng):
    import mlir.ir as ir, structure as ST
    ctx = ir.Context(); st = collections.Counter(); miss = collections.Counter()
    for _ in range(n):
        bs, ft = rgraph(rng)
        text = OR.emit_core(bs, ft)
        if text is None: st['不可发射'] += 1; continue
        truth = mlir_ok(ctx, text)
        full_ft = '(i8, i32, index, f32, tensor<4xi8>) -> ' + ft.split('->')[1].strip()
        sd = list(ST.check(ST.M(ST.F('f', bs, ftype=full_ft))))
        C.ORACLE_ON = False; sym = C.typecheck(bs, ft); C.ORACLE_ON = True
        full = C.typecheck(bs, ft)
        mine = not sd and not full; nooracle = not sd and not sym
        st['图'] += 1; st['MLIR 接受'] += truth; st['多块'] += len(bs) > 1
        st['结构 + 类型(含预言机) ' + ('一致' if mine == truth else '误收' if mine else '误拒')] += 1
        st['结构 + 符号层(无预言机) ' + ('一致' if nooracle == truth else '误收' if nooracle else '误拒')] += 1
        st['拒于结构层' if sd else '拒于符号层' if sym else '拒于预言机' if full else '接受'] += 1
        st['调用了预言机'] += any(n['op'] in ('scf.if', 'scf.for', 'cf.cond_br') for b in bs for n in b['ops']) and not sd and not sym
        if nooracle and not truth:
            ds = [m for _, m in full][:1]
            miss[re.sub(r"'[^']*'", "'…'", ds[0].split(': ', 1)[-1])[:70] if ds else '?'] += 1
    return st, miss

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
    st5, miss = O5(n, rng)
    print('── O5 区域/后继/多块:', dict(st5)); print('    无预言机时漏掉的（MLIR 原话归类）:', miss.most_common(8))
    print('    预言机不可用的原因:', OR.WHY_NONE)
    print('── core 计数:', {k: v for k, v in C.TC_STATS.items() if not k.endswith('例') and k != '发射缺陷'}); print('    人工签名误拒例:', C.TC_STATS.get('人工签名误拒例', [])[:3])
