"""复合 op 多态签名的检验（第 8 轮）。
  C1 三方一致：推导签名 Φ(T) ≡ 按体检查（内联语义）≡ MLIR verifier（把复合 op 在 T 上内联成 func）
  C2 嵌套 ≡ 内联：外层复合 op 调用内层复合 op 的签名，与把内层体展开后的签名逐 T 一致
  C3 端口投影（connectable）：可靠（存在可接受的补全 ⇒ 判可连）；精度（判可连但找不到补全的比例）
  C4 多态的价值：同一个复合 op 被接受的类型种类；'with' 与字面量结果类型的对照
  C5 代价
用法：PYTHONPATH=model python3 model/compsig_test.py [复合 op 数]
"""
import random, sys, time, collections, itertools
import typesig as TS, compsig as CS
from typesig_test import mlir_ok, CRASH
P = TS.parse
BASES = ['i1', 'i8', 'i32', 'i64', 'index', 'f32']
FORMS = ['{}', 'tensor<4x{}>', 'tensor<?x{}>', 'tensor<2x3x{}>', 'tensor<*x{}>', 'vector<4x{}>', 'tensor<{}>']
POOL = [P(f.format(b)) for b in BASES for f in FORMS]
INFER = ['arith.addi', 'arith.subi', 'arith.muli', 'arith.xori', 'arith.cmpi', 'arith.select', 'arith.addf', 'tensor.extract', 'tensor.insert', 'tensor.dim']
CHOSEN = ['arith.trunci', 'arith.extui', 'arith.index_cast']
_n = itertools.count()

def _witness(rng, n):
    b = rng.choice(['i8', 'i32', 'i64', 'index', 'f32']); f = rng.choice(FORMS[:4] + FORMS[5:])
    ws = [P(f.format(b)) for _ in range(n)]
    if rng.random() < 0.5: ws[-1] = P('index')
    return ws

def rand_comp(rng, nports=None, inner=None, use_with=True, tries=40):
    """类型导向生成：先取见证类型 W，只接受在 W 下良类型的节点，再抹掉类型 ⇒ 复合 op 一定可实例化（W ⊨ Φ）。"""
    ports = [f'p{i}' for i in range(nports or rng.choice([2, 3]))]
    W = _witness(rng, len(ports)); ty = dict(zip(ports, W)); vals, body = list(ports), []
    want = rng.choice([2, 3, 4]); k = 0
    while len(body) < want and k < tries:
        k += 1; nid = f'n{len(body)}'
        pick = lambda: rng.choice(vals[-3:] if rng.random() < 0.6 else vals)
        if inner and not body:
            c = TS.COMPS[inner]; args = [pick() for _ in c.ins]
            if c.check([ty[a] for a in args]): continue
            outs = c.infer([ty[a] for a in args]); body.append((nid, inner, args, [None] * len(outs)))
            for i, t in enumerate(outs):
                v = f'{nid}#{i}' if len(outs) > 1 else nid; vals.append(v); ty[v] = t
            continue
        op = rng.choice(INFER + CHOSEN)
        if op == 'arith.select': args = [pick(), pick(), pick()]
        elif op == 'tensor.extract': args = [pick()] + [pick() for _ in range(rng.choice([1, 2]))]
        elif op == 'tensor.insert': args = [pick(), pick()] + [pick() for _ in range(rng.choice([1, 2]))]
        elif op in CHOSEN: args = [pick()]
        else: args = [pick(), pick()]
        if op in CHOSEN:
            e = {'arith.trunci': 'i8', 'arith.extui': 'i64', 'arith.index_cast': rng.choice(['index', 'i64'])}[op]
            x = ty[args[0]]; rt = (x[0], x[1], P(e)) if TS.shaped(x) else P(e)
            spec = [('like', args[0], e)] if use_with else [TS.show(rt)]
        else:
            rt = _infer_concrete(op, [ty[a] for a in args]); spec = [None]
        if rt is None or TS.check(op, [ty[a] for a in args], [rt]): continue
        body.append((nid, op, args, spec)); vals.append(nid); ty[nid] = rt
    if not body: return rand_comp(rng, nports, inner, use_with, tries)
    c = CS.Comp(f'c{next(_n)}', ports, body, [vals[-1]]); c.W = W
    return c

def rand_T(rng, c):
    r = rng.random()
    if r < 0.3: return [rng.choice(POOL) for _ in c.ins]
    W = list(c.W)
    if r < 0.6:                                      # 换容器形式、保留元素：检验形状多态
        f = rng.choice(FORMS); return [t if t == ('index',) else P(f.format(TS.show(TS.elt(t)))) for t in W]
    if r < 0.8:                                      # 换元素、保留形式：检验元素多态
        b = rng.choice(BASES)
        return [t if t == ('index',) else ((t[0], t[1], P(b)) if TS.shaped(t) else P(b)) for t in W]
    j = rng.randrange(len(W)); W[j] = rng.choice(POOL); return W        # 改一个端口

def mlir_inline(c, T, rng):
    """直接按 Comp.nodes 的值名展开（不依赖表达式反查）。"""
    tys = dict(zip(c.ins, T)); names = {p: f'%{p}' for p in c.ins}; lines = []
    def go(src_nodes, rename, prefix):
        for nid, op, args, spec in src_nodes:
            argn = [rename(v) for v in args]
            if op in TS.COMPS:
                ic = TS.COMPS[op]; m = dict(zip(ic.ins, argn)); outs = ic.outs
                go(ic.body_src, lambda v, m=m, pf=prefix + nid + '_': m.get(v, pf + v), prefix + nid + '_')
                for i, o in enumerate(outs):
                    key = f'{nid}#{i}' if len(outs) > 1 else nid
                    names[prefix + key] = names[m.get(o, prefix + nid + '_' + o)]; tys[prefix + key] = tys[m.get(o, prefix + nid + '_' + o)]
                continue
            at = [tys[v] for v in argn]
            inf = None
            try:
                inf = None if any(t is None for t in at) else _infer_concrete(op, at)
            except Exception: inf = None
            sp = spec[0]
            if sp is None: rt = inf
            elif isinstance(sp, tuple):
                x = tys[rename(sp[1])]; rt = None if x is None else ((x[0], x[1], P(sp[2])) if TS.shaped(x) else P(sp[2]))
            else: rt = P(sp)
            if rt is None: rt = rng.choice(POOL)
            vn = prefix + nid; names[vn] = '%' + vn.replace('#', '_'); tys[vn] = rt
            attrs = ' {predicate = 0 : i64}' if op == 'arith.cmpi' else ''
            lines.append(f'  {names[vn]} = "{op}"(' + ', '.join(names[v] for v in argn) + f'){attrs} : ('
                         + ', '.join(TS.show(t) for t in at) + ') -> ' + TS.show(rt))
    go(c.body_src, lambda v: v, '')
    args = ', '.join(f'%{p}: {TS.show(t)}' for p, t in zip(c.ins, T))
    return f'func.func @f({args}) {{\n' + '\n'.join(lines) + '\n  return\n}'
def _infer_concrete(op, at):
    e = CS.infer_expr(op, [('p', f'a{i}') for i in range(len(at))])
    return None if e is None else CS.ev(e[0], {f'a{i}': t for i, t in enumerate(at)})

def C1(nc, rng):
    import mlir.ir as ir
    ctx = ir.Context(); st = collections.Counter(); ex = []
    for _ in range(nc):
        c = rand_comp(rng)
        for _ in range(12):
            T = rand_T(rng, c)
            phi, body = not c.check(T), c.check_body(T)
            nc0 = len(CRASH); truth = mlir_ok(ctx, mlir_inline(c, T, rng))
            if len(CRASH) > nc0: st['MLIR 崩溃（不计）'] += 1; st['崩溃例中 Φ 拒绝'] += not phi; continue
            st['实例'] += 1; st['MLIR 接受'] += truth
            st['Φ≡体'] += phi == body; st['Φ≡MLIR'] += phi == truth
            if phi != truth and len(ex) < 3: ex.append((c.show(), [TS.show(t) for t in T], phi, truth))
    return dict(st), ex

def flatten(outer):
    """把外层第一个节点（内层复合 op）展开，得到不含嵌套的等价复合 op。"""
    nid, op, args, spec = outer.body_src[0]; ic = TS.COMPS[op]; m = dict(zip(ic.ins, args))
    ren = lambda v: m.get(v, f'{nid}_{v}')
    body = [(f'{nid}_{x}', o, [ren(a) for a in aa], [(s[0], ren(s[1]), s[2]) if isinstance(s, tuple) else s for s in ss])
            for x, o, aa, ss in ic.body_src]
    outmap = {(f'{nid}#{i}' if len(ic.outs) > 1 else nid): ren(o) for i, o in enumerate(ic.outs)}
    for x, o, aa, ss in outer.body_src[1:]:
        body.append((x, o, [outmap.get(a, a) for a in aa], [(s[0], outmap.get(s[1], s[1]), s[2]) if isinstance(s, tuple) else s for s in ss]))
    return CS.Comp(outer.name + '_flat', outer.ins, body, [outmap.get(o, o) for o in outer.outs])

def C2(nc, rng):
    st = collections.Counter()
    for _ in range(nc):
        inner = rand_comp(rng, 2); outer = rand_comp(rng, inner=inner.name); flat = flatten(outer)
        for _ in range(20):
            T = rand_T(rng, outer); a, b, d = not outer.check(T), not flat.check(T), flat.check_body(T)
            st['实例'] += 1; st['嵌套≡展开'] += a == b; st['展开 Φ≡体'] += b == d; st['接受'] += a
    return dict(st)

def C3(nc, rng):
    st = collections.Counter()
    for _ in range(nc):
        c = rand_comp(rng, 2)
        for j, p in enumerate(c.ins):
            for t in POOL:
                conn = c.connectable(p, t)
                exists = any(not c.check([t if i == j else u for i in range(2)]) for u in POOL)
                st['（端口, 类型）'] += 1; st['判可连'] += conn; st['存在补全'] += exists
                st['不可靠（有补全却判不可连）'] += exists and not conn; st['判可连但无补全'] += conn and not exists
    return dict(st)

def C4(nc, rng):
    st = collections.Counter()
    for _ in range(nc):
        seed = rng.random()
        cw = rand_comp(random.Random(seed), 2, use_with=True); cl = rand_comp(random.Random(seed), 2, use_with=False)
        if not any(op in CHOSEN for _, op, _, _ in cw.body_src): continue            # 只看含“结果类型由用户选择”的 op 的复合 op
        st['复合 op'] += 1
        kinds_w, kinds_l = set(), set()
        for T in itertools.product(POOL, repeat=2):
            if not cw.check(list(T)): kinds_w.add(tuple(t[0] if TS.shaped(t) else 's' for t in T))
            if not cl.check(list(T)): kinds_l.add(tuple(t[0] if TS.shaped(t) else 's' for t in T))
            st['with 接受'] += not cw.check(list(T)); st['字面量 接受'] += not cl.check(list(T))
        st['可实例化（with）'] += bool(kinds_w); st['可实例化（字面量）'] += bool(kinds_l)
        st['多种容器（with）'] += len({k for k in kinds_w}) > 1; st['多种容器（字面量）'] += len(kinds_l) > 1
    return dict(st)

def rand_untyped(rng):
    """不做类型导向的随机复合 op（像用户随手连的图）：检验定义期能否发现“永远无法实例化”。"""
    ports = ['p0', 'p1']; vals, body = list(ports), []
    for k in range(rng.choice([2, 3])):
        op = rng.choice(['arith.addi', 'arith.cmpi', 'arith.select', 'tensor.extract', 'tensor.dim', 'arith.addf'])
        n = {'arith.select': 3, 'tensor.extract': 2}.get(op, 2)
        body.append((f'n{k}', op, [rng.choice(vals) for _ in range(n)], [None])); vals.append(f'n{k}')
    return CS.Comp(f'u{next(_n)}', ports, body, [vals[-1]])
def C6(nc, rng):
    st = collections.Counter(); t0 = time.perf_counter()
    for _ in range(nc):
        c = rand_untyped(rng)
        cand = [[t for t in POOL if c.connectable(p, t)] for p in c.ins]       # 先用端口投影剪枝，再在乘积上找见证
        sat = any(not c.check(list(T)) for T in itertools.product(*cand))
        st['复合 op'] += 1; st['在类型池上可实例化'] += sat; st['剪枝后的候选组合'] += len(cand[0]) * len(cand[1])
    st['平均耗时（ms）'] = round((time.perf_counter() - t0) / nc * 1e3, 1); st['剪枝后的候选组合'] //= nc
    return dict(st)

def C5(rng):
    cs = [rand_comp(rng, 3) for _ in range(30)]; Ts = [[rand_T(rng, c) for _ in range(30)] for c in cs]
    t0 = time.perf_counter(); [rand_comp(random.Random(i), 3) for i in range(200)]; t1 = time.perf_counter()
    [c.check(T) for c, ts in zip(cs, Ts) for T in ts]; t2 = time.perf_counter()
    [c.check_body(T) for c, ts in zip(cs, Ts) for T in ts]; t3 = time.perf_counter()
    [c.connectable('p0', T[0]) for c, ts in zip(cs, Ts) for T in ts]; t4 = time.perf_counter()
    n = 900
    return {'推导一个签名（µs）': round((t1 - t0) / 200 * 1e6), 'Φ 检查（µs）': round((t2 - t1) / n * 1e6, 1),
            '按体检查（µs）': round((t3 - t2) / n * 1e6, 1), 'connectable（µs）': round((t4 - t3) / n * 1e6, 1)}

if __name__ == '__main__':
    nc = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = random.Random(11)
    print('── 示例'); print(rand_comp(random.Random(3)).show())
    st, ex = C1(nc, rng); print('── C1 Φ ≡ 体 ≡ MLIR:', st)
    for e in ex: print('     例', e)
    print('── C2 嵌套 ≡ 展开:', C2(nc // 2, rng))
    print('── C3 端口投影:', C3(nc // 3, rng))
    print('── C4 with 与字面量:', C4(nc // 3, rng))
    print('── C5 代价:', C5(rng))
    print('── C6 定义期可实例化判定（随手连的图）:', C6(nc, rng))
