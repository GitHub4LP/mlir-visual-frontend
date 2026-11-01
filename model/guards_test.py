"""第 40 轮（L24 落地，spec/values.md §13）：降低时插守卫，core 的 tensor 语义（越界 ⇒ 毒值）对 JIT。
  G0 规划：core.refines(core 语义, 插守卫后语义) 判定可降低；毒值下标（nsw 溢出等）流进守卫 ⇒ 拒绝（不发射）。
  G1 生成器（只作补充；真值是 JIT）：tensor 参数 + index 参数，extract/insert/dim/cast 用变量/算术下标；
     插守卫（guards.insert_guards，对全部 extract/insert/dim/cast）后的模块 vs core 语义，**包括越界输入**（此前一律“守卫”跳过）。
  G2 A/B：同一批输入跑**未插守卫**的原模块：越界输入上出现不一致/崩溃 ⇒ 守卫是必要的（否则 G1 的一致只是巧合）。
  G0′ 定向：nsw 溢出得到的毒值下标进守卫 ⇒ 必须拒绝；普通下标 ⇒ 必须接受。
用法：PYTHONPATH=model python3 model/guards_test.py [G1 程序数] [语料文件数上限]
"""
import sys, re, random, collections, glob, os
import tensorjit_test as TJ

def gen(rng):
    T = rng.choice([('tensor<?xi8>', 1), ('tensor<3xi8>', 1), ('tensor<?x2xi8>', 2)])
    ty, r = T
    L = []; k = [0]; scal = []; tens = ['%t']
    def nv(p='%v'):
        k[0] += 1; return f'{p}{k[0]}'
    def idx():
        c = rng.random()
        if c < 0.5: return rng.choice(['%i0', '%i1'])
        if c < 0.7:
            v = nv('%c'); L.append(f'{v} = arith.constant {rng.choice([0, 1, 2, 3, 5])} : index'); return v
        v = nv('%x'); L.append(f'{v} = arith.{rng.choice(["addi", "subi", "muli"])} %i0, %i1 : index'); return v
    for _ in range(rng.randrange(2, 6)):
        c = rng.random(); t = rng.choice(tens)
        tty = ty if t == '%t' or not t.startswith('%k') else TSTAT[t]
        if c < 0.35:
            ixs = [idx() for _ in range(r)]
            o = nv(); L.append(f'{o} = tensor.extract {t}[' + ', '.join(ixs) + f'] : {tty}')
            if rng.random() < 0.7:                                      # 无分支的掩码：core 语义下未选中的越界读只是毒值；MLIR 原生则是 UB
                cs = []
                for kk, x in enumerate(ixs):
                    kc, dd, cc = nv('%kc'), nv('%dd'), nv('%cc')
                    L += [f'{kc} = arith.constant {kk} : index', f'{dd} = tensor.dim {t}, {kc} : {tty}', f'{cc} = arith.cmpi ult, {x}, {dd} : index']; cs.append(cc)
                c0 = cs[0]
                for x in cs[1:]:
                    a2 = nv('%and'); L.append(f'{a2} = arith.andi {c0}, {x} : i1'); c0 = a2
                z = nv('%z'); m2 = nv('%m'); L += [f'{z} = arith.constant {rng.randrange(-5, 6)} : i8', f'{m2} = arith.select {c0}, {o}, {z} : i8']; o = m2
            scal.append(o)
        elif c < 0.6:
            v = rng.choice(scal) if scal else '%s'
            o = nv('%u'); L.append(f'{o} = tensor.insert {v} into {t}[' + ', '.join(idx() for _ in range(r)) + f'] : {tty}')
            if tty == ty: tens.append(o)
        elif c < 0.75:
            o = nv('%d'); L.append(f'{o} = tensor.dim {t}, {idx()} : {tty}')
            w = nv(); L.append(f'{w} = arith.index_cast {o} : index to i8'); scal.append(w)
        elif c < 0.85 and '?' in ty:
            st = ty.replace('?', rng.choice(['2', '3']))
            o = nv('%k'); L.append(f'{o} = tensor.cast {t} : {tty} to {st}'); TSTAT[o] = st
            e = nv(); L.append(f'{e} = tensor.extract {o}[' + ', '.join(['%c0z'] * r) + f'] : {st}'); scal.append(e)
        elif len(scal) >= 2:
            o = nv(); L.append(f'{o} = arith.{rng.choice(["addi", "xori", "muli"])} {scal[-1]}, {scal[-2]} : i8'); scal.append(o)
    if not scal: o = nv(); L.append(f'{o} = tensor.extract %t[' + ', '.join(['%i0'] * r) + f'] : {ty}'); scal.append(o)
    outs_v = [scal[-1]]; outs_t = ['i8']
    last_t = [x for x in tens if x.startswith('%u')]
    if last_t: outs_v.append(last_t[-1]); outs_t.append(ty)
    body = '\n  '.join(['%c0z = arith.constant 0 : index'] + L)
    return (f'func.func @main(%t: {ty}, %i0: index, %i1: index, %s: i8) -> (' + ', '.join(outs_t) + ') {\n  ' + body +
            '\n  return ' + ', '.join(outs_v) + ' : ' + ', '.join(outs_t) + '\n}', [ty, 'index', 'index', 'i8'], outs_t)
TSTAT = {}

def lower(txt, fn, tag):
    """规划 + 插守卫。返回 (守卫后文本 | None, 原因, S_core)。"""
    import mlir.ir as ir, guards as G, core as C
    from valimport_test import derive_imported
    k = [0]
    def der():
        k[0] += 1; S, _ = derive_imported(f'{tag}_{k[0]}_{C.GUARD_LOWER}', fn); return S
    ok, why, S, S2 = G.plan(der)
    if not ok: return None, why, S
    names = set()
    def walk(blocks):
        for b in blocks:
            for n in b['ops']:
                if n['op'] in G.GUARDABLE and n['outs']: names.add(n['outs'][0][0])
                for rg in n['regions']: walk(rg)
    walk(fn['regions'][0])
    m = ir.Module.parse(txt, TJ.ctx1())
    n, w = G.insert_guards(m, names)
    if w or not m.operation.verify(): return None, '插守卫失败 ' + str(w), S
    return str(m.body.operations[0]), '', S

def oob_inputs(rng, ins, n=6):
    """一半常规输入、一半刻意越界（下标取 -1、大数、等于 extent）。"""
    xs = [TJ.mk_inputs(rng, ins) for _ in range(n)]
    for x in xs[n // 2:]:
        for j, t in enumerate(ins):
            if t == 'index': x[j] = rng.choice([-1, 3, 4, 7, 1 << 40, -(1 << 62)])
    return xs

def G1(N, seed=17):
    import mlir.ir as ir, irimport as I
    rng = random.Random(seed); st = collections.Counter(); ab = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 3: ex[k].append(v)
    for t in range(N):
        TSTAT.clear(); txt, ins, outs = gen(rng)
        try: m = ir.Module.parse(txt, TJ.ctx1()); assert m.operation.verify()
        except Exception: st['不合法（跳过）'] += 1; continue
        fn = I.import_module(m)['regions'][0][0]['ops'][0]
        gtxt, why, S = lower(txt, fn, f'g1_{t}')
        if S is None: st['未知（core）'] += 1; continue
        if gtxt is None: st['规划拒绝/' + why[:30]] += 1; note('拒绝', why + '\n' + txt); continue
        st['程序'] += 1
        xs = oob_inputs(rng, ins)
        xg = [x for x in xs if TJ.model_run(S, ins, outs, x, random.Random(0))[1] == '守卫']
        xn = [x for x in xs if x not in xg]
        st['输入中原先因守卫跳过'] += len(xg)
        sg = collections.Counter()
        if xn: TJ.check_fn(S, gtxt, 'main', ins, outs, random.Random(t), st, lambda k, v: note(k, v + '\n' + gtxt), f'g1_{t}', guarded=True, inputs=xn)
        if xg: TJ.check_fn(S, gtxt, 'main', ins, outs, random.Random(t), sg, lambda k, v: note(k, v + '\n' + gtxt), f'g1_{t}g', guarded=True, inputs=xg)
        for k2, v2 in sg.items(): st[k2] += v2; st['越界输入/' + k2] += v2
        TJ.check_fn(S, txt, 'main', ins, outs, random.Random(t), ab, lambda k, v: note('A/B ' + k, v), f'g1_{t}', guarded=True, inputs=xs)
    return st, ab, ex

def G0():
    import mlir.ir as ir, irimport as I
    cases = {'nsw 毒值下标（须拒绝）': ('''func.func @main(%t: tensor<?xi8>, %i0: index, %i1: index, %s: i8) -> (i8) {
  %j = arith.addi %i0, %i1 overflow<nsw> : index
  %v = tensor.extract %t[%j] : tensor<?xi8>
  return %v : i8
}''', False), '普通下标（须接受）': ('''func.func @main(%t: tensor<?xi8>, %i0: index, %i1: index, %s: i8) -> (i8) {
  %v = tensor.extract %t[%i0] : tensor<?xi8>
  return %v : i8
}''', True)}
    out = {}
    for nm, (txt, want) in cases.items():
        fn = I.import_module(ir.Module.parse(txt, TJ.ctx1()))['regions'][0][0]['ops'][0]
        g, why, S = lower(txt, fn, 'g0_' + str(want))
        out[nm] = (g is not None) == want
    return out

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    g0 = G0(); print('── G0′:', g0)
    st, ab, ex = G1(N)
    print('── G1 插守卫后 vs core（含越界输入）:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
    print('── G2 A/B 未插守卫（同一批输入）:'); [print(f'   {k}: {v}') for k, v in sorted(ab.items())]
    for k, v in ex.items(): print('──', k); [print('    ', x[:600]) for x in v]
    ok = all(g0.values()) and st['越界输入/一致'] > 0 and st['不一致'] == 0 and st['一致'] > 0 and st['程序'] > 0 and (ab['不一致'] > 0 or any('JIT 不可用' in k for k in ab))
    sys.exit(0 if ok else 1)
