"""第 40 轮（L24 落地，spec/values.md §13）：降低时插守卫，core 的 tensor 语义（越界 ⇒ 毒值）对 JIT。
  G0 规划：core.refines(core 语义, 插守卫后语义) 判定可降低；毒值下标（nsw 溢出等）流进守卫 ⇒ 拒绝（不发射）。
  G1 生成器（只作补充；真值是 JIT）：tensor 参数 + index 参数，extract/insert/dim/cast 用变量/算术下标；
     插守卫（guards.insert_guards，对全部 extract/insert/dim/cast）后的模块 vs core 语义，**包括越界输入**（此前一律“守卫”跳过）。
  G2 A/B：同一批输入跑**未插守卫**的原模块：越界输入上出现不一致/崩溃 ⇒ 守卫是必要的（否则 G1 的一致只是巧合）。
  G3 语料（第 41 轮）：tensor 签名语料函数中需要守卫的（此前越界输入一律跳过），插守卫后用常规 + 越界输入对 JIT；
     报：需守卫函数数、规划接受/拒绝（按原因）、比较数（其中越界输入）。
  G4 S3 改写规则外部化（第 43 轮）：shape_test S3 的 6 条有效规则与 2 条错误规则（对照），左边按 core 语义求值，
     右边插守卫后由 JIT 执行，输入含越界/负尺寸；有效规则须“左边非毒值处右边相同”，错误规则须被找到反例（灵敏度）。
  G5 全形式切片（第 43 轮）：tensorjit_test.gen_slice 的程序（步长、降秩、动态偏移/尺寸/步长、写回 insert_slice）插守卫后，
     对此前因守卫跳过的输入比较；同一批输入跑未插守卫的原程序作 A/B。
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

def _g3_file(path, q, seed):
    import mlir.ir as ir, irimport as I, core as C
    from compcorpus_test import _walk, _funcs
    from valimport_test import derive_imported
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 3: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    covered = set(C.REG) | {'scf.for', 'scf.if', 'scf.yield'}
    for ci, ch in enumerate(I.split_chunks(src)):
        if not ch.strip() or I.expected(ch): continue
        try: m = ir.Module.parse(ch, TJ.ctx1()); root = I.import_module(m)
        except Exception: continue
        for fi, fn in enumerate(_funcs(root)):
            blocks = fn['regions'][0]; ft = fn['props']['_attrs'].get('function_type')
            if len(blocks) != 1 or not ft or not re.search(r'tensor<', ft): continue
            if any(x['op'] not in covered and x['op'] != 'func.return' for x in _walk(fn) if x is not fn): continue
            ins = [t for _, t in blocks[0]['args']]
            mm = re.fullmatch(r'\((.*)\) -> \(?([^()]*?)\)?', ft)
            outs = [x.strip() for x in mm.group(2).split(',')] if mm and mm.group(2).strip() else []
            if not all(TJ.ok_ty(t) for t in ins + outs): continue
            S, why = derive_imported(f'g3{seed}_{ci}_{fi}', fn)
            if S is None: continue
            st['有事实（候选）'] += 1
            if not C.guards(S): continue
            st['需守卫的函数'] += 1
            name = fn['props']['_attrs']['sym_name'].strip('"')
            ftxt = next((str(o.operation) for o in m.body.operations if o.operation.name == 'func.func' and
                         str(o.operation.attributes['sym_name']).strip('"') == name), None)
            gtxt, why, _ = lower(ftxt, fn, f'g3{seed}_{ci}_{fi}')
            if gtxt is None: st['规划拒绝/' + why[:28]] += 1; note('拒绝', f'{path}:{name} {why}'); continue
            st['规划接受'] += 1
            fname = name if re.fullmatch(r'[A-Za-z_][\w.$]*', name) else f'"{name}"'
            xs = oob_inputs(rng, ins, 8)
            xg = [x for x in xs if TJ.model_run(S, ins, outs, x, random.Random(0))[1] == '守卫']
            xn = [x for x in xs if x not in xg]
            st['输入中原先因守卫跳过'] += len(xg)
            sg = collections.Counter()
            if xn: TJ.check_fn(S, gtxt, fname, ins, outs, random.Random(seed), st, note, f'{path}:{name}', guarded=True, inputs=xn)
            if xg: TJ.check_fn(S, gtxt, fname, ins, outs, random.Random(seed), sg, note, f'{path}:{name}', guarded=True, inputs=xg)
            for k2, v2 in sg.items(): st[k2] += v2; st['越界输入/' + k2] += v2
    q.put((st, dict(ex)))

def G3(lim=10 ** 9):
    import multiprocessing as mp
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
        q = ctxm.Queue(); pr = ctxm.Process(target=_g3_file, args=(path, q, i)); pr.start()
        try: s, ex = q.get(timeout=600)
        except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; continue
        pr.join(); tot.update(s)
        for k, v in ex.items(): exs[k] += v[:2]
    return tot, exs

TD = 'tensor<?xi8>'
def _f(args, outs, body):
    sig = ', '.join(f'%{a}: {t}' for a, t in args)
    return f'func.func @main({sig}) -> (' + ', '.join(t for _, t in outs) + ') {\n  ' + '\n  '.join(body) + '\n  return ' + ', '.join(v for v, _ in outs) + ' : ' + ', '.join(t for _, t in outs) + '\n}'
X5 = [('x', TD), ('o1', 'index'), ('n1', 'index'), ('o2', 'index'), ('n2', 'index')]
RULES = {
  'slice∘slice → slice(o1+o2)': (True, X5, [('%s2', TD)],
      [f'%s1 = tensor.extract_slice %x[%o1] [%n1] [1] : {TD} to {TD}', f'%s2 = tensor.extract_slice %s1[%o2] [%n2] [1] : {TD} to {TD}'],
      [('%s', TD)], ['%o = arith.addi %o1, %o2 : index', f'%s = tensor.extract_slice %x[%o] [%n2] [1] : {TD} to {TD}']),
  '✗ slice∘slice → slice(o1−o2)': (False, X5, [('%s2', TD)],
      [f'%s1 = tensor.extract_slice %x[%o1] [%n1] [1] : {TD} to {TD}', f'%s2 = tensor.extract_slice %s1[%o2] [%n2] [1] : {TD} to {TD}'],
      [('%s', TD)], ['%o = arith.subi %o1, %o2 : index', f'%s = tensor.extract_slice %x[%o] [%n2] [1] : {TD} to {TD}']),
  'extract(slice(x,o,n), i) → extract(x, o+i)': (True, [('x', TD), ('o', 'index'), ('n', 'index'), ('i', 'index')], [('%e', 'i8')],
      [f'%s = tensor.extract_slice %x[%o] [%n] [1] : {TD} to {TD}', f'%e = tensor.extract %s[%i] : {TD}'],
      [('%e', 'i8')], ['%k = arith.addi %o, %i : index', f'%e = tensor.extract %x[%k] : {TD}']),
  'extract(insert(v,x,i), i) → v': (True, [('v', 'i8'), ('x', TD), ('i', 'index'), ('j', 'index')], [('%e', 'i8')],
      [f'%t = tensor.insert %v into %x[%i] : {TD}', f'%e = tensor.extract %t[%i] : {TD}'],
      [('%v', 'i8')], []),
  '✗ extract(insert(v,x,i), j) → extract(x, j)': (False, [('v', 'i8'), ('x', TD), ('i', 'index'), ('j', 'index')], [('%e', 'i8')],
      [f'%t = tensor.insert %v into %x[%i] : {TD}', f'%e = tensor.extract %t[%j] : {TD}'],
      [('%e', 'i8')], [f'%e = tensor.extract %x[%j] : {TD}']),
  'insert_slice(slice(x,o,n), x, o) → x': (True, [('x', TD), ('o', 'index'), ('n', 'index')], [('%t', TD)],
      [f'%s = tensor.extract_slice %x[%o] [%n] [1] : {TD} to {TD}', f'%t = tensor.insert_slice %s into %x[%o] [%n] [1] : {TD} into {TD}'],
      [('%x', TD)], []),
  'dim(insert_slice(s,x,o), 0) → dim(x, 0)': (True, [('s', TD), ('x', TD), ('o', 'index')], [('%d', 'index')],
      ['%c0 = arith.constant 0 : index', f'%n = tensor.dim %s, %c0 : {TD}', f'%t = tensor.insert_slice %s into %x[%o] [%n] [1] : {TD} into {TD}', f'%d = tensor.dim %t, %c0 : {TD}'],
      [('%d', 'index')], ['%c0 = arith.constant 0 : index', f'%d = tensor.dim %x, %c0 : {TD}']),
  'from_elements(v,v) → splat(v)': (True, [('v', 'i8')], [('%t', 'tensor<2xi8>')],
      ['%t = tensor.from_elements %v, %v : tensor<2xi8>'], [('%t', 'tensor<2xi8>')], ['%t = tensor.splat %v : tensor<2xi8>']),
}
def rule_inputs(rng, ins, n=160):
    xs = []
    for _ in range(n):
        x = TJ.mk_inputs(rng, ins)
        for j, t in enumerate(ins):
            if t == 'index': x[j] = rng.choice([0, 0, 1, 1, 1, 2, 2, 3]) if rng.random() < 0.8 else rng.choice([-2, -1, 4, 6, 1 << 40])   # 多数在界内（才有非毒值可比），少数越界
        xs.append(x)
    return xs

def G4(seed=29):
    import mlir.ir as ir, irimport as I
    from valimport_test import derive_imported
    out = {}
    for name, (valid, args, lo, lb, ro, rb) in RULES.items():
        rng = random.Random(seed); st = collections.Counter()
        ins = [t for _, t in args]; outs = [t for _, t in lo]
        ltxt = _f(args, lo, lb); rtxt = _f(args, ro, rb)
        fl = I.import_module(ir.Module.parse(ltxt, TJ.ctx1()))['regions'][0][0]['ops'][0]
        Sl, why = derive_imported('g4l_' + str(abs(hash(name))), fl)
        fr = I.import_module(ir.Module.parse(rtxt, TJ.ctx1()))['regions'][0][0]['ops'][0]
        gr, whyr, _ = lower(rtxt, fr, 'g4r_' + str(abs(hash(name))))
        if Sl is None or gr is None: out[name] = ('未知', why or whyr); continue
        good, bad = [], []                               # 拒绝采样：左边有非毒值结果的输入 60 组 + 全毒值（越界）的 20 组
        for _ in range(4000):
            if len(good) >= 60 and len(bad) >= 20: break
            x = rule_inputs(rng, ins, 1)[0]
            r = TJ.model_run(Sl, ins, outs, x, random.Random(0), True)
            if r[0] == 'val' and any(m[0] != 'x' for m in r[1]):
                if len(good) < 60: good.append(x)
            elif len(bad) < 20: bad.append(x)
        TJ.check_fn(Sl, gr, 'main', ins, outs, random.Random(seed), st, lambda k, v: None, name, guarded=True, inputs=good + bad)
        out[name] = (valid, st['一致'], st['不一致'], st['跳过/毒值'] + st['跳过/毒值结果'])
    return out

FAILS = []
def G5(N=80, seed=37):
    import mlir.ir as ir, irimport as I
    rng = random.Random(seed); st = collections.Counter(); ab = collections.Counter()
    for t in range(N):
        txt, ins, outs = TJ.gen_slice(rng)
        try: m = ir.Module.parse(txt, TJ.ctx1()); assert m.operation.verify()
        except Exception: st['不合法（跳过）'] += 1; continue
        fn = I.import_module(m)['regions'][0][0]['ops'][0]
        gtxt, why, S = lower(txt, fn, f'g5_{t}')
        if S is None: st['未知（core）'] += 1; continue
        if gtxt is None: st['规划拒绝/' + why[:24]] += 1; continue
        st['程序'] += 1
        cands = [TJ.mk_inputs(rng, ins) for _ in range(12)]
        for x in cands:
            for j, ty in enumerate(ins):
                if ty == 'index': x[j] = rng.choice([0, 1, 2, 3, -1, 5, 9])
        xg = [x for x in cands if TJ.model_run(S, ins, outs, x, random.Random(0))[1] == '守卫']
        st['输入中原先因守卫跳过'] += len(xg)
        if xg:                                           # 越界输入下输出整体是毒值，无值可比；比较的是“执行是否安全”：读出结果的 extent 与一个元素
            qs = []
            for j, ty in enumerate(outs):
                r = len(TJ.tparse(ty)[0]); qs += [(f'q_{j}_d{d}', j, 'dim', d) for d in range(r)] + [(f'q_{j}_e', j, 'el', r)]
            for nm2, src, cnt in (('插守卫', gtxt, st), ('未插守卫', txt, ab)):
                prog, npar, nq = TJ.build(src, 'main', ins, outs, qs)
                if isinstance(TJ.jit_forked(prog, []), str):    # 只编译：MLIR 管线自身崩溃（与守卫无关，原程序同样）⇒ 单列，不计
                    cnt['MLIR 编译崩溃（不计）'] += 1; continue
                calls = [(q[0], TJ.flat(ins, x, [0] * q[3] if q[2] == 'el' else [], nq)) for x in xg for q in qs]
                got = TJ.jit_forked(prog, calls)
                cnt['执行（越界输入）/' + ('运行崩溃或异常' if isinstance(got, str) else '正常结束')] += 1
                if isinstance(got, str) and nm2 == '插守卫': FAILS.append((got[:300], gtxt, xg[:2]))
    return st, ab

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
    if len(sys.argv) > 3:
        r4 = G4(); print('── G4 S3 改写规则（左 core、右插守卫 JIT）：(有效?, 一致, 不一致, 左边毒值跳过)')
        for k, v in r4.items(): print(f'   {k}: {v}')
        s5, a5 = G5(); print('── G5 全形式切片（越界输入）:', dict(sorted(s5.items()))); print('   A/B 未插守卫:', dict(sorted(a5.items())))
        import verdict as V
        V.check('G4 S3 改写规则外部化', all(v[0] != '未知' and ((v[0] and v[2] == 0 and v[1] > 0) or (not v[0] and v[2] > 0)) for v in r4.values()), '有效规则 0 不一致、错误规则有反例')
        V.check('G5 全形式切片', s5.get('执行（越界输入）/运行崩溃或异常', 0) == 0 and s5.get('执行（越界输入）/正常结束', 0) > 0, f"正常 {s5.get('执行（越界输入）/正常结束', 0)}，运行崩溃 {s5.get('执行（越界输入）/运行崩溃或异常', 0)}")
        V.finish()
    if len(sys.argv) > 2:
        t3, e3 = G3(int(sys.argv[2]))
        print('── G3 语料（插守卫）:'); [print(f'   {k}: {v}') for k, v in sorted(t3.items())]
        for k, v in e3.items(): print('──', k); [print('    ', x[:300]) for x in v[:10]]
        if N == 0: sys.exit(0 if t3['不一致'] == 0 else 1)
    g0 = G0(); print('── G0′:', g0)
    st, ab, ex = G1(N)
    print('── G1 插守卫后 vs core（含越界输入）:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
    print('── G2 A/B 未插守卫（同一批输入）:'); [print(f'   {k}: {v}') for k, v in sorted(ab.items())]
    for k, v in ex.items(): print('──', k); [print('    ', x[:600]) for x in v]
    import verdict as V
    V.check('G0′ 定向', all(g0.values()), str(g0))
    V.check('G1 生成器', st['越界输入/一致'] > 0 and st['不一致'] == 0 and st['一致'] > 0 and st['程序'] > 0 and (ab['不一致'] > 0 or any('JIT 不可用' in k for k in ab)),
            f"一致 {st['一致']}，越界一致 {st['越界输入/一致']}，不一致 {st['不一致']}")
    V.finish()
