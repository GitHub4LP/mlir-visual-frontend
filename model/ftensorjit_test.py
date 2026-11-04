"""第 60 轮：浮点元素 tensor 签名的语料函数，值层 vs 同一提交 MLIR 的 JIT（M0.2 分母缺口）。

背景：tensorjit_test 的 T 语料只构造整数 tensor 输入；签名含浮点 tensor/浮点标量的函数，模型给出了事实，
却记为“有事实但签名含非整数 tensor”跳过——即模型在这类函数上的事实**从未经外部对照**。这里补上。
  - 浮点 tensor 输入：元素位模式 = (seed + Σ c_k·i_k) mod 2^BW，seed 取 floatjit 的特殊值/随机位模式，c_k ∈ [-3, 3]：
    相邻元素是相邻位模式，覆盖 ±0、次正规、±Inf、NaN 附近。JIT 侧 i64 计算后 trunci + bitcast；模型侧 Int2BV + fpBVToFP。
  - 浮点标量输入：floatjit.rand_in；结果经 bitcast 为位模式比较；NaN 只比较“是 NaN”（载荷不可观测，values §10）。
  - f16 预先排除（本机 aarch64 上 LLVM f16 往返误编译，第 35b 轮），单列计数。
  - 灵敏度：模型侧输入 seed+1（元素位模式整体偏移），元素比较须发现。
第 60c 轮：60b 的 19 个未执行中 6 个守卫恒假（out_of_bounds_*、invalid_slice_ops，MLIR 恒 UB，正确），13 个要求动态维 > 8；动态维上界 8 → 40（事后，只改采样），阈值不变。
第 60b 轮（阈值运行前写定）：随机 16 组全被跳过的函数改由求解器构造输入；FT 求解输入：求解后执行的函数 ≥ 24（47 的一半）且不一致 0。
首跑（每函数 4 组输入）：候选 94、执行 36、比较 562、不一致 0；二跑改为至多 16 组取 4 组（事后，只改采样，阈值不变）。
阈值（运行前写定）：FT 比较 ≥ 300、不一致 0、文件超时/崩溃 0；FT 灵敏度：seed+1 比较 ≥ 50 且发现率 ≥ 50%。
用法：PYTHONPATH=model python3 model/ftensorjit_test.py [文件数上限]
"""
import sys, re, glob, random, collections, multiprocessing as mp
import z3
import tensorjit_test as TJ, floatjit_test as FJ

FPT = {'bf16': (8, 8), 'f32': (8, 24), 'f64': (11, 53)}
IELEM = TJ.ELEM
ELEM = IELEM + tuple(FPT)
def BW(t): return 64 if t == 'index' else sum(FPT[t]) if t in FPT else int(t[1:])
def tparse(t):
    m = re.fullmatch(r'tensor<((?:(?:\d+|\?)x)*)(\w+)>', t)
    if not m or m.group(2) not in ELEM: return None
    return ([None if d == '?' else int(d) for d in m.group(1).rstrip('x').split('x')] if m.group(1) else []), m.group(2)
def ok_ty(t): return t in ELEM or tparse(t) is not None
def has_float(ts): return any(t in FPT or (tparse(t) or (0, ''))[1] in FPT for t in ts)

def mk_inputs(rng, ins):
    xs = []
    for t in ins:
        if t in FPT: xs.append(FJ.rand_in(rng, t))
        elif t in IELEM:
            from valimport_test import consts
            xs.append(rng.choice(consts(t)))
        else:
            dims, e = tparse(t); ex = [d if d is not None else rng.randrange(0, 4) for d in dims]
            seed = FJ.rand_in(rng, e) if e in FPT else rng.randrange(-8, 9)
            xs.append((ex, seed, [rng.randrange(-3, 4) for _ in dims]))
    first = {}
    for k, t in enumerate(ins):
        if t in ELEM: continue
        if t in first and rng.random() < 0.8: xs[k] = (list(xs[first[t]][0]),) + xs[k][1:]
        first.setdefault(t, k)
    return xs

def ztensor(t, x):
    import core as C
    dims, e = tparse(t); ex, seed, cs = x; S_ = C.zsort(t)
    iv = [z3.Int(f'_g{k}') for k in range(len(dims))]
    lin = seed + z3.Sum([c * i for c, i in zip(cs, iv)]) if iv else z3.IntVal(seed)
    if e == 'index': val = lin
    elif e in FPT: val = z3.fpBVToFP(z3.Int2BV(lin, BW(e)), z3.FPSort(*FPT[e]))
    else: val = z3.Int2BV(lin, BW(e))
    return C.mkt(S_, [z3.IntVal(d) for d in ex], C.lam(iv, val), C.lam(iv, z3.BoolVal(False)))

def scal(vv, t):
    """模型值 → ('nan',) | ('bits', b) | None（未定）。"""
    if t in FPT:
        if z3.is_true(z3.simplify(z3.fpIsNaN(vv))): return ('nan',)
        b = z3.simplify(z3.fpToIEEEBV(vv))
        return ('bits', b.as_long()) if z3.is_bv_value(b) else None
    if z3.is_bv_value(vv): return ('bits', vv.as_long())
    if z3.is_int_value(vv): return ('bits', vv.as_long() % (1 << 64))
    return None

def model_run(S, ins, outs, xs, rng):
    import core as C
    pairs = []
    for (h, hp), t, x in zip(S.holes, ins, xs):
        if t in FPT: z = z3.simplify(z3.fpBVToFP(z3.BitVecVal(x % (1 << BW(t)), BW(t)), z3.FPSort(*FPT[t])))
        elif t in IELEM: z = z3.IntVal(C.sx(x % (1 << 64), 64)) if t == 'index' else z3.BitVecVal(x % (1 << BW(t)), BW(t))
        else: z = ztensor(t, x)
        pairs += [(h, z), (hp, z3.BoolVal(False))]
    sub = lambda f: z3.simplify(z3.substitute(f, pairs)) if pairs else z3.simplify(f)
    def decide(f):
        f = sub(f)
        if z3.is_true(f) or z3.is_false(f): return z3.is_true(f)
        so = z3.Solver(); so.set('timeout', 5000); so.add(f); r = so.check()
        if r == z3.unknown: return None
        so2 = z3.Solver(); so2.set('timeout', 5000); so2.add(z3.Not(f)); r2 = so2.check()
        return None if r2 == z3.unknown or (r == z3.sat and r2 == z3.sat) else r == z3.sat
    gs = [decide(f) for _, f in S.guards]
    if None in gs: return 'skip', '未定'
    if not all(gs): return 'skip', '守卫'
    u = decide(S.ubf)
    if u is None: return 'skip', '未定'
    if u: return 'skip', 'UB'
    res = []
    for (v, p), t in zip(S.outv, outs):
        if not z3.is_false(sub(p)): return 'skip', '毒值'
        if t in ELEM:
            s = scal(sub(v), t)
            if s is None: return 'skip', '未定'
            res.append(('s', s)); continue
        e_t = tparse(t)[1]
        ex = [sub(e) for e in C.ext(v)]
        if not all(z3.is_int_value(e) for e in ex): return 'skip', '未定'
        ex = [e.as_long() for e in ex]; els = {}; n = 1
        for e in ex: n *= e
        pts = set()
        if n: [pts.add(tuple(rng.randrange(e) for e in ex)) for _ in range(8)]
        for ix in sorted(pts):
            I = tuple(z3.IntVal(i) for i in ix)
            if not z3.is_false(sub(C.sel(C.tpz(v), I))): continue
            s = scal(sub(C.sel(C.dat(v), I)), e_t)
            if s is None: return 'skip', '未定'
            els[ix] = s
        res.append(('t', ex, els))
    return 'val', res

def _in_cast(x, t, out):
    """i64 参数 x → 类型 t 的值，追加指令到 out，返回值名。"""
    if t == 'i64': return x
    if t == 'index': out.append(f'{x}__c = arith.index_cast {x} : i64 to index'); return f'{x}__c'
    if t in FPT:
        w = BW(t)
        if w < 64: out.append(f'{x}__t = arith.trunci {x} : i64 to i{w}'); x = f'{x}__t'
        out.append(f'{x}__f = arith.bitcast {x} : i{w} to {t}'); return f'{x}__f'
    out.append(f'{x}__c = arith.trunci {x} : i64 to {t}'); return f'{x}__c'
def _out_cast(x, t, out):
    if t in FPT:
        w = BW(t); out.append(f'%ob = arith.bitcast {x} : {t} to i{w}'); x = '%ob'; t = f'i{w}'
    if t == 'i64': out.append(f'%o = arith.addi {x}, %c0_i64 : i64')
    elif t == 'index': out.append(f'%o = arith.index_cast {x} : index to i64')
    else: out.append(f'%o = arith.extui {x} : {t} to i64')

def build(ftxt, fname, ins, outs, queries):
    P, L = [], []
    def p():
        n = f'%p{len(P)}'; P.append(n); return n
    args = []
    for k, t in enumerate(ins):
        if t in ELEM: args.append(_in_cast(p(), t, L)); continue
        dims, e = tparse(t); r = len(dims); dyn = []
        for d_i, d in enumerate(dims):
            x = p()
            if d is None: L.append(f'%d{k}_{d_i} = arith.index_cast {x} : i64 to index'); dyn.append(f'%d{k}_{d_i}')
        seed = p(); cs = [p() for _ in dims]
        ivs = ', '.join(f'%i{k}_{j}: index' for j in range(r))
        body = [f'%acc{k}_00 = arith.addi {seed}, %c0_i64 : i64']; cur = f'%acc{k}_00'
        for j in range(r):
            body += [f'%ii{k}_{j} = arith.index_cast %i{k}_{j} : index to i64', f'%m{k}_{j} = arith.muli %ii{k}_{j}, {cs[j]} : i64',
                     f'%acc{k}_{j + 1} = arith.addi {cur}, %m{k}_{j} : i64']; cur = f'%acc{k}_{j + 1}'
        el = _in_cast(cur, e, body)
        L.append(f'%a{k} = tensor.generate ' + ', '.join(dyn) + ' {\n  ^bb0(' + ivs + '):\n    ' + '\n    '.join(body) +
                 f'\n    tensor.yield {el} : {e}\n  }} : {t}')
        args.append(f'%a{k}')
    nq = max([q[3] for q in queries if q[2] == 'el'] + [0])
    Q = [p() for _ in range(nq)]
    for j in range(nq): L.append(f'%q{j} = arith.index_cast {Q[j]} : i64 to index')
    fty = '(' + ', '.join(ins) + ') -> (' + ', '.join(outs) + ')'
    res = ', '.join(f'%r{j}' for j in range(len(outs)))
    call = (f'{res} = ' if outs else '') + f'func.call @{fname}(' + ', '.join(args) + f') : {fty}'
    mains = []
    for nm, j, kind, d in queries:
        t = outs[j]; tail = []
        if kind == 's': _out_cast(f'%r{j}', t, tail)
        elif kind == 'dim': tail += [f'%dc = arith.constant {d} : index', f'%x = tensor.dim %r{j}, %dc : {t}']; _out_cast('%x', 'index', tail)
        else: tail.append(f'%x = tensor.extract %r{j}[' + ', '.join(f'%q{i}' for i in range(d)) + f'] : {t}'); _out_cast('%x', tparse(t)[1], tail)
        tail.append('return %o : i64')
        mains.append(f'func.func @{nm}(' + ', '.join(f'{q}: i64' for q in P) + ') -> i64 attributes {llvm.emit_c_interface} {\n  %c0_i64 = arith.constant 0 : i64\n  '
                     + '\n  '.join(L + [call] + tail) + '\n}')
    return ftxt + '\n' + '\n'.join(mains), len(P), nq

def flat(ins, xs, qidx, nq):
    out = []
    for t, x in zip(ins, xs):
        if t in ELEM: out.append(C64(x)); continue
        ex, seed, cs = x; out += ex + [C64(seed)] + cs
    return out + list(qidx) + [0] * (nq - len(qidx))
def C64(v): v %= 1 << 64; return v - (1 << 64) if v >= 1 << 63 else v

def agree(t, mv, jv): return FJ.agree(t, mv, jv % (1 << 64)) if t in FPT else mv[1] % (1 << BW(t)) == jv % (1 << BW(t))

SOLVE = True
def solve_inputs(S, ins, rng, k=4, dmax=40, bounded=True):
    """守卫 ∧ ¬UB 在洞上求解：动态维 ∈ [0, 8]、index/整数标量符号化；tensor 元素公式与浮点标量取随机具体值。返回至多 k 组输入（逐次阻塞已得解）。"""
    import core as C
    base = mk_inputs(rng, ins); pairs = []; syms = []        # syms：(参数号, 维号|None, z3 变量)
    for a, ((h, hp), t, x) in enumerate(zip(S.holes, ins, base)):
        if t in FPT: z = z3.simplify(z3.fpBVToFP(z3.BitVecVal(x % (1 << BW(t)), BW(t)), z3.FPSort(*FPT[t])))
        elif t == 'index': z = z3.Int(f'_s{a}'); syms.append((a, None, z))
        elif t in IELEM: z = z3.BitVec(f'_s{a}', BW(t)); syms.append((a, None, z))
        else:
            dims, e = tparse(t); ex = []
            for d_i, d in enumerate(dims):
                if d is None: v = z3.Int(f'_s{a}_{d_i}'); syms.append((a, d_i, v)); ex.append(v)
                else: ex.append(z3.IntVal(d))
            tz = ztensor(t, ([0] * len(dims), x[1], x[2]))
            z = C.mkt(C.zsort(t), ex, C.dat(tz), C.tpz(tz))
        pairs += [(h, z), (hp, z3.BoolVal(False))]
    so = z3.Solver(); so.set('timeout', 10000)
    for _, f in S.guards: so.add(z3.substitute(f, pairs))
    so.add(z3.Not(z3.substitute(S.ubf, pairs)))
    for a, d, v in syms:
        if d is not None: so.add(v >= 0, *([v <= dmax] if bounded else []))
        elif z3.is_int(v): so.add(*([v >= -4, v <= 64] if bounded else [v >= -(1 << 63), v < (1 << 63)]))
    out = []
    for _ in range(k):
        if so.check() != z3.sat: break
        mo = so.model()
        xs = [((list(x[0]),) + x[1:]) if isinstance(x, tuple) else x for x in base]
        blk = []
        for a, d, v in syms:
            val = mo.eval(v, model_completion=True).as_long(); blk.append(v != val)
            if d is None: xs[a] = val
            else: xs[a][0][d] = val
        out.append(xs)
        if not blk: break
        so.add(z3.Or(blk))
    return out

def check_fn(S, ftxt, fname, ins, outs, rng, st, note, tag):
    runs = []
    for _ in range(16):                                  # 至多 16 组输入取前 4 组可执行的（首跑只试 4 组：94 个候选只执行 36，主要被守卫跳过；事后改动，只改采样）
        if len(runs) == 4: break
        xs = mk_inputs(rng, ins); r = model_run(S, ins, outs, xs, rng)
        if r[0] == 'skip': st['跳过/' + r[1]] += 1; continue
        runs.append((xs, r[1]))
    if not runs and SOLVE:                               # 第 60b 轮：随机输入全被跳过 ⇒ 求解器构造满足守卫且非 UB 的输入（形状/标量由求解器给，元素公式仍随机）
        for xs in solve_inputs(S, ins, rng):
            r = model_run(S, ins, outs, xs, rng)
            if r[0] == 'skip': st['求解输入/仍被跳过·' + r[1]] += 1; continue
            runs.append((xs, r[1]))
        if runs: st['求解输入/函数'] += 1
        elif solve_inputs(S, ins, rng, 1, bounded=False): st['未执行/界外才有解'] += 1
        else: st['未执行/守卫∧¬UB 恒假（MLIR 恒 UB）'] += 1
    if not runs: st['未执行/所有输入被跳过'] += 1
    if not runs: return None
    queries = []
    for j, t in enumerate(outs):
        if t in ELEM: queries.append((f'q_{j}_s', j, 's', 0))
        else:
            r = len(tparse(t)[0]); queries += [(f'q_{j}_d{d}', j, 'dim', d) for d in range(r)] + [(f'q_{j}_e', j, 'el', r)]
    src, _, nq = build(ftxt, fname, ins, outs, queries)
    calls, keys = [], []
    for ri, (xs, mres) in enumerate(runs):
        for (nm, j, kind, d) in queries:
            m = mres[j]
            if kind == 's': calls.append((nm, flat(ins, xs, [], nq))); keys.append((ri, j, 's', None, m[1]))
            elif kind == 'dim': calls.append((nm, flat(ins, xs, [], nq))); keys.append((ri, j, 'dim', d, ('bits', m[1][d])))
            else:
                for ix, v in m[2].items(): calls.append((nm, flat(ins, xs, ix, nq))); keys.append((ri, j, 'el', ix, v))
    got = TJ.jit_forked(src, calls)
    if isinstance(got, str): st['JIT 不可用' + ('（崩溃）' if got.startswith('崩溃') else '')] += 1; note('JIT 不可用', f'{tag} {got[:160]}'); return None
    st['函数（已执行）'] += 1; st['输入组'] += len(runs)
    for (ri, j, kind, ix, mv), jv in zip(keys, got):
        t = outs[j]; et = t if kind == 's' else ('index' if kind == 'dim' else tparse(t)[1])
        st['比较/' + kind] += 1; st['比较'] += 1
        if et in FPT: st['比较/浮点值'] += 1; st['比较/浮点值·NaN'] += mv[0] == 'nan'
        if agree(et, mv, jv): st['一致'] += 1
        else: st['不一致'] += 1; note('不一致', f'{tag} 结果{j} {kind}{ix} 模型 {mv} JIT {jv % (1 << 64):#x}')
    return runs, keys, got

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, core as C
    from compcorpus_test import _walk, _funcs
    from valimport_test import derive_imported
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
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
            if all(TJ.ok_ty(t) for t in ins + outs): continue                   # 整数签名：T 语料已覆盖
            S, why = derive_imported(f'ft{seed}_{ci}_{fi}', fn)
            if S is None: continue                                              # 判未知：合规（M0.2 “否则未知”），不在本检验分母
            st['有事实且签名含非整数 tensor'] += 1
            if re.search(r'\bf16\b', ft): st['排除/f16（LLVM AArch64 误编译）'] += 1; continue
            if not all(ok_ty(t) for t in ins + outs): st['排除/其它类型：' + ','.join(sorted({re.sub(r'tensor<[^>]*?(\w+)>', r'tensor<…\1>', t) for t in ins + outs if not ok_ty(t)}))[:60]] += 1; continue
            st['候选'] += 1
            name = fn['props']['_attrs']['sym_name'].strip('"')
            ftxt = next((str(o.operation) for o in m.body.operations if o.operation.name == 'func.func' and
                         str(o.operation.attributes['sym_name']).strip('"') == name), None)
            fname = name if re.fullmatch(r'[A-Za-z_][\w.$]*', name) else f'"{name}"'
            tag = f'{path}:{name}'
            res = check_fn(S, ftxt, fname, ins, outs, rng, st, note, tag)
            if not res: continue
            runs, keys, got = res
            for (ri, j, kind, ix, mv), jv in zip(keys, got):                     # 灵敏度：模型侧 seed+1
                if kind != 'el' or not any(t not in ELEM for t in ins): continue
                xs2 = [x if t in ELEM else (x[0], x[1] + 1, x[2]) for t, x in zip(ins, runs[ri][0])]
                r2 = model_run(S, ins, outs, xs2, random.Random(0))
                if r2[0] != 'val' or ix not in r2[1][j][2]: continue
                st['seed+1 比较'] += 1; st['seed+1 发现'] += not agree(tparse(outs[j])[1], r2[1][j][2][ix], jv)
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
        q = ctxm.Queue(); pr = ctxm.Process(target=run_file, args=(path, q, i)); pr.start()
        try: s_, ex = q.get(timeout=600)
        except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; print('  ! 超时/崩溃', path, flush=True); continue
        pr.join(); tot.update(s_)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── FT 语料：浮点元素 tensor 签名函数 值层 vs JIT'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:10]]
    V.check('FT 语料', tot['比较'] >= 300 and tot['不一致'] == 0 and tot['文件超时/崩溃'] == 0,
            f"候选 {tot['候选']}，执行 {tot['函数（已执行）']}，比较 {tot['比较']}（浮点值 {tot['比较/浮点值']}，其中 NaN {tot['比较/浮点值·NaN']}），不一致 {tot['不一致']}")
    V.check('FT 灵敏度', tot['seed+1 比较'] >= 50 and tot['seed+1 发现'] >= 0.5 * tot['seed+1 比较'],
            f"seed+1 发现 {tot['seed+1 发现']}/{tot['seed+1 比较']}")
    V.check('FT 求解输入', tot['求解输入/函数'] >= 24 and tot['不一致'] == 0, f"求解后执行 {tot['求解输入/函数']}，仍未执行 {tot['未执行/所有输入被跳过']}，不一致 {tot['不一致']}")
    V.finish()
