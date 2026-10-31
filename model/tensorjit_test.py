"""G1.4（第 32 轮）：tensor 签名的语料函数，值层 vs 同一提交 MLIR 的 JIT。
此前 A2 只对照标量签名（valimport_test）。这里把签名里有 ranked 整数 tensor 的语料函数也纳入：
  - 输入 tensor 用 tensor.generate 按公式 e(i) = trunc(seed + Σ c_k·i_k) 生成（动态维随机取 0..3），
    模型侧用同一公式构造具体的 z3 tensor 值（extent、Lambda 元素、元素毒值全假）代入洞；
  - 外包 main 只有 i64 参数：标量结果直接返回，tensor 结果先比较 tensor.dim，再在模型给出的 extent 内抽样下标比较元素；
  - 模型判 UB/毒值/守卫不成立的输入跳过，毒值元素跳过（JIT 不可观测）。
同时统计：tensor 签名函数中有多少因值层未建模而判未知（按原因），这就是 G1.4 的分母缺口。
灵敏度：valimport 的 pred / static_zero 变异必须被发现（只在本函数上）。
第 32 轮还修了 core：无秩 tensor、浮点元素、带 encoding 的 tensor 作为操作数/结果时，此前以 z3/Python 异常告终（被当作“core 异常”判未知，
  路径正确但原因不可审计），现在显式 ValueUnknown。
用法：PYTHONPATH=model python3 model/tensorjit_test.py [文件数上限]
"""
import sys, re, glob, random, ctypes, collections, os, multiprocessing as mp
import z3

ELEM = ('i1', 'i8', 'i16', 'i32', 'i64', 'index')
def W(t): return 64 if t == 'index' else int(t[1:])
def tparse(t):
    m = re.fullmatch(r'tensor<((?:(?:\d+|\?)x)+)(\w+)>', t)
    if not m or m.group(2) not in ELEM: return None
    return [None if d == '?' else int(d) for d in m.group(1).rstrip('x').split('x')], m.group(2)
def ok_ty(t): return t in ELEM or tparse(t) is not None

def mk_inputs(rng, ins):
    """每个参数：标量 → 值；tensor → (extent, seed, coeffs)。"""
    from valimport_test import consts
    xs = []
    for t in ins:
        if t in ELEM: xs.append(rng.choice(consts(t)))
        else:
            dims, e = tparse(t); ex = [d if d is not None else rng.randrange(0, 4) for d in dims]
            xs.append((ex, rng.randrange(-8, 9), [rng.randrange(-3, 4) for _ in dims]))
    return xs

def ztensor(t, x):
    import core as C
    dims, e = tparse(t); ex, seed, cs = x; S_ = C.zsort(t)
    iv = [z3.Int(f'_g{k}') for k in range(len(dims))]
    lin = seed + z3.Sum([c * i for c, i in zip(cs, iv)]) if iv else z3.IntVal(seed)
    val = lin if e == 'index' else z3.Int2BV(lin, W(e))
    return C.mkt(S_, [z3.IntVal(d) for d in ex], z3.Lambda(iv, val), z3.Lambda(iv, z3.BoolVal(False)))

def model_run(S, ins, outs, xs, rng):
    """返回 ('skip', 原因) 或 ('val', [每个结果：('s', 值) | ('t', extent, {下标: 值})])。"""
    import core as C
    pairs = []
    for (h, hp), t, x in zip(S.holes, ins, xs):
        if t in ELEM: z = z3.IntVal(C.sx(x % (1 << 64), 64)) if t == 'index' else z3.BitVecVal(x % (1 << W(t)), W(t))
        else: z = ztensor(t, x)
        pairs += [(h, z), (hp, z3.BoolVal(False))]
    sub = lambda f: z3.simplify(z3.substitute(f, pairs)) if pairs else z3.simplify(f)
    if not all(z3.is_true(sub(f)) for _, f in S.guards): return 'skip', '守卫'
    if not z3.is_false(sub(S.ubf)): return 'skip', 'UB'
    res = []
    for (v, p), t in zip(S.outv, outs):
        if not z3.is_false(sub(p)): return 'skip', '毒值'
        if t in ELEM:
            vv = sub(v)
            if not (z3.is_int_value(vv) or z3.is_bv_value(vv)): return 'skip', '未定'
            res.append(('s', vv.as_long())); continue
        ex = [sub(e) for e in C.ext(v)]
        if not all(z3.is_int_value(e) for e in ex): return 'skip', '未定'
        ex = [e.as_long() for e in ex]; els = {}
        n = 1
        for e in ex: n *= e
        pts = set()
        if n: [pts.add(tuple(rng.randrange(e) for e in ex)) for _ in range(8)]
        for ix in sorted(pts):
            I = tuple(z3.IntVal(i) for i in ix)
            pz = sub(C.tpz(v)[I] if I else C.tpz(v))
            if not z3.is_false(pz): continue                         # 毒值元素：JIT 不可观测
            vv = sub(C.dat(v)[I])
            if not (z3.is_int_value(vv) or z3.is_bv_value(vv)): return 'skip', '未定'
            els[ix] = vv.as_long()
        res.append(('t', ex, els))
    return 'val', res

def to64(x, t):
    if t == 'i64': return x
    if t == 'index': return f'arith.index_cast {x} : index to i64'
    return f'arith.extui {x} : {t} to i64'

def build(ftxt, fname, ins, outs, queries):
    """queries：[(名字, 结果号, 种类 's'|'dim'|'el', 维号/秩)] → 每个一个 main_*(i64...)->i64。参数布局见 flat()。"""
    P, L = [], []
    def p():
        n = f'%p{len(P)}'; P.append(n); return n
    args = []
    for k, t in enumerate(ins):
        if t in ELEM:
            x = p()
            if t == 'i64': args.append(x)
            elif t == 'index': L.append(f'%a{k} = arith.index_cast {x} : i64 to index'); args.append(f'%a{k}')
            else: L.append(f'%a{k} = arith.trunci {x} : i64 to {t}'); args.append(f'%a{k}')
            continue
        dims, e = tparse(t); r = len(dims)
        dyn = []
        for d_i, d in enumerate(dims):
            x = p()
            if d is None: L.append(f'%d{k}_{d_i} = arith.index_cast {x} : i64 to index'); dyn.append(f'%d{k}_{d_i}')
        seed = p(); cs = [p() for _ in dims]
        ivs = ', '.join(f'%i{k}_{j}: index' for j in range(r))
        body = [f'%acc{k}_0 = arith.addi {seed}, {seed} : i64', f'%acc{k}_00 = arith.subi %acc{k}_0, {seed} : i64']
        cur = f'%acc{k}_00'
        for j in range(r):
            body += [f'%ii{k}_{j} = arith.index_cast %i{k}_{j} : index to i64', f'%m{k}_{j} = arith.muli %ii{k}_{j}, {cs[j]} : i64',
                     f'%acc{k}_{j + 1} = arith.addi {cur}, %m{k}_{j} : i64']; cur = f'%acc{k}_{j + 1}'
        if e == 'index': body.append(f'%el{k} = arith.index_cast {cur} : i64 to index')
        elif e == 'i64': body.append(f'%el{k} = arith.addi {cur}, %c0_i64 : i64')
        else: body.append(f'%el{k} = arith.trunci {cur} : i64 to {e}')
        L.append(f'%a{k} = tensor.generate ' + ', '.join(dyn) + ' {\n  ^bb0(' + ivs + '):\n    ' + '\n    '.join(body) +
                 f'\n    tensor.yield %el{k} : {e}\n  }} : {t}')
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
        if kind == 's': x = f'%r{j}'; et = t
        elif kind == 'dim': tail.append(f'%dc = arith.constant {d} : index'); tail.append(f'%x = tensor.dim %r{j}, %dc : {t}'); x, et = '%x', 'index'
        else: tail.append(f'%x = tensor.extract %r{j}[' + ', '.join(f'%q{i}' for i in range(d)) + f'] : {t}'); x, et = '%x', tparse(t)[1]
        cv = to64(x, et)
        tail += [f'%o = {cv}' if cv != x else f'%o = arith.addi {x}, %c0_i64 : i64', 'return %o : i64']
        mains.append(f'func.func @{nm}(' + ', '.join(f'{q}: i64' for q in P) + ') -> i64 attributes {llvm.emit_c_interface} {\n  %c0_i64 = arith.constant 0 : i64\n  '
                     + '\n  '.join(L + [call] + tail) + '\n}')
    return ftxt + '\n' + '\n'.join(mains), len(P), nq

def flat(ins, xs, qidx, nq):
    out = []
    for t, x in zip(ins, xs):
        if t in ELEM: out.append(x); continue
        ex, seed, cs = x; out += ex + [seed] + cs
    return out + list(qidx) + [0] * (nq - len(qidx))

def jit_eval(src, fns_and_args):
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    from valimport_test import PIPE as P0
    PIPE = P0.replace('bufferize-function-boundaries', 'bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map').replace('convert-bufferization-to-memref,', 'convert-bufferization-to-memref,func.func(convert-linalg-to-loops),')   # tensor.generate 缓冲化为 linalg.map； 函数边界用恒等布局：否则参数的 strided 布局 cast 降不到 LLVM
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse(PIPE).run(m.operation)
        lib = os.path.join(os.path.dirname(ir.__file__), '_mlir_libs', 'libmlir_c_runner_utils.so')
        ee = ExecutionEngine(m, opt_level=0, shared_libs=[lib]); out = []
        for fn, a in fns_and_args:
            ps = [ctypes.pointer(ctypes.c_int64(v)) for v in a]; r = ctypes.c_int64(0)
            ee.invoke(fn, *ps, ctypes.pointer(r)); out.append(r.value)
        return out

def check_fn(S, ftxt, fname, ins, outs, rng, st, note, tag, extra=()):
    """对一个函数做 4 组输入；返回 [(输入, 模型结果)] 供变异复用。"""
    runs = []
    for _ in range(4):
        xs = mk_inputs(rng, ins); r = model_run(S, ins, outs, xs, rng)
        if r[0] == 'skip': st['跳过/' + r[1]] += 1; continue
        runs.append((xs, r[1]))
    if not runs: return []
    queries = []
    for j, t in enumerate(outs):
        if t in ELEM: queries.append((f'q_{j}_s', j, 's', 0))
        else:
            r = len(tparse(t)[0]); queries += [(f'q_{j}_d{d}', j, 'dim', d) for d in range(r)] + [(f'q_{j}_e', j, 'el', r)]
    src, npar, nq = build(ftxt, fname, ins, outs, queries)
    calls, keys = [], []
    for ri, (xs, mres) in enumerate(runs):
        for (nm, j, kind, d) in queries:
            m = mres[j]
            if kind == 's': calls.append((nm, flat(ins, xs, [], nq))); keys.append((ri, j, 's', None, m[1]))
            elif kind == 'dim': calls.append((nm, flat(ins, xs, [], nq))); keys.append((ri, j, 'dim', d, m[1][d]))
            else:
                for ix, v in m[2].items(): calls.append((nm, flat(ins, xs, ix, nq))); keys.append((ri, j, 'el', ix, v))
    try: got = jit_eval(src, calls)
    except Exception as e: st['JIT 不可用'] += 1; note('JIT 不可用', f'{tag} {str(e)[:120]}'); return []
    st['函数（已执行）'] += 1; st['输入组'] += len(runs)
    for (ri, j, kind, ix, mv), jv in zip(keys, got):
        t = outs[j]; w = W(t if kind == 's' else ('index' if kind == 'dim' else tparse(t)[1]))
        st['比较/' + kind] += 1
        if mv % (1 << w) == jv % (1 << w): st['一致'] += 1
        else: st['不一致'] += 1; note('不一致', f'{tag} 结果{j} {kind}{ix} 模型 {mv} JIT {jv}')
    return [(runs, keys, got)]

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
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fi, fn in enumerate(_funcs(root)):
            blocks = fn['regions'][0]; ft = fn['props']['_attrs'].get('function_type')
            if len(blocks) != 1 or not ft or not re.search(r'tensor<', ft): continue
            st['tensor 签名函数（单块）'] += 1
            if any(x['op'] not in covered and x['op'] != 'func.return' for x in _walk(fn) if x is not fn):
                st['含值层未覆盖 op'] += 1; continue
            ins = [t for _, t in blocks[0]['args']]
            mm = re.fullmatch(r'\((.*)\) -> \(?([^()]*?)\)?', ft)
            outs = [x.strip() for x in mm.group(2).split(',')] if mm and mm.group(2).strip() else []
            S, why = derive_imported(f'tj{seed}_{ci}_{fi}', fn)
            if S is None: st['未知/' + re.sub(r'[：:].*', '', why.split('：', 1)[-1] if why.startswith('core') else why)[:36]] += 1; note('未知', f'{path} {why[:90]}'); continue
            if not all(ok_ty(t) for t in ins + outs): st['有事实但签名含非整数 tensor'] += 1; continue
            st['有事实（候选）'] += 1
            name = fn['props']['_attrs']['sym_name'].strip('"')
            ftxt = next((str(o.operation) for o in m.body.operations if o.operation.name == 'func.func' and
                         str(o.operation.attributes['sym_name']).strip('"') == name), None)
            fname = name if re.fullmatch(r'[A-Za-z_][\w.$]*', name) else f'"{name}"'
            tag = f'{path}:{name}'
            res = check_fn(S, ftxt, fname, ins, outs, rng, st, note, tag)
            if res:                                                                 # 比较器灵敏度：模型侧输入 seed+1（tensor 元素整体偏移），元素比较必须发现
                runs, keys, got = res[0]
                for (ri, j, kind, ix, mv), jv in zip(keys, got):
                    if kind != 'el' or not any(t not in ELEM for t in ins): continue
                    xs2 = [x if t in ELEM else (x[0], x[1] + 1, x[2]) for t, x in zip(ins, runs[ri][0])]
                    r2 = model_run(S, ins, outs, xs2, random.Random(0))
                    if r2[0] != 'val' or ix not in r2[1][j][2]: continue
                    w = W(tparse(outs[j])[1]); st['变异 seed+1 比较'] += 1; st['变异 seed+1 发现'] += r2[1][j][2][ix] % (1 << w) != jv % (1 << w)
            for flip in ('pred', 'static_zero'):                                    # 变异：用同一组输入与 JIT 结果
                S2, _ = derive_imported(f'tj{seed}_{ci}_{fi}_{flip}', fn, flip)
                if S2 is None or not res: continue
                runs, keys, got = res[0]
                for (ri, j, kind, ix, mv), jv in zip(keys, got):
                    r2 = model_run(S2, ins, outs, runs[ri][0], random.Random(0))
                    if r2[0] != 'val': continue
                    m2 = r2[1][j]
                    v2 = m2[1] if kind == 's' else (m2[1][ix] if kind == 'dim' else m2[2].get(ix))
                    if v2 is None: continue
                    t = outs[j]; w = W(t if kind == 's' else ('index' if kind == 'dim' else tparse(t)[1]))
                    st[f'变异 {flip} 比较'] += 1; st[f'变异 {flip} 发现'] += v2 % (1 << w) != jv % (1 << w)
    q.put((st, dict(ex)))

if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
        q = ctxm.Queue(); pr = ctxm.Process(target=run_file, args=(path, q, i)); pr.start()
        try: s, ex = q.get(timeout=600)
        except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; print('  ! 超时/崩溃', path, flush=True); continue
        pr.join(); tot.update(s)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── T 语料 tensor 签名函数 值层 vs JIT'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:12]]
