"""第 35 轮：浮点值层 vs MLIR JIT（spec/values.md §10）。
  F1 定向：f16/bf16/f32/f64 标量程序（addf/subf/mulf/divf/remf/negf/maximumf/minimumf/cmpf+select/extf/truncf/
     sitofp/uitofp/fptosi/fptoui/bitcast i→f/常量，fastmath none|nnan|ninf），输入取特殊值（±0、±Inf、qNaN/sNaN、次正规、
     最大有限值、最小正规）与随机位模式。JIT 经包装函数 @w（iN 位模式进出，bitcast 后 func.call @main）执行；
     比较：模型 NaN ⇔ JIT 位模式是 NaN（载荷/符号不比较——LLVM 的 NaN 结果载荷不确定），否则位模式完全相等。
  F2 语料：MLIR 测试集中单块、标量（整数/index/浮点）签名、值层有事实且含浮点的函数，同样包装执行。
  F3 探针（不计入一致性）：truncf 的 roundingmode 在 JIT 中是否生效（值层对它判未知的依据）。
  变异：cmpf 谓词编号异或 1（pred）。
用法：PYTHONPATH=model python3 model/floatjit_test.py [F1 程序数] [语料文件上限]
"""
import sys, re, glob, random, ctypes, collections, multiprocessing as mp
import z3
from valimport_test import derive_imported, PIPE as _P0
PIPE = _P0.replace('convert-arith-to-llvm', 'convert-math-to-llvm,convert-arith-to-llvm')   # 第 35b 轮：math 方言

FPT = {'f16': (5, 11), 'bf16': (8, 8), 'f32': (8, 24), 'f64': (11, 53)}
def BW(t): return 64 if t == 'index' else sum(FPT[t]) if t in FPT else int(t[1:])
CTB = {1: ctypes.c_bool, 8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32, 64: ctypes.c_int64}
def ity(t): return 'i' + str(BW(t)) if t in FPT else ('i64' if t == 'index' else t)

def specials(t):
    if t not in FPT:
        w = BW(t); return [0, 1, 2, 3, 7, -1, -2, (1 << (w - 1)) - 1, -(1 << (w - 1))] if w > 1 else [0, 1]
    e, m = FPT[t]; w = e + m; mb = m - 1; emax = (1 << e) - 1
    one = ((1 << (e - 1)) - 1) << mb
    v = [0, 1 << (w - 1), one, one | (1 << (w - 1)), one | (1 << (mb - 1)), 1, 1 | (1 << (w - 1)), (1 << mb) - 1,
         1 << mb, ((emax - 1) << mb) | ((1 << mb) - 1), emax << mb, (emax << mb) | (1 << (w - 1)),
         (emax << mb) | (1 << (mb - 1)), (emax << mb) | 1, one + (3 << (mb - 2)), (one + (2 << mb)) | 5]
    return v
def rand_in(rng, t):
    if rng.random() < 0.6: return rng.choice(specials(t))
    return rng.getrandbits(BW(t))

# ───────────────────────── 模型求值：位模式 → FP；输出 ('nan',) / ('bits', b) / skip
def model_eval(S, ins_ty, xs, out_ty):
    pairs = []
    for (h, hp), t, x in zip(S.holes, ins_ty, xs):
        if t in FPT: hv = z3.simplify(z3.fpBVToFP(z3.BitVecVal(x % (1 << BW(t)), BW(t)), z3.FPSort(*FPT[t])))
        elif t == 'index': hv = z3.IntVal(x)
        else: hv = z3.BitVecVal(x % (1 << BW(t)), BW(t))
        pairs.append((h, hv)); pairs.append((hp, z3.BoolVal(False)))
    sub = lambda f: z3.simplify(z3.substitute(f, pairs)) if pairs else z3.simplify(f)
    if not all(z3.is_true(sub(f)) for _, f in S.guards): return 'skip', '守卫'
    if not z3.is_false(sub(S.ubf)): return 'skip', 'UB'
    out = []
    for (v, p), t in zip(S.outv, out_ty):
        if not z3.is_false(sub(p)): return 'skip', '毒值'
        vv = sub(v)
        if t in FPT:
            if z3.is_true(z3.simplify(z3.fpIsNaN(vv))): out.append(('nan',)); continue
            b = z3.simplify(z3.fpToIEEEBV(vv))
            if not z3.is_bv_value(b): return 'skip', '未定'
            out.append(('bits', b.as_long()))
        elif z3.is_bv_value(vv): out.append(('bits', vv.as_long()))
        elif z3.is_int_value(vv): out.append(('bits', vv.as_long() % (1 << 64)))
        else: return 'skip', '未定'
    return 'val', out
def is_nan_bits(b, t):
    e, m = FPT[t]; mb = m - 1
    return (b >> mb) & ((1 << e) - 1) == (1 << e) - 1 and b & ((1 << mb) - 1) != 0
def agree(t, mv, jb):
    jb %= 1 << BW(t)
    if mv[0] == 'nan': return t in FPT and is_nan_bits(jb, t)
    return mv[1] % (1 << BW(t)) == jb

# ───────────────────────── JIT：@w 包装（位模式进出）
def wrap(fn_txt, name, ins_ty, outs_ty):
    """每个结果一个包装 @__wJ（单结果：多结果的 C 接口 sret 打包在 ExecutionEngine.invoke 下会挂起/崩溃，第 35 轮踩过）。"""
    a = ', '.join(f'%x{i}: {ity(t)}' for i, t in enumerate(ins_ty)); txt = fn_txt
    for j, tj in enumerate(outs_ty):
        L, args = [], []
        for i, t in enumerate(ins_ty):
            if t in FPT: L.append(f'%y{i} = arith.bitcast %x{i} : {ity(t)} to {t}'); args.append(f'%y{i}')
            elif t == 'index': L.append(f'%y{i} = arith.index_cast %x{i} : i64 to index'); args.append(f'%y{i}')
            else: args.append(f'%x{i}')
        n = len(outs_ty)
        L.append(f'%r{":" + str(n) if n > 1 else ""} = func.call @{name}({", ".join(args)}) : ({", ".join(ins_ty)}) -> ({", ".join(outs_ty)})')
        r = f'%r#{j}' if n > 1 else '%r'
        if tj in FPT: L.append(f'%z = arith.bitcast {r} : {tj} to {ity(tj)}'); r = '%z'
        elif tj == 'index': L.append(f'%z = arith.index_cast {r} : index to i64'); r = '%z'
        L.append(f'return {r} : {ity(tj)}')
        txt += f'\nfunc.func @__w{j}({a}) -> {ity(tj)} {{\n  ' + '\n  '.join(L) + '\n}\n'
    return txt

def _jit_child(text, ins_ty, outs_ty, inputs, q):
    try:
        import mlir.ir as ir, os
        from mlir.passmanager import PassManager
        from mlir.execution_engine import ExecutionEngine
        with ir.Context() as c:
            c.enable_multithreading(False)
            m = ir.Module.parse(text)
            for op in m.body.operations:
                if op.operation.name == 'func.func': op.operation.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
            PassManager.parse(PIPE).run(m.operation)
            lib = os.path.join(os.path.dirname(ir.__file__), '_mlir_libs', 'libmlir_c_runner_utils.so')
            ee = ExecutionEngine(m, opt_level=0, shared_libs=[lib])
            res = []
            for xs in inputs:
                a = [ctypes.pointer(CTB[BW(t)](x if BW(t) > 1 else bool(x & 1))) for t, x in zip(ins_ty, xs)]
                row = []
                for j, t in enumerate(outs_ty):
                    r = CTB[BW(t)](0); ee.invoke(f'__w{j}', *a, ctypes.pointer(r)); row.append(int(r.value))
                res.append(row)
            q.put(('ok', res))
    except Exception as e: q.put(('err', str(e)[:200]))
def jit(text, ins_ty, outs_ty, inputs):
    ctxm = mp.get_context('fork'); q = ctxm.Queue(); p = ctxm.Process(target=_jit_child, args=(text, ins_ty, outs_ty, inputs, q)); p.start()
    try: r = q.get(timeout=120)
    except Exception: r = ('crash', f'退出码 {p.exitcode}')
    p.join(5)
    if p.is_alive(): p.kill()
    return r

# ───────────────────────── F1 生成器
BIN = ['addf', 'subf', 'mulf', 'divf', 'remf', 'maximumf', 'minimumf']
PRED = ['false', 'oeq', 'ogt', 'oge', 'olt', 'ole', 'one', 'ord', 'ueq', 'ugt', 'uge', 'ult', 'ule', 'une', 'uno', 'true']
LIT = {'f16': ['1.5', '-0.0', '0.1', '6.550400e+04', '0x7C00', '0xFE00', '5.960464e-08'], 'bf16': ['1.5', '-0.0', '0.1', '3.389531e+38', '0x7F80', '0xFFC1'],
       'f32': ['1.5', '-0.0', '0.1', '3.40282347E+38', '0x7F800000', '0xFFC00001', '1.401298e-45', '16777217.0'],
       'f64': ['1.5', '-0.0', '0.1', '1.7976931348623157E+308', '0x7FF0000000000000', '4.940656e-324', '9007199254740993.0']}
MUN = ['absf', 'ceil', 'floor', 'trunc', 'round', 'roundeven', 'sqrt']
MATH = True
WIDER = {'f16': ['f32', 'f64'], 'bf16': ['f32', 'f64'], 'f32': ['f64'], 'f64': []}
def gen(rng):
    t = rng.choice(list(FPT)); vs = ['%a', '%b', '%c']; L = []; k = [0]
    def nv():
        k[0] += 1; return f'%v{k[0]}'
    def fm():
        r = rng.random()
        return '' if r < 0.7 else ' fastmath<' + rng.choice(['nnan', 'ninf', 'nnan,ninf', 'none']) + '>'
    for _ in range(rng.randrange(1, 5)):
        x, y = rng.choice(vs), rng.choice(vs); o = nv(); r = rng.random()
        if r < 0.45: L.append(f'{o} = arith.{rng.choice(BIN)} {x}, {y}{fm()} : {t}')
        elif r < 0.6:
            c = nv(); L.append(f'{c} = arith.cmpf {rng.choice(PRED)}, {x}, {y}{fm()} : {t}'); L.append(f'{o} = arith.select {c}, {x}, {y} : {t}')
        elif r < 0.67:
            if rng.random() < 0.25: L.append(f'{o} = arith.negf {x}{fm()} : {t}')
            else: L.append(f'{o} = math.{rng.choice(MUN)} {x}{fm()} : {t}')
        elif r < 0.71 and MATH:
            if rng.random() < 0.5: L.append(f'{o} = math.fma {x}, {y}, {rng.choice(vs)}{fm()} : {t}')
            else:
                c = nv(); L.append(f'{c} = math.{rng.choice(["isnan", "isfinite"])} {x} : {t}'); L.append(f'{o} = arith.select {c}, {x}, {y} : {t}')   # 汇编格式不带 fastmath
        elif r < 0.75:
            c = nv(); L.append(f'{c} = arith.constant {rng.choice(LIT[t])} : {t}'); L.append(f'{o} = arith.{rng.choice(BIN[:4])} {x}, {c} : {t}')
        elif r < 0.85 and WIDER[t]:
            w = rng.choice(WIDER[t]); e = nv()
            L.append(f'{e} = arith.extf {x} : {t} to {w}')
            if rng.random() < 0.5: e2 = nv(); L.append(f'{e2} = arith.mulf {e}, {e} : {w}'); e = e2
            L.append(f'{o} = arith.truncf {e} : {w} to {t}')
        elif r < 0.85:
            L.append(f'{o} = arith.addf {x}, {y} : {t}')
        else:
            it = rng.choice(['i8', 'i16', 'i32', 'i64']); s = rng.choice(['si', 'ui']); e = nv()
            L.append(f'{e} = arith.fpto{s} {x} : {t} to {it}')
            if rng.random() < 0.3: e2 = nv(); L.append(f'{e2} = arith.addi {e}, %i : {it}'.replace('%i', e)); e = e2
            if MATH and rng.random() < 0.5: e2 = nv(); L.append(f'{e2} = math.{rng.choice(["absi", "ctlz", "cttz", "ctpop"])} {e} : {it}'); e = e2
            L.append(f'{o} = arith.{rng.choice(["si", "ui"])}tofp {e} : {it} to {t}')
        vs.append(o)
    body = '\n  '.join(L)
    return f'func.func @main(%a: {t}, %b: {t}, %c: {t}) -> {t} {{\n  {body}\n  return {vs[-1]} : {t}\n}}', [t] * 3, [t]

# 第 35b 轮：LLVM AArch64 误编译（本提交）——`sitofp (fptosi f16 → i32/i64) → f16` 被模式 SCVTFv1i16∘FCVTZSv1f16 选成
# 16 位有符号中间值（饱和于 ±32768）；MLIR 发出的 llvm.fptosi/llvm.sitofp 正确。归因 = A/B：插入不可内联的恒等调用隔断模式后
# 同一输入的 JIT 结果与模型一致，才计入“已归因”，否则仍是不一致。
F16PAT = re.compile(r'(%[\w#]+) = arith\.sitofp (%[\w#]+) : (i32|i64) to f16')
def barrier(txt):
    ws = set()
    def rep(m):
        ws.add(m.group(3)); return f'{m.group(2)}__o = func.call @__id_{m.group(3)}({m.group(2)}) : ({m.group(3)}) -> {m.group(3)}\n  {m.group(1)} = arith.sitofp {m.group(2)}__o : {m.group(3)} to f16'
    t2 = F16PAT.sub(rep, txt)
    return t2 + ''.join(f'\nfunc.func private @__id_{w}(%x: {w}) -> {w} attributes {{no_inline}} {{\n  return %x : {w}\n}}' for w in ws)
def run_prog(S, txt, name, ins, outs, rng, st, note, tag, S2=None, ninp=10):
    inputs = [[rand_in(rng, t) for t in ins] for _ in range(ninp)]
    ev = [model_eval(S, ins, xs, outs) for xs in inputs]
    run = [(xs, r[1]) for xs, r in zip(inputs, ev) if r[0] == 'val']
    for r in ev:
        if r[0] == 'skip': st['跳过/' + r[1]] += 1
    if not run: return
    jr = jit(wrap(txt, name, ins, outs), ins, outs, [xs for xs, _ in run])
    if jr[0] != 'ok': st['JIT 不可用/' + jr[0]] += 1; note('JIT 不可用', f'{tag} {jr[1]}'); return
    st['函数（已执行）'] += 1
    jb = None
    if F16PAT.search(txt) and any(not all(agree(t, a, b) for t, a, b in zip(outs, mv, jv)) for (_, mv), jv in zip(run, jr[1])):
        rb = jit(wrap(barrier(txt), name, ins, outs), ins, outs, [xs for xs, _ in run]); jb = rb[1] if rb[0] == 'ok' else None
    for k, ((xs, mv), jv) in enumerate(zip(run, jr[1])):
        for j, (t, a, b) in enumerate(zip(outs, mv, jv)):
            st['比较'] += 1; ok = agree(t, a, b)
            if not ok and jb is not None and agree(t, a, jb[k][j]):
                st['不一致·已归因 LLVM AArch64 f16 fptosi→sitofp 误编译（A/B 隔断后一致）'] += 1; continue
            st['一致' if ok else '不一致'] += 1
            if a[0] == 'nan': st['其中 NaN 结果'] += 1
            if not ok: note('不一致', f'{tag} 输入 {[hex(x % (1 << BW(u))) for x, u in zip(xs, ins)]} 模型 {a if a[0] == "nan" else hex(a[1])} JIT {hex(b % (1 << BW(t)))}')
    if S2 is not None:
        for (xs, _), jv in zip(run, jr[1]):
            r2 = model_eval(S2, ins, xs, outs)
            if r2[0] == 'val':
                st['变异 pred 比较'] += 1; st['变异 pred 发现'] += not all(agree(t, a, b) for t, a, b in zip(outs, r2[1], jv))

def F1(n, seed=17):
    import mlir.ir as ir, irimport as I
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    for i in range(n):
        txt, ins, outs = gen(rng)
        try: m = ir.Module.parse(txt, ir.Context()); assert m.operation.verify()
        except Exception as e: st['不合法（跳过）'] += 1; note('不合法', str(e)[:80] + '\n' + txt); continue
        fn = I.import_module(m)['regions'][0][0]['ops'][0]
        S, why = derive_imported(f'f1_{i}', fn)
        if S is None: st['未知/' + why[:40]] += 1; note('未知', why); continue
        st['程序'] += 1; st['程序/' + ins[0]] += 1
        S2 = derive_imported(f'f1_{i}_p', fn, 'pred')[0] if 'cmpf' in txt else None
        run_prog(S, txt, 'main', ins, outs, rng, st, lambda k, v: note(k, v + '\n' + txt), f'f1_{i}', S2)
    return st, ex

# ───────────────────────── F3 探针：truncf roundingmode
def F3():
    out = {}
    for rm in ('to_nearest_even', 'downward', 'upward', 'toward_zero', 'to_nearest_away'):
        txt = f'func.func @main(%a: f64) -> f32 {{\n  %r = arith.truncf %a {rm} : f64 to f32\n  return %r : f32\n}}'
        x = 0x3FF0000010000000                      # 1 + 2^-24·…：不能精确表示为 f32，RNE 向下、upward 向上
        jr = jit(wrap(txt, 'main', ['f64'], ['f32']), ['f64'], ['f32'], [[x], [x | (1 << 63)]])
        out[rm] = jr if jr[0] != 'ok' else [hex(v[0] % (1 << 32)) for v in jr[1]]
    return out

# ───────────────────────── F2 语料
def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, core as C
    from compcorpus_test import _walk, _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    covered = set(C.REG) | {'scf.for', 'scf.if', 'scf.yield'}
    scal = lambda t: t in FPT or t == 'index' or re.fullmatch(r'i(1|8|16|32|64)', t)
    for ci, ch in enumerate(I.split_chunks(src)):
        if not ch.strip() or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fi, fn in enumerate(_funcs(root)):
            blocks = fn['regions'][0]; ft = fn['props']['_attrs'].get('function_type')
            if len(blocks) != 1 or not ft: continue
            mm = re.fullmatch(r'\((.*)\) -> \(?([^()]*?)\)?', ft)
            ins = [t for _, t in blocks[0]['args']]
            outs = [x.strip() for x in mm.group(2).split(',')] if mm and mm.group(2).strip() else []
            if not outs or not all(scal(t) for t in ins + outs): continue
            if not any(t in FPT for t in ins + outs) and not any(re.search(r'\b(bf16|f16|f32|f64)\b', str(x['outs'])) for x in _walk(fn)): continue
            st['含浮点的标量签名函数'] += 1
            if any(x['op'] not in covered and x['op'] != 'func.return' for x in _walk(fn) if x is not fn):
                st['含值层未覆盖 op'] += 1
                for x in _walk(fn):
                    if x is not fn and x['op'] not in covered and x['op'] != 'func.return': st['  未覆盖/' + x['op']] += 1
                continue
            S, why = derive_imported(f'f2_{seed}_{ci}_{fi}', fn)
            if S is None: st['未知/' + re.sub(r'%\w+', '%', why)[:44]] += 1; note('未知', f'{path} {why[:90]}'); continue
            st['有事实'] += 1
            name = fn['props']['_attrs']['sym_name'].strip('"')
            ftxt = next((str(o.operation) for o in m.body.operations if o.operation.name == 'func.func' and
                         str(o.operation.attributes['sym_name']).strip('"') == name), None)
            S2 = derive_imported(f'f2_{seed}_{ci}_{fi}_p', fn, 'pred')[0] if 'cmpf' in ftxt else None
            run_prog(S, ftxt, name, ins, outs, rng, st, lambda k, v: note(k, v + '\n' + ftxt[:400]), f'{path}:{name}', S2, ninp=16)
    q.put((st, dict(ex)))

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    import verdict as V
    f3 = F3(); print('── F3 truncf roundingmode 探针（输入 ±(1+2^-24)，恰在两 f32 中点）:'); [print('   ', k, v) for k, v in f3.items()]
    vals = [tuple(v) for v in f3.values()]
    V.check('F3 truncf roundingmode', len(vals) >= 2 and len(set(vals)) == 1, f'{len(vals)} 种舍入模式 JIT 结果相同 {vals[0]} ⇒ valimport 对非默认 roundingmode 判未知（A0 探针另验）')
    if n:
        st, ex = F1(n)
        print('── F1 浮点定向:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
        att = sum(v for k, v in st.items() if k.startswith('不一致·已归因'))
        V.check('F1 浮点定向', st['比较'] > 0 and st['不一致'] == 0 and st.get('变异 pred 发现', 1) > 0, f"比较 {st['比较']}，一致 {st['一致']}，不一致 {st['不一致']}，A/B 归因 {att}")
        V.check('MUN =', MATH and st['不一致'] == 0 and st['一致'] > 0, 'math 精确子集混入 F1 生成器（MATH=True），不单独计数')
        V.check('barrier(', st['不一致'] == 0 and att >= 1, f'A/B 隔断归因 {att} 例：插入不可内联恒等调用后与模型一致')
        for k, v in ex.items(): print('──', k); [print('    ', x) for x in v]
    if lim:
        tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
        for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
            q = ctxm.Queue(); pr = ctxm.Process(target=run_file, args=(path, q, i)); pr.start()
            try: s, ex = q.get(timeout=900)
            except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; continue
            pr.join(); tot.update(s)
            for k, v in ex.items(): exs[k] += v[:2]
        print('── F2 语料（含浮点的标量签名函数）:'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
        V.check('F2 语料', tot['比较'] > 0 and tot['不一致'] == 0 and tot['文件超时/崩溃'] == 0, f"比较 {tot['比较']}，不一致 {tot['不一致']}")
        for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:6]]
    V.finish()
