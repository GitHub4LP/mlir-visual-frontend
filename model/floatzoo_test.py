"""第 58 轮：M5.1 管线白名单扩到浮点程序族（L55 的未覆盖项）。判据同 pipezoo：Sem(g) ⊑ Sem(pipeline(⟦g⟧))。

程序族 F：floatjit_test.gen（f16/bf16/f32/f64；addf/subf/mulf/divf/remf/maximumf/minimumf/negf、cmpf+select、math 一元/fma/isnan/isfinite、
  常量、extf/truncf、fptosi/fptoui→整数 math→sitofp/uitofp；fastmath none|nnan|ninf 随机）。模型 = valimport.derive_imported。
比较 = floatjit.agree：模型 NaN ⇔ JIT 是 NaN（载荷不比较），否则位模式相等。
排除（运行前写定）：含 `sitofp i32/i64 → f16` 的程序不进分母（本机 aarch64，LLVM f16 误编译 35b 会污染基线与归因），单列计数。
管线清单（运行前写定，PIPES_F）：pipezoo 中与浮点相关的通用 pass，加浮点专用 pass（arith-expand 两种、math-expand-ops、math-uplift-to-fma、
  arith-emulate-unsupported-floats）。每个程序在 fork 子进程里跑全部管线；管线作用于含包装函数的整个模块，再接 floatjit.PIPE 降低。
分类（A/B 归因）：模型反例（基线 O0 ≠ 模型）；变换偏离（基线 = 模型、管线后 ≠ 模型，同时记基线 O3 是否 = 模型）；管线失败单列不计分母。
阈值（运行前写定）：
  FZ0 基线：比较 ≥ 1500，模型反例 0。
  FZ1 每条可用管线执行 ≥ 300 个程序，偏离全部可归因（基线 = 模型）；偏离 > 0 的管线排出白名单（不判失败），须在 spec 写明。
用法：PYTHONPATH=model python3 model/floatzoo_test.py [程序数]
"""
import sys, os, random, ctypes, collections, pickle, traceback
import floatjit_test as FJ, verdict as V

PIPES_F = ['', 'inline', 'canonicalize', 'cse', 'sccp', 'loop-invariant-code-motion', 'control-flow-sink', 'symbol-dce',
           'func.func(int-range-optimizations)', 'func.func(arith-expand)', 'func.func(arith-expand{include-bf16=true})',
           'func.func(math-expand-ops)', 'func.func(math-uplift-to-fma)',
           'func.func(arith-emulate-unsupported-floats{source-types=bf16,f16 target-type=f32})',
           'inline,canonicalize,cse,sccp,canonicalize']

def available():
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    ok = []
    with ir.Context():
        for p in PIPES_F[1:]:
            try: PassManager.parse(f'builtin.module({p})'); ok.append(p)
            except Exception: pass
    return [''] + ok

def run_all(text, ins_ty, outs_ty, inputs, pipes, opts=(0, 3)):
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    lib = None; out = {}
    for p in pipes:
        for opt in (opts if p == '' else (0,)):
            try:
                with ir.Context() as c:
                    c.enable_multithreading(False)
                    m = ir.Module.parse(text)
                    for op in m.body.operations:
                        if op.operation.name == 'func.func': op.operation.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
                    if p: PassManager.parse(f'builtin.module({p})').run(m.operation)
                    if not m.operation.verify(): out[(p, opt)] = 'fail:verify'; continue
                    PassManager.parse(FJ.PIPE).run(m.operation)
                    lib = lib or os.path.join(os.path.dirname(ir.__file__), '_mlir_libs', 'libmlir_c_runner_utils.so')
                    ee = ExecutionEngine(m, opt_level=opt, shared_libs=[lib]); res = []
                    for xs in inputs:
                        a = [ctypes.pointer(FJ.CTB[FJ.BW(t)](x if FJ.BW(t) > 1 else bool(x & 1))) for t, x in zip(ins_ty, xs)]
                        row = []
                        for j, t in enumerate(outs_ty):
                            r = FJ.CTB[FJ.BW(t)](0); ee.invoke(f'__w{j}', *a, ctypes.pointer(r)); row.append(int(r.value))
                        res.append(row)
                    out[(p, opt)] = res
            except Exception as e: out[(p, opt)] = 'fail:' + type(e).__name__ + ':' + str(e).split('\n')[0][:100]
    return out

def forked(*a):
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r)
        try: data = pickle.dumps(run_all(*a))
        except Exception: data = pickle.dumps('crash:' + traceback.format_exc()[-200:])
        with os.fdopen(w, 'wb') as f: f.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, 'rb') as f: data = f.read()
    _, status = os.waitpid(pid, 0)
    if not data: return f'crash:status={status}'
    return pickle.loads(data)

def run(n, seed=58, ninp=6):
    import mlir.ir as ir, irimport as I
    pipes = available(); rng = random.Random(seed)
    st = collections.Counter(); per = {p: collections.Counter() for p in pipes}; dev = []
    for i in range(n):
        txt, ins, outs = FJ.gen(rng)
        if FJ.F16PAT.search(txt): st['排除：sitofp→f16（LLVM 35b）'] += 1; continue
        try: m = ir.Module.parse(txt, ir.Context()); assert m.operation.verify()
        except Exception: st['不合法'] += 1; continue
        S, why = FJ.derive_imported(f'fz_{i}', I.import_module(m)['regions'][0][0]['ops'][0])
        if S is None: st['值层未知'] += 1; continue
        inputs = [[FJ.rand_in(rng, t) for t in ins] for _ in range(ninp)]
        md = [FJ.model_eval(S, ins, xs, outs) for xs in inputs]
        for r in md:
            if r[0] != 'val': st['模型跳过/' + r[1]] += 1
        res = forked(FJ.wrap(txt, 'main', ins, outs), ins, outs, inputs, pipes)
        if isinstance(res, str): st['子进程崩溃'] += 1; continue
        base = res[('', 0)]
        if isinstance(base, str): st['基线失败'] += 1; continue
        st['程序'] += 1; st['程序/' + ins[0]] += 1
        good = []
        for k, mv in enumerate(md):
            if mv[0] != 'val': continue
            st['基线比较'] += 1
            if all(FJ.agree(t, a, b) for t, a, b in zip(outs, mv[1], base[k])): good.append(k)
            else: st['模型反例'] += 1; dev.append(('模型反例', txt, inputs[k], mv[1], base[k]))
        o3 = res[('', 3)]
        for p in pipes[1:]:
            r = res[(p, 0)]
            if isinstance(r, str): per[p]['失败'] += 1; per[p]['失败:' + r[:50]] += 1; continue
            per[p]['程序'] += 1
            for k in good:
                per[p]['确定值'] += 1
                mv = md[k][1]
                if not all(FJ.agree(t, a, b) for t, a, b in zip(outs, mv, r[k])):
                    o3ok = not isinstance(o3, str) and all(FJ.agree(t, a, b) for t, a, b in zip(outs, mv, o3[k]))
                    per[p]['偏离'] += 1; per[p]['偏离且基线O3一致'] += o3ok
                    per[p]['偏离/' + ins[0]] += 1
                    if per[p]['偏离'] <= 3: dev.append((p, txt, [hex(x) for x in inputs[k]], mv, [hex(x % (1 << 64)) for x in r[k]]))
    return pipes, st, per, dev

# FZ2（首跑 FZ1 之后加入的定向检验，非预注册）：三处偏离的最小例，确认归因（基线 O0 = 模型 = IEEE 定义，管线后不同）。
FZ2_CASES = {   # 名字：(管线, 函数文本, 类型, 输入位模式, 期望（IEEE/模型）位模式)
  'arith-expand maximumf(+0,-0)': ('func.func(arith-expand)', 'func.func @main(%a: f32, %b: f32) -> f32 {\n  %r = arith.maximumf %a, %b : f32\n  return %r : f32\n}', 'f32', [0, 0x80000000], 0),
  'arith-expand minimumf(-0,+0)': ('func.func(arith-expand)', 'func.func @main(%a: f32, %b: f32) -> f32 {\n  %r = arith.minimumf %a, %b : f32\n  return %r : f32\n}', 'f32', [0x80000000, 0], 0x80000000),
  'math-expand-ops ceil(1e300)': ('func.func(math-expand-ops)', 'func.func @main(%a: f64) -> f64 {\n  %r = math.ceil %a : f64\n  return %r : f64\n}', 'f64', [0x7E37E43C8800759C], 0x7E37E43C8800759C),
  'math-expand-ops ceil(+inf)': ('func.func(math-expand-ops)', 'func.func @main(%a: f64) -> f64 {\n  %r = math.ceil %a : f64\n  return %r : f64\n}', 'f64', [0x7FF0000000000000], 0x7FF0000000000000),
  'math-expand-ops fma 不再融合': ('func.func(math-expand-ops)', 'func.func @main(%a: f64, %b: f64, %c: f64) -> f64 {\n  %r = math.fma %a, %b, %c : f64\n  return %r : f64\n}', 'f64',
                                   [0x3FF0000000000001, 0x3FEFFFFFFFFFFFFE, 0xBFF0000000000000], 0xB970000000000000),
}
def FZ2():
    out = {}
    for k, (p, txt, t, xs, want) in FZ2_CASES.items():
        ins = [t] * len(xs)
        r = run_all(FJ.wrap(txt, 'main', ins, [t]), ins, [t], [xs], ['', p], opts=(0,))
        b, a = r[('', 0)], r[(p, 0)]
        mask = (1 << FJ.BW(t)) - 1
        out[k] = (not isinstance(b, str) and not isinstance(a, str) and b[0][0] & mask == want and a[0][0] & mask != want,
                  f"基线 {hex(b[0][0] & mask) if not isinstance(b, str) else b}，{p} 后 {hex(a[0][0] & mask) if not isinstance(a, str) else a}，IEEE {hex(want)}")
    return out

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'fz2':
    z = FZ2()
    for k, (ok, m) in z.items(): print('  FZ2', '✓' if ok else '✗', k, m)
    V.check('FZ2 偏离定向（事后）', all(ok for ok, _ in z.values()), f"{sum(ok for ok, _ in z.values())}/{len(z)}")
    V.finish(); sys.exit()

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    pipes, st, per, dev = run(n)
    print('可用管线', len(pipes) - 1, '/', len(PIPES_F) - 1, [p for p in PIPES_F if p not in pipes])
    print('总', dict(sorted(st.items())))
    for p in pipes[1:]:
        d = per[p]
        print(f'  {p:80s} 程序 {d["程序"]} 确定值 {d["确定值"]} 偏离 {d["偏离"]}（基线O3一致 {d["偏离且基线O3一致"]}） 失败 {d["失败"]}',
              {k: v for k, v in d.items() if k.startswith('失败:') or k.startswith('偏离/')})
    for e in dev[:16]: print('  ✗', e[0], e[2:], '\n', e[1])
    V.check('FZ0 浮点基线', st['基线比较'] >= 1500 and st['模型反例'] == 0, f"基线比较 {st['基线比较']}，模型反例 {st['模型反例']}，排除 {st['排除：sitofp→f16（LLVM 35b）']}")
    bad = [p for p in pipes[1:] if per[p]['程序'] < 300]
    white = [p for p in pipes[1:] if per[p]['程序'] >= 300 and per[p]['偏离'] == 0]
    V.check('FZ1 浮点管线差分', not bad and len(pipes) == len(PIPES_F) and st['模型反例'] == 0,
            f"可用 {len(pipes) - 1}/{len(PIPES_F) - 1}；执行不足 {bad}；0 偏离 {len(white)} 条；偏离 {[(p, per[p]['偏离']) for p in pipes[1:] if per[p]['偏离']]}")
    V.finish()
