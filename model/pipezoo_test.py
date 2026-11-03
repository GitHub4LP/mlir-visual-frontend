"""第 56 轮：M5.1 管线差分扩面。判据 Sem(g) ⊑ Sem(pipeline(⟦g⟧))：模型对 g 给出确定值的输入上，
经管线 P（再接标准降低）后的 JIT 结果必须等于该值。

程序两族：
  A 复合 op 不透明落地（func + func.call，M5.2），调用方随机注入依赖数据的毒值（poison_test 的六种来源，也可能不注入）；
  B loop_test.gen：i8/i32/i64/index、类型转换、select、静态 scf.for（含嵌套）。
管线清单在看数据前写定（PIPES）。每个程序在 fork 子进程里跑全部管线（MLIR 断言崩溃不会带走父进程）。
分类（A/B 归因，L52）：
  模型反例：基线（无管线）JIT ≠ 模型确定值 ⇒ 模型错误；
  变换偏离：基线 = 模型、管线后 ≠ 模型 ⇒ 记入 MLIR 变换缺陷候选（再看基线 O3 是否 = 模型，O3 也一致则更像 MLIR 变换错误）；
  管线失败：解析/运行/校验失败或子进程崩溃，单列，不计入分母。
阈值（运行前写定）：
  Z0 基线：模型确定值 ≥ 1000，模型反例 0。
  Z1 每条可用管线：两族合计执行 ≥ 300 个程序；输出逐管线的确定值输入数、偏离数、失败数。判定只要求分类完整（偏离全部可归因：基线 = 模型）。
     偏离 > 0 的管线不视为 Z1 失败，而是写入 spec 的“前端不得使用 / 须加前置条件”的清单。
用法：PYTHONPATH=model python3 model/pipezoo_test.py [每族程序数] [族：AB]
"""
import sys, os, random, ctypes, collections, pickle, traceback
import core as C, loop_test as LT, poison_test as PT, verdict as V

LOWER = 'convert-scf-to-cf,convert-ub-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts'
PIPES = ['', 'inline', 'canonicalize', 'cse', 'sccp', 'loop-invariant-code-motion', 'control-flow-sink', 'remove-dead-values', 'symbol-dce',
         'func.func(scf-for-loop-canonicalization)', 'func.func(scf-for-loop-specialization)', 'func.func(scf-for-loop-peeling)',
         'func.func(int-range-optimizations)', 'func.func(arith-unsigned-when-equivalent)', 'func.func(arith-expand)',
         'inline,canonicalize,cse,sccp,loop-invariant-code-motion,canonicalize']

def available():
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    ok = []
    with ir.Context():
        for p in PIPES[1:]:
            try: PassManager.parse(f'builtin.module({p})'); ok.append(p)
            except Exception: pass
    return [''] + ok

def run_all(src, inputs, pipes, opts=(0,)):
    """子进程：每条管线 → {(p, opt): [结果] | 'fail:原因'}。"""
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    out = {}
    for p in pipes:
        for opt in (opts if p == '' else (0,)):
            try:
                with ir.Context() as ctx:
                    ctx.enable_multithreading(False)
                    m = ir.Module.parse(src)
                    if p: PassManager.parse(f'builtin.module({p})').run(m.operation)
                    if not m.operation.verify(): out[(p, opt)] = 'fail:verify'; continue
                    PassManager.parse(f'builtin.module({LOWER})').run(m.operation)
                    ee = ExecutionEngine(m, opt_level=opt); res = []
                    for xs in inputs:
                        a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
                        ee.invoke('main', *a, r); res.append(r[0] % (1 << 64))
                    out[(p, opt)] = res
            except Exception as e: out[(p, opt)] = 'fail:' + type(e).__name__ + ':' + str(e)[:120]
    return out

def forked(src, inputs, pipes):
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r)
        try: data = pickle.dumps(run_all(src, inputs, pipes, opts=(0, 3)))
        except Exception: data = pickle.dumps('crash:' + traceback.format_exc()[-200:])
        with os.fdopen(w, 'wb') as f: f.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, 'rb') as f: data = f.read()
    _, status = os.waitpid(pid, 0)
    if not data: return f'crash:status={status}'
    return pickle.loads(data)

def prog_A(rng, t):
    psrc = rng.choice(['none'] + PT.SOURCES)
    T = rng.choice(['i8', 'i16', 'i32'] if psrc == 'trunci_nuw' else ['i8', 'i16', 'i32', 'i64'])
    fb = PT.gen_f(rng, T); fn = f'ZF{t}'
    C.derive(fn, [('x', T), ('y', T), ('z', T)], [('q', T)], fb)
    mask = [0, 0, 0] if psrc == 'none' else [int(rng.random() < 0.5) for _ in range(3)]
    mb = PT.gen_main(rng, T, fn, mask, psrc if psrc != 'none' else 'shl')
    G = C.derive(f'ZM{t}', PT.INS, [('q', 'i64')], mb)
    src = PT.emit_fn(fn, fb, [('x', T), ('y', T), ('z', T)], T) + '\n' + PT.emit_fn('main', mb, PT.INS, 'i64', callee=(fn, fn), iface=True)
    return G, src

def prog_B(rng, t):
    g = LT.gen(rng, 56000 + t)
    G = C.derive(f'ZB{t}', LT.INS, [('q', 'i64')], g)
    return G, LT.emit(g)

EDGE_IN = [0, 1, 2, 3, 5, 7, 100, 255, -1, -128, 1 << 31, (1 << 63) - 1, -(1 << 63)]

def run(n, fams='AB', seed=56):
    pipes = available(); rng = random.Random(seed)
    st = collections.Counter(); per = {p: collections.Counter() for p in pipes}; dev = []
    for fam in fams:
        for t in range(n):
            try: G, src = (prog_A if fam == 'A' else prog_B)(rng, t)
            except C.ValueUnknown: st[f'{fam}:值层未知'] += 1; continue
            ins = [[rng.choice(EDGE_IN + [rng.randrange(-1 << 63, 1 << 63)]) for _ in range(3)] for _ in range(4)]
            md = [LT.model_eval(G, xs) for xs in ins]
            res = forked(src, ins, pipes)
            if isinstance(res, str): st[f'{fam}:子进程崩溃'] += 1; continue
            base = res[('', 0)]
            if isinstance(base, str): st[f'{fam}:基线失败'] += 1; continue
            st[f'{fam}:程序'] += 1
            for i, mv in enumerate(md):
                if mv[0] != 'val': st[f'模型 {mv[0]}'] += 1; continue
                st['基线确定值'] += 1
                if base[i] != mv[1]: st['模型反例'] += 1; dev.append(('模型反例', fam, src, ins[i], mv[1], base[i]))
            for p in pipes[1:]:
                r = res[(p, 0)]
                if isinstance(r, str): per[p]['失败'] += 1; per[p]['失败:' + r[:40]] += 1; continue
                per[p]['程序'] += 1; per[p][f'{fam}:程序'] += 1
                for i, mv in enumerate(md):
                    if mv[0] != 'val' or base[i] != mv[1]: continue
                    per[p]['确定值'] += 1
                    if r[i] != mv[1]:
                        o3 = res[('', 3)]; o3ok = not isinstance(o3, str) and o3[i] == mv[1]
                        per[p]['偏离'] += 1; per[p]['偏离且基线O3一致'] += o3ok
                        if per[p]['偏离'] <= 2: dev.append((p, fam, src, ins[i], mv[1], r[i]))
    return pipes, st, per, dev

PEEL_MIN = '''func.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {
  %lb = arith.constant 2 : index
  %ub = arith.constant 1 : index
  %st = arith.constant 2 : index
  %c1 = arith.constant 1 : i64
  %r = scf.for %i = %lb to %ub step %st iter_args(%a = %arg0) -> (i64) {
    %n = arith.addi %a, %c1 : i64
    scf.yield %n : i64
  }
  return %r : i64
}'''
RDV_MIN = '''func.func @f(%x: i64, %y: i64) -> i64 {
  return %y : i64
}
func.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {
  %c = arith.constant 3 : i64
  %d = arith.addi %arg0, %c : i64
  %r = func.call @f(%d, %arg1) : (i64, i64) -> i64
  return %r : i64
}'''
def Z2():
    """MLIR 缺陷定向（9bae84b0）：scf-for-loop-peeling 对 lb > ub 的空循环，拆分点 ub − ((ub − lb) mod step) 落到 lb 之下，剥出的一次迭代被执行。"""
    r = run_all(PEEL_MIN, [[10, 0, 0]], ['', 'func.func(scf-for-loop-peeling)', 'func.func(scf-for-loop-peeling{peel-front=true})'])
    return r[('', 0)], r[('func.func(scf-for-loop-peeling)', 0)], r[('func.func(scf-for-loop-peeling{peel-front=true})', 0)]
def Z3():
    """MLIR 缺陷定向（9bae84b0）：remove-dead-values 删掉公开 callee 的死参数在调用方的定义，留下空操作数的 func.call（非法 IR）。"""
    return run_all(RDV_MIN, [[1, 2, 3]], ['', 'remove-dead-values'])

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    fams = sys.argv[2] if len(sys.argv) > 2 else 'AB'
    pipes, st, per, dev = run(n, fams)
    print('可用管线', len(pipes) - 1, '/', len(PIPES) - 1, [p for p in PIPES if p not in pipes])
    print('总', dict(sorted(st.items())))
    for p in pipes[1:]:
        d = per[p]; print(f'  {p:70s} 程序 {d["程序"]} 确定值 {d["确定值"]} 偏离 {d["偏离"]}（基线O3一致 {d["偏离且基线O3一致"]}） 失败 {d["失败"]}',
                          {k: v for k, v in d.items() if k.startswith('失败:')})
    for e in dev[:12]: print('  ✗', e[0], e[1], e[3], '模型', e[4], 'JIT', e[5], '\n', e[2])
    need = 300 if fams == 'AB' else 150
    V.check('Z0 基线', st['基线确定值'] >= 1000 and st['模型反例'] == 0, f"基线确定值 {st['基线确定值']}，模型反例 {st['模型反例']}")
    V.check('Z1 管线差分', all(per[p]['程序'] >= need for p in pipes[1:]),
            '; '.join(f"{p} 程序 {per[p]['程序']} 确定值 {per[p]['确定值']} 偏离 {per[p]['偏离']} 失败 {per[p]['失败']}" for p in pipes[1:]))
    z2 = Z2(); print('Z2', z2)
    V.check('Z2 peeling lb>ub 定向', z2[0] == [10] and z2[1] == [11] and z2[2] == [10],
            f"基线 {z2[0]}，peel-back {z2[1]}（多执行 1 次），peel-front {z2[2]}；upstream main 已改为 min(lb, ub) 形式")
    z3 = Z3(); print('Z3', z3)
    V.check('Z3 remove-dead-values 定向', z3[('', 0)] == [2] and isinstance(z3[('remove-dead-values', 0)], str),
            f"基线 {z3[('', 0)]}，remove-dead-values {z3[('remove-dead-values', 0)]}")
    V.finish()
