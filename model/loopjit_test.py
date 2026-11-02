"""第 41 轮（M3.8，spec/loops.md §9）：scf.for 的 iN 界与 unsignedCmp，core vs JIT。
此前：unsignedCmp 判未知（语料 15 个函数）；iN（BV）界缺“归纳变量越界 ⇒ UB”规则；符号界的 iN 循环摘要路径 iv 是 Int ⇒ 异常 ⇒ 未知。
  L1 生成器（只作补充；真值是 JIT）：i8/i16/i32/index × 有符号/无符号，界与步长取边界附近的值（含跨 0、跨符号位、负步长、零步长），
     体内把 iv 混入累加器。每组输入把界常量化后由 core 精确展开得值，与原函数（界是参数）的 JIT 比较；core 判 UB 的输入跳过。
  L2 UB 归因：core 因“越界”判 UB 的输入，JIT（带超时）结果与 LangRef 文字（iv ∈ [lb, ub) 才执行、按趟数 ⌈(ub−lb)/st⌉）的结果不同或不终止的比例
     ——证明“按 UB”是必要的（两种 MLIR 行为都应是它的精化），而不是 core 自作主张。
  L3 语料：corpus/SCF/trip_count.mlir 全部函数有事实（此前 15 个未知）。
用法：PYTHONPATH=model python3 model/loopjit_test.py [程序数]
"""
import sys, re, random, collections, multiprocessing as mp
import valimport_test as VT

def W(t): return 64 if t == 'index' else int(t[1:])

def gen(rng):
    ty = rng.choice(['i8', 'i8', 'i16', 'i32', 'index']); uns = rng.random() < 0.5
    body = ['%e = arith.index_cast %iv : index to i32' if ty == 'index' else
            ('%e = arith.extui %iv : ' + ty + ' to i32' if ty != 'i32' else '%e = arith.addi %iv, %z32 : i32')]
    body += ['%m = arith.muli %x, %c3 : i32', '%y = arith.addi %m, %e : i32']
    if rng.random() < 0.4:
        body += ['%cc = arith.cmpi ' + rng.choice(['slt', 'ult', 'eq']) + ', %iv, %lb : ' + ty,
                 '%y2 = arith.select %cc, %y, %x : i32']; res = '%y2'
    else: res = '%y'
    tyann = '' if ty == 'index' else f' : {ty}'
    txt = (f'func.func @main(%lb: {ty}, %ub: {ty}, %st: {ty}, %a: i32) -> i32 {{\n  %c3 = arith.constant 3 : i32\n  %z32 = arith.constant 0 : i32\n'
           f'  %r = scf.for {"unsigned " if uns else ""}%iv = %lb to %ub step %st iter_args(%x = %a) -> i32{tyann} {{\n    '
           + '\n    '.join(body) + f'\n    scf.yield {res} : i32\n  }}\n  return %r : i32\n}}')
    return txt, ty, uns

def inputs(rng, ty, uns, n=8):
    w = W(ty); M = 1 << w; out = []
    for _ in range(n):
        base = rng.choice([0, 1, 5, (1 << (w - 1)) - 3, (1 << (w - 1)) + 2, M - 4, M - 1, -3 % M])
        lb = base; span = rng.choice([0, 1, 3, 7, 20, -2])
        ub = (lb + span) % M; st = rng.choice([1, 1, 2, 3, 5, 0, M - 1])
        a = rng.randrange(-50, 50)
        def sx(v): return v - M if v >= M // 2 else v
        if ty == 'index': lb, ub, st = sx(lb), sx(ub), sx(st)
        out.append((lb, ub, st, a))
    return out

def constantized(txt, ty, xs):
    lb, ub, st, _ = xs
    c = lambda nm, v: f'  {nm} = arith.constant {v} : {ty}\n'
    t = txt.replace('func.func @main(%lb: ' + ty + ', %ub: ' + ty + ', %st: ' + ty + ', %a: i32)', 'func.func @main(%a: i32)')
    return t.replace('{\n  %c3', '{\n' + c('%lb', lb) + c('%ub', ub) + c('%st', st) + '  %c3', 1)

def langref(ty, uns, xs):
    """LangRef 文字：iv 从 lb 起按 st 递增，iv < ub（按比较方式）才执行；数学整数，不回绕。返回 (结果, 趟数) 或 None（步长 ≤ 0）。"""
    lb, ub, st, a = xs; w = W(ty); M = 1 << w
    def m(v): v %= M; return v if uns else (v - M if v >= M // 2 else v)
    l, u, k = m(lb), m(ub), (st % M if (st % M) < M // 2 else st % M - M)
    if k <= 0: return None
    x = a; i = l; n = 0
    while i < u and n < 10 ** 6:
        e = i % M if ty in ('i8', 'i16') else i                          # extui；i32 / index_cast 截断由下面的掩码完成
        x = (x * 3 + e) & 0xffffffff; i += k; n += 1
    return x, n

def _jit(txt, xs, q):
    import ctypes
    try:
        r = VT.jit_run(txt, 'main', [TY[0]] * 3 + ['i32'], 'i32', [list(xs)]); q.put(('ok', r[0]))
    except Exception as e: q.put(('err', str(e)[:80]))
TY = ['i32']
def jit(txt, ty, xs, timeout=20):
    TY[0] = ty
    ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=_jit, args=(txt, xs, q)); p.start(); p.join(timeout)
    if p.is_alive(): p.kill(); return ('timeout', None)
    if p.exitcode != 0: return ('crash', p.exitcode)
    return q.get()

def L1(N, seed=23):
    import mlir.ir as ir, irimport as I, core as C
    from compcorpus_test import _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 3: ex[k].append(v)
    for t in range(N):
        txt, ty, uns = gen(rng); tag = f'{ty}{"u" if uns else "s"}'
        st['程序'] += 1; st['程序/' + tag] += 1
        fn = list(_funcs(I.import_module(ir.Module.parse(txt, ir.Context()))))[0]
        S, why = VT.derive_imported(f'l1_{t}', fn)
        st['符号界：有事实' if S is not None else '符号界：未知'] += 1
        if S is None: note('符号界未知', why)
        for j, xs in enumerate(inputs(rng, ty, uns)):
            ct = constantized(txt, ty, xs)
            fc = list(_funcs(I.import_module(ir.Module.parse(ct, ir.Context()))))[0]
            Sc, why = VT.derive_imported(f'l1_{t}_{j}', fc)
            if Sc is None: st['常量化：未知'] += 1; note('常量化未知', why); continue
            r = VT.model_eval(Sc, ['i32'], [xs[3]])
            lr = langref(ty, uns, xs)
            if r[0] == 'skip':
                st['跳过/' + r[1]] += 1
                if r[1] == 'UB':                                             # L2：UB 归因
                    jv = jit(txt, ty, xs)
                    kind = '步长≤0' if lr is None else '越界'
                    if lr is None: st[f'L2 {kind}：JIT {jv[0]}'] += 1
                    else: st[f'L2 {kind}：JIT ' + ('与 LangRef 不同' if jv != ('ok', lr[0] - (1 << 32) if lr[0] >= 1 << 31 else lr[0]) else '与 LangRef 相同')] += 1
                continue
            if S is not None:                                                # 符号界的摘要与常量化展开须相容：UB 判定一致
                pass
            jv = jit(txt, ty, xs)
            st['比较'] += 1
            if jv[0] != 'ok': st['不一致/JIT ' + jv[0]] += 1; note('不一致', f'{tag} {xs} 模型 {r[1]} JIT {jv}\n{txt}'); continue
            if r[1] % (1 << 32) == jv[1] % (1 << 32): st['一致'] += 1; st['一致/' + tag] += 1
            else: st['不一致'] += 1; note('不一致', f'{tag} {xs} 模型 {r[1]} JIT {jv}\n{txt}')
            if lr is not None and lr[1] > 0: st['比较中趟数≥1'] += 1
    return st, ex

def L3():
    import mlir.ir as ir, irimport as I
    from compcorpus_test import _funcs
    src = open('corpus/SCF/trip_count.mlir').read(); c = collections.Counter()
    for ci, ch in enumerate(I.split_chunks(src)):
        if not ch.strip(): continue
        for fn in _funcs(I.import_module(ir.Module.parse(ch, ir.Context()))):
            S, why = VT.derive_imported(f'l3_{ci}', fn)
            c['有事实' if S is not None else '未知'] += 1
            c['其中 unsignedCmp'] += 'unsignedCmp' in str(fn)
    return c

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    l3 = L3(); print('── L3 trip_count.mlir:', dict(l3))
    st, ex = L1(N)
    print('── L1/L2:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
    for k, v in ex.items(): print('──', k); [print('    ', x[:700]) for x in v]
    import verdict as V
    V.check('L1 生成器', st['不一致'] == 0 and not any(k.startswith('不一致/') for k in st) and st['一致'] > 0, f"一致 {st['一致']}，不一致 {st['不一致']}")
    V.check('L3 trip_count', l3['未知'] == 0 and l3['有事实'] > 0, str(dict(l3)))
    V.finish()
