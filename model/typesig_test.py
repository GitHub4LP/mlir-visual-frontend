"""类型层的检验（第 7 轮）。
  T1 与 MLIR verifier 的差分：23 个 arith/tensor op，随机类型实例（合法模板 + 一处变异），按来源分级比较
     ods = 只用从 .td 自动导入的部分；+cpred = 加上登记的 C++ 片段；+declared = 再加上只在 C++ verify() 里的条件
  T2 推断：推出的结果类型被 MLIR 接受
  T3 查询代价：connectable / candidates / check 的耗时
用法：PYTHONPATH=model python3 model/typesig_test.py [每个 op 的实例数]
"""
import random, sys, time, collections
import typesig as TS
from typesig import parse as P, show

KDYN = -9223372036854775808
INTS = ['i1', 'i8', 'i32', 'i64', 'index', 'si8']
ALL_B = INTS + ['f32']
def form(rng, b):
    k = rng.choice(['s', 's', 't4', 'tq', 't23', 'tu', 'v4', 'm4', 't0'])
    return {'s': b, 't4': f'tensor<4x{b}>', 'tq': f'tensor<?x{b}>', 't23': f'tensor<2x3x{b}>', 'tu': f'tensor<*x{b}>',
            'v4': f'vector<4x{b}>', 'm4': f'memref<4x{b}>', 't0': f'tensor<{b}>'}[k]
def rand_t(rng): return P(form(rng, rng.choice(ALL_B)))
def same_shape(t, e):
    return (t[0], t[1], e) if TS.shaped(t) else e
def rtensor(rng, e, dyn_ok=True):
    r = rng.choice([1, 1, 2]); dims = tuple(rng.choice([2, 3, 4] + ([None] if dyn_ok else [])) for _ in range(r))
    return ('tensor', dims, e)
IX = ('index',)

def template(rng, op):
    """返回 (ins, outs, seg)：大体合法的实例。"""
    ib = lambda: P(rng.choice(['i8', 'i32', 'i64', 'index']))
    if op in ('arith.addi', 'arith.subi', 'arith.muli', 'arith.divui', 'arith.andi', 'arith.xori'):
        t = same_shape(P(form(rng, 'i8')), ib()); return [t, t], [t], None
    if op == 'arith.addf': t = same_shape(P(form(rng, 'f32')), ('f', 'f32')); return [t, t], [t], None
    if op == 'arith.cmpi': t = same_shape(P(form(rng, 'i8')), ib()); return [t, t], [TS.i1same(t)], None
    if op == 'arith.select':
        t = rand_t(rng); c = TS.i1same(t) if TS.shaped(t) and rng.random() < 0.5 else ('i', 1, ''); return [c, t, t], [t], None
    if op in ('arith.trunci', 'arith.extui', 'arith.extsi'):
        a, b = sorted(rng.sample([1, 8, 16, 32, 64], 2))
        if op == 'arith.trunci': a, b = b, a
        sh = P(form(rng, 'i8')); return [same_shape(sh, ('i', a, ''))], [same_shape(sh, ('i', b, ''))], None
    if op in ('arith.index_cast', 'arith.index_castui'):
        sh = P(form(rng, 'i8')); x, y = same_shape(sh, ('i', rng.choice([8, 32, 64]), '')), same_shape(sh, IX)
        return ([x], [y], None) if rng.random() < 0.5 else ([y], [x], None)
    e = P(rng.choice(['i8', 'i32', 'f32', 'index']))
    if op == 'tensor.dim': return [rtensor(rng, e), IX], [IX], None
    if op == 'tensor.extract': t = rtensor(rng, e); return [t] + [IX] * len(t[1]), [e], None
    if op == 'tensor.insert': t = rtensor(rng, e); return [e, t] + [IX] * len(t[1]), [t], None
    if op == 'tensor.empty': t = rtensor(rng, e); return [IX] * sum(d is None for d in t[1]), [t], None
    if op == 'tensor.splat': t = rtensor(rng, e); return [e] + [IX] * sum(d is None for d in t[1]), [t], None
    if op == 'tensor.from_elements': t = rtensor(rng, e, dyn_ok=False); return [e] * TS._numel(t), [t], None
    if op == 'tensor.cast':
        t = rtensor(rng, e); u = ('tensor', tuple(None if rng.random() < 0.5 else d for d in t[1]), e)
        return [t], [u if rng.random() < 0.8 else ('tensor', None, e)], None
    if op == 'tensor.extract_slice':
        t = rtensor(rng, e); r = len(t[1]); return [t] + [IX] * (3 * r), [('tensor', (None,) * r, e)], [1, r, r, r]
    if op == 'tensor.insert_slice':
        d = rtensor(rng, e); r = len(d[1]); s = ('tensor', (None,) * r, e)
        return [s, d] + [IX] * (3 * r), [d], [1, 1, r, r, r]
    raise KeyError(op)

def mutate(rng, op, ins, outs, seg):
    r = rng.random()
    if r < 0.15: return ins, outs, seg
    if r < 0.3 and op in ('tensor.extract', 'tensor.insert', 'tensor.empty', 'tensor.splat', 'tensor.from_elements'):
        return (ins + [ins[-1] if ins else IX]) if rng.random() < 0.5 or len(ins) < 2 else ins[:-1], outs, seg     # 变长组个数 ±1
    if r < 0.3 and seg:
        j = rng.randrange(len(seg) - 3, len(seg)); seg = list(seg); start = sum(seg[:j])
        if rng.random() < 0.5: seg[j] += 1; ins = ins[:start] + [IX] + ins[start:]
        elif seg[j] > 0: seg[j] -= 1; ins = ins[:start] + ins[start + 1:]
        return ins, outs, seg
    k = rng.randrange(len(ins) + len(outs))
    t = rand_t(rng) if rng.random() < 0.6 else (same_shape(rand_t(rng), TS.elt((ins + outs)[k])) if rng.random() < 0.5
                                              else same_shape((ins + outs)[k], TS.elt(rand_t(rng))))
    if k < len(ins): ins = ins[:k] + [t] + ins[k + 1:]
    else: outs = [t]
    return ins, outs, seg

def mlir_text(op, ins, outs, seg):
    attrs = ''
    if op == 'arith.cmpi': attrs = ' {predicate = 0 : i64}'
    if op in ('tensor.extract_slice', 'tensor.insert_slice'):
        n = seg[-3:]
        arr = lambda k: 'array<i64' + (': ' + ', '.join([str(KDYN)] * k) if k else '') + '>'
        attrs = (' <{operandSegmentSizes = array<i32: ' + ', '.join(map(str, seg)) + '>, static_offsets = ' + arr(n[0])
                 + ', static_sizes = ' + arr(n[1]) + ', static_strides = ' + arr(n[2]) + '}>')
    args = ', '.join(f'%a{i}: {show(t)}' for i, t in enumerate(ins))
    return (f'func.func @f({args}) {{\n  %r = "{op}"(' + ', '.join(f'%a{i}' for i in range(len(ins))) + f'){attrs} : ('
            + ', '.join(map(show, ins)) + ') -> ' + show(outs[0]) + '\n  return\n}')

CRASH = []; CRASHOPS = collections.Counter()
def mlir_ok(ctx, text):
    """在子进程里解析并验证：MLIR 的 verifier 会在某些非法实例上断言崩溃（记入 CRASH，按拒绝处理）。"""
    import os
    pid = os.fork()
    if pid == 0:
        import mlir.ir as ir
        os.close(2); os.open(os.devnull, os.O_WRONLY)
        try: ok = ir.Module.parse(text, ctx).operation.verify()
        except Exception: ok = False
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    if os.WIFSIGNALED(status): CRASH.append(text); return False
    return os.WEXITSTATUS(status) == 0

def T1(n, rng):
    import mlir.ir as ir
    ctx = ir.Context()
    lv = {'ods': ('ods',), '+cpred': ('ods', 'cpred'), '+declared': TS.LEVELS}
    st = {k: collections.Counter() for k in lv}; per = collections.defaultdict(collections.Counter); ex = []
    for op in TS.load():
        for _ in range(n):
            ins, outs, seg = mutate(rng, op, *template(rng, op))
            nc = len(CRASH); truth = mlir_ok(ctx, mlir_text(op, ins, outs, seg))
            if len(CRASH) > nc:                                              # MLIR 自身崩溃：单独计数，并检查前端是否拒绝
                CRASHOPS[op] += 1; CRASHOPS['前端拒绝'] += bool(TS.check(op, ins, outs, TS.LEVELS, seg)); continue
            for k, L in lv.items():
                ours = not TS.check(op, ins, outs, L, seg)
                tag = '一致' if ours == truth else ('误收（前端放行、MLIR 拒绝）' if ours else '误拒')
                st[k][tag] += 1
                if k == '+declared':
                    per[op][tag] += 1
                    if tag != '一致' and len(ex) < 6: ex.append((op, [show(t) for t in ins], show(outs[0]), seg, truth))
            st['ods']['MLIR 接受'] += truth
    return st, {op: dict(c) for op, c in per.items() if c['一致'] != sum(c.values())}, ex

def T2(rng):
    import mlir.ir as ir
    ctx = ir.Context(); c = collections.Counter()
    for op in TS.load():
        for _ in range(20):
            ins, outs, seg = template(rng, op)
            if not mlir_ok(ctx, mlir_text(op, ins, outs, seg)): continue
            t = TS.infer(op, ins, seg)
            if t is None: c['推不出（结果类型由用户选择）'] += 1; continue
            c['推断被 MLIR 接受' if mlir_ok(ctx, mlir_text(op, ins, [t], seg)) else '推断被拒'] += 1
    return dict(c)

def T3():
    t = P('tensor<?xi32>'); k = 2000
    t0 = time.perf_counter(); [TS.connectable('tensor.extract', 'tensor', t) for _ in range(k)]; t1 = time.perf_counter()
    [TS.candidates(t) for _ in range(200)]; t2 = time.perf_counter()
    [TS.check('tensor.extract', [t, ('index',)], [('i', 32, '')]) for _ in range(k)]; t3 = time.perf_counter()
    return {'connectable（µs）': round((t1 - t0) / k * 1e6, 1), 'candidates，23 个 op（µs）': round((t2 - t1) / 200 * 1e6, 1),
            'check 一个节点（µs）': round((t3 - t2) / k * 1e6, 1), 'tensor<?xi32> 的补全': len(TS.candidates(t))}

def T4():
    """core 接入：病类型图得到诊断，不进值层（第 6 轮之前，这两例直接让 z3 抛异常）。"""
    import core as C
    from core import blk, node
    a = C.derive('bad_w', [('a', 'i8'), ('b', 'i32')], [('q', 'i8')],
                 [blk('e', [C.op2('s', 'arith.addi', 'a', 'b', 'i8'), C.ret('r', 's')], args=[('a', 'i8'), ('b', 'i32')])])
    b = C.derive('bad_t', [('a', 'i8')], [('q', 'i8')],
                 [blk('e', [node('x', 'tensor.extract', ins=['a'], outs=[('x', 'i8')]), C.ret('r', 'x')], args=[('a', 'i8')])])
    return {'addi(i8, i32)': (a.type_diag[:1], a.value_known), 'extract(i8)': (b.type_diag[:1], b.value_known)}

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    st, per, ex = T1(n, random.Random(7))
    print('── T1 与 MLIR verifier 的差分（每个 op', n, '个实例）')
    for k, c in st.items(): print('     ', k, dict(c))
    print('      +declared 仍不一致的 op:', per)
    for e in ex: print('        例', e)
    print('── MLIR verifier 断言崩溃的实例:', len(CRASH), dict(CRASHOPS), '含无秩 tensor:', sum('tensor<*' in c for c in CRASH)); [print(c) for c in CRASH[:2]]
    print('── T2 推断:', T2(random.Random(8)))
    print('── T3 代价:', T3())
    print('── T4 接入 core:', T4())
