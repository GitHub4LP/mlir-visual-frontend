"""位宽、index、循环的检验：
  W1 pattern 逐位宽证明（i3…i64）
  W2 与真实 MLIR 执行的差分：随机程序（i8/i32/i64/index、类型转换、scf.if、静态 scf.for），LLVM JIT 执行 vs 模型求值
  W3 循环摘要（L18）：确定性、嵌套 ≡ 内联、透明模式调用点特化、不透明模式只可靠
  W4 不变式（L19）：归纳检查、收窄 Pre、错误不变式被拒、公理与具体执行一致
  W5 index 回绕（L17″，第 21 轮依 LangRef 更正 L17′）：模型值 = MLIR 执行值；nsw/nuw 才产生毒值；L17′ 的反例
用法：PYTHONPATH=model python3 model/loop_test.py [W2 的程序数]
"""
import random, sys, ctypes, z3
import core as C, transform as T
from core import blk, node

I64 = 'i64'; INS = [('x', I64), ('y', I64), ('z', I64)]
EDGE = {'i8': [0, 1, 2, 127, 128, 255, 7], 'i32': [0, 1, 3, (1 << 31) - 1, 1 << 31, (1 << 32) - 1],
        'index': [0, 1, 2, 5, -1, (1 << 62), -(1 << 63)], 'i64': [0, 1, (1 << 63) - 1, 1 << 63]}

def W1():
    out = {}
    for w in ('i3', 'i8', 'i16', 'i32', 'i64'):
        bad = [p.name for p in T.PATTERNS + T.EXTRA + T.WRONG
               if (T.check(p, w) is True) != (not p.name.startswith('✗') or p.name == '✗sub_self_dup')]
        out[w] = '全部符合' if not bad else bad
    return out

# ───────────────────────── W2 随机程序
CASTS = [('i64', 'i32', 'trunci'), ('i64', 'i8', 'trunci'), ('i32', 'i8', 'trunci'), ('i8', 'i32', 'extui'), ('i8', 'i64', 'extsi'),
         ('i32', 'i64', 'extsi'), ('i32', 'i64', 'extui'), ('i64', 'index', 'index_cast'), ('i32', 'index', 'index_cast'),
         ('i32', 'index', 'index_castui'), ('index', 'i32', 'index_cast'), ('index', 'i64', 'index_castui')]
def gen(rng, t):
    k = [0]
    def fresh(): k[0] += 1; return f'v{t}_{k[0]}'
    def body(pool, d, n_ops):
        ops = []
        for _ in range(n_ops):
            r = rng.random(); v = fresh()
            tys = [ty for ty in ('i8', 'i32', 'i64', 'index') if pool[ty]]
            T_ = rng.choice(tys)
            if r < 0.15:
                ty = rng.choice(['i8', 'i32', 'index', 'i64'])
                ops.append(C.cst(v, rng.choice(EDGE[ty]) if rng.random() < 0.5 else rng.randrange(0, 9), ty)); pool[ty].append(v)
            elif r < 0.5:
                op = rng.choice(['arith.addi', 'arith.subi', 'arith.muli', 'arith.andi', 'arith.xori', 'arith.divui'])
                a, b = rng.choice(pool[T_]), rng.choice(pool[T_])
                pr = {'nuw': True} if op in ('arith.addi', 'arith.subi', 'arith.muli') and rng.random() < 0.3 else {}
                ops.append(C.op2(v, op, a, b, T_, **pr)); pool[T_].append(v)
            elif r < 0.6:
                a, b = rng.choice(pool[T_]), rng.choice(pool[T_])
                ops.append(C.op2(v, 'arith.cmpi', a, b, 'i1', predicate=rng.randrange(10))); pool['i1'].append(v)
            elif r < 0.68 and pool['i1']:
                ops.append(node(v, 'arith.select', ins=[rng.choice(pool['i1']), rng.choice(pool[T_]), rng.choice(pool[T_])], outs=[(v, T_)]))
                pool[T_].append(v)
            elif r < 0.88:
                src, dst, kd = rng.choice([c for c in CASTS if pool[c[0]]])
                ops.append(node(v, f'arith.{kd}', ins=[rng.choice(pool[src])], outs=[(v, dst)])); pool[dst].append(v)
            elif d > 0:
                lb, ub, st = fresh(), fresh(), fresh()
                ops += [C.cst(lb, rng.randrange(0, 3), 'index'), C.cst(ub, rng.randrange(0, 7), 'index'), C.cst(st, rng.randrange(1, 3), 'index')]
                it_t = rng.choice(tys); init = rng.choice(pool[it_t])
                iv, it = fresh(), fresh()
                sub = {ty: list(x) for ty, x in pool.items()}; sub['index'].append(iv); sub[it_t].append(it)
                bo = body(sub, d - 1, rng.randint(1, 4))
                y = sub[it_t][-1]
                ops.append(node(v, 'scf.for', ins=[lb, ub, st, init], outs=[(v, it_t)],
                                regions=[[blk('b', bo + [node(fresh(), 'scf.yield', ins=[y])], args=[(iv, 'index'), (it, it_t)])]]))
                pool[it_t].append(v)
        return ops
    pool = {'i64': ['x', 'y', 'z'], 'i32': [], 'i8': [], 'index': [], 'i1': []}
    ops = body(pool, 1, rng.randint(4, 12))
    last_t = rng.choice([ty for ty in ('i8', 'i32', 'index', 'i64') if pool[ty]])
    last = pool[last_t][-1]; r = f'r{t}'
    if last_t in ('i8', 'i32'): ops.append(node(r, 'arith.extsi', ins=[last], outs=[(r, I64)]))
    elif last_t == 'index': ops.append(node(r, 'arith.index_cast', ins=[last], outs=[(r, I64)]))
    else: r = last
    return [blk('entry', ops + [C.ret(f'ret{t}', r)], args=INS)]

def emit(blocks):
    nm = {'x': '%arg0', 'y': '%arg1', 'z': '%arg2'}; ty = dict(INS)
    v = lambda x: nm.setdefault(x, '%' + x)
    def ops(b, ind):
        for a, t in b['args']: ty[a] = t
        out = []
        for n in b['ops']:
            for o, t in n['outs']: ty[o] = t
            op, pr = n['op'], n.get('props', {})
            if op == 'func.return': out.append(f'{ind}return {v(n["ins"][0])} : i64'); continue
            if op == 'scf.yield': out.append(f'{ind}scf.yield {v(n["ins"][0])} : {ty[n["ins"][0]]}'); continue
            if op == 'scf.for':
                bb = n['regions'][0][0]; (iv, _), (it, itt) = bb['args']
                out.append(f'{ind}{v(n["outs"][0][0])} = scf.for {v(iv)} = {v(n["ins"][0])} to {v(n["ins"][1])} step {v(n["ins"][2])} '
                           f'iter_args({v(it)} = {v(n["ins"][3])}) -> ({itt}) {{')
                out += ops(bb, ind + '  '); out.append(ind + '}'); continue
            at = ''
            if op == 'arith.constant': at = f'<{{value = {pr["value"]} : {pr["type"]}}}>'
            elif op == 'arith.cmpi': at = f'<{{predicate = {pr["predicate"]} : i64}}>'
            elif pr.get('nuw'): at = '<{overflowFlags = #arith.overflow<nuw>}>'
            sig = '(' + ', '.join(ty[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
            out.append(f'{ind}{v(n["outs"][0][0])} = "{op}"(' + ', '.join(v(x) for x in n['ins']) + f') {at} : {sig}')
        return out
    return ('module {\nfunc.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {\n'
            + '\n'.join(ops(blocks[0], '  ')) + '\n}\n}')

PIPE = 'builtin.module(convert-scf-to-cf,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'
def jit(src, inputs):
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse(PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=0)
        res = []
        for xs in inputs:
            a = [(ctypes.c_int64 * 1)(sx) for sx in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); res.append(r[0] % (1 << 64))
        return res

def model_eval(G, xs):
    pairs = [(h[0], z3.BitVecVal(x % (1 << 64), 64)) for h, x in zip(G.holes, xs)] + [(h[1], z3.BoolVal(False)) for h in G.holes]
    sub = lambda t: z3.simplify(z3.substitute(t, pairs))
    ub, (val, poi) = sub(G.ubf), G.outv[0]
    if not z3.is_false(ub): return ('UB' if z3.is_true(ub) else '?',)
    p = sub(poi)
    if not z3.is_false(p): return ('poison' if z3.is_true(p) else '?',)
    vv = sub(val)
    return ('val', vv.as_long()) if z3.is_bv_value(vv) else ('?',)

def W2(n, rng):
    st = dict(程序=0, 输入=0, 一致=0, 不一致=0, 模型UB=0, 模型毒值=0, 未定=0, 含循环=0)
    for t in range(n):
        g = gen(rng, t); G = C.derive(f'J{t}', INS, [('q', I64)], g)
        assert not G.struct_diag, G.struct_diag
        st['程序'] += 1; st['含循环'] += any(x['op'] == 'scf.for' for x in g[0]['ops'])
        ins = [[rng.choice([0, 1, 2, 3, 5, 200, -1, 1 << 40, (1 << 63) - 1, -(1 << 63), rng.randrange(-1 << 63, 1 << 63)]) for _ in range(3)] for _ in range(4)]
        outs = jit(emit(g), ins)
        for xs, o in zip(ins, outs):
            st['输入'] += 1; r = model_eval(G, xs)
            if r[0] == 'val':
                if r[1] == o: st['一致'] += 1
                else:
                    st['不一致'] += 1
                    if st['不一致'] <= 2: print('  ✗ 不一致', xs, '模型', r[1], 'MLIR', o, '\n', emit(g))
            elif r[0] == 'UB': st['模型UB'] += 1
            elif r[0] == 'poison': st['模型毒值'] += 1
            else: st['未定'] += 1
    return st

# ───────────────────────── W3 / W4 循环摘要与不变式
IX = 'index'
def counter(name, inc=1, mode='transparent', inv=None, div=False):
    """name(lb, n) = for i = lb to n step 1 iter c = 0: c + inc（div=True 时体内还有 100 / (i+1)，结果加进 c）。"""
    ops = [C.cst('k', inc, IX), C.op2('c2', 'arith.addi', 'c', 'k', IX)]
    if div: ops = [C.cst('one', 1, IX), C.cst('h', 100, IX), C.op2('i1', 'arith.addi', 'i', 'one', IX),
                   C.op2('q', 'arith.divui', 'h', 'i1', IX), C.cst('k', 0, IX), C.op2('c2', 'arith.addi', 'k', 'q', IX)]
    lp = node('L', 'scf.for', ins=['lb', 'n', 's', 'z'], outs=[('r', IX)], props={'inv': inv} if inv else {},
              regions=[[blk('b', ops + [node('y', 'scf.yield', ins=['c2'])], args=[('i', IX), ('c', IX)])]])
    args = [('lb', IX), ('n', IX)]
    return C.derive(name, args, [('q', IX)], [blk('entry', [C.cst('s', 1, IX), C.cst('z', 0, IX), lp, C.ret('rt', 'r')], args=args)], mode=mode)

def host(name, callee, a, b, op='arith.subi', consts=()):
    """name(p, q) = callee(0, a) op callee(0, b)，a/b 是端口名或常量。"""
    ops = [C.cst('zero', 0, IX)] + [C.cst(f'k{v}', v, IX) for v in consts]
    ops += [node('u', callee, ins=['zero', a], outs=[('u', IX)]), node('w', callee, ins=['zero', b], outs=[('w', IX)]),
            C.op2('d', op, 'u', 'w', IX), C.ret('rt', 'd')]
    return C.derive(name, [('p', IX), ('q', IX)], [('o', IX)], [blk('entry', ops, args=[('p', IX), ('q', IX)])])

def out_is(G, val, assume=True):
    v, p = G.outv[0]
    return C.prove(z3.Implies(z3.And(G.ax, assume, z3.Not(G.ubf), z3.Not(p)), v == val))      # 不 UB、不是毒值时

def W3():
    cnt = counter('cnt'); cnt2 = counter('cnt2', inc=2); cop = counter('cnt_op', mode='opaque')
    r = {}
    r['摘要是精确的'] = cnt.exact
    r['cnt(p) - cnt(p) = 0（确定性）'] = out_is(host('h1', 'cnt', 'p', 'p'), 0)
    r['cnt(p) - cnt(q) = 0 不可证'] = not out_is(host('h2', 'cnt', 'p', 'q'), 0)
    h3 = C.derive('h3', [('p', IX), ('q', IX)], [('o', IX)],
                  [blk('entry', [C.cst('zero', 0, IX), node('u', 'cnt', ins=['zero', 'p'], outs=[('u', IX)]),
                                 node('w', 'cnt2', ins=['zero', 'p'], outs=[('w', IX)]), C.op2('d', 'arith.subi', 'u', 'w', IX), C.ret('rt', 'd')],
                       args=[('p', IX), ('q', IX)])])
    r['cnt(p) - cnt2(p) = 0 不可证（不同的循环是不同的函数）'] = not out_is(h3, 0)
    hn = host('h4', 'cnt', 'p', 'q', op='arith.addi')
    hi = C.derive('h4i', [('p', IX), ('q', IX)], [('o', IX)], C.inline_all(hn.body))
    r['嵌套 ≡ 内联（摘要按结构命名）'] = C.equivalent(hn, hi)
    r['透明：cnt(5) 在调用点特化 = 5'] = out_is(host('h5', 'cnt', 'k5', 'k5', op='arith.addi', consts=(5,)), 10)
    r['不透明：cnt_op(5) + cnt_op(5) = 10 不可证（只可靠）'] = not out_is(host('h6', 'cnt_op', 'k5', 'k5', op='arith.addi', consts=(5,)), 10)
    return r

def W4(rng):
    r = {}
    # L17″ 下 c+1 回绕；“c = i − lb”要靠范围事实维持：lb < 0 时 i − lb 可以超过 2^63 而回绕
    weak = lambda iv, its, B: z3.And(its[0][0] == iv - B['lb'], z3.Not(its[0][1]), iv >= B['lb'])
    r['c = i − lb（不带范围）：不是归纳的'] = counter('cnt_weak', mode='opaque', inv=weak).loop_diag
    good = lambda iv, its, B: z3.And(its[0][0] == iv - B['lb'], z3.Not(its[0][1]), iv >= B['lb'], B['lb'] >= 0)
    ci = counter('cnt_inv', mode='opaque', inv=good)
    v, p = ci.outv[0]; lb, n = ci.holes[0][0], ci.holes[1][0]
    nopz = z3.And(z3.Not(ci.holes[0][1]), z3.Not(ci.holes[1][1]))
    r['不变式 c = i − lb ∧ lb ≥ 0：n > lb ⇒ 输出 = n − lb'] = C.prove(z3.Implies(z3.And(ci.ax, nopz, n > lb, lb >= 0, z3.Not(ci.ubf)), v == n - lb))
    r['无不变式时同一事实不可证'] = not C.prove(z3.Implies(z3.And(C.REG['cnt_op'].ax, z3.Not(C.REG['cnt_op'].ubf), C.REG['cnt_op'].holes[1][0] > C.REG['cnt_op'].holes[0][0]),
                                                     C.REG['cnt_op'].outv[0][0] == C.REG['cnt_op'].holes[1][0] - C.REG['cnt_op'].holes[0][0]))
    bad = lambda iv, its, B: its[0][0] == iv               # 只有 lb = 0 时成立，且不是归纳的（步长 1 时 c+1 = i+1 是归纳的！）
    cb = counter('cnt_bad', mode='opaque', inv=bad)
    r['c = i：归纳但入口不一定成立 ⇒ 仍可靠，给诊断'] = cb.loop_diag
    worse = lambda iv, its, B: its[0][0] == 2 * iv
    cw = counter('cnt_worse', mode='opaque', inv=worse); r['c = 2i：非归纳 ⇒ 忽略'] = cw.loop_diag
    # 收窄 Pre：体内 100 / (i+1)；不变式 i ≥ lb ≥ 0 ⇒ 不 UB
    dv0 = counter('dv0', mode='opaque', div=True)
    dv = counter('dv', mode='opaque', div=True, inv=lambda iv, its, B: z3.And(iv >= B['lb'], B['lb'] >= 0))
    for nm, s in (('无不变式', dv0), ('有不变式', dv)):
        lb, n = s.holes[0][0], s.holes[1][0]
        r[f'{nm}：lb ≥ 0 ⇒ 不 UB'] = C.prove(z3.Implies(z3.And(s.ax, lb >= 0, n < 1000, z3.Not(s.holes[0][1]), z3.Not(s.holes[1][1])), z3.Not(s.ubf)))
    # 公理与具体执行一致：随机具体输入展开得到的值，满足摘要的约束
    agree = 0
    for _ in range(30):
        a, b = rng.randrange(-5, 6), rng.randrange(-5, 12)
        Cc = C.derive(f'cc{_}', [], [('q', IX)], [blk('entry', [C.cst('lb', a, IX), C.cst('n', b, IX), C.cst('s', 1, IX), C.cst('z', 0, IX),
                      node('L', 'scf.for', ins=['lb', 'n', 's', 'z'], outs=[('r', IX)], regions=ci.body[0]['ops'][2]['regions']), C.ret('rt', 'r')])])
        conc = z3.simplify(Cc.outv[0][0]).as_long()
        sv = z3.Solver(); sv.add(ci.ax, ci.holes[0][0] == a, ci.holes[1][0] == b, nopz, z3.Not(ci.ubf), ci.outv[0][0] == conc)
        agree += sv.check() == z3.sat
    r['公理与具体展开一致'] = f'{agree}/30'
    return r

def W5():
    """L17″（第 21 轮）：index 运算按 2^IW 回绕（LangRef：结果为数学值 mod 2^n），只有 nsw/nuw 使溢出成为毒值。
    对照 L17′（第 6–20 轮，溢出 = 毒值）：它能“证明” (x+1) > x，MLIR 执行在 x = 2^63−1 时给出假。"""
    def ix(nm, ops, ins=(('a', IX), ('b', IX)), out=IX): return C.derive(nm, list(ins), [('q', out)], [blk('entry', ops, args=list(ins))])
    run = lambda body, xs: jit('module {\nfunc.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {\n'
                               '  %a = arith.index_cast %arg0 : i64 to index\n  %b = arith.index_cast %arg1 : i64 to index\n' + body + '\n}\n}', [xs])[0]
    sgn = lambda v: v - (1 << 64) if v >= 1 << 63 else v
    f = ix('ixadd', [C.op2('s', 'arith.addi', 'a', 'b', IX), C.ret('r', 's')])
    fs = ix('ixadd_nsw', [C.op2('s', 'arith.addi', 'a', 'b', IX, nsw=True), C.ret('r', 's')])
    a, b = f.holes[0][0], f.holes[1][0]; nop = z3.And(z3.Not(f.holes[0][1]), z3.Not(f.holes[1][1]))
    wrap = sgn(run('  %s = arith.addi %a, %b : index\n  %r = arith.index_cast %s : index to i64\n  return %r : i64', [1 << 62, 1 << 62, 0]))
    g = ix('ixgt', [C.cst('one', 1, IX), C.op2('s', 'arith.addi', 'a', 'one', IX), C.op2('c', 'arith.cmpi', 's', 'a', 'i1', predicate=4), C.ret('r', 'c')],
           ins=(('a', IX),), out='i1')
    gt_proved = C.prove(z3.Implies(z3.And(g.ax, z3.Not(g.holes[0][1]), z3.Not(g.outv[0][1])), g.outv[0][0] == 1))
    gt_mlir = run('  %one = arith.constant 1 : index\n  %s = arith.addi %a, %one : index\n  %c = arith.cmpi sgt, %s, %a : index\n'
                  '  %r = arith.extui %c : i1 to i64\n  return %r : i64', [(1 << 63) - 1, 0, 0])
    rt = C.derive('rt', [('x', I64)], [('q', I64)], [blk('entry', [node('i', 'arith.index_cast', ins=['x'], outs=[('i', IX)]),
                   node('o', 'arith.index_cast', ins=['i'], outs=[('o', I64)]), C.ret('r', 'o')], args=[('x', I64)])])
    r = {'语义': C.IXOVF,
         'UB ≡ ⊥，MayPoison(a + b : index) ⇔ 输入毒值': C.prove(z3.Implies(f.ax, z3.Not(f.ubf))) and C.prove(z3.Implies(f.ax, f.outv[0][1] == z3.Not(nop))),
         'nsw：MayPoison ⇔ 输入毒值 ∨ 超出 [−2^63, 2^63)': C.prove(z3.Implies(fs.ax, fs.outv[0][1] == z3.Or(fs.holes[0][1], fs.holes[1][1],
                                                     fs.holes[0][0] + fs.holes[1][0] < C.LO, fs.holes[0][0] + fs.holes[1][0] >= C.HI))),
         '2^62 + 2^62：模型值 = MLIR 执行值': C.prove(z3.Implies(z3.And(f.ax, a == 1 << 62, b == 1 << 62, nop), f.outv[0][0] == wrap)) and wrap == -(1 << 63),
         'index addi 可推测（与 MLIR 的 Pure 一致）': C.caps(f)['Speculatable'],
         '(x+1) > x 不可证（MLIR 在 x=2^63−1 执行得假）': (not gt_proved) and gt_mlir == 0,
         'i64 → index → i64 往返恒等': C.prove(z3.Implies(z3.Not(rt.holes[0][1]), rt.outv[0][0] == rt.holes[0][0]))}
    old = C.IXOVF; C.IXOVF = 'poison'
    g2 = ix('ixgt_old', [C.cst('one', 1, IX), C.op2('s', 'arith.addi', 'a', 'one', IX), C.op2('c', 'arith.cmpi', 's', 'a', 'i1', predicate=4), C.ret('r', 'c')],
            ins=(('a', IX),), out='i1')
    C.IXOVF = old
    r['对照 L17′：(x+1) > x 被“证明”（与 MLIR 执行矛盾）'] = C.prove(z3.Implies(z3.And(g2.ax, z3.Not(g2.holes[0][1]), z3.Not(g2.outv[0][1])), g2.outv[0][0] == 1))
    return r

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    rng = random.Random(1)
    print('── W1 pattern 逐位宽证明:', W1())
    print('── W2 与 MLIR 执行的差分:', W2(n, rng))
    print('── W3 循环摘要:'); [print('     ', k, v) for k, v in W3().items()]
    print('── W4 不变式:'); [print('     ', k, v) for k, v in W4(rng).items()]
    print('── W5 index 溢出:'); [print('     ', k, v) for k, v in W5().items()]
    print('── 求解器 unknown/超时次数（按“未证明”处理）:', C.STATS['unknown'])
