"""形状层的检验（第 6 轮）。
  S1 Pre 与 MayPoison：切片、取元素的条件与手写公式等价
  S2 L22 静态维：端口、内部边、调用点的义务；tensor.cast
  S3 改写规则（L23 域内相等）：有效规则证明，错误规则反驳；与“整体相等”对照
  S4 可推测性与 MLIR 的一致性；L17′ 的对照（index 溢出是 UB 时 addi 不可推测，slice 融合不成立）
  S5 降低守卫（L24）：上下文能证明在界内的节点不需要守卫
  S6 与真实 MLIR 执行的差分：随机 tensor 程序，bufferize 后 JIT 执行；只比较模型判定“无守卫、非毒值”的输入
用法：PYTHONPATH=model python3 model/shape_test.py [S6 的程序数]
"""
import random, sys, ctypes, z3
import core as C
from core import blk, node

IX, I8, I64 = 'index', 'i8', 'i64'
TD = 'tensor<?xi8>'
def D(name, ins, outs, ops): return C.derive(name, ins, outs, [blk('entry', ops, args=ins)])
def ci(v, k): return C.cst(v, k, IX)
def nd(v, op, ins, ty): return node(v, op, ins=ins, outs=[(v, ty)])
def H(s, i): return s.holes[i][0]
def nop(s): return C.no_poison(s)

def S1():
    f = D('xs', [('x', TD), ('o', IX), ('n', IX), ('i', IX)], [('q', I8)],
          [nd('s', 'tensor.extract_slice', ['x', 'o', 'n'], TD), nd('e', 'tensor.extract', ['s', 'i'], I8), C.ret('r', 'e')])
    x, o, n, i = (H(f, k) for k in range(4)); d = C.ext(x)[0]
    may = z3.Or(o < 0, n < 0, o + n > d, i < 0, i >= n, C.tpz(x)[o + i])
    return {'UB ≡ ⊥（tensor 层没有 UB，L24）': C.prove(z3.Implies(f.ax, z3.Not(f.ubf))),
            'MayPoison ≡ 切片越界 ∨ 下标越界 ∨ 元素毒值': C.prove(z3.Implies(z3.And(f.ax, nop(f)), f.outv[0][1] == may)),
            '值 = x[o+i]': C.prove(z3.Implies(z3.And(f.ax, nop(f), z3.Not(may)), f.outv[0][0] == C.dat(x)[o + i]))}

def S2():
    r = {}
    g = D('g', [('x', TD), ('n', IX)], [('q', 'tensor<4xi8>')], [ci('z', 0), nd('s', 'tensor.extract_slice', ['x', 'z', 'n'], TD), C.ret('r', 's')])
    r['出口 tensor<4>，尺寸 n 动态 ⇒ 诊断'] = C.shape_diag(g)
    g2 = D('g2', [('x', TD)], [('q', 'tensor<4xi8>')], [ci('z', 0), ci('n', 4), nd('s', 'tensor.extract_slice', ['x', 'z', 'n'], TD), C.ret('r', 's')])
    r['尺寸为常量 4 ⇒ 无诊断'] = C.shape_diag(g2)
    g3 = D('g3', [('x', TD), ('n', IX)], [('q', 'tensor<4xi8>')],
           [ci('z', 0), nd('s', 'tensor.extract_slice', ['x', 'z', 'n'], TD), nd('c', 'tensor.cast', ['s'], 'tensor<4xi8>'), C.ret('r', 'c')])
    r['显式 tensor.cast ⇒ 无诊断，n ≠ 4 时输出是毒值'] = (C.shape_diag(g3), C.prove(z3.Implies(z3.And(g3.ax, nop(g3), H(g3, 1) != 4), g3.outv[0][1])))
    k = D('k4', [('x', 'tensor<4xi8>')], [('q', I8)], [ci('three', 3), nd('e', 'tensor.extract', ['x', 'three'], I8), C.ret('r', 'e')])
    r['端口 tensor<4> 是体的假设：x[3] 不越界'] = C.prove(z3.Implies(z3.And(k.ax, nop(k), C.static_ok(H(k, 0), 'tensor<4xi8>'),
                                                               z3.Not(C.tpz(H(k, 0))[3])), z3.Not(k.outv[0][1])))
    c1 = D('c1', [('x', TD)], [('q', I8)], [nd('u', 'k4', ['x'], I8), C.ret('r', 'u')])
    r['调用点传 tensor<?> 给 tensor<4> ⇒ 调用点诊断'] = C.shape_diag(c1)
    c2 = D('c2', [('a', I8)], [('q', I8)], [nd('t', 'tensor.from_elements', ['a', 'a', 'a', 'a'], 'tensor<4xi8>'), nd('u', 'k4', ['t'], I8), C.ret('r', 'u')])
    r['调用点传 from_elements(4 个) ⇒ 无诊断'] = C.shape_diag(c2)
    return r

def rule(name, ins, outs, src_ops, tgt_ops, assume=None):
    s = D(name + '.src', ins, outs, src_ops); t = D(name + '.tgt', ins, outs, tgt_ops)
    a = assume(s) if assume else True
    return C.refines(s, t, a), s, t

def S3():
    X = [('x', TD), ('o1', IX), ('n1', IX), ('o2', IX), ('n2', IX)]
    ss = [nd('s1', 'tensor.extract_slice', ['x', 'o1', 'n1'], TD), nd('s2', 'tensor.extract_slice', ['s1', 'o2', 'n2'], TD), C.ret('r', 's2')]
    fused = lambda op: [C.op2('o', op, 'o1', 'o2', IX), nd('s', 'tensor.extract_slice', ['x', 'o', 'n2'], TD), C.ret('r', 's')]
    out = {}
    out['slice∘slice → slice(o1+o2)'] = rule('ss', X, [('q', TD)], ss, fused('arith.addi'))[0]
    out['✗ slice∘slice → slice(o1−o2)'] = rule('ssw', X, [('q', TD)], ss, fused('arith.subi'))[0]
    Y = [('x', TD), ('o', IX), ('n', IX), ('i', IX)]
    out['extract(slice(x,o,n), i) → extract(x, o+i)'] = rule('es', Y, [('q', I8)],
        [nd('s', 'tensor.extract_slice', ['x', 'o', 'n'], TD), nd('e', 'tensor.extract', ['s', 'i'], I8), C.ret('r', 'e')],
        [C.op2('k', 'arith.addi', 'o', 'i', IX), nd('e', 'tensor.extract', ['x', 'k'], I8), C.ret('r', 'e')])[0]
    Z = [('v', I8), ('x', TD), ('i', IX), ('j', IX)]
    out['extract(insert(v,x,i), i) → v'] = rule('ei', Z, [('q', I8)],
        [nd('t', 'tensor.insert', ['v', 'x', 'i'], TD), nd('e', 'tensor.extract', ['t', 'i'], I8), C.ret('r', 'e')], [C.ret('r', 'v')])[0]
    out['✗ extract(insert(v,x,i), j) → extract(x, j)'] = rule('eiw', Z, [('q', I8)],
        [nd('t', 'tensor.insert', ['v', 'x', 'i'], TD), nd('e', 'tensor.extract', ['t', 'j'], I8), C.ret('r', 'e')],
        [nd('e', 'tensor.extract', ['x', 'j'], I8), C.ret('r', 'e')])[0]
    W = [('x', TD), ('o', IX), ('n', IX)]
    out['insert_slice(slice(x,o,n), x, o) → x'] = rule('is', W, [('q', TD)],
        [nd('s', 'tensor.extract_slice', ['x', 'o', 'n'], TD), nd('t', 'tensor.insert_slice', ['s', 'x', 'o'], TD), C.ret('r', 't')], [C.ret('r', 'x')])[0]
    out['dim(insert_slice(s,x,o), 0) → dim(x, 0)'] = rule('di', [('s', TD), ('x', TD), ('o', IX)], [('q', IX)],
        [ci('z', 0), nd('t', 'tensor.insert_slice', ['s', 'x', 'o'], TD), nd('d', 'tensor.dim', ['t', 'z'], IX), C.ret('r', 'd')],
        [ci('z', 0), nd('d', 'tensor.dim', ['x', 'z'], IX), C.ret('r', 'd')])[0]
    ok, s, t = rule('fe', [('v', I8)], [('q', 'tensor<2xi8>')],
                    [nd('t', 'tensor.from_elements', ['v', 'v'], 'tensor<2xi8>'), C.ret('r', 't')],
                    [nd('t', 'tensor.splat', ['v'], 'tensor<2xi8>'), C.ret('r', 't')])
    whole = C.prove(z3.Implies(z3.And(s.ax, nop(s)), s.outv[0][0] == z3.substitute(t.outv[0][0], [(t.holes[0][0], s.holes[0][0])])))
    out['from_elements(v,v) → splat(v)：域内相等 / 整体相等'] = (ok, whole)
    return out

def S4():
    def wrap(op, ins, ty, extra=()):
        return C.caps(D(f'w.{op}.{len(extra)}', ins, [('q', ty)], list(extra) + [nd('y', op, [p for p, _ in ins] + [e['id'] for e in extra], ty), C.ret('r', 'y')]))['Speculatable']
    r = {}
    r['index addi 可推测（MLIR: Pure）'] = C.caps(D('ia', [('a', IX), ('b', IX)], [('q', IX)], [C.op2('s', 'arith.addi', 'a', 'b', IX), C.ret('r', 's')]))['Speculatable']
    r['extract_slice 可推测（MLIR: Pure）'] = wrap('tensor.extract_slice', [('x', TD), ('o', IX), ('n', IX)], TD)
    r['extract 可推测（MLIR: Pure）'] = wrap('tensor.extract', [('x', TD), ('i', IX)], I8)
    r['dim(x, i) 可推测（MLIR: 下标未知时不可推测；本模型更强，代价在降低守卫）'] = wrap('tensor.dim', [('x', TD), ('i', IX)], IX)
    # 对照：旧 L17（index 溢出 = UB）
    C.IXOVF = 'ub'
    old_add = C.caps(D('ia_old', [('a', IX), ('b', IX)], [('q', IX)], [C.op2('s', 'arith.addi', 'a', 'b', IX), C.ret('r', 's')]))['Speculatable']
    X = [('x', TD), ('o1', IX), ('n1', IX), ('o2', IX), ('n2', IX)]
    old_fuse = rule('ss_old', X, [('q', TD)],
                    [nd('s1', 'tensor.extract_slice', ['x', 'o1', 'n1'], TD), nd('s2', 'tensor.extract_slice', ['s1', 'o2', 'n2'], TD), C.ret('r', 's2')],
                    [C.op2('o', 'arith.addi', 'o1', 'o2', IX), nd('s', 'tensor.extract_slice', ['x', 'o', 'n2'], TD), C.ret('r', 's')])[0]
    C.IXOVF = 'poison'
    r['对照 旧 L17：index addi 可推测'] = old_add
    r['对照 旧 L17：slice∘slice 融合成立'] = old_fuse
    return r

def S5():
    a = D('ga', [('x', TD), ('i', IX)], [('q', I8)], [nd('e', 'tensor.extract', ['x', 'i'], I8), C.ret('r', 'e')])
    b = D('gb', [('v', I8)], [('q', I8)], [nd('t', 'tensor.splat', ['v'], 'tensor<8xi8>'), ci('k', 5), nd('e', 'tensor.extract', ['t', 'k'], I8), C.ret('r', 'e')])
    c = D('gc', [('x', TD), ('n', IX)], [('q', IX)],
          [ci('z', 0), ci('one', 1), nd('d', 'tensor.dim', ['x', 'z'], IX), C.op2('h', 'arith.divui', 'd', 'one', IX),
           nd('s', 'tensor.extract_slice', ['x', 'z', 'h'], TD), nd('dd', 'tensor.dim', ['s', 'z'], IX), C.ret('r', 'dd')])
    return {'extract(x, i)：需要守卫': C.guards(a), 'extract(splat:8, 5)：不需要': C.guards(b),
            'slice(x, 0, dim x)、dim(·, 0)：不需要': C.guards(c)}

# ───────────────────────── S6 与真实 MLIR 的差分
N0 = 6
def gen(rng, t):
    """main(x, y, z: i64) -> i64：from_elements 建 tensor<6xi64>，随机 insert / extract_slice / insert_slice / extract / dim，返回一个 i64。"""
    k = [0]
    def f(): k[0] += 1; return f'v{t}_{k[0]}'
    ops, ix, sz = [], [], []                      # ix：偏移与下标（小常量与输入），sz：尺寸（小常量与 dim）
    for a in ('x', 'y', 'z'):
        v = f(); ops.append(nd(v, 'arith.index_castui', [a], IX)); ix.append(v)
    for m in (0, 1, 2):
        c = f(); ops.append(ci(c, m)); ix.append(c)
    for m in (1, 2, 3):
        c = f(); ops.append(ci(c, m)); sz.append(c)
    base = f(); ops.append(nd(base, 'tensor.from_elements', ['x', 'y', 'z', 'x', 'y', 'z'], 'tensor<6xi64>')); TY[base] = 'tensor<6xi64>'
    ts, sl, sc = [base], [], ['x', 'y', 'z']
    for _ in range(rng.randint(3, 8)):
        r, v = rng.random(), f()
        if r < 0.25:
            src = rng.choice(ts); TY[v] = TY[src]; ts.append(v)
            ops.append(node(v, 'tensor.insert', ins=[rng.choice(sc), src, rng.choice(ix)], outs=[(v, TY[src])]))
        elif r < 0.5:
            src = rng.choice(ts); TY[v] = 'tensor<?xi64>'; ts.append(v); sl.append(v)
            ops.append(node(v, 'tensor.extract_slice', ins=[src, rng.choice(ix), rng.choice(sz)], outs=[(v, TY[v])]))
        elif r < 0.75 and sl:
            s_ = rng.choice(sl); d_ = rng.choice([x for x in ts if x != s_]); TY[v] = TY[d_]; ts.append(v)
            ops.append(node(v, 'tensor.insert_slice', ins=[s_, d_, rng.choice(ix)], outs=[(v, TY[d_])]))
        elif r < 0.85:
            z = f(); ops += [ci(z, 0), nd(v, 'tensor.dim', [rng.choice(ts), z], IX)]; sz.append(v)
        else: ops.append(nd(v, 'tensor.extract', [rng.choice(ts), rng.choice(ix)], I64)); sc.append(v)
    e = f(); ops.append(nd(e, 'tensor.extract', [ts[-1] if rng.random() < 0.7 else rng.choice(ts), rng.choice(ix)], I64))
    k2 = f(); ops.append(nd(k2, 'arith.index_cast', [rng.choice(sz[3:] or sz)], I64))      # 把一个尺寸（多半是 dim）也带进结果
    s2 = f(); ops.append(C.op2(s2, 'arith.addi', e, k2, I64))
    ops.append(C.ret(f'ret{t}', s2))
    return [blk('entry', ops, args=[('x', I64), ('y', I64), ('z', I64)])]
TY = {}

def emit(blocks):
    nm = {'x': '%arg0', 'y': '%arg1', 'z': '%arg2'}; ty = {'x': I64, 'y': I64, 'z': I64}
    v = lambda x: nm.setdefault(x, '%' + x)
    out = []
    for n in blocks[0]['ops']:
        op, pr, ins = n['op'], n.get('props', {}), n['ins']
        for o, t in n['outs']: ty[o] = t
        o = v(n['outs'][0][0]) if n['outs'] else None
        if op == 'func.return': out.append(f'  return {v(ins[0])} : i64')
        elif op == 'arith.constant': out.append(f'  {o} = arith.constant {pr["value"]} : {pr["type"]}')
        elif op == 'arith.index_castui': out.append(f'  {o} = arith.index_castui {v(ins[0])} : i64 to index')
        elif op == 'arith.index_cast': out.append(f'  {o} = arith.index_cast {v(ins[0])} : index to i64')
        elif op == 'arith.addi': out.append(f'  {o} = arith.addi {v(ins[0])}, {v(ins[1])} : i64')
        elif op == 'tensor.from_elements': out.append(f'  {o} = tensor.from_elements ' + ', '.join(map(v, ins)) + ' : tensor<6xi64>')
        elif op == 'tensor.insert': out.append(f'  {o} = tensor.insert {v(ins[0])} into {v(ins[1])}[{v(ins[2])}] : {ty[ins[1]]}')
        elif op == 'tensor.extract': out.append(f'  {o} = tensor.extract {v(ins[0])}[{v(ins[1])}] : {ty[ins[0]]}')
        elif op == 'tensor.dim': out.append(f'  {o} = tensor.dim {v(ins[0])}, {v(ins[1])} : {ty[ins[0]]}')
        elif op == 'tensor.extract_slice':
            out.append(f'  {o} = tensor.extract_slice {v(ins[0])}[{v(ins[1])}] [{v(ins[2])}] [1] : {ty[ins[0]]} to tensor<?xi64>')
        elif op == 'tensor.insert_slice':
            out.append(f'  {o} = tensor.insert_slice {v(ins[0])} into {v(ins[1])}[{v(ins[2])}] [{_dimof(ins[0], ty, out, v)}] [1] : {ty[ins[0]]} into {ty[ins[1]]}')
        else: raise ValueError(op)
    return ('module {\nfunc.func @main(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 attributes {llvm.emit_c_interface} {\n'
            + '\n'.join(out) + '\n}\n}')
_dk = [0]
def _dimof(t, ty, out, v):
    if ty[t] == 'tensor<6xi64>': return '6'
    _dk[0] += 1; c, d = f'%dc{_dk[0]}', f'%dd{_dk[0]}'
    out.append(f'  {c} = arith.constant 0 : index'); out.append(f'  {d} = tensor.dim {v(t)}, {c} : {ty[t]}'); return d

PIPE = ('builtin.module(one-shot-bufferize{bufferize-function-boundaries},func.func(buffer-deallocation-simplification),'
        'convert-bufferization-to-memref,expand-strided-metadata,lower-affine,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,'
        'convert-cf-to-llvm,reconcile-unrealized-casts)')
def jit(src, inputs):
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse(PIPE).run(m.operation)
        import os, mlir; lib = os.path.join(os.path.dirname(ir.__file__), '_mlir_libs', 'libmlir_c_runner_utils.so')   # memrefCopy
        ee = ExecutionEngine(m, opt_level=0, shared_libs=[lib])
        res = []
        for xs in inputs:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); res.append(r[0] % (1 << 64))
        return res

def S6(n, rng, quiet=False):
    st = dict(程序=0, 输入=0, 一致=0, 不一致=0, 模型毒值=0, 需守卫跳过=0, 未定=0)
    for t in range(n):
        TY.clear(); g = gen(rng, t)
        G = C.derive(f'T{t}', [('x', I64), ('y', I64), ('z', I64)], [('q', I64)], g)
        st['程序'] += 1
        ins = [[rng.randrange(0, 4) for _ in range(3)] for _ in range(6)]
        runnable = []
        for xs in ins:
            pairs = [(h[0], z3.BitVecVal(x, 64)) for h, x in zip(G.holes, xs)] + [(h[1], z3.BoolVal(False)) for h in G.holes]
            sub = lambda f: z3.simplify(z3.substitute(f, pairs))
            gd = [sub(f) for _, f in G.guards]
            if not all(z3.is_true(x) for x in gd):          # MLIR 中会 UB（越界访存）：不能执行，跳过
                st['需守卫跳过'] += 1; continue
            p = sub(G.outv[0][1])
            if z3.is_true(p): st['模型毒值'] += 1; continue
            vv = sub(G.outv[0][0])
            if not z3.is_bv_value(vv): st['未定'] += 1; continue
            runnable.append((xs, vv.as_long()))
        if not runnable: continue
        outs = jit(emit(g), [xs for xs, _ in runnable])
        for (xs, mv), o in zip(runnable, outs):
            st['输入'] += 1
            if mv == o: st['一致'] += 1
            else:
                st['不一致'] += 1
                if st['不一致'] <= 2 and not quiet: print('  ✗', xs, '模型', mv, 'MLIR', o, '\n', emit(g))
    return st

def S7(n):
    """变异灵敏度：把模型的一条语义改错，S6 必须能发现。"""
    out = {}
    def es_off(p, a):                                    # extract_slice 偏移差一
        (t, pt) = a[0]; (o, po), (m, pm) = a[1], a[2]
        return C._extract_slice(p, [(t, pt), (o + 1, po), (m, pm)])
    def is_win(p, a):                                    # insert_slice 窗口多一格
        (s_, ps), (t, pt), (o, po) = a; iv = C._ivars(1); I = tuple(iv); J = (iv[0] - o,)
        win = z3.And(iv[0] >= o, iv[0] <= o + C.ext(s_)[0])
        res = C.mkt(t.sort(), C.ext(t), z3.Lambda(iv, z3.If(win, C.dat(s_)[J], C.dat(t)[I])), z3.Lambda(iv, z3.If(win, C.tpz(s_)[J], C.tpz(t)[I])))
        return [(res, z3.Or(ps, pt, po, C._slice_oob([o], C.ext(s_), C.ext(t))))], C.F_
    def dim_m1(p, a): return [(C.ext(a[0][0])[0] - 1, a[0][1])], C.F_
    for nm, op, fn in [('extract_slice 偏移差一', 'tensor.extract_slice', es_off), ('insert_slice 窗口多一格', 'tensor.insert_slice', is_win),
                       ('dim 少一', 'tensor.dim', dim_m1)]:
        orig = C.REG[op].fn; C.REG[op].fn = fn
        try: r = S6(n, random.Random(2), quiet=True)
        finally: C.REG[op].fn = orig
        out[nm] = f"发现 {r['不一致']}/{r['输入']}"
    return out

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    for nm, f in [('S1 Pre 与 MayPoison', S1), ('S2 L22 静态维', S2), ('S3 改写规则（L23）', S3), ('S4 可推测性', S4), ('S5 降低守卫（L24）', S5)]:
        print('──', nm); [print('     ', k, '→', v) for k, v in f().items()]
    print('── S6 与 MLIR 执行的差分:', S6(n, random.Random(2)))
    print('── S7 变异灵敏度:', S7(150))
    print('── 求解器 unknown:', C.STATS['unknown'])
