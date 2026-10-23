"""第 26 轮：L18 循环摘要的外部化（spec/loops.md §6）。
L18 说：不能展开的 scf.for，其结果 = 以“结构”命名的未解释函数作用于（边界、初值、捕获值）。可靠性要求：
名字相同 ⇒ 语义相同；结构之外凡影响语义的东西都必须作为实参传入，或编码进名字。
  U0 已发现的反例（回归）：复合 op 参数 $k 只出现在循环体内 ⇒ 旧实现“证明” addk<1>(n) ≡ addk<2>(n)；
     JIT（内联后的两段循环）给出 n=1 时 1 ≠ 2。另有正对照：addk<1> ≡ addk<1> 仍可证。
  U1 证伪审计：MLIR 文本 → 解析 → irimport → valimport → core.derive（第 25 轮的导入路径）。
     同一函数里放两段循环 P、Q（Q = P 或 P 的一处变异），返回 cmpi eq。模型若证明“恒为 1（或毒值/UB）”，
     就用 JIT 在随机输入上找 0：找到即不可靠。变异：常量、op、操作数次序、谓词、初值、捕获值、下界、步长、语义开关（IW）。
  U2 完备性：Q = P（换名）时模型能否证明相等（摘要同一性是 L18 的全部价值）。
用法：PYTHONPATH=model python3 model/loopsum_test.py [U1 程序数]
"""
import sys, random, collections
import z3
import core as C
from structure import node, blk

def prove_true(S):
    v, p = S.outv[0]
    return C.prove(z3.Implies(S.ax, z3.Or(p, S.ubf, v == 1)))

# ───────────────────────── U0
def addk_body(kval, I='i64'):
    kk = node('k', 'arith.constant', outs=[('k', I)], props={'value': kval, 'type': I})
    inner = [blk('b', [kk, node('s', 'arith.addi', ins=['acc', 'k'], outs=[('s', I)]), node('y', 'scf.yield', ins=['s'])], args=[('i', 'index'), ('acc', I)])]
    return [node('c0', 'arith.constant', outs=[('c0', 'index')], props={'value': 0, 'type': 'index'}),
            node('c1', 'arith.constant', outs=[('c1', 'index')], props={'value': 1, 'type': 'index'}),
            node('z', 'arith.constant', outs=[('z', I)], props={'value': 0, 'type': I}),
            node('u', 'arith.index_cast', ins=['n'], outs=[('u', 'index')]),
            node('f', 'scf.for', ins=['c0', 'u', 'c1', 'z'], outs=[('r', I)], regions=[inner])]
ADDK_MLIR = '''func.func @main(%n: i64) -> i64 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0 : i64
  %u = arith.index_cast %n : i64 to index
  %r1 = scf.for %i = %c0 to %u step %c1 iter_args(%acc = %z) -> (i64) {
    %k = arith.constant 1 : i64
    %s = arith.addi %acc, %k : i64
    scf.yield %s : i64 }
  %r2 = scf.for %i = %c0 to %u step %c1 iter_args(%acc = %z) -> (i64) {
    %k = arith.constant 2 : i64
    %s = arith.addi %acc, %k : i64
    scf.yield %s : i64 }
  %d = arith.subi %r1, %r2 : i64
  return %d : i64
}'''
def U0():
    import valimport_test as T
    C.ORACLE_ON = False; I = 'i64'; out = {}
    comp = C.derive('addk', [('n', I)], [('q', I)], [blk('entry', addk_body('$k') + [C.ret('ret', 'r')], args=[('n', I)])], params=[('k', 'int')])
    for mode in ('transparent', 'opaque'):
        comp.mode = mode
        for k2 in (2, 1):
            host = [blk('entry', [node('x1', 'addk', ins=['n'], outs=[('x1', I)], props={'k': 1}),
                                  node('x2', 'addk', ins=['n'], outs=[('x2', I)], props={'k': k2}),
                                  node('e', 'arith.cmpi', ins=['x1', 'x2'], outs=[('e', 'i1')], props={'predicate': 0}), C.ret('rr', 'e')], args=[('n', I)])]
            h = C.derive(f'u0_{mode}_{k2}', [('n', I)], [('e', 'i1')], host)
            out[f'{mode} addk<1> ≡ addk<{k2}> 被证明'] = prove_true(h)
    comp.mode = 'transparent'
    out['JIT 内联：n=0..3 时 addk<1>−addk<2>'] = T.jit_run(ADDK_MLIR, 'main', ['i64'], 'i64', [[0], [1], [2], [3]])
    return out

# ───────────────────────── U1/U2
BIN = ['addi', 'subi', 'muli', 'xori', 'andi', 'ori']
def gen_body(rng):
    """循环体：2–4 个 op，用 %acc、%iv64、捕获值 %b、体内常量。返回 [(op, a, b, const)]。"""
    ops = []
    for j in range(rng.randrange(2, 5)):
        k = rng.random()
        if k < 0.75: ops.append(('bin', rng.choice(BIN), rng.choice(['acc', 'iv', 'b', 'K', 'prev']), rng.choice(['acc', 'iv', 'b', 'K', 'prev']), rng.randrange(-3, 9)))
        else: ops.append(('sel', rng.randrange(10), rng.choice(['acc', 'iv', 'b', 'prev']), rng.choice(['acc', 'b', 'K', 'prev']), rng.randrange(-3, 9)))
    return ops
def emit_loop(tag, spec):
    ops, init, cap, lb, st = spec['ops'], spec['init'], spec['cap'], spec['lb'], spec['st']
    L = [f'  %{tag}r = scf.for %{tag}i = %c{lb} to %u step %c{st} iter_args(%{tag}acc = %{init}) -> (i64) {{',
         f'    %{tag}iv = arith.index_cast %{tag}i : index to i64']
    prev = f'%{tag}acc'
    def ref(x, j, c):
        if x == 'acc': return f'%{tag}acc'
        if x == 'iv': return f'%{tag}iv'
        if x == 'b': return f'%{cap}'
        if x == 'prev': return prev
        L.append(f'    %{tag}k{j}_{len(L)} = arith.constant {c} : i64'); return f'%{tag}k{j}_{len(L) - 1}'
    for j, o in enumerate(ops):
        if o[0] == 'bin':
            a, b = ref(o[2], j, o[4]), ref(o[3], j, o[4] + 1)
            L.append(f'    %{tag}v{j} = arith.{o[1]} {a}, {b} : i64')
        else:
            a, b = ref(o[2], j, o[4]), ref(o[3], j, o[4] + 1)
            L.append(f'    %{tag}p{j} = arith.cmpi {C.PRED[o[1]]}, {a}, {b} : i64')
            L.append(f'    %{tag}v{j} = arith.select %{tag}p{j}, {a}, {b} : i64')
        prev = f'%{tag}v{j}'
    L.append(f'    scf.yield {prev} : i64'); L.append('  }')
    return L
def program(P, Q):
    L = ['func.func @main(%a: i64, %b: i64, %n: i64) -> i1 {'] + [f'  %c{k} = arith.constant {k} : index' for k in (0, 1, 2)]
    L.append('  %u = arith.index_cast %n : i64 to index')
    L += emit_loop('P', P) + emit_loop('Q', Q)
    L += ['  %e = arith.cmpi eq, %Pr, %Qr : i64', '  return %e : i1', '}']
    return '\n'.join(L)
def mutate(rng, P):
    Q = dict(P, ops=list(P['ops'])); kind = rng.choice(['常量', 'op', '操作数次序', '谓词', '初值', '捕获值', '下界', '步长'])
    j = rng.randrange(len(Q['ops'])); o = list(Q['ops'][j])
    if kind == '常量': o[4] += rng.choice([1, 2, -1])
    elif kind == 'op':
        if o[0] == 'bin': o[1] = rng.choice([x for x in BIN if x != o[1]])
        else: kind = '谓词'
    elif kind == '操作数次序': o[2], o[3] = o[3], o[2]
    if kind == '谓词':
        if o[0] == 'sel': o[1] = (o[1] + rng.randrange(1, 10)) % 10
        else: kind = '常量'; o[4] += 1
    Q['ops'][j] = tuple(o)
    if kind == '初值': Q['init'] = 'b' if P['init'] == 'a' else 'a'
    if kind == '捕获值': Q['cap'] = 'a' if P['cap'] == 'b' else 'b'
    if kind == '下界': Q['lb'] = 1 - P['lb']
    if kind == '步长': Q['st'] = 2 if P['st'] == 1 else 1
    return Q, kind

def U1(n, seed=26):
    import mlir.ir as ir, irimport as I, valimport_test as T
    rng = random.Random(seed); st = collections.Counter(); bad = []
    C.ORACLE_ON = False
    for t in range(n):
        P = dict(ops=gen_body(rng), init=rng.choice(['a', 'b']), cap=rng.choice(['a', 'b']), lb=rng.choice([0, 1]), st=rng.choice([1, 2]))
        same = rng.random() < 0.3
        Q, kind = (dict(P), '相同') if same else mutate(rng, P)
        if emit_loop('X', P) == emit_loop('X', Q): kind = '相同'          # 变异没改到文本（如改了未用到的常量）
        txt = program(P, Q)
        try: m = ir.Module.parse(txt, ir.Context())
        except Exception as e: st['MLIR 不接受（生成器错误）'] += 1; print(e, txt); continue
        fn = I.import_module(m)['regions'][0][0]['ops'][0]
        S, why = T.derive_imported(f'u1_{t}', fn)
        if S is None: st['值层未知'] += 1; continue
        st[f'{kind}: 程序'] += 1
        if not prove_true(S): continue
        st[f'{kind}: 模型证明相等'] += 1
        if kind == '相同': continue
        inputs = [[rng.choice([0, 1, 2, 3, 5, -1, 7]) for _ in range(2)] + [rng.randrange(0, 6)] for _ in range(12)]
        ev = [(xs, T.model_eval(S, ['i64'] * 3, xs)) for xs in inputs]
        run = [xs for xs, r in ev if r[0] == 'val']
        if not run: continue
        outs = T.jit_run(txt, 'main', ['i64'] * 3, 'i1', run)
        diff = [(xs, o) for xs, o in zip(run, outs) if o == 0]
        st[f'{kind}: 已执行'] += 1
        if diff:
            st[f'{kind}: 不可靠（JIT 不等）'] += 1
            if len(bad) < 3: bad.append((kind, diff[0], txt))
        else: st[f'{kind}: JIT 也相等（变异恰好等价）'] += 1
    return st, bad

def U3():
    """语义开关进名字：同一 Schema 在 IW=64 与 IW=32 下推导的摘要函数不同名。"""
    I = 'i64'; names = []
    for iw in (64, 32):
        C.set_iw(iw)
        S = C.derive(f'u3_{iw}', [('n', I)], [('q', I)], [blk('entry', addk_body(1) + [C.ret('ret', 'r')], args=[('n', I)])])
        fs = set()
        def coll(e):
            if z3.is_app(e):
                if e.decl().kind() == z3.Z3_OP_UNINTERPRETED and e.decl().name().startswith('loop_'): fs.add(e.decl().name())
                for c in e.children(): coll(c)
        coll(S.outv[0][0]); names.append(fs)
    C.set_iw(64)
    return {'IW=64 摘要名': sorted(names[0]), 'IW=32 摘要名': sorted(names[1]), '不相交': not (names[0] & names[1])}

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    print('── U0 复合 op 参数（回归）:', U0())
    print('── U3 语义开关进名字:', U3())
    st, bad = U1(n)
    print('── U1/U2 证伪审计与完备性:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
    for b in bad: print('  ✗', b[0], b[1], '\n', b[2])
    print('── 求解器 unknown:', C.STATS['unknown'])
