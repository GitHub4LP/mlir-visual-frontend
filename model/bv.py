"""值层用 QF_BV 精确求 Pre：op 是确定性的（Sem 是函数），所以 wp 只需要做项代换，不需要投影。
Pre = ¬UB(f(x)) 本身就是一个无量词的位向量公式；只在“问它”时才求解（是否恒真、是否可满足、取反例）。
这里拿 test_oracle 中 Presburger 层只能做到可靠（S）的用例来对比。
"""
import random, sys, z3
import test_oracle as T

def sym(nodes, out, a, b):
    W = T.W; env = {'a': (a, z3.BoolVal(False)), 'b': (b, z3.BoolVal(False))}; ub = []
    for nid, name, w in nodes:
        g = {p: env[s] for p, s in w.items()}
        if isinstance(name, int): v, p = z3.BitVecVal(name, W), z3.BoolVal(False)
        elif name == 'ult':
            v, p = z3.If(z3.ULT(g['x'][0], g['y'][0]), z3.BitVecVal(1, W), z3.BitVecVal(0, W)), z3.Or(g['x'][1], g['y'][1])
        elif name == 'select':
            c, pc = g['c']; t, f = g['t'], g['f']
            v = z3.If(c == 1, t[0], f[0]); p = z3.Or(pc, z3.If(c == 1, t[1], f[1]))
        else:
            (x, px), (y, py) = g['x'], g['y']; p = z3.Or(px, py)
            if name in ('divui', 'remui'): ub.append(z3.Or(py, y == 0))
            v = {'add': x + y, 'add_nuw': x + y, 'sub': x - y, 'sub_nuw': x - y, 'mul': x * y,
                 'divui': z3.UDiv(x, y), 'remui': z3.URem(x, y)}[name]
            if name == 'add_nuw': p = z3.Or(p, z3.Not(z3.BVAddNoOverflow(x, y, False)))
            if name == 'sub_nuw': p = z3.Or(p, z3.ULT(x, y))
        env[nid + '.r'] = (v, p)
    return z3.Not(z3.Or(ub)) if ub else z3.BoolVal(True), env[out][1]

if __name__ == '__main__':
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    a, b = z3.BitVecs('a b', T.W); n_abs = n_ok = 0; n_spec = 0
    for seed in range(N):
        ok, ex, loss, nodes, out = T.check(seed, random.Random(seed).randint(2, 6))
        if ex: continue
        n_abs += 1; pre, poi = sym(nodes, out, a, b)
        good = True
        for x in range(T.M):                       # 与解释器真值逐点比对
            for y in range(T.M):
                t = T.run(nodes, out, x, y)
                s = {a: z3.BitVecVal(x, T.W), b: z3.BitVecVal(y, T.W)}
                zp = z3.is_true(z3.simplify(z3.substitute(pre, *s.items())))
                zq = zp and z3.is_true(z3.simplify(z3.substitute(poi, *s.items())))
                if zp != (t is not None) or (zp and zq != t): good = False; break
            if not good: break
        n_ok += good
        sv = z3.Solver(); sv.add(z3.Not(pre)); n_spec += sv.check() == z3.unsat     # 可推测 ⇔ ¬Pre 不可满足
    print(f'W={T.W}  Presburger 层只能做到可靠的 {n_abs} 个用例：QF_BV 精确 {n_ok}/{n_abs}；其中可推测 {n_spec} 个')
    # 规模试探：i64 下问一次“是否可推测”需要多久
    import time; A, B = z3.BitVecs('A B', 64); T.W = 64
    nodes = [('n0', 'mul', {'x': 'a', 'y': 'b'}), ('n1', 1, {}), ('n2', 'add', {'x': 'n0.r', 'y': 'n1.r'}),
             ('n3', 'divui', {'x': 'a', 'y': 'n2.r'})]
    pre, _ = sym(nodes, 'n3.r', A, B); sv = z3.Solver(); sv.add(z3.Not(pre)); t0 = time.time(); r = sv.check()
    print(f'  i64：a / (a*b + 1) 是否可推测 → {"否" if r == z3.sat else "是"}，反例 {sv.model() if r == z3.sat else "-"}，用时 {time.time()-t0:.2f}s')
