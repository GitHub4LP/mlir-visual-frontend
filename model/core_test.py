"""统一核心 v4 的检验：
  T1 随机复合 op（嵌套调用、scf.if、nuw、divui、select），三方比较：
       摘要推导（嵌套） ≡ 全内联推导       —— 闭包（组合律）
       推导 vs 穷举解释器（独立的预言机）  —— 可靠性；exact 时还要相等
  T2 跨层用例 X1–X6（每条对应一个组合律，见 spec/core-v4.md）
用法：PYTHONPATH=model python3 model/core_test.py [用例数] [种子]
"""
import random, sys, itertools, z3
import core as C, structure as ST
from core import blk, node, cst, op2, ret, I3, derive, user_pre, caps, inline_all

class UB(Exception): pass

def run(blocks, env):
    for n in blocks[0]['ops']:
        op = n['op']; p = n.get('props', {})
        if op in ('scf.yield', 'func.return'): return [env[v] for v in n['ins']]
        a = [env[v] for v in n['ins']]
        if op == 'scf.if':
            (c, pc), = a[:1]
            if pc: raise UB('if 的条件是毒值')
            outs = run(n['regions'][0 if c else 1], env)
        elif op == 'arith.constant': outs = [(p['value'], False)]
        elif op == 'arith.addi':
            (x, px), (y, py) = a; outs = [((x + y) % C.M, px or py or (p.get('nuw', False) and x + y > C.MAX))]
        elif op == 'arith.subi':
            (x, px), (y, py) = a; outs = [((x - y) % C.M, px or py or (p.get('nuw', False) and x < y))]
        elif op == 'arith.divui':
            (x, px), (y, py) = a
            if py or y == 0: raise UB('除数为 0 或毒值')
            outs = [(x // y, px)]
        elif op == 'arith.cmpi':
            (x, px), (y, py) = a; sx, sy = [v - C.M if v >= C.M // 2 else v for v in (x, y)]
            rel = [x == y, x != y, sx < sy, sx <= sy, sx > sy, sx >= sy, x < y, x <= y, x > y, x >= y]
            outs = [(int(rel[p['predicate']]), px or py)]
        elif op == 'arith.select':
            (c, pc), (x, px), (y, py) = a; outs = [(x if c else y, pc or (px if c else py))]
        else:
            s = C.REG[op]
            outs = run(s.body, dict(zip([v for v, _ in s.body[0]['args']], a)))
        for (o, _), val in zip(n['outs'], outs): env[o] = val
    raise AssertionError('块没有终结子')

def oracle(s):
    """穷举：输入都不是毒值时，哪些输入安全，哪些输入让输出成为毒值。"""
    safe, pois = set(), set()
    for xs in itertools.product(range(C.M), repeat=len(s.ins)):
        try: r = run(s.body, {v: (x, False) for (v, _), x in zip(s.body[0]['args'], xs)})
        except UB: continue
        safe.add(xs)
        if r[0][1]: pois.add(xs)
    return safe, pois

def model_sets(s):
    """推导结果在每个输入点上求值（输入都不是毒值）。"""
    safe, pois = set(), set()
    pre, mp = C.user_pre(s), C.may_poison(s)
    for xs in itertools.product(range(C.M), repeat=len(s.ins)):
        sub = [(v, C.bv(x)) for (v, _), x in zip(s.holes, xs)] + [(p, C.F_) for _, p in s.holes]
        if z3.is_true(z3.simplify(z3.substitute(pre, sub))):
            safe.add(xs)
            if z3.is_true(z3.simplify(z3.substitute(mp, sub))): pois.add(xs)
    return safe, pois

# ───────────────────────── T1 随机复合 op
def gen_block(rng, pfx, pool, comps, depth, uid):
    ops = []
    def fresh(): uid[0] += 1; return f'{pfx}v{uid[0]}'
    for _ in range(rng.randint(1, 4)):
        r = rng.random(); pick = lambda: rng.choice(pool)
        if r < 0.15: v = fresh(); ops.append(cst(v, rng.randrange(C.M))); pool.append(v)
        elif r < 0.35: v = fresh(); ops.append(op2(v, 'arith.addi', pick(), pick(), nuw=rng.random() < 0.6)); pool.append(v)
        elif r < 0.45: v = fresh(); ops.append(op2(v, 'arith.subi', pick(), pick(), nuw=rng.random() < 0.5)); pool.append(v)
        elif r < 0.6: v = fresh(); ops.append(op2(v, 'arith.divui', pick(), pick())); pool.append(v)
        elif r < 0.7:
            c, v = fresh(), fresh()
            ops += [op2(c, 'arith.cmpi', pick(), pick(), 'i1', predicate=rng.randrange(10)), node(v, 'arith.select', ins=[c, pick(), pick()], outs=[(v, I3)])]
            pool.append(v)
        elif r < 0.85 and depth > 0:
            c, v = fresh(), fresh(); ops.append(op2(c, 'arith.cmpi', pick(), pick(), 'i1', predicate=rng.randrange(10)))
            regs = []
            for side in 'te':
                bo, bp = gen_block(rng, f'{pfx}{side}{uid[0]}_', list(pool), comps, depth - 1, uid)
                regs.append([blk(side, bo + [node(fresh(), 'scf.yield', ins=[rng.choice(bp)])])])
            ops.append(node(v, 'scf.if', ins=[c], outs=[(v, I3)], regions=regs)); pool.append(v)
        elif comps:
            s = C.REG[rng.choice(comps)]; v = fresh()
            ops.append(node(v, s.name, ins=[pick() for _ in s.ins], outs=[(v, I3)])); pool.append(v)
    return ops, pool

def gen(rng, tag, ncomp=3):
    comps = []
    for i in range(ncomp):
        name = f'{tag}c{i}'; k = rng.randint(1, 3 if i < ncomp - 1 else 3)
        args = [(f'{name}a{j}', I3) for j in range(k)]
        ops, pool = gen_block(rng, name + '_', [a for a, _ in args], comps, 2, [0])
        body = [blk('entry', ops + [ret(name + '_ret', rng.choice(pool[k:] or pool))], args=args)]
        derive(name, args, [('q', I3)], body); comps.append(name)
    return C.REG[comps[-1]]

def T1(n, seed):
    rng = random.Random(seed); stat = dict(n=0, 闭包=0, 不可靠=0, 精确相等=0, 仅可靠=0)
    for t in range(n):
        h = gen(rng, f't{seed}_{t}_')
        flat = derive(h.name + '_flat', h.ins, h.outs, inline_all(h.body))
        stat['n'] += 1
        if C.equivalent(h, flat): stat['闭包'] += 1
        else: print('  ✗ 闭包失败', h.name)
        safe, pois = oracle(h); P, MP = model_sets(h)
        if not P <= safe or not pois <= MP: stat['不可靠'] += 1; print('  ✗ 不可靠', h.name)
        elif P == safe and MP == pois: stat['精确相等'] += 1
        else: stat['仅可靠'] += 1
    return stat

# ───────────────────────── T2 跨层用例
def X1():
    """L6 毒值敏感性必须进入摘要。h(x,b,c) = f(x, b +nuw c)，f(x,y) = x / y。"""
    f = derive('X1f', [('x', I3), ('y', I3)], [('q', I3)],
               [blk('entry', [op2('d', 'arith.divui', 'x', 'y'), ret('r', 'd')], args=[('x', I3), ('y', I3)])])
    body = [blk('entry', [op2('s', 'arith.addi', 'b', 'c', nuw=True), node('k', 'X1f', ins=['x', 's'], outs=[('k', I3)]),
                          ret('r', 'k')], args=[('x', I3), ('b', I3), ('c', I3)])]
    h = derive('X1h', [('x', I3), ('b', I3), ('c', I3)], [('q', I3)], body)
    fl = derive('X1h_flat', h.ins, h.outs, inline_all(body))
    import copy
    v3 = copy.copy(f); v3.name = 'X1f_v3'                       # v3 的摘要：不含输入毒值标志（sensitive 被丢掉）
    nz = [(p, C.F_) for _, p in f.holes]
    v3.ubf = z3.substitute(f.ubf, nz); v3.outv = [(v, z3.Or(z3.substitute(p, nz), C.anyp(f.holes))) for v, p in f.outv]
    C.reg(v3)
    old = derive('X1h_v3', h.ins, h.outs, [dict(body[0], ops=[dict(o, op='X1f_v3') if o['op'] == 'X1f' else o for o in body[0]['ops']])])
    ok = C.equivalent(h, fl) and model_sets(h)[0] == oracle(h)[0] and not C.equivalent(old, fl)
    return ok, f'v4 嵌套 ≡ 内联 ≡ 预言机；v3 式摘要与内联不等价（漏掉 b+c 溢出产生的毒值除数）'

def X2():
    """编辑操作“提取为复合 op”保持语义：if 分支里用到外层值 x（捕获）→ 自动提升为端口（L3）。"""
    def host():
        return [blk('entry', [
            cst('one', 1), op2('c', 'arith.cmpi', 'one', 'b', 'i1', predicate=6),
            node('if', 'scf.if', ins=['c'], outs=[('r', I3)], regions=[
                [blk('t', [op2('s', 'arith.addi', 'x', 'b', nuw=True), op2('d', 'arith.divui', 's', 'b'),
                           node('y1', 'scf.yield', ins=['d'])])],
                [blk('e', [node('y2', 'scf.yield', ins=['x'])])]]),
            ret('rr', 'r')], args=[('x', I3), ('b', I3)])]
    before = derive('X2_before', [('x', I3), ('b', I3)], [('q', I3)], host())
    hb = host(); tb = hb[0]['ops'][2]['regions'][0][0]
    comp, newtb = C.extract('X2_ext', tb, 0, 2, ST._types_of(hb))
    hb[0]['ops'][2]['regions'][0][0] = newtb
    after = derive('X2_after', [('x', I3), ('b', I3)], [('q', I3)], hb)
    return C.equivalent(before, after), f'提取出的端口 = {[p for p, _ in comp.ins]}，Pre = {C.pretty(user_pre(after))}'

def X3():
    """caps 逐层派生：g = b≠0 ? a/b : 0。g 本身在非毒值输入上可推测，体内的 divui 不可以；毒值输入下 g 也不可推测。"""
    g = derive('X3g', [('a', I3), ('b', I3)], [('q', I3)],
               [blk('entry', [cst('z', 0), op2('c', 'arith.cmpi', 'z', 'b', 'i1', predicate=6),
                              node('if', 'scf.if', ins=['c'], outs=[('r', I3)], regions=[
                                  [blk('t', [op2('d', 'arith.divui', 'a', 'b'), node('y1', 'scf.yield', ins=['d'])])],
                                  [blk('e', [node('y2', 'scf.yield', ins=['z'])])]]),
                              ret('rr', 'r')], args=[('a', I3), ('b', I3)])])
    k = caps(g)
    return k['Speculatable|非毒值输入'] and not k['Speculatable'], f'非毒值输入上可推测 = {k["Speculatable|非毒值输入"]}，允许毒值输入时 = {k["Speculatable"]}'

def X4():
    """效应 × 区域：alloc 在外、只在 then 分支 dealloc ⇒ else 路径泄漏；同块 alloc/dealloc ⇒ 抵消，Eff = ∅。"""
    def mk(name, free_in_branch):
        ops = [node('m', 'memref.alloc', outs=[('m', 'memref')]), node('st', 'memref.store', ins=['x', 'm']),
               node('ld', 'memref.load', ins=['m'], outs=[('l', I3)])]
        tb = [node('fr', 'memref.dealloc', ins=['m'])] if free_in_branch else []
        ops += [cst('one', 1), op2('c', 'arith.cmpi', 'one', 'x', 'i1', predicate=6),
                node('if', 'scf.if', ins=['c'], outs=[], regions=[[blk('t', tb + [node('y1', 'scf.yield')])],
                                                                    [blk('e', [node('y2', 'scf.yield')])]])]
        if not free_in_branch: ops.append(node('fr', 'memref.dealloc', ins=['m']))
        return derive(name, [('x', I3)], [('q', I3)], [blk('entry', ops + [ret('r', 'l')], args=[('x', I3)])])
    a, b = mk('X4_branch', True), mk('X4_same', False)
    return bool(a.eff_diag) and not b.eff_diag and not b.eff, f'分支内释放：{a.eff_diag}；同块释放：Eff={set(b.eff) or "∅"}'

def X5():
    """L5：体内含 cf 跳转 ⇒ needs_ssacfg；预测“内联进 Graph 区域会违反 R7”，与内联后真实检查一致。"""
    ST.STRUCT['test.graph'] = ST.StructSchema('test.graph', regions=[ST.RegionSpec(kind='Graph', term=None)])
    body = [blk('entry', [node('b1', 'cf.br', succs=[('x', [])])], args=[('a', I3)]),
            blk('x', [ret('r', 'a')])]
    s = derive('X5c', [('a', I3)], [('q', I3)], body)
    host = ST.M(ST.F('h', [blk('entry', [
        node('gr', 'test.graph', regions=[[blk('entry', [cst('k', 1)] + [dict(n, id='in_' + n['id']) for n in body[0]['ops']])]]),
        node('r', 'func.return')])]))
    host['regions'][0][0]['ops'][0]['regions'][0][0]['ops'][0]['regions'][0].append(dict(body[1], ops=[node('rr', 'test.end')]))
    real = {o for _, o, _ in ST.check(host)}
    return s.summary.needs_ssacfg and 'R7' in real and not s.value_known, f'预测 needs_ssacfg={s.summary.needs_ssacfg}；内联后实际违例 {sorted(real)}；值层降为未知（多块 CFG）'

def X6():
    """L4：透明模式下自由符号改由宿主的符号表解析。复合 op 定义在模块 A，被用在模块 B，两边各有一个 @h。"""
    body = [blk('entry', [node('c1', 'func.call', callee='@h'), ret('r')])]
    s = derive('X6c', [], [], body)
    def resolve(table, name): return ST._visible_syms(table).get(name.lstrip('@'))
    A = node('A', 'builtin.module', sym='A', regions=[[blk('entry', [ST.F('h', [blk('entry', [ret('r0')])], nid='hA')])]])
    B = node('B', 'builtin.module', sym='B', regions=[[blk('entry', [ST.F('h', [blk('entry', [ret('r1')])], nid='hB')])]])
    differ = any(resolve(A, f)[1]['id'] != resolve(B, f)[1]['id'] for f in s.summary.free_syms)
    return differ and s.summary.free_syms == ['@h'], f'自由符号 {s.summary.free_syms}：定义处解析到 hA，使用处解析到 hB ⇒ 透明模式语义改变，必须拒绝或改为不透明'

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seeds = [int(sys.argv[2])] if len(sys.argv) > 2 else [0, 1, 2]
    print('── T1 随机复合 op：嵌套 ≡ 内联，推导 ⊑ 预言机')
    for sd in seeds: print(f'  种子 {sd}:', T1(n, sd))
    print('── T2 跨层组合律')
    for f in (X1, X2, X3, X4, X5, X6):
        ok, msg = f(); print(f'  {"✓" if ok else "✗"} {f.__name__}: {f.__doc__.strip().splitlines()[0]}\n      {msg}')
