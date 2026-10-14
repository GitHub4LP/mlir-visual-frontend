"""用穷举预言机检验 core.derive：随机生成 DAG（W 位无符号整数），逐个输入执行带毒值/UB 的解释器，
再与推导出的 Pre 和毒值集合比较。
判据：Sem 精确 ⇒ 相等；Sem 被抽象 ⇒ Pre ⊆ 真值，且毒值集合 ⊇ 真值（可靠方向）。
"""
import random, sys, islpy as isl
from core import Schema, derive

W = int(sys.argv[1]) if len(sys.argv) > 1 else 6; M = 1 << W; H = M - 1
def op(name, ins, sem, ub='false', poison=None, sens=(), prop=None, exact=True, outh=H):
    return Schema(name, [(i, H if i != 'c' else 1) for i in ins], [('r', outh)], sem, ub, poison or {}, set(sens),
                  prop=prop, exact=exact)
L = {
 'add':     (op('add', ['l', 'r'], f'{{r}} = ({{l}} + {{r}}) mod {M}'.replace('{r} =', '{r} =')), lambda a, b: (a + b) % M, lambda a, b: False),
}
# 端口名不能和输出 r 冲突，统一用 x、y
def bop(name, sem, f, pois=None, pf=lambda a, b: False, ub='false', uf=lambda a, b: False, exact=True):
    return (op(name, ['x', 'y'], sem, ub, {'r': pois} if pois else None, ['y'] if ub != 'false' else (), exact=exact), f, pf, uf)
L = {
 'add':     bop('add', f'{{r}} = ({{x}} + {{y}}) mod {M}', lambda a, b: (a + b) % M),
 'add_nuw': bop('add_nuw', f'{{r}} = ({{x}} + {{y}}) mod {M}', lambda a, b: (a + b) % M, f'{{x}} + {{y}} > {H}', lambda a, b: a + b > H),
 'sub':     bop('sub', f'{{r}} = ({{x}} - {{y}}) mod {M}', lambda a, b: (a - b) % M),
 'sub_nuw': bop('sub_nuw', f'{{r}} = ({{x}} - {{y}}) mod {M}', lambda a, b: (a - b) % M, '{x} < {y}', lambda a, b: a < b),
 'mul':     bop('mul', 'true', lambda a, b: (a * b) % M, exact=False),                 # 非线性：抽象
 'divui':   bop('divui', '0 <= {r} <= {x}', lambda a, b: a // b if b else 0, ub='{y} = 0', uf=lambda a, b: b == 0, exact=False),
 'remui':   bop('remui', '0 <= {r} <= {x} and {r} < {y}', lambda a, b: a % b if b else 0, ub='{y} = 0', uf=lambda a, b: b == 0, exact=False),
}
CMP = (op('ult', ['x', 'y'], '({x} < {y} and {r} = 1) or ({x} >= {y} and {r} = 0)', outh=1), lambda a, b: int(a < b))
SEL = op('select', ['c', 't', 'f'], '({c} = 1 and {r} = {t}) or ({c} = 0 and {r} = {f})',
         prop={'r': '({P:c}) or ({c} = 1 and {P:t}) or ({c} = 0 and {P:f})'})
def const(k): return op(f'c{k}', [], f'{{r}} = {k}')

def gen(rng, n):
    vals = [('a', 'i'), ('b', 'i')]; nodes = []
    for k in range(n):
        nid = f'n{k}'; ints = [v for v, t in vals if t == 'i']; bools = [v for v, t in vals if t == 'b']
        kind = rng.choice(['bin'] * 5 + ['cmp', 'const'] + (['sel'] * 2 if bools else []))
        if kind == 'bin':
            name = rng.choice(list(L)); nodes.append((nid, name, {'x': rng.choice(ints), 'y': rng.choice(ints)})); vals.append((nid + '.r', 'i'))
        elif kind == 'cmp':
            nodes.append((nid, 'ult', {'x': rng.choice(ints), 'y': rng.choice(ints)})); vals.append((nid + '.r', 'b'))
        elif kind == 'const':
            nodes.append((nid, rng.choice([0, 1, 2, H]), {})); vals.append((nid + '.r', 'i'))
        else:
            nodes.append((nid, 'select', {'c': rng.choice(bools), 't': rng.choice(ints), 'f': rng.choice(ints)})); vals.append((nid + '.r', 'i'))
    out = [v for v, t in vals if t == 'i'][-1]
    return nodes, out

def run(nodes, out, a, b):
    env = {'a': (a, False), 'b': (b, False)}
    for nid, name, w in nodes:
        g = {p: env[s] for p, s in w.items()}
        if isinstance(name, int): v, p = name, False
        elif name == 'ult': v, p = CMP[1](g['x'][0], g['y'][0]), g['x'][1] or g['y'][1]
        elif name == 'select':
            c, pc = g['c']; v, pv = g['t'] if c == 1 else g['f']; p = pc or pv
        else:
            sch, f, pf, uf = L[name]; (x, px), (y, py) = g['x'], g['y']
            if sch.sensitive and (py or uf(x, y)): return None                 # UB
            v, p = f(x, y), px or py or pf(x, y)
        env[nid + '.r'] = (v, p)
    return env[out][1]

def schema_of(name):
    if isinstance(name, int): return const(name)
    return {'ult': CMP[0], 'select': SEL}.get(name) or L[name][0]

def points(st):
    pts = set(); st.foreach_point(lambda p: pts.add(tuple(p.get_coordinate_val(isl.dim_type.set, i).to_python() for i in range(2))))
    return pts

def check(seed, n):
    rng = random.Random(seed); nodes, out = gen(rng, n)
    sch, pre = derive('t', [('a', H), ('b', H)], {'o': out}, [(nid, schema_of(nm), w) for nid, nm, w in nodes])
    truth = {(a, b): run(nodes, out, a, b) for a in range(M) for b in range(M)}
    t_pre = {k for k, v in truth.items() if v is not None}; t_poi = {k for k, v in truth.items() if v}
    c_pre = points(pre)
    c_poi = points(isl.Set('{ [a, b] : %s }' % sch.poison['o'].replace('{a}', 'a').replace('{b}', 'b')))
    if sch.exact: ok = c_pre == t_pre and c_poi == t_poi
    else: ok = c_pre <= t_pre and c_poi >= t_poi
    loss = (len(t_pre) - len(c_pre)) / max(1, len(t_pre))
    return ok, sch.exact, loss, nodes, out

if __name__ == '__main__':
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    stats = {True: [0, 0], False: [0, 0]}; losses = []; fails = []
    for seed in range(N):
        ok, ex, loss, nodes, out = check(seed, random.Random(seed).randint(2, 6))
        stats[ex][0] += 1; stats[ex][1] += ok
        if not ex: losses.append(loss)
        if not ok: fails.append((seed, nodes, out))
    print(f'W={W}  精确 {stats[True][1]}/{stats[True][0]} 一致；抽象 {stats[False][1]}/{stats[False][0]} 可靠')
    if losses: print(f'  抽象用例的 Pre 损失（真值可接受而被拒的比例）：均值 {sum(losses)/len(losses):.1%}，最大 {max(losses):.1%}，为 0 的 {sum(l == 0 for l in losses)}/{len(losses)}')
    for f in fails[:3]: print('  失败', f)
