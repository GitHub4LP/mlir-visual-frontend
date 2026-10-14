"""变换层检验：
  R1 定义期：每条 pattern 对全部参数符号地证明（或反驳）精化；度量是否下降
  R2 局部 → 整体：随机宿主程序上，在随机匹配点应用 pattern，整个程序被精化；贪心到不动点时度量逐步下降
  R3 fold 由 Sem 派生（含复合 op）
  R4 翻译确认：随机函数交给真实 MLIR 的 canonicalize，读回结果，用本模型判定 refines(原程序, 结果)
用法：PYTHONPATH=model python3 model/transform_test.py [随机用例数]
"""
import random, sys, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
import core as C, transform as T
from core import blk, node, I3

INS = [('a', I3), ('b', I3), ('c', 'i1')]

def gen(rng, t, depth=1):
    k = [0]
    def fresh(): k[0] += 1; return f'p{t}_{k[0]}'
    def body(pool, d):
        ops = []
        for _ in range(rng.randint(2, 6)):
            i3, i1 = pool[I3], pool['i1']
            pick = lambda: rng.choice(i3[-3:] if rng.random() < 0.6 else i3)
            v = fresh(); r = rng.random()
            if r < 0.18: ops.append(C.cst(v, rng.choice([0, 1, 2, 7]))); i3.append(v); continue
            if r < 0.7:
                op = rng.choice(['arith.addi', 'arith.subi', 'arith.divui', 'arith.muli', 'arith.addi'])
                x = pick(); y = x if rng.random() < 0.3 else pick()
                ops.append(C.op2(v, op, x, y, **({'nuw': rng.random() < 0.4} if op in ('arith.addi', 'arith.subi') else {})))
                i3.append(v)
            elif r < 0.82:
                x = pick(); y = x if rng.random() < 0.3 else pick()
                ops.append(C.op2(v, 'arith.cmpi', x, y, 'i1', predicate=rng.randrange(10))); i1.append(v)
            elif r < 0.92 or d == 0:
                x = pick(); y = x if rng.random() < 0.3 else pick()
                ops.append(node(v, 'arith.select', ins=[rng.choice(i1), x, y], outs=[(v, I3)])); i3.append(v)
            else:
                regs = []
                for s in 'te':
                    sub = {I3: list(i3), 'i1': list(i1)}; bo = body(sub, d - 1)
                    regs.append([blk(s, bo + [node(fresh(), 'scf.yield', ins=[sub[I3][-1]])])])
                ops.append(node(v, 'scf.if', ins=[rng.choice(i1)], outs=[(v, I3)], regions=regs)); i3.append(v)
        return ops
    pool = {I3: ['a', 'b'], 'i1': ['c']}
    ops = body(pool, depth)
    return [blk('entry', ops + [C.ret(f'ret{t}', pool[I3][-1])], args=INS)]

def drv(nm, blocks): return C.derive(nm, INS, [('q', I3)], blocks)

def R1():
    rows = []
    for pt in T.PATTERNS + T.EXTRA + T.WRONG:
        rows.append((pt.name, T.check(pt), T.decreasing(pt)))
    pats = list(rows)
    # 有效应时值层精化不作数：返回 None（降为声明）
    ms = [('m', 'memref')]
    for nm, body in (('E0', [node('m0', 'memref.alloc', outs=[('m0', 'memref')]), node('d', 'memref.dealloc', ins=['m0']), C.cst('z', 0), C.ret('r', 'z')]),
                     ('E1', [C.cst('z', 0), node('s', 'memref.store', ins=['z', 'm']), node('s2', 'memref.store', ins=['z', 'm']), C.ret('r', 'z')]),
                     ('E2', [C.cst('z', 0), node('s', 'memref.store', ins=['z', 'm']), C.ret('r', 'z')]),
                     ('E3', [C.cst('z', 0), C.ret('r', 'z')])):
        C.derive(nm, ms, [('q', I3)], [blk('entry', body, args=ms)])
    rows.append(('同块 alloc+dealloc → 删除', C.refines(C.REG['E0'], C.REG['E3']), 'L7 抵消，Eff=∅'))
    rows.append(('两次 store → 一次（有效应）', C.refines(C.REG['E1'], C.REG['E2']), '降为声明'))
    ok = rows[-2][1] is True and rows[-1][1] is None and all(c and d for n, c, d in pats if not n.startswith('✗')) and \
         all(c is False for n, c, d in pats if n.startswith('✗') and n != '✗sub_self_dup') and \
         dict((n, (c, d)) for n, c, d in pats)['✗sub_self_dup'] == (True, False)
    return ok, rows

def R2(n, rng):
    st = dict(宿主=0, 应用=0, 整体精化=0, 错误规则应用=0, 错误被整体抓到=0, 不动点=0, 度量逐步下降=0, 不动点精化=0, 平均步数=0)
    for t in range(n):
        g = T.dce(gen(rng, t)); G = drv(f'G{t}', g); st['宿主'] += 1      # 先删死代码：匹配点都影响输出
        cand = [(pt, s) for pt in T.PATTERNS for s in T.sites(pt, g)]
        if cand:
            pt, s = rng.choice(cand); h = T.rewrite(g, pt, s, f'w{t}')
            st['应用'] += 1; st['整体精化'] += C.refines(G, drv(f'H{t}', h)) is True
        bad = [(pt, s) for pt in T.WRONG[:4] for s in T.sites(pt, g)]
        if bad:
            pt, s = rng.choice(bad); h = T.rewrite(g, pt, s, f'x{t}')
            st['错误规则应用'] += 1; st['错误被整体抓到'] += C.refines(G, drv(f'X{t}', h)) is False
        # 贪心到不动点（随机顺序），每步检查度量严格下降
        cur, m, steps, mono = g, T.measure(g), 0, True
        while steps < 200:
            cand = [(pt, s) for pt in T.PATTERNS for s in T.sites(pt, cur)]
            if not cand: break
            pt, s = rng.choice(cand); cur = T.dce(T.rewrite(cur, pt, s, f'g{t}_{steps}')); steps += 1
            m2 = T.measure(cur); mono &= m2 < m; m = m2
        st['不动点'] += steps < 200; st['度量逐步下降'] += mono; st['平均步数'] += steps
        st['不动点精化'] += C.refines(G, drv(f'F{t}', cur)) is True
    st['平均步数'] = round(st['平均步数'] / max(n, 1), 2)
    # 规范形是否唯一：同一宿主按两种随机顺序做到不动点，比较结果
    for name, rules in (('仅 PATTERNS', T.PATTERNS), ('加上 drop_nuw', T.PATTERNS + T.EXTRA)):
        diff = 0
        for t in range(n):
            g = T.dce(gen(random.Random(t), t)); nfs = set()
            for seed in range(4):
                r2, cur = random.Random(seed), g
                for step in range(200):
                    cand = [(pt, s) for pt in rules for s in T.sites(pt, cur)]
                    if not cand: break
                    pt, s = r2.choice(cand); cur = T.dce(T.rewrite(cur, pt, s, f'c{seed}_{step}'))
                nfs.add(shape(cur))
            diff += len(nfs) > 1
        st[f'规范形不唯一（{name}）'] = diff
    return st

def shape(blocks, D=None):
    """与命名无关的形状：从返回值出发展开的项（DAG 展开为树）。D：整个程序的定义表（区域里能看到外层定义）。"""
    D = D or T.defs_of(blocks)
    def tr(v):
        n = D.get(v)
        if n is None: return v
        pr = tuple(sorted((k, str(x)) for k, x in n.get('props', {}).items()))
        return (n['op'], pr, tuple(tr(x) for x in n['ins']), tuple(tuple(shape(r, D)) for r in n['regions']))
    rets = [n for b in blocks for n in b['ops'] if n['op'] in ('func.return', 'scf.yield')]
    return tuple(tr(x) for n in rets for x in n['ins'])

def R3():
    C.derive('f', [('x', I3), ('y', I3)], [('q', I3)],
             [blk('entry', [C.op2('d', 'arith.divui', 'x', 'y'), C.op2('s', 'arith.addi', 'd', 'x', nuw=True), C.ret('r', 's')],
                  args=[('x', I3), ('y', I3)])])
    return {'divui(6,2)': T.fold('arith.divui', [6, 2]), 'divui(1,0)': T.fold('arith.divui', [1, 0]),
            'addi nuw(5,3)': T.fold('arith.addi', [5, 3], {'nuw': True}), 'addi(5,3)': T.fold('arith.addi', [5, 3]),
            '复合 f(6,2)=6/2+nuw 6': T.fold('f', [6, 2]), '复合 f(2,1)=2/1+nuw 2': T.fold('f', [2, 1])}

# ───────────────────────── R4 翻译确认
def emit(blocks):
    nm = {'a': '%arg0', 'b': '%arg1', 'c': '%arg2'}; ty = dict(INS)
    def v(x): return nm.setdefault(x, '%' + x)
    def ops(bs, ind):
        out = []
        for n in bs[0]['ops']:
            for o, t in n['outs']: ty[o] = t
            res = ', '.join(v(o) for o, _ in n['outs'])
            pr = n.get('props', {}); op = n['op']
            if op == 'arith.constant': at = f'<{{value = {pr["value"] % (2 if pr.get("type") == "i1" else 8)} : {pr.get("type", "i3")}}}>'
            elif op == 'arith.cmpi': at = f'<{{predicate = {pr["predicate"]} : i64}}>'
            elif op in ('arith.addi', 'arith.subi'): at = '<{overflowFlags = #arith.overflow<nuw>}>' if pr.get('nuw') else ''
            else: at = ''
            sig = '(' + ', '.join(ty[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
            if op == 'func.return': out.append(ind + 'return ' + ', '.join(v(x) for x in n['ins']) + ' : i3'); continue
            if op == 'scf.yield': out.append(ind + f'scf.yield {v(n["ins"][0])} : i3'); continue
            if op == 'scf.if':
                out.append(ind + f'{res} = scf.if {v(n["ins"][0])} -> (i3) {{')
                out += ops(n['regions'][0], ind + '  '); out.append(ind + '} else {'); out += ops(n['regions'][1], ind + '  '); out.append(ind + '}')
                continue
            out.append(ind + f'{res} = "{op}"(' + ', '.join(v(x) for x in n['ins']) + f') {at} : {sig}')
        return out
    return 'func.func @f(%arg0: i3, %arg1: i3, %arg2: i1) -> i3 {\n' + '\n'.join(ops(blocks, '  ')) + '\n}'

KNOWN = {'arith.constant', 'arith.addi', 'arith.subi', 'arith.divui', 'arith.muli', 'arith.cmpi', 'arith.select',
         'arith.andi', 'arith.ori', 'arith.xori', 'arith.extui', 'scf.if', 'scf.yield', 'func.return'}
def read_back(func):
    names = {}; cnt = [0]
    for i, a in enumerate(func.regions[0].blocks[0].arguments): names[a.get_name()] = 'abc'[i]
    def vid(val):
        k = val.get_name()
        if k not in names: cnt[0] += 1; names[k] = f'm{cnt[0]}'
        return names[k]
    def rd(block):
        ops = []
        for op in block.operations:
            o = op.operation; name = o.name
            if name not in KNOWN: raise KeyError(name)
            outs = [(vid(r), str(r.type)) for r in o.results]; ins = [vid(x) for x in o.operands]; pr = {}
            if name == 'arith.constant':
                t = str(o.results[0].type); w = 1 if t == 'i1' else 3
                pr = {'value': ir.IntegerAttr(o.attributes['value']).value % (1 << w), 'type': t}
            elif name == 'arith.cmpi': pr = {'predicate': ir.IntegerAttr(o.attributes['predicate']).value}
            elif name in ('arith.addi', 'arith.subi', 'arith.muli'):
                pr = {'nuw': 'nuw' in str(o.attributes['overflowFlags'])} if 'overflowFlags' in o.attributes else {}
            elif name == 'arith.extui': pr = {'type': str(o.results[0].type)}
            regs = [[blk(f'r{cnt[0]}{j}', rd(rg.blocks[0]))] for j, rg in enumerate(o.regions)]
            ops.append(node(f'n{cnt[0]}_{len(ops)}_{name}', name, ins=ins, outs=outs, props=pr, regions=regs))
            cnt[0] += 1
        return ops
    return [blk('entry', rd(func.regions[0].blocks[0]), args=INS)]

def R4(n, rng):
    st = dict(n=0, 精化=0, 不可靠=0, 含未建模op=0, 变小=0)
    for t in range(n):
        g = gen(rng, 10000 + t); G = drv(f'M{t}', g)
        with ir.Context():
            m = ir.Module.parse(emit(g)); assert m.operation.verify()
            PassManager.parse('builtin.module(func.func(canonicalize))').run(m.operation)
            func = m.body.operations[0]
            try: h = read_back(func)
            except KeyError as e: st['含未建模op'] += 1; continue
            txt = str(m)
        st['n'] += 1; H = drv(f'MC{t}', h); r = C.refines(G, H)
        if r: st['精化'] += 1
        else: st['不可靠'] += 1; print('  ✗ canonicalize 结果不是精化\n', emit(g), '\n  →\n', txt)
        st['变小'] += T.measure(h) < T.measure(g)
    return st

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = random.Random(0)
    ok, rows = R1(); print(f'── R1 {"✓" if ok else "✗"} 定义期检查（规则, 精化, 度量下降）')
    for r in rows: print('     ', r)
    print('── R2 局部 → 整体:', R2(n, rng))
    print('── R3 派生的 fold:', R3())
    print('── R4 MLIR canonicalize 的翻译确认:', R4(n, rng))
