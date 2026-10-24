"""变换层检验：
  R1 定义期：每条 pattern 对全部参数符号地证明（或反驳）精化；度量是否下降
  R2 局部 → 整体：随机宿主程序上，在随机匹配点应用 pattern，整个程序被精化；贪心到不动点时度量逐步下降
  R3 fold 由 Sem 派生（含复合 op）
  R4 翻译确认：随机函数交给真实 MLIR 的 canonicalize，读回结果，用本模型判定 refines(原程序, 结果)
  R5 L13/L14 外部化（第 29 轮）：模型证明的规则在随机宿主中应用（单点、贪心到不动点），原程序与改写后程序
     都交给同一提交 MLIR 降级 + JIT，穷举 128 个输入。凡是模型判定原程序有定义（无 UB、非毒值）的输入，
     要求 JIT(原) = 模型值（翻译健全）且 JIT(改写) = JIT(原)（整体精化）。错误规则作灵敏度对照：
     值层可观测的错误（✗sel_true、✗mul_div）必须被 JIT 抓到差异；只差在毒值上的错误（nuw）JIT 不可观测，单列。
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

JPIPE = 'builtin.module(convert-scf-to-cf,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)'
ALL_IN = [(a, b, c) for a in range(8) for b in range(8) for c in range(2)]
def jit_all(blocks):
    """f(i3,i3,i1)->i3 外包 main(i64,i64,i64)->i64，穷举 128 个输入。"""
    import ctypes
    from mlir.execution_engine import ExecutionEngine
    wrap = ('func.func @main(%x: i64, %y: i64, %z: i64) -> i64 attributes {llvm.emit_c_interface} {\n'
            '  %a = arith.trunci %x : i64 to i3\n  %b = arith.trunci %y : i64 to i3\n  %c = arith.trunci %z : i64 to i1\n'
            '  %r = func.call @f(%a, %b, %c) : (i3, i3, i1) -> i3\n  %e = arith.extui %r : i3 to i64\n  return %e : i64\n}')
    with ir.Context():
        m = ir.Module.parse(emit(blocks) + '\n' + wrap); assert m.operation.verify()
        PassManager.parse(JPIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=0); out = []
        for xs in ALL_IN:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); out.append(r[0] & 7)
        return out

def model_vals(S):
    from valimport_test import model_eval
    return [model_eval(S, [I3, I3, 'i1'], xs) for xs in ALL_IN]

def R5(n, rng):
    st = dict(宿主=0, 定义输入=0, 翻译不符=0, 单点改写=0, 单点模型精化=0, 单点JIT不符=0, 不动点=0, 不动点模型精化=0, 不动点JIT不符=0,
              值错规则应用=0, 值错模型拒=0, 值错JIT抓到=0, 毒值错规则应用=0, 毒值错模型拒=0, 毒值错JIT可见=0)
    VAL_WRONG = {'✗sel_true', '✗mul_div'}
    for t in range(n):
        g = T.dce(gen(rng, 20000 + t)); G = drv(f'J{t}', g); mv = model_vals(G); jg = jit_all(g)
        dfd = [i for i, r in enumerate(mv) if r[0] == 'val']; st['宿主'] += 1; st['定义输入'] += len(dfd)
        st['翻译不符'] += sum(mv[i][1] != jg[i] for i in dfd)
        def diff(h): jh = jit_all(h); return sum(jh[i] != jg[i] for i in dfd)
        cand = [(pt, s) for pt in T.PATTERNS for s in T.sites(pt, g)]
        if cand:
            pt, s = rng.choice(cand); h = T.rewrite(g, pt, s, f'jw{t}'); st['单点改写'] += 1
            st['单点模型精化'] += C.refines(G, drv(f'JH{t}', h)) is True
            d = diff(h); st['单点JIT不符'] += d
            if d: print('  ✗ 模型证明的规则在宿主中被 JIT 反驳:', pt.name, '\n', emit(g), '\n →\n', emit(h))
        cur = g
        for step in range(200):
            cand = [(pt, s) for pt in T.PATTERNS for s in T.sites(pt, cur)]
            if not cand: break
            pt, s = rng.choice(cand); cur = T.dce(T.rewrite(cur, pt, s, f'jg{t}_{step}'))
        if cur is not g:
            st['不动点'] += 1; st['不动点模型精化'] += C.refines(G, drv(f'JF{t}', cur)) is True
            d = diff(cur); st['不动点JIT不符'] += d
            if d: print('  ✗ 不动点被 JIT 反驳\n', emit(g), '\n →\n', emit(cur))
        for pt in T.WRONG:
            ss = T.sites(pt, g)
            if not ss or pt.name == '✗sub_self_dup': continue
            h = T.rewrite(g, pt, rng.choice(ss), f'jx{t}'); r = C.refines(G, drv(f'JX{t}', h)); d = diff(h)
            k = '值错' if pt.name in VAL_WRONG else '毒值错'
            st[k + '规则应用'] += 1; st[k + '模型拒'] += r is False
            if r is False: st[k + ('JIT抓到' if k == '值错' else 'JIT可见')] += d > 0
            else:                                   # 模型接受（宿主上下文里恰好成立）：JIT 不得有差异，否则不可靠
                st[k + '模型接受'] = st.get(k + '模型接受', 0) + 1; st['接受但JIT不符'] = st.get('接受但JIT不符', 0) + (d > 0)
                if d: print('  ✗ 模型接受的错误规则被 JIT 反驳:', pt.name, r, '\n', emit(g), '\n →\n', emit(h))
    return st

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rng = random.Random(0)
    ok, rows = R1(); print(f'── R1 {"✓" if ok else "✗"} 定义期检查（规则, 精化, 度量下降）')
    for r in rows: print('     ', r)
    print('── R2 局部 → 整体:', R2(n, rng))
    print('── R3 派生的 fold:', R3())
    print('── R4 MLIR canonicalize 的翻译确认:', R4(n, rng))
    print('── R5 L13/L14 JIT 外部化:', R5(n, random.Random(29)))
