"""自动导入的签名与 MLIR verifier 的差分（第 9 轮）。
  L0 导入账：11 个方言 305 个 op 按 auto / cpred / declared / unknown 分类；端口约束用到的原子与变换数
  L1 差分：无 region/successor/结构 trait、且无必需属性（或只有 I64Enum 必需属性）的 op，随机实例，比较
         “自动签名（ods / ods+cpred）”与 MLIR verifier；分方言、分账目统计误收/误拒
用法：PYTHONPATH=model python3 model/odssig_test.py [每个 op 的实例数]
"""
import random, sys, collections, os
import odsjson as O, odssig as G, typesig as TS
from typesig_test import mlir_ok, CRASH
P = TS.parse
BASES = ['i1', 'i8', 'i32', 'si8', 'index', 'f16', 'f32', 'complex<f32>', '!shape.shape', '!shape.size']
FORMS = ['{}', 'tensor<4x{}>', 'tensor<?x{}>', 'tensor<2x3x{}>', 'tensor<*x{}>', 'tensor<{}>', 'vector<4x{}>', 'vector<{}>', 'vector<2x3x{}>',
         'memref<4x{}>', 'memref<?x{}>', 'memref<*x{}>']

def pool(ctx):
    """只保留 MLIR 能解析的类型（例：vector<4xcomplex<f32>>、tensor<4x!shape.shape> 非法）。"""
    import mlir.ir as ir
    out = []
    with ctx:
        for b in BASES:
            for f in FORMS:
                s = f.format(b)
                if f != '{}' and b.startswith('!'): continue
                try: ir.Type.parse(s); out.append(P(s))
                except Exception: pass
    return out

def eligible(S):
    if S['regions'] or S['successors']: return False
    if [t for t in S['structural'] if 'Segments' not in t]: return False
    return all(not req or kind == 'I64EnumAttr' for _, _, kind, req in S['attrs'])

def instance(rng, S, POOL):
    b = rng.choice(BASES); f = rng.choice(FORMS)
    def one():
        r = rng.random()
        if r < 0.55:
            s = f.format(b) if not (b.startswith('!') and f != '{}') else b
        elif r < 0.7: s = f.format('i1') if f != '{}' else 'i1'
        elif r < 0.8: s = 'index'
        else: return rng.choice(POOL)
        try: t = P(s)
        except Exception: return rng.choice(POOL)
        return t if t in POOL else rng.choice(POOL)
    cnt = lambda m: 1 if m == 'one' else rng.choice([0, 1]) if m == 'optional' else rng.choice([0, 1, 2, 3])
    ins = [[one() for _ in range(cnt(m))] for _, _, m, _ in S['ins']]
    outs = [[one() for _ in range(cnt(m))] for _, _, m, _ in S['outs']]
    return ins, outs

def draw(rng, S, POOL, acc, k):
    """一半实例取前端接受的（拒绝采样，最多 60 次），其中一半再改动一个端口：让样本集中在边界附近。"""
    if k % 2 == 0:
        for _ in range(60):
            ins, outs = instance(rng, S, POOL)
            if not G.check(S, ins, outs): acc.append((ins, outs)); return ins, outs
    if acc and k % 4 == 3:
        ins, outs = [list(g) for g in acc[-1][0]], [list(g) for g in acc[-1][1]]
        gs = [g for g in ins + outs if g]
        if gs: g = rng.choice(gs); g[rng.randrange(len(g))] = rng.choice(POOL)
        return ins, outs
    return instance(rng, S, POOL)

def text(S, ins, outs):
    opd = [t for g in ins for t in g]; res = [t for g in outs for t in g]
    props = [f'{nm} = 0 : i64' for nm, _, kind, req in S['attrs'] if req and kind == 'I64EnumAttr']
    if any('AttrSizedOperandSegments' in t for t in S['structural']): props.append('operandSegmentSizes = array<i32: ' + ', '.join(str(len(g)) for g in ins) + '>')
    if any('AttrSizedResultSegments' in t for t in S['structural']): props.append('resultSegmentSizes = array<i32: ' + ', '.join(str(len(g)) for g in outs) + '>')
    args = ', '.join(f'%a{i}: {TS.show(t)}' for i, t in enumerate(opd))
    lhs = (f'%r:{len(res)} = ' if len(res) > 1 else '%r = ') if res else ''
    pr = (' <{' + ', '.join(props) + '}>') if props else ''
    return (f'func.func @f({args}) {{\n  {lhs}"{S["op"]}"(' + ', '.join(f'%a{i}' for i in range(len(opd))) + f'){pr} : ('
            + ', '.join(TS.show(t) for t in opd) + ') -> (' + ', '.join(TS.show(t) for t in res) + ')\n  return\n}')

def L0():
    tab = collections.defaultdict(collections.Counter); unk = collections.Counter(); atoms = set()
    for dia in O.DIALECTS:
        d = O.records(dia)
        for op in O.ops(dia):
            c, S = G.ledger(dia, op); tab[dia][c] += 1; tab[dia]['op'] += 1
            for u in S['unknown']: unk[u] += 1
            r = d[O.ops(dia)[op]]
            for cc, _ in O.dag_args(r['arguments']) + O.dag_args(r['results']):
                if cc in d and 'TypeConstraint' in d[cc]['!superclasses'] and O.ref(d[cc].get('predicate')):
                    s = set(); O.leaves(d, O.ref(d[cc]['predicate']), s); atoms |= s
    return tab, unk, atoms

def L1(n, rng):
    import mlir.ir as ir
    ctx = ir.Context(); POOL = pool(ctx)
    st = collections.defaultdict(collections.Counter); per = collections.defaultdict(collections.Counter); ex = {}
    for dia in O.DIALECTS:
        for op in O.ops(dia):
            cls, S = G.ledger(dia, op)
            if not eligible(S): st[dia]['不参与（region/后继/结构 trait/必需属性）'] += 1; continue
            st[dia]['参与 op'] += 1
            acc = []
            for k in range(n):
                ins, outs = draw(rng, S, POOL, acc, k)
                nc = len(CRASH); truth = mlir_ok(ctx, text(S, ins, outs))
                if len(CRASH) > nc:
                    st[dia]['MLIR 崩溃'] += 1; st[dia]['崩溃例前端拒绝'] += bool(G.check(S, ins, outs)); continue
                st[dia]['实例'] += 1; st[dia]['MLIR 接受'] += truth
                for lv, L in (('ods', ('ods',)), ('+cpred', ('ods', 'cpred'))):
                    fe = not G.check(S, ins, outs, L)
                    key = f'{lv}:' + ('一致' if fe == truth else '误收' if fe else '误拒')
                    st[dia][key] += 1
                    if lv == '+cpred':
                        per[(op, cls)][key] += 1
                        if fe != truth and (op, fe) not in ex: ex[(op, fe)] = (text(S, ins, outs).split('\n')[1].strip(), G.check(S, ins, outs, L))
    return st, per, ex

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    tab, unk, atoms = L0()
    tot = collections.Counter()
    for v in tab.values(): tot.update(v)
    print('── L0 导入账（op 数）'); [print('   ', k, dict(v)) for k, v in tab.items()]; print('    合计', dict(tot))
    print('    端口约束用到的不同 C++ 片段:', len(atoms), '（原子', sum(a[0] == 'atom' for a in atoms), '，变换', sum(a[0] != 'atom' for a in atoms), '）')
    print('    unknown 条目:', dict(unk))
    st, per, ex = L1(n, random.Random(5))
    print('── L1 差分（每个 op', n, '个实例）')
    for k, v in st.items(): print('   ', k, dict(v))
    bycls = collections.defaultdict(collections.Counter)
    for (op, cls), v in per.items(): bycls[cls].update(v); bycls[cls]['op'] += 1; bycls[cls]['有分歧的 op'] += any('一致' not in k and c for k, c in v.items())
    print('── L1 按账目（ods+cpred）'); [print('   ', k, dict(v)) for k, v in bycls.items()]
    print('── 非 declared 却有分歧的 op（应为 0；否则是导入缺陷或未登记的 C++）')
    for (op, cls), v in per.items():
        if cls in ('auto', 'cpred') and any(('误收' in k or '误拒' in k) and c for k, c in v.items()):
            print('   ', op, cls, dict(v), [ex.get((op, b)) for b in (True, False) if (op, b) in ex])
    print('── declared op 的误收（C++ verify 的缺口）:', sorted(((op, v.get('+cpred:误收', 0)) for (op, cls), v in per.items() if cls == 'declared'), key=lambda x: -x[1])[:15])
