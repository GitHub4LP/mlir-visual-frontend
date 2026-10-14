"""编辑代数与增量查询的检验。
  E1 随机编辑序列：每一步后，增量结果（结构诊断、效应、值层、caps、Pre）≡ 从零全量推导；分别测语法截断和语义截断
  E2 L20：连接洞的项代换 ≡ 连接后重推导；补全分类 ≡ 连接后重推导的分类
  E3 代价：每种编辑重推导了几个定义、早截断省了多少；推导与查询的耗时对比
  E4 定向用例：x+0 → x（语法截断），x*2 → x+x（只有语义截断能截住）
用法：PYTHONPATH=model python3 model/edit_test.py [序列数] [每序列编辑数]
"""
import random, sys, time, copy, z3
import core as C, edit as E
from core import blk, node

T8 = 'i8'; ARGS = [('a', T8), ('b', T8)]
BIN = ['arith.addi', 'arith.subi', 'arith.muli', 'arith.andi', 'arith.xori', 'arith.divui']

def gen_def(rng, name, callees, tag):
    k = [0]
    def f(): k[0] += 1; return f'{tag}{k[0]}'
    pool, ops = ['a', 'b'], []
    for _ in range(rng.randint(3, 7)):
        r, v = rng.random(), f()
        if r < 0.2: ops.append(C.cst(v, rng.choice([0, 1, 2, 255, rng.randrange(256)]), T8))
        elif r < 0.65:
            op = rng.choice(BIN); pr = {'nuw': True} if op in BIN[:3] and rng.random() < 0.3 else {}
            ops.append(C.op2(v, op, rng.choice(pool), rng.choice(pool), T8, **pr))
        elif r < 0.8 and callees:
            ops.append(node(v, rng.choice(callees), ins=[rng.choice(pool), rng.choice(pool)], outs=[(v, T8)]))
        else:
            c = f(); ops.append(C.op2(c, 'arith.cmpi', rng.choice(pool), rng.choice(pool), 'i1', predicate=rng.randrange(10)))
            ops.append(node(v, 'arith.select', ins=[c, rng.choice(pool), rng.choice(pool)], outs=[(v, T8)]))
        pool.append(v)
    for c in callees:                                       # 每个被调者至少被用到一次，结果参与返回值
        v = f(); ops.append(node(v, c, ins=[rng.choice(pool), pool[-1]], outs=[(v, T8)])); pool.append(v)
    ops.append(C.ret(f'{tag}ret', pool[-1]))
    return dict(ins=ARGS, outs=[('q', T8)], mode='transparent', blocks=[blk('entry', ops, args=ARGS)])

def gen_ws(rng, t):
    n0, n1, n2 = f'c0_{t}', f'c1_{t}', f'main_{t}'
    return {n0: gen_def(rng, n0, [], 'x'), n1: gen_def(rng, n1, [n0], 'y'), n2: gen_def(rng, n2, [n0, n1], 'z')}

def random_edit(rng, ws):
    name = rng.choice(list(ws.defs)); ops = E._ops(ws, name); ty = E._types(ws, name)
    body = [n for n in ops if n['op'] != 'func.return']
    holes = [n['id'] for n in ops if n['op'] == 'edit.hole']
    kind = rng.choice(['set_prop', 'set_op', 'rewire', 'rewire', 'insert', 'delete', 'disconnect', 'fill', 'fill'])
    if kind == 'fill' and holes:
        hid = rng.choice(holes); cands = E.complete(ws, name, hid)
        if cands: v, _ = rng.choice(cands); E.fill(ws, name, hid, v); return kind, name
        kind = 'rewire'
    if kind == 'set_prop' and body:
        n = rng.choice(body)
        if n['op'] == 'arith.constant': E.set_prop(ws, name, n['id'], 'value', rng.randrange(256)); return kind, name
        if n['op'] == 'arith.cmpi': E.set_prop(ws, name, n['id'], 'predicate', rng.randrange(10)); return kind, name
        if n['op'] in BIN[:3]: E.set_prop(ws, name, n['id'], 'nuw', not n.get('props', {}).get('nuw', False)); return kind, name
    if kind == 'set_op':
        bs = [n for n in body if n['op'] in BIN]
        if bs: E.set_op(ws, name, rng.choice(bs)['id'], rng.choice(BIN)); return kind, name
    if kind == 'insert':
        i = rng.randrange(len(ops)); sc = E.in_scope(ws, name, ops[i]['id'], T8)
        v = f'n{E._uid()}'
        E.insert(ws, name, i, C.op2(v, rng.choice(BIN), rng.choice(sc), rng.choice(sc), T8) if rng.random() < 0.7 else C.cst(v, rng.randrange(256), T8))
        return kind, name
    if kind == 'delete' and body:
        E.delete(ws, name, rng.choice(body)['id']); return kind, name
    cand = [n for n in ops if n['ins'] and n['op'] != 'edit.hole']
    n = rng.choice(cand); k = rng.randrange(len(n['ins']))
    if kind == 'disconnect': E.disconnect(ws, name, n['id'], k); return kind, name
    sc = [v for v in E.in_scope(ws, name, n['id'], ty[n['ins'][k]])]
    E.rewire(ws, name, n['id'], k, rng.choice(sc)); return 'rewire', name

def agree(a, b):
    if a.struct_diag != b.struct_diag or a.eff != b.eff or a.value_known != b.value_known: return False
    if not a.value_known: return True
    return C.equivalent(a, b) and C.caps(a) == C.caps(b)

def E1(nseq, nedit, rng, semantic):
    st = dict(编辑=0, 一致=0, 不一致=0, 推导=0, 全量推导=0)
    per = {}
    for t in range(nseq):
        ws = E.Workspace(gen_ws(rng, f'{t}{"s" if semantic else ""}'), semantic=semantic); ws.all()
        for _ in range(nedit):
            d0 = ws.stats['推导']
            kind, name = random_edit(rng, ws)
            inc = ws.all(); dn = ws.stats['推导'] - d0
            deps = {'c0': 2, 'c1': 1, 'main': 0}[name.split('_')[0]]
            st['依赖者'] = st.get('依赖者', 0) + deps; st['依赖者被截断'] = st.get('依赖者被截断', 0) + (deps if dn == 0 else deps - (dn - 1))
            full_ws = E.Workspace(ws.defs); full = full_ws.all()
            ok = all(agree(inc[n], full[n]) for n in inc)
            ok = ok and all(C.prove(ws.query(n, 'pre') == C.user_pre(full[n])) for n in inc if full[n].value_known)
            ws.all()                                             # 全量推导改写了 REG；增量工作区的下一步会按需恢复
            st['编辑'] += 1; st['一致' if ok else '不一致'] += 1; st['推导'] += dn; st['全量推导'] += len(inc)
            pk = per.setdefault(kind, [0, 0]); pk[0] += 1; pk[1] += dn
        st.setdefault('截断', 0); st['截断'] += ws.stats['语义截断']
    st['每种编辑平均重推导（共 3 个定义）'] = {k: round(b / a, 2) for k, (a, b) in sorted(per.items())}
    return st

def E2(n, rng):
    st = dict(洞=0, 候选=0, 代换等价=0, 分类一致=0, 分类={'安全': 0, '有条件': 0, '必然UB': 0}, 排除_成环=0)
    for t in range(n):
        ws = E.Workspace(gen_ws(rng, f'h{t}')); ws.all()
        name = rng.choice(list(ws.defs)); ops = E._ops(ws, name)
        cand = [x for x in ops if x['ins']]
        x = rng.choice(cand); hid = None
        if t % 2:                                                # 删除一个被使用的节点：洞在原位，后面的值可能依赖它
            used = [y for y in ops if y['op'] != 'func.return' and any(o in z['ins'] for z in ops for o, _ in y['outs'])]
            if used: hs = E.delete(ws, name, rng.choice(used)['id']); hid = hs[0] if hs else None
        if hid is None: hid = E.disconnect(ws, name, x['id'], rng.randrange(len(x['ins'])))
        sch = ws.schema(name); st['洞'] += 1
        ops2 = E._ops(ws, name)
        first = min(i for i, m in enumerate(ops2) if hid in m['ins'])
        all_scope = E.in_scope(ws, name, ops2[first]['id'], E._types(ws, name)[hid])
        cs = E.complete(ws, name, hid); st['排除_成环'] += len([v for v in all_scope if v != hid]) - len(cs)
        for v, cls in cs:
            st['候选'] += 1; st['分类'][cls] += 1
            w2 = E.Workspace(ws.defs); E.fill(w2, name, hid, v); w2.all(); re_ = w2.schema(name)
            sub = E.subst_fill(sch, hid, *sch.env[v])
            st['代换等价'] += C.equivalent(sub, re_); st['分类一致'] += E.classify(re_) == cls
            ws.all()
    return st

def E3(rng):
    ws = E.Workspace(gen_ws(rng, 'cost')); t = time.time(); ws.all(); td = time.time() - t
    t = time.time(); [C.caps(s) for s in ws.all().values()]; tq = time.time() - t
    return {'推导 3 个定义（ms）': round(td * 1000, 1), 'caps 查询 3 个定义（ms）': round(tq * 1000, 1)}

def E4():
    out = {}
    for semantic in (False, True):
        def mk(body_ops):
            return {'k0': dict(ins=ARGS, outs=[('q', T8)], mode='transparent', blocks=[blk('entry', body_ops, args=ARGS)]),
                    'k1': dict(ins=ARGS, outs=[('q', T8)], mode='transparent',
                               blocks=[blk('entry', [node('u', 'k0', ins=['a', 'b'], outs=[('u', T8)]), C.op2('w', 'arith.divui', 'u', 'b', T8),
                                                     C.ret('r', 'w')], args=ARGS)])}
        ws = E.Workspace(mk([C.cst('z', 0, T8), C.op2('t', 'arith.addi', 'a', 'z', T8), C.ret('r', 't')]), semantic=semantic); ws.all()
        d0 = ws.stats['推导']; E.rewire(ws, 'k0', 'r', 0, 'a'); ws.all()
        a = ws.stats['推导'] - d0
        ws2 = E.Workspace(mk([C.op2('t', 'arith.xori', 'a', 'b', T8), C.ret('r', 't')]), semantic=semantic); ws2.all()
        d0 = ws2.stats['推导']
        E.insert(ws2, 'k0', 0, C.op2('o', 'arith.ori', 'a', 'b', T8)); E.insert(ws2, 'k0', 1, C.op2('n', 'arith.andi', 'a', 'b', T8))
        E.set_op(ws2, 'k0', 't', 'arith.subi'); E.rewire(ws2, 'k0', 't', 0, 'o'); E.rewire(ws2, 'k0', 't', 1, 'n'); ws2.all()
        b = ws2.stats['推导'] - d0
        out['语义截断' if semantic else '语法截断'] = {'x+0 → x 重推导数': a, 'a^b → (a|b)−(a&b) 重推导数': b}
    return out

if __name__ == '__main__':
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    ne = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    rng = random.Random(3)
    print('── E1 增量 ≡ 全量（语法截断）:', E1(ns, ne, rng, False))
    print('── E1 增量 ≡ 全量（语义截断）:', E1(ns // 2, ne, rng, True))
    print('── E2 L20 连接洞 = 项代换；补全:', E2(ns, rng))
    print('── E3 代价:', E3(rng))
    print('── E4 早截断:', E4())
    print('── 求解器 unknown:', C.STATS['unknown'])
