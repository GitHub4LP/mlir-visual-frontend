"""定义层检验：
  D1 attribute 项 vs 真实 MLIR 解析器（接受与否、规范打印）
  D2 op 参数与元数据 vs 真实 MLIR verifier
  D3 复合 op 的参数：推导出的约束 vs MLIR；参数化摘要的闭包
  D4 用户类型：生成的降级满足数据精化；错误降级被抓到；程序级 1:N 降级的精化（随机）
  D5 interface：定律定向检查、外部实现、泛型实例化、declared 不能用于推导
用法：PYTHONPATH=model python3 model/defs_test.py [随机用例数]
"""
import random, sys, z3
import mlir.ir as ir
import core as C, defs as D
from core import blk, node, I3
from defs import Term, iT, INDEX, int_lit, OF

def mlir_attr(txt):
    with ir.Context():
        try: return str(ir.Attribute.parse(txt))
        except Exception: return None

def mlir_verify(src):
    with ir.Context() as c:
        try: m = ir.Module.parse(src); return m.operation.verify(), ''
        except Exception as e:
            msg = [l for l in str(e).split('\n') if 'error' in l]; return False, (msg[0] if msg else str(e))[:140]

def emit(n, types):
    v = lambda x: '%' + x.replace('.', '_').replace('#', '_')
    res = ', '.join(v(o) for o, _ in n['outs'])
    pr = ', '.join(f'{k} = {t}' for k, t in n.get('props', {}).items())
    at = ', '.join(f'{k} = {t}' for k, t in n.get('attrs', {}).items())
    sig = '(' + ', '.join(types[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
    return (f'{res} = ' if res else '') + f'"{n["op"]}"(' + ', '.join(v(x) for x in n['ins']) + ')' + \
           (f' <{{{pr}}}>' if pr else '') + (f' {{{at}}}' if at else '') + f' : {sig}'
def emit_func(args, ops):
    types = dict(args)
    for n in ops: types.update(dict(n['outs']))
    return 'func.func @f(' + ', '.join(f'%{a}: {t}' for a, t in args) + ') {\n' + \
           '\n'.join('  ' + emit(n, types) for n in ops) + '\n  return\n}'

# ───────────────────────── D1
def D1(n, rng):
    st = dict(n=0, 一致=0, 不一致=0)
    for _ in range(n):
        w = rng.randint(1, 9); sg = rng.choice(D.SIGN); ty = iT(w, sg)
        v = rng.randint(-(1 << w) - 2, (1 << w) + 2)
        t = int_lit(v, ty); m = mlir_attr(f'{v} : {ty}')
        if rng.random() < 0.3 and t is not None:               # 嵌进数组，混入别的种类
            t = Term('ArrayAttr', ((t, Term('StringAttr', ('s',)), Term('TypeAttr', (ty,)),
                                    Term('SymbolRefAttr', ('a', ('b',))), Term('ArrayAttr', ((),))),))
            m = mlir_attr(f'[{v} : {ty}, "s", {ty}, @a::@b, []]')
        st['n'] += 1
        ok = (t is None and m is None) or (t is not None and not D.verify(t) and str(t) == m)
        st['一致' if ok else '不一致'] += 1
        if not ok: print(f'  ✗ {v} : {ty}  模型={t}  MLIR={m}')
    same = int_lit(3, iT(2)) == int_lit(-1, iT(2))
    return st, f"'3 : i2' 与 '-1 : i2' 是同一个项: {same}；打印为 {int_lit(3, iT(2))}"

# ───────────────────────── D2
def rand_node(rng, i):
    op = rng.choice(['arith.constant', 'arith.cmpi', 'arith.addi'])
    props, attrs = {}, {}
    if op == 'arith.constant':
        ty = rng.choice([iT(3), iT(4), iT(64)])
        if rng.random() < 0.9: props['value'] = int_lit(rng.randint(0, 3), ty) if rng.random() < 0.85 else Term('StringAttr', ('x',))
        outs, ins = [(f'c{i}', 'i3')], []
    elif op == 'arith.cmpi':
        if rng.random() < 0.9:
            r = rng.random()
            props['predicate'] = int_lit(rng.randint(-2, 12), iT(64)) if r < 0.7 else (
                int_lit(rng.randint(0, 9), iT(32)) if r < 0.85 else OF('nuw'))
        outs, ins = [(f'c{i}', 'i1')], ['a', 'b']
    else:
        r = rng.random()
        if r < 0.7: props['overflowFlags'] = OF(*rng.choice([(), ('nsw',), ('nuw',), ('nsw', 'nuw')]))
        elif r < 0.8: props['overflowFlags'] = int_lit(3, iT(32))
        outs, ins = [(f'c{i}', 'i3')], ['a', 'b']
    if rng.random() < 0.15: props['bogus'] = int_lit(1, iT(64))
    if rng.random() < 0.3: attrs['foo'] = int_lit(1, iT(64))
    if rng.random() < 0.1 and D.OPSIG[op]:
        p = D.OPSIG[op][0].name; attrs[p] = props.get(p) or int_lit(1, iT(64))
    return node(f'c{i}', op, ins=ins, outs=outs, props=props, attrs=attrs)

def D2(n, rng):
    st = dict(n=0, 一致=0, 保守=0, 不可靠=0)
    for i in range(n):
        nd = rand_node(rng, i)
        errs, _ = D.verify_node(nd)
        ok_r, msg = mlir_verify(emit_func([('a', 'i3'), ('b', 'i3')], [nd]))
        st['n'] += 1
        if (not errs) == ok_r: st['一致'] += 1
        elif not errs and not ok_r: st['不可靠'] += 1; print('  ✗ 不可靠', emit(nd, {'a': 'i3', 'b': 'i3'}), msg)
        else:
            cats = {c for c, _ in errs}
            if cats & set(D.CONSERVATIVE): st['保守'] += 1        # 由 P1/P2 引起（P2 时缺参数 P3 是同一原因）
            else: st['不可靠'] += 1; print('  ✗ 未登记的保守', errs, emit(nd, {'a': 'i3', 'b': 'i3'}))
    return st

# ───────────────────────── D3
def D3():
    body = [blk('entry', [node('c', 'arith.cmpi', ins=['a', 'b'], outs=[('c', 'i1')], props={'predicate': '$k'}),
                          node('s', 'arith.addi', ins=['a', 'b'], outs=[('s', I3)], props={'overflowFlags': '$f'}),
                          node('r', 'arith.select', ins=['c', 's', 'b'], outs=[('r', I3)]), C.ret('ret', 'r')],
                args=[('a', I3), ('b', I3)])]
    errs, rng_ = D.derive_params('cmpk', [('k', Term('IntegerAttr', (iT(64), None))), ('f', OF())], body)
    k = z3.Int('cmpk.$k')
    agree = 0
    for kv in range(-3, 13):
        inl = [dict(n, props={'predicate': int_lit(kv, iT(64))}) if n['op'] == 'arith.cmpi' else
               dict(n, props={'overflowFlags': OF('nuw')}) if n['op'] == 'arith.addi' else n for n in body[0]['ops'][:-1]]
        ok_r, _ = mlir_verify(emit_func([('a', 'i3'), ('b', 'i3')], inl))
        ours = z3.is_true(z3.simplify(z3.substitute(rng_, (k, z3.IntVal(kv)))))
        agree += ours == ok_r
    # 参数化摘要：每个合法参数下，嵌套调用 ≡ 内联
    comp = C.derive('cmpk', [('a', I3), ('b', I3)], [('q', I3)], D.to_core(body), params=[('k', 'int'), ('f', 'bool')])
    closure = 0
    for kv in range(10):
        for fv in (False, True):
            host = [blk('entry', [node('x', 'cmpk', ins=['a', 'b'], outs=[('x', I3)], props={'k': kv, 'f': fv}), C.ret('r', 'x')],
                        args=[('a', I3), ('b', I3)])]
            h = C.derive(f'h{kv}{fv}', comp.ins, comp.outs, host)
            fl = C.derive(f'h{kv}{fv}f', comp.ins, comp.outs, C.inline_all(host))
            closure += C.equivalent(h, fl)
    bad_errs, _ = D.derive_params('bad', [('k', Term('IntegerAttr', (iT(3), None))), ('f', OF())], body)
    return agree == 16 and closure == 20 and bad_errs and not errs, \
        f'推导出的约束 {rng_}；与 MLIR 一致 {agree}/16；参数化摘要闭包 {closure}/20；声明成 i3 参数时报 {bad_errs[0][1]}'

# ───────────────────────── D4
def D4(n, rng):
    opt = D.define_adt('option', [('none', []), ('some', [('val', I3)])])
    pair = D.define_adt('pair', [('mk', [('fst', I3), ('snd', I3)])])
    chk = {**D.check_adt_lowering(opt), **D.check_adt_lowering(pair)}
    # 用“表示函数”检查 none 的降级：要求载荷是 0 且不是毒值 ⇒ 错误地拒绝正确的降级
    L = C.REG['lower[option.none]']
    rep_ok = C.prove(z3.And(L.outv[0][0] == 0, L.outv[1][0] == 0, z3.Not(L.outv[1][1])))
    # 两个错误的降级
    saved = dict(opt.lower_ops)
    C.derive('bad_is_some', [('v0', 'i1'), ('v1', I3)], [('r', 'i1')],
             [blk('entry', [C.cst('k', 1, 'i1'), node('eq', 'arith.cmpi', ins=['v0', 'k'], outs=[('eq', 'i1')], props={'predicate': 1}),
                            C.ret('rr', 'eq')], args=[('v0', 'i1'), ('v1', I3)])])
    opt.lower_ops['option.is_some'] = 'bad_is_some'
    bad1 = D.check_adt_lowering(opt)['option.is_some']; opt.lower_ops = saved
    C.derive('bad_mk', [('fst', I3), ('snd', I3)], [('o0', I3), ('o1', I3)], [blk('entry', [C.ret('r', 'snd', 'fst')], args=[('fst', I3), ('snd', I3)])])
    sp = dict(pair.lower_ops); pair.lower_ops['pair.mk'] = 'bad_mk'
    bad2 = D.check_adt_lowering(pair)['pair.mk']; pair.lower_ops = sp
    # 程序级：随机程序，逐 op 精化 ⇒ 整体精化
    st = dict(n=0, 精化=0)
    for t in range(n):
        ins, ops, body = gen_opt(rng, t)
        A = C.derive(f'P{t}', ins, [('q', body[1])], [blk('entry', ops + [C.ret('r', body[0])], args=ins)])
        lb = D.lower_blocks([blk('entry', ops + [C.ret('r', body[0])], args=ins)])
        Lw = C.derive(f'P{t}.low', lb[0]['args'], [(f'o{i}', ty) for i, ty in enumerate(D.ADTS[body[1]].lowered if body[1] in D.ADTS else [body[1]])], lb)
        st['n'] += 1; st['精化'] += D.refines_lowered(A, Lw, [ty for _, ty in ins], [body[1]])
    return all(chk.values()) and not rep_ok and not bad1 and not bad2 and st['精化'] == st['n'], \
        f'生成的降级逐 op 精化 {sum(chk.values())}/{len(chk)}；表示函数检查会错误拒绝 none：{not rep_ok}；' \
        f'错误的 is_some 被抓：{not bad1}；交换字段的 pair.mk 被抓：{not bad2}；随机程序整体精化 {st["精化"]}/{st["n"]}'

def gen_opt(rng, t):
    ins = [('a', I3), ('b', I3), ('o', 'option')]
    pool = {I3: ['a', 'b'], 'option': ['o'], 'i1': []}; ops = []; k = [0]
    def fresh(): k[0] += 1; return f'v{t}_{k[0]}'
    def step(ops, pool, depth):
        r = rng.random(); v = fresh()
        if r < 0.15: ops.append(node(v, 'option.some', ins=[rng.choice(pool[I3])], outs=[(v, 'option')])); pool['option'].append(v)
        elif r < 0.25: ops.append(node(v, 'option.none', outs=[(v, 'option')])); pool['option'].append(v)
        elif r < 0.45: ops.append(node(v, 'option.is_some', ins=[rng.choice(pool['option'])], outs=[(v, 'i1')])); pool['i1'].append(v)
        elif r < 0.55: ops.append(node(v, 'option.some.val', ins=[rng.choice(pool['option'])], outs=[(v, I3)])); pool[I3].append(v)
        elif r < 0.7: ops.append(C.op2(v, rng.choice(['arith.addi', 'arith.divui']), rng.choice(pool[I3]), rng.choice(pool[I3]))); pool[I3].append(v)
        elif depth > 0:
            if pool['i1'] and rng.random() < 0.6: c = rng.choice(pool['i1'])          # 条件来自 is_some，或来自比较
            else: c = fresh(); ops.append(C.op2(c, 'arith.cmpi', rng.choice(pool[I3]), rng.choice(pool[I3]), 'i1', predicate=rng.randrange(10)))
            ty = rng.choice([I3, 'option']); regs = []
            for side in 'te':
                bo, bp = [], {kk: list(vv) for kk, vv in pool.items()}
                for _ in range(rng.randint(0, 2)): step(bo, bp, depth - 1)
                regs.append([blk(side, bo + [node(fresh(), 'scf.yield', ins=[rng.choice(bp[ty])])])])
            ops.append(node(v, 'scf.if', ins=[c], outs=[(v, ty)], regions=regs)); pool[ty].append(v)
    for _ in range(rng.randint(2, 6)): step(ops, pool, 2)
    ty = rng.choice([I3, 'option'])
    return ins, ops, (rng.choice(pool[ty]), ty)

# ───────────────────────── D5
def D5():
    def comp(nm, ins, outs, ops, r):
        return C.derive(nm, ins, outs, [blk('entry', ops + [C.ret(nm + '.r', *r)], args=ins)])
    comp('i3.zero', [], [('q', I3)], [C.cst('z', 0)], ['z'])
    for nm, op, fl in [('i3.add', 'arith.addi', False), ('i3.add_nuw', 'arith.addi', True), ('i3.sub', 'arith.subi', False)]:
        comp(nm, [('a', I3), ('b', I3)], [('q', I3)], [C.op2('s', op, 'a', 'b', nuw=fl)], ['s'])
    comp('option.first', [('a', 'option'), ('b', 'option')], [('q', 'option')],
         [node('c', 'option.is_some', ins=['a'], outs=[('c', 'i1')]),
          node('r', 'scf.if', ins=['c'], outs=[('r', 'option')], regions=[[blk('t', [node('y1', 'scf.yield', ins=['a'])])],
                                                                             [blk('e', [node('y2', 'scf.yield', ins=['b'])])]])], ['r'])
    rows = {}
    for ty, u, c in [('i3', 'i3.zero', 'i3.add'), ('i3', 'i3.zero', 'i3.add_nuw'), ('i3', 'i3.zero', 'i3.sub'),
                     ('option', 'option.none', 'option.first')]:
        rows[c] = D.implement('Monoid', ty, {'unit': u, 'combine': c})
    D.implement('Monoid', 'i3', {'unit': 'i3.zero', 'combine': 'i3.add'})        # 最后登记的实现生效
    # 泛型复合 op：reduce3<T: Monoid>
    gen = [blk('entry', [node('l', 'Monoid.combine', ins=['x', 'y'], outs=[('l', 'Self')]),
                         node('q', 'Monoid.combine', ins=['l', 'z'], outs=[('q', 'Self')]), C.ret('r', 'q')],
               args=[('x', 'Self'), ('y', 'Self'), ('z', 'Self')])]
    r3 = {ty: D.monomorphize('reduce3', gen, 'Monoid', ty, [('x', 'Self'), ('y', 'Self'), ('z', 'Self')], [('q', 'Self')])
          for ty in ('i3', 'option')}
    try: D.monomorphize('reduce3', gen, 'Monoid', 'pair', [('x', 'Self')] * 3, [('q', 'Self')]); no_impl = False
    except TypeError: no_impl = True
    D.IMPLS[('Monoid', 'pair')] = ({}, 'declared', {})                  # 例如来自 ODS 的声明，前端没有可执行实现
    gate = (D.law_ok('Monoid', 'option', 'right_id'), D.law_ok('Monoid', 'pair', 'assoc'))
    derived = {op: C.caps(C.REG[op])['Pure'] for op in ('option.some', 'option.some.val', 'option.is_some')}
    expect = {'i3.add': {'assoc': (True, True), 'left_id': (True, True), 'right_id': (True, True)},
              'i3.sub': {'assoc': (False, False), 'left_id': (False, False), 'right_id': (True, True)},
              'option.first': {'assoc': (False, True), 'left_id': (True, True), 'right_id': (True, False)}}
    ok = all(rows[k] == v for k, v in expect.items()) and no_impl and gate == (True, False) and all(derived.values())
    fmt = lambda r: ' '.join(f'{law}:{"→" if a else "·"}{"←" if b else "·"}' for law, (a, b) in r.items())
    return ok, '定律（→ 表示左改写成右有效，← 表示反向有效）\n' + '\n'.join(f'        {k:14s} {fmt(v)}' for k, v in rows.items()) + \
        f'\n      reduce3 在 i3 和 option 上实例化，caps={C.caps(r3["option"])["Speculatable|非毒值输入"]}（option 的非毒值可推测）；' \
        f'pair 没有实现时拒绝实例化：{no_impl}；declared 实现不能用于改写：{not gate[1]}；生成的 ADT op 派生为 Pure：{all(derived.values())}'

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    rng = random.Random(0)
    st, msg = D1(n, rng); print('── D1 attribute 项 vs MLIR 解析器:', st, '\n     ', msg)
    print('── D2 op 参数与元数据 vs MLIR verifier:', D2(n, rng), '\n      保守类别:', D.CONSERVATIVE)
    for f in (D3, lambda: D4(max(n // 3, 20), rng), D5):
        ok, msg = f(); print(f'── {"✓" if ok else "✗"} {msg}')
