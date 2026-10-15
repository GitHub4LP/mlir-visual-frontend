"""第 15 轮：结构义务审计——每条义务在真实 IR 上被触发多少次，以及能不能抓住只违反它的变异（spec/obligations.md）。

动机：第 14 轮发现 S1–S4 在真实 IR 上从未触发，“误拒 0”是空检验。本测试对 R1–R8、S1–S4 全部做同样的复查。
  Z1 触发数：MLIR 接受的 2108 个模块整模块判定，structure.CHECKED 记录每条义务的求值次数；按带区域 op 统计哪些由 R8 覆盖。
  Z2 变异差分：以真实模块为底，每种变异只违反一条义务，交 MLIR（预言机子进程）判定。
     我们拒、MLIR 收 = 误拒（必须 0）；MLIR 拒、我们收 = 漏判，按变异种类与父 op 分账。
用法：PYTHONPATH=model python3 model/obligations_test.py [每模块变异上限=12]
"""
import sys, re, glob, collections, copy, random
import mlir.ir as ir
import irimport as I, structure as ST, symtab as SY, oracle as OR, core as C

def walk(n, parent=None):
    yield n, parent
    for r in n['regions']:
        for b in r:
            for m in b['ops']: yield from walk(m, n)

def blocks_of(g):
    for n, _ in walk(g):
        for ri, r in enumerate(n['regions']):
            for b in r: yield n, ri, r, b

def mutations(g, rnd, cap):
    """(种类, 父 op, 变异图)。每种只动一处。"""
    out = []
    def M(fn, kind, par):
        h = copy.deepcopy(g); ix = {}
        for n, _ in walk(h): ix[n['id']] = n
        try:
            if fn(h, ix) is False: return
        except (StopIteration, IndexError): return
        # 通用形式在 op 上写出操作数类型：改了操作数就按值的类型刷新 _types（否则是发射伪影，MLIR 报的是解析错）。
        # 类型环境逐区域下行（兄弟区域复用值名）
        def refresh(n, env):
            for r in n['regions']:
                e2 = ST._types_of(r, {} if n['op'] in ('func.func', 'builtin.module') else dict(env), deep=False)
                for b in r:
                    for x in b['ops']:
                        flat = x['ins'] + [v for _, sa in x['succs'] for v in sa]
                        old = x.get('props', {}).get('_types', [])
                        if len(old) == len(flat): x['props']['_types'] = [e2.get(v, t) for v, t in zip(flat, old)]
                        refresh(x, e2)
        refresh(h, {})
        out.append((kind, par, h))
    bl = list(blocks_of(g))
    for n, ri, r, b in bl:
        if not b['ops']: continue
        last = b['ops'][-1]
        if ST.st(last['op']).terminator:
            # R2a：删掉终结子
            M(lambda h, ix, n=n, ri=ri, bn=b['name']: next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn)['ops'].pop() and None,
              'R2 删终结子', n['op'])
            # R2b：终结子挪到块首
            if len(b['ops']) >= 2 and not last['outs']:
                def f(h, ix, n=n, ri=ri, bn=b['name']):
                    ops = next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn)['ops']; ops.insert(0, ops.pop())
                M(f, 'R2 终结子不在块尾', n['op'])
            # R8/R4：终结子的一个操作数换成同作用域内另一类型的值
            if last['ins'] or any(sa for _, sa in last['succs']):
                tys = ST._types_of(r, deep=False)
                if n['op'] == 'func.func' or 'IsIsolatedFromAbove' in (SY.interfaces(n['op']) or ()): scope = tys
                else: scope = tys     # 外层值也可见，但只取本区域内的值即足够构造类型错
                def f(h, ix, n=n, ri=ri, bn=b['name'], tys=tys):
                    t = next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn)['ops'][-1]
                    slots = [('ins', i) for i in range(len(t['ins']))] + [('succ', (si, ai)) for si, (_, sa) in enumerate(t['succs']) for ai in range(len(sa))]
                    kind, at = rnd.choice(slots)
                    cur = t['ins'][at] if kind == 'ins' else t['succs'][at[0]][1][at[1]]
                    defs = ST._defs_of(next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn) and [next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn)])
                    cands = [v for v in list(defs) if tys.get(v) and tys.get(v) != tys.get(cur)]
                    if not cands: return False
                    v = rnd.choice(cands)
                    if kind == 'ins': t['ins'][at] = v
                    else: t['succs'][at[0]][1][at[1]] = v
                M(f, ('R4 后继实参换类型' if last['succs'] else 'R8 终结子操作数换类型'), n['op'])
        # R5：把一个被同块后继者使用的定义挪到使用者之后
        for k in range(len(b['ops']) - 1):
            a, c = b['ops'][k], b['ops'][k + 1]
            if any(v in c['ins'] for v, _ in a['outs']) and not ST.st(c['op']).terminator:
                def f(h, ix, n=n, ri=ri, bn=b['name'], k=k):
                    ops = next(x for x in ix[n['id']]['regions'][ri] if x['name'] == bn)['ops']; ops[k], ops[k + 1] = ops[k + 1], ops[k]
                M(f, 'R5 定义挪到使用之后', n['op']); break
    # R1：给带区域的 op 删掉区域里的全部块（区域变空）
    for n, _ in walk(g):
        if n['op'] != 'builtin.module' and n['regions'] and n['regions'][0]:
            M(lambda h, ix, n=n: ix[n['id']]['regions'][0].clear(), 'R1 区域置空', n['op'])
    # R6：区域内定义的值在区域 op 之后被使用（越出作用域）
    for n, par in walk(g):
        if par is None or not n['regions'] or n['op'] == 'func.func': continue
        inner = [v for r in n['regions'] for b in r for o in b['ops'] for v, _ in o['outs']]
        if not inner: continue
        def f(h, ix, n=n, v=inner[0]):
            for r in [x for p, _ in walk(h) for x in p['regions']]:
                for b in r:
                    ids = [o['id'] for o in b['ops']]
                    if n['id'] in ids:
                        k = ids.index(n['id'])
                        for o in b['ops'][k + 1:]:
                            if o['ins']: o['ins'][0] = v; return
            return False
        M(f, 'R6 区域内的值越出作用域', n['op'])
    rnd.shuffle(out)
    return out[:cap]

def full(h):
    """前端的完整无预言机判定：结构 + 符号（整模块）+ 每个函数体的类型层（自动签名、属性层；预言机关闭）。"""
    out = [('struct',) + tuple(x) for x in ST.check(h)]
    if out: return out
    C.ORACLE_ON = False
    for n, _ in walk(h):
        if n['op'] == 'func.func' and n['regions'] and n['regions'][0]:
            C.TYPE_WARN.clear()
            try: out += [('type', x) for x in C.typecheck(n['regions'][0], n['props']['_attrs'].get('function_type'))]
            except Exception: pass
    return out

def main(cap=12):
    rnd = random.Random(15)
    z1 = collections.Counter(); byop = collections.Counter(); st = collections.Counter()
    byk = collections.Counter(); ex = collections.defaultdict(list); orc = OR.Oracle()
    def note(k, v):
        if len(ex[k]) < 6: ex[k].append(v)
    R8_COVERED = {'func.func', 'scf.if', 'scf.for', 'scf.execute_region'}
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read(); unreg = 'allow-unregistered-dialect' in src
        for k, ch in enumerate(I.split_chunks(src)):
            if not re.search(r'\S', I.strip_comments(ch)): continue
            ctx = ir.Context(); ctx.allow_unregistered_dialects = unreg
            try: m = ir.Module.parse(ch, ctx); g = I.import_module(m)
            except Exception: continue
            ST.CHECKED.clear(); dd = full(g); z1.update(ST.CHECKED); z1['模块'] += 1
            if dd: z1['误拒'] += 1
            for n, _ in walk(g):
                if n['regions'] and n['op'] != 'builtin.module':
                    byop[('R8 覆盖' if n['op'] in R8_COVERED else 'R8 未覆盖（区域分支类型交给自动签名/预言机）', n['op'])] += 1
            if unreg or dd: continue
            base = '\n'.join(I.emit_generic(g))
            if orc.verify(base)[0] != 'ok': continue
            for kind, par, h in mutations(g, rnd, cap):
                try: ours = full(h)
                except Exception as e: st['我们异常'] += 1; note('我们异常', f'{kind} {par}: {type(e).__name__} {e}'[:160]); continue
                s_, ds = orc.verify('\n'.join(I.emit_generic(h)))
                rej = s_ != 'ok'; st['变异'] += 1
                key = ('拒' if ours else '收') + '/' + ('拒' if rej else '收')
                byk[(kind, key)] += 1
                if ours and not rej: st['误拒'] += 1; note('误拒 ' + kind, f'{par}: {ours[0]}'[:200])
                if rej and not ours:
                    st['漏判'] += 1; byk[(kind + ' 漏判父 op', par)] += 1
                    note('漏判 ' + kind, f"{par}: {' | '.join(x for _, x in ds)[:150]}")
                if ours and rej: st['两者都拒'] += 1
                if not ours and not rej: st['两者都收'] += 1
    print('── Z1 触发数（真实 IR，MLIR 接受）', dict(z1))
    agg = collections.Counter()
    for (cls, op), v in byop.items(): agg[cls] += v
    print('   带区域 op 实例', dict(agg))
    print('   R8 未覆盖的 op（实例数前 15）', sorted(((v, op) for (c, op), v in byop.items() if c.startswith('R8 未')), reverse=True)[:15])
    print('── Z2 变异差分', dict(st)); print('   预言机', orc.stats)
    for kk, v in sorted(byk.items(), key=lambda x: (x[0][0], -x[1])): print('     ', kk, v)
    for kk, v in ex.items(): print('──', kk); [print('    ', x) for x in v]
    orc.close()

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
