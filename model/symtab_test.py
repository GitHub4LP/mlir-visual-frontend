"""第 14 轮：符号层的外部真值检验（spec/symbols.md）。

  Y0 导入账：定义者 / 符号表 / 使用者（ODS trait/interface，含接口继承）。
  Y1 真实模块不误拒：MLIR 测试集里 MLIR 接受的每个段，把**整个模块**交给 structure.check（第 12、13 轮的 C2 只把单个函数
     装进新模块，且导入节点没有手工符号字段 ⇒ 符号义务在真实 IR 上从未触发）。统计实际被检查的符号使用数。
  Y2 变异差分：以真实模块为底，只改符号层的东西，交 MLIR（预言机子进程）判定：
       引用改成不存在的名字 / 改成同模块里另一个符号（种类或类型不同）/ 目标改成 private、nested / 让两个符号重名。
     我们拒 ⇒ MLIR 拒（L36a）；MLIR 以符号诊断拒 ⇒ 我们拒（L36b，对登记的使用义务精确）。
用法：PYTHONPATH=model python3 model/symtab_test.py [每段最多变异数=40]
"""
import sys, re, glob, collections, copy
import mlir.ir as ir
import irimport as I, structure as ST, symtab as SY, oracle as OR

SYMMSG = re.compile(r"does not reference a valid|reference to undefined|incorrect number of (operands|results) for callee|"
                    r"type mismatch|mismatched type|does not match type|redefinition of symbol|not a symbol|"
                    r"could not be found|references a symbol|visibility|private|nested symbol|expected .*symbol")

def walk(n):
    yield n
    for r in n['regions']:
        for b in r:
            for m in b['ops']: yield from walk(m)

def diags(g, attrs=False):
    """结构 + 符号义务；attrs=True 时再加属性层（attrsig，第 13 轮）——前端的完整符号层判定。"""
    out = list(ST.check(g))
    if attrs:
        import attrsig as A
        for n in walk(g):
            if '_attrs' in n.get('props', {}): out += [(n['id'], 'attr', m) for _, m in A.check(n['op'], n['props']['_attrs'])]
    return out

def mutations(g):
    """(说明, 变异后的图) 生成器：只动符号层。"""
    nodes = list(walk(g))
    users = [n for n in nodes if n['op'] in SY.USES and SY.ref_of(n)]
    syms = [n for n in nodes if SY.sym_of(n) and n['op'] != 'builtin.module']
    names = [SY.sym_of(n) for n in syms]
    def mut(f, why):
        h = copy.deepcopy(g); M = {x['id']: x for x in walk(h)}; M['__root__'] = h; f(M); return why, h
    for u in users:
        attr = SY.USES[u['op']][0]
        yield mut(lambda M, u=u, a=attr: M[u['id']]['props']['_attrs'].__setitem__(a, '@__nope'), f"{u['op']} → @__nope")
        for nm in names:
            if '@' + nm != u['props']['_attrs'][attr]:
                yield mut(lambda M, u=u, a=attr, nm=nm: M[u['id']]['props']['_attrs'].__setitem__(a, f'@{nm}'), f"{u['op']} → 另一个符号")
    for s_ in syms:
        for v in ('"private"', '"nested"'):
            yield mut(lambda M, s_=s_, v=v: M[s_['id']]['props']['_attrs'].__setitem__('sym_visibility', v), f"{s_['op']} 可见性 {v}")
    # 嵌套符号表：把被引用的目标移进新的嵌套模块 @inner，引用改成 @inner::@f（或保留扁平引用），目标可见性三种
    top = g['regions'][0][0]['ops']
    for u in users[:2]:
        attr = SY.USES[u['op']][0]; nm = u['props']['_attrs'][attr].lstrip('@')
        tgt = next((x for x in top if SY.sym_of(x) == nm), None)
        if tgt is None: continue
        for vis in ('"public"', '"private"', '"nested"'):
            for nested_ref in (True, False):
                def f(M, u=u, attr=attr, nm=nm, tgt=tgt, vis=vis, nested_ref=nested_ref):
                    body = M['__root__']['regions'][0][0]['ops']
                    t = next(x for x in body if x['id'] == tgt['id']); body.remove(t)
                    t['props']['_attrs']['sym_visibility'] = vis
                    body.insert(0, ST.node(f'inner#{u["id"]}', 'builtin.module', props={'_attrs': {'sym_name': '"inner"'}, '_types': []},
                                           regions=[[ST.blk('bi', [t])]]))
                    M[u['id']]['props']['_attrs'][attr] = f'@inner::@{nm}' if nested_ref else f'@{nm}'
                yield mut(f, f"嵌套 {'@inner::@f' if nested_ref else '@f'} 可见性 {vis}")
    for a, b in zip(syms, syms[1:]):
        yield mut(lambda M, a=a, b=b: M[b['id']]['props']['_attrs'].__setitem__('sym_name', M[a['id']]['props']['_attrs']['sym_name']), '重名')

def main(cap=40):
    print('── Y0 导入账', {k: v for k, v in SY.ledger().items()})
    st = collections.Counter(); ex = collections.defaultdict(list); orc = OR.Oracle(); byk = collections.Counter()
    def note(k, v):
        if len(ex[k]) < 8: ex[k].append(v)
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read()
        unreg = 'allow-unregistered-dialect' in src
        for k, ch in enumerate(I.split_chunks(src)):
            if not re.search(r'\S', I.strip_comments(ch)): continue
            ctx = ir.Context(); ctx.allow_unregistered_dialects = unreg
            try: m = ir.Module.parse(ch, ctx); g = I.import_module(m)
            except Exception: continue
            st['Y1 模块'] += 1
            uses = sum(1 for n in walk(g) if SY.ref_of(n))
            st['Y1 被检查的符号使用'] += uses
            dd = diags(g)
            if dd: st['Y1 误拒'] += 1; note('Y1 误拒', f'{path.split("/")[-2]}/{path.split("/")[-1]}#{k}: {dd[0]}'[:220])
            if unreg or not any(n['op'] in SY.USES or SY.sym_of(n) for n in walk(g) if n['op'] != 'builtin.module'): continue
            base = '\n'.join(I.emit_generic(g))
            if orc.verify(base)[0] != 'ok': st['Y2 底本 MLIR 不接受（跳过）'] += 1; continue
            for i, (why, h) in enumerate(mutations(g)):
                if i >= cap: break
                ours = diags(h, attrs=True); s_, ds = orc.verify('\n'.join(I.emit_generic(h)))
                msg = ' | '.join(x for _, x in ds); rej = s_ != 'ok'
                st['Y2 变异'] += 1
                kind = re.sub(r'^\S+ ', '', why) if '→' in why or '可见性' in why else why
                byk[(kind, ('拒' if ours else '收') + '/' + ('拒' if rej else '收'))] += 1
                if ours and not rej: st['Y2 我们拒、MLIR 收（误拒）'] += 1; note('Y2 误拒', f'{why}: {ours[0]}'[:220])
                elif ours and rej: st['Y2 两者都拒'] += 1
                elif not ours and not rej: st['Y2 两者都收'] += 1
                elif SYMMSG.search(msg):
                    st['Y2 MLIR 以符号诊断拒、我们收（漏判）'] += 1; note('Y2 漏判', f'{why}: {msg[:160]}')
                else: st['Y2 MLIR 以其它理由拒'] += 1; note('Y2 其它', f'{why}: {msg[:140]}')
    print('── 合计', dict(st)); print('   预言机', orc.stats)
    print('   按变异种类（我们/MLIR）'); [print('     ', k, v) for k, v in sorted(byk.items())]
    for k, v in ex.items():
        print('──', k); [print('    ', x) for x in v]
    orc.close()

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40)
