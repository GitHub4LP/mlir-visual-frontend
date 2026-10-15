"""第 13 轮：属性层的外部真值检验（spec/attributes.md）。

  A0 导入账：12 方言全部 op 的属性参数，精确 / 上近似（接口原子）/ 未知。
  A1 真实属性不误拒：MLIR 测试集（corpus/）中 MLIR 接受的段里，每个已导入 op 实例的属性都过 attrsig.check。
  A2 变异差分：以真实 op 实例为底（外部真值的分布），把一个属性参数换成取值池里的值或删掉，交 MLIR（预言机子进程）判定：
      - 我们拒绝 ⇒ MLIR 必须拒绝（不误拒，L34a）；
      - MLIR 以 ODS 属性层的诊断（requires attribute 'x' / attribute 'x' failed to satisfy constraint / 属性转属性存储失败）拒绝 ⇒ 我们必须拒绝（对 ODS 层精确，L34b）；
      - MLIR 因 C++ verify 拒绝（读属性的 verify）⇒ 归 declared，记账，不要求我们抓。
用法：PYTHONPATH=model python3 model/attrsig_test.py [每个 op 的实例数=3]
"""
import sys, re, glob, collections, copy
import mlir.ir as ir
import attrsig as A, irimport as I, oracle as OR

POOL = ['0 : i64', '12 : i64', '-1 : i64', '2 : i64', '3 : i32', '4 : i32', '0 : i32', '1 : index', 'true', '"s"', 'unit',
        '[1, 2]', '[]', 'array<i64: 0, 1>', 'array<i64>', 'array<i32: 1>', '#arith.overflow<nsw>', '#arith.fastmath<fast>',
        '@foo', 'i32', '(i32) -> i32', 'memref<4xf32>', 'affine_map<(d0) -> (d0)>', 'affine_map<(d0, d1) -> (d1, d0)>',
        'dense<[0, 1]> : tensor<2xindex>', 'dense<1> : tensor<2xi64>', '{a = 1}', '[true, false]', '#vector.kind<add>',
        '1.000000e+00 : f32', '[0 : i64, 1 : i64]', '#arith.cmpi_predicate<eq>' ]

def walk(n):
    yield n
    for r in n['regions']:
        for b in r:
            for m in b['ops']: yield from walk(m)

def main(per_op=3):
    print('── A0 导入账', A.ledger()[0])
    st = collections.Counter(); ex = collections.defaultdict(list); bases = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 8: ex[k].append(v)
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read()
        if 'allow-unregistered-dialect' in src: continue
        for k, ch in enumerate(I.split_chunks(src)):
            if not re.search(r'\S', I.strip_comments(ch)): continue
            ctx = ir.Context()
            try: m = ir.Module.parse(ch, ctx); g = I.import_module(m)
            except Exception: continue
            for n in walk(g):
                aa = A.attr_args(n['op'])
                if aa is None: continue
                st['A1 op 实例'] += 1
                d = A.check(n['op'], n['props']['_attrs'])
                if d: st['A1 误拒'] += 1; note('A1 误拒', f'{path.split("/")[-2]}/{path.split("/")[-1]}#{k} {n["op"]}: {d[0][1]} = {n["props"]["_attrs"].get(d[0][0])}')
                if aa and len(bases[n['op']]) < per_op: bases[n['op']].append((g, n['id']))
    print('── A1 真实属性', {k: v for k, v in st.items() if k.startswith('A1')})
    orc = OR.Oracle(); s2 = collections.Counter(); gaps = collections.Counter(); tmpl = collections.Counter(); tmpl2 = collections.Counter()
    def find(g, nid): return next(x for x in walk(g) if x['id'] == nid)
    for op, lst in sorted(bases.items()):
        for g0, nid in lst:
            base_txt = '\n'.join(I.emit_generic(g0))
            if orc.verify(base_txt)[0] != 'ok': s2['底本不被 MLIR 接受（跳过）'] += 1; continue
            for e in A.attr_args(op):
                for val in POOL + [None]:
                    g = copy.deepcopy(g0); n = find(g, nid); at = n['props']['_attrs']
                    if val is None:
                        if e['name'] not in at: continue
                        del at[e['name']]
                    else:
                        try: A.parse(val)
                        except Exception: continue
                        at[e['name']] = val
                    ours = A.check(op, at)
                    try:
                        import typesig as TS
                        ours = ours + [('rel', m) for m in A.rel_check(op, at, [TS.parse(t) for t in n['props']['_types']], [TS.parse(t) for _, t in n['outs']])]
                    except ValueError: pass
                    st_, ds = orc.verify('\n'.join(I.emit_generic(g)))
                    msg = ' | '.join(m for _, m in ds)
                    s2['变异'] += 1
                    mlir_rej = st_ != 'ok'
                    ods = bool(re.search(rf"requires attribute '{e['name']}'|attribute '{e['name']}' failed to satisfy|property conversion", msg))
                    rel = bool(re.search(r"failed to verify that", msg))     # ODS 的 op 级关系谓词（属性与类型的关系）
                    if ours and not mlir_rej: s2['我们拒、MLIR 收（误拒）'] += 1; note('A2 误拒', f'{op}.{e["name"]} = {val}: {ours[0][1]}')
                    elif ours and mlir_rej: s2['两者都拒'] += 1
                    elif not ours and not mlir_rej: s2['两者都收'] += 1
                    elif ods:
                        s2['MLIR 以 ODS 属性层拒、我们收（漏判）'] += 1; gaps[f'{op}.{e["name"]}'] += 1
                        tmpl[re.sub(r"'[^']*'|\d+|\(.*?\)|`[^`]*`", '_', msg.split(' | ')[0])[:90]] += 1
                        note('A2 ODS 层漏判', f'{op}.{e["name"]} = {val}: {msg[:140]}')
                    elif rel:
                        s2['MLIR 以 ODS 关系谓词拒（属性-类型关系，下一步）'] += 1; tmpl2[re.sub(r"'[^']*'", '_', msg.split(' | ')[0])[:90]] += 1
                    else:
                        s2['MLIR 以 C++ verify 拒（declared，不要求）'] += 1; note('A2 C++ verify', f'{op}.{e["name"]} = {val}: {msg[:120]}')
    print('── A2 变异差分', dict(s2), '预言机', orc.stats, '求值', A.STATS)
    print('   漏判最多的属性参数', gaps.most_common(12))
    print('   漏判的 MLIR 诊断模板'); [print('     ', v, k) for k, v in tmpl.most_common(25)]
    print('   关系谓词'); [print('     ', v, k) for k, v in tmpl2.most_common(12)]
    for k, v in ex.items():
        print('──', k); [print('    ', x[:220]) for x in v]
    orc.close()

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
