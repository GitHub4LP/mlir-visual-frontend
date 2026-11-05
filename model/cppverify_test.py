"""第 78 轮：C++ verify 解释执行（cppverify）的批量检验，对全部有源码的 op 一次进行（closure §9.18）。

在真实 IR 的单个顶层 op 上（端口 = 操作数），变异端口类型、结果类型、结果个数、整数数组/整数属性，逐例比较。判据（运行前写定）：
  C1 误拒 0：MLIR 接受 ⇒ 解释器不拒绝（全部 op，完备与否）。
  C2 漏判 0：静态完备的 op 上，MLIR 在该 op（loc = 节点 id）报出来自其 verify 的诊断（去掉 "'op' op " 前缀后以 verify 源码字面量开头）⇒ 解释器拒绝。
  变异属性在 MLIR Python 中不可解析的用例跳过（不在 IR 域内）。
  C3 未定 0：静态完备的 op，解释器运行时不得 CppUnknown。
  C5 完整模型≡MLIR：op 的残差（declared + unknown）为空时，core.typecheck（不开预言机）与 MLIR 一致：误拒 0、误收 0。
  非空：比较 ≥ 1000，完备 op 上的 verify 诊断 ≥ 100；文件崩溃 0。
  只报告：不完备 op 的判定率、拒绝时消息与 MLIR 诊断一致的比例。
原类型独立成函数后 MLIR 即拒绝的实例（父 op 要求、符号等）整例跳过。MLIR 崩溃按拒绝计，不计入 C2。
运行：PYTHONPATH=model python3 model/cppverify_test.py [语料文件上限] [起点]
"""
import sys, re, glob, random, collections, multiprocessing as mp, copy
from scf_test import variants

def attr_variants(txt):
    out = []
    m = re.fullmatch(r'array<i64(?::\s*(.*))?>', txt.strip())
    if m:
        v = [x.strip() for x in (m.group(1) or '').split(',') if x.strip()]
        mk = lambda xs: 'array<i64' + (': ' + ', '.join(xs) if xs else '') + '>'
        if v: out += [mk(v[:-1]), mk(v[:-1] + [str(int(v[-1]) + 1) if v[-1].lstrip('-').isdigit() else '1'])]
        out.append(mk(v + ['1']))
        return out
    m = re.fullmatch(r'(-?\d+)\s*:\s*(i\d+|index)', txt.strip())
    if m: return [f'{int(m.group(1)) + 1} : {m.group(2)}', f'{int(m.group(1)) * 2 + 3} : {m.group(2)}']
    return out

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odssig as G, typesig as TS, nfrules as NF, attrsig as A, core as C, cppverify as CV
    from compcorpus_test import _funcs
    from regcomp_test import free_values, types_of
    import structure as ST
    st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 6: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']; tyof = types_of(fn)
            for n in body:
                if n.get('succs') or n['op'] in ('func.return', 'func.call') or n.get('regions'): continue
                dia = n['op'].split('.')[0]
                lk = G.lookup(n['op'])
                if lk is None or not CV.has_source(dia, n['op']): continue
                S0 = G.sig(dia, n['op'])
                if 'C++ verify()' not in S0['declared']: continue
                if any(isinstance(v, str) and '@' in v for v in ((n.get('props') or {}).get('_attrs') or {}).values()): continue
                S = lk[1]
                complete = CV.completeness(dia, n['op'])[0]
                full = not S['declared'] and not S['unknown']
                lits = CV.verify_literals(dia, n['op'])
                ports = list(dict.fromkeys(n['ins']))
                if any(v not in tyof for v in ports): continue
                T0 = [tyof[v] for v in ports]
                cases = [('原类型', T0, None)]
                for j, t in enumerate(T0):
                    for t2 in variants(t): cases.append(('端口', T0[:j] + [t2] + T0[j + 1:], None))
                for k, (o, t) in enumerate(n['outs']):
                    for t2 in variants(t): cases.append(('结果', T0, ('res', k, t2)))
                at0 = (n.get('props') or {}).get('_attrs') or {}
                for an, av in at0.items():
                    if isinstance(av, str):
                        for v2 in attr_variants(av): cases.append(('属性', T0, ('attr', an, v2)))
                tag = 'complete' if complete else 'partial'
                st['op 实例'] += 1; st[f'op 实例/{tag}'] += 1
                for kind, T, mu in cases:
                    node = copy.deepcopy(n)
                    tys = dict(zip(ports, T))
                    node['props'] = dict(node.get('props') or {}); node['props']['_types'] = [tys[v] for v in node['ins']]
                    if mu is not None:
                        if mu[0] == 'res': node['outs'] = [(o, mu[2] if i == mu[1] else t) for i, (o, t) in enumerate(node['outs'])]
                        elif mu[0] == 'attr': node['props']['_attrs'] = dict(node['props']['_attrs']); node['props']['_attrs'][mu[1]] = mu[2]
                    try:
                        for v_ in (node['props'].get('_attrs') or {}).values():
                            if isinstance(v_, str): CV._parse_attr(v_)
                    except CV.CppUnknown: st['属性不可解析（跳过）'] += 1; continue
                    bs = [ST.blk('entry', [node, ST.node('ret', 'func.return', ins=[])], args=list(zip(ports, T)))]
                    try:
                        ins_t = [TS.parse(tys[v]) for v in node['ins']]; outs_t = [TS.parse(t) for _, t in node['outs']]
                    except Exception: st['类型不可解析'] += 1; continue
                    seg = None
                    a_ = node['props'].get('_attrs', {}).get('operandSegmentSizes')
                    if a_: seg = I._segs(a_) or None
                    gi = G.bind(S, ins_t, S['ins'], seg); go = G.bind(S, outs_t, S['outs'])
                    verdict = 'skip'
                    if gi is not None and go is not None:
                        try: at = A.from_core(node)
                        except Exception: at = None
                        if at is not None:
                            try:
                                okv, msgv = CV.decide(dia, n['op'], S, gi + go, at); verdict = okv
                            except CV.CppUnknown as e: verdict = None; note(f'未定/{tag}', f"{n['op']}: {str(e)[:90]}")
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    ok = tr == 'ok'
                    if kind == '原类型' and not ok: st['原类型独立后被拒'] += 1; note('原类型被拒', f"{n['op']} {' '.join(d[1] for d in diags)[:100]}"); break
                    # 诊断来自 verify：落在本节点，且去掉 "'op' op " 前缀后以 verify 源码里的某个字面量开头（ODS 生成的约束诊断不算）
                    strip = lambda m_: re.sub(r"^'[\w.]+' op ", '', m_)
                    vd = tr != 'crash' and any(l == node['id'] and any(strip(m_).startswith(s_[:40]) for s_ in lits) for l, m_ in diags)
                    st['比较'] += 1; st[f'比较/{tag}'] += 1; st['MLIR 接受'] += ok; st['MLIR 崩溃'] += tr == 'crash'
                    st[f'verify 诊断/{tag}'] += vd
                    if verdict == 'skip': st['无法绑定/属性'] += 1; continue
                    if verdict is None:
                        st[f'未定/{tag}'] += 1
                        if complete: st['C3 完备 op 未定'] += 1; note('C3', f"{n['op']} {kind} {T} {mu}")
                    else:
                        st[f'判定/{tag}'] += 1; st[f'拒绝/{tag}'] += (verdict is False)
                        if ok and verdict is False: st['C1 误拒'] += 1; note('C1 误拒', f"{n['op']} {kind} {T} {mu} {msgv!r}")
                        if vd and verdict is not False and complete: st['C2 漏判'] += 1; note('C2 漏判', f"{n['op']} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
                        if vd and verdict is False and msgv and msgv != '\x00crash':
                            st['消息比较'] += 1; st['消息一致'] += any(msgv[:20] in m_ for _, m_ in diags)
                    if full:
                        old = C.ORACLE_ON; C.ORACLE_ON = False
                        try: full_rej = bool(C.typecheck(bs, '(' + ', '.join(T) + ') -> ()'))
                        except Exception as e: full_rej = None; note('完整模型异常', f"{n['op']} {type(e).__name__}: {str(e)[:80]}")
                        finally: C.ORACLE_ON = old
                        if full_rej is not None:
                            st['C5 比较'] += 1; st['C5 MLIR 拒绝'] += not ok
                            if ok and full_rej: st['C5 误拒'] += 1; note('C5 误拒', f"{n['op']} {kind} {T} {mu}")
                            if not ok and not full_rej: st['C5 误收'] += 1; note('C5 误收', f"{n['op']} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
    st['python 异常'] += CV.STATS['python 异常']
    for x in CV.STATS.get('例', [])[:3]: note('python 异常', x)
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = (sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/Linalg/*.mlir')))[start:lim]
    for i, path in enumerate(files, start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:3]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:10]]
    V.check('C1 解释器无误拒', tot['比较'] >= 1000 and tot['C1 误拒'] == 0 and tot['文件超时/崩溃'] == 0,
            f"比较 {tot['比较']}（op 实例 {tot['op 实例']}），MLIR 接受 {tot['MLIR 接受']}，误拒 {tot['C1 误拒']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('C2 完备 op 无漏判', tot['verify 诊断/complete'] >= 100 and tot['C2 漏判'] == 0,
            f"完备 op 比较 {tot['比较/complete']}，verify 诊断 {tot['verify 诊断/complete']}，漏判 {tot['C2 漏判']}，拒绝 {tot['拒绝/complete']}")
    V.check('C3 完备 op 无未定', tot['C3 完备 op 未定'] == 0 and tot['python 异常'] == 0, f"未定 {tot['C3 完备 op 未定']}，解释器 Python 异常 {tot['python 异常']}")
    V.check('C5 残差为空 完整模型≡MLIR', tot['C5 比较'] >= 100 and tot['C5 误拒'] == 0 and tot['C5 误收'] == 0,
            f"比较 {tot['C5 比较']}（MLIR 拒绝 {tot['C5 MLIR 拒绝']}），误拒 {tot['C5 误拒']}，误收 {tot['C5 误收']}")
    print(f"不完备 op：比较 {tot['比较/partial']}，判定 {tot['判定/partial']}，拒绝 {tot['拒绝/partial']}，未定 {tot['未定/partial']}；消息一致 {tot['消息一致']}/{tot['消息比较']}")
    V.finish()
