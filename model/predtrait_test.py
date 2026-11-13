"""第 90 轮：PredOpTrait 谓词树 → C++ 表达式 → cppverify 解释执行（odssig.pred_cpp、cppverify.rule_pred）的检验。

在真实 IR 的单个顶层 op 上（端口 = 操作数），变异端口类型、结果类型、整数数组/整数属性，逐例比较。判据（运行前写定，2026-11-13 16:30 CST）：
  P1 误拒 0：MLIR 接受 ⇒ 每条 PredOpTrait 规则不判 False；比较 ≥ 200。
  P2 漏判 0：静态完备的规则上，MLIR 在该 op 报 "failed to verify that <该 trait 的 summary>" ⇒ 规则判 False；此类诊断 ≥ 20。
  P3 未定 0：静态完备的规则运行时不得 Unknown。
  P4 残差为空的 op（因本轮规则而清空者计入）：core.typecheck（不开预言机）与 MLIR 一致，误拒 0、误收 0；只在比较 ≥ 50 时判 PASS，不足则 FAIL（数量不足）。
MLIR 崩溃按拒绝计，不计入 P2。原类型独立后被拒的实例整例跳过；属性变异不可解析的用例跳过。
运行：PYTHONPATH=model python3 model/predtrait_test.py
"""
import sys, re, glob, collections, multiprocessing as mp, copy
from scf_test import variants
from cppverify_test import attr_variants

def run_file(path, q):
    import mlir.ir as ir, irimport as I, oracle as OR, odssig as G, typesig as TS, attrsig as A, core as C, cppverify as CV, nfrules as NF
    from compcorpus_test import _funcs
    from regcomp_test import types_of
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
                if lk is None: continue
                S0 = G.sig(dia, n['op']); pt = S0.get('predtrait') or {}
                if not pt: continue
                S = lk[1]
                rules = {}
                for k_, s_ in pt.items():
                    got = CV.rule_pred(S0, k_, s_)
                    if got: rules[k_] = (got[0][4], got[1])
                if not rules: continue
                full = not S['declared'] and not S['unknown']
                ports = list(dict.fromkeys(n['ins']))
                if any(v not in tyof for v in ports): continue
                T0 = [tyof[v] for v in ports]
                cases = [('原类型', T0, None)]
                for j, t in enumerate(T0):
                    for t2 in variants(t): cases.append(('端口', T0[:j] + [t2] + T0[j + 1:], None))
                for k, (o, t) in enumerate(n['outs']):
                    for t2 in variants(t): cases.append(('结果', T0, ('res', k, t2)))
                for an, av in ((n.get('props') or {}).get('_attrs') or {}).items():
                    if isinstance(av, str):
                        for v2 in attr_variants(av): cases.append(('属性', T0, ('attr', an, v2)))
                st['op 实例'] += 1
                for kind, T, mu in cases:
                    node = copy.deepcopy(n); tys = dict(zip(ports, T))
                    node['props'] = dict(node.get('props') or {}); node['props']['_types'] = [tys[v] for v in node['ins']]
                    if mu is not None:
                        if mu[0] == 'res': node['outs'] = [(o, mu[2] if i == mu[1] else t) for i, (o, t) in enumerate(node['outs'])]
                        else: node['props']['_attrs'] = dict(node['props']['_attrs']); node['props']['_attrs'][mu[1]] = mu[2]
                    try:
                        for v_ in (node['props'].get('_attrs') or {}).values():
                            if isinstance(v_, str): CV._parse_attr(v_)
                    except CV.CppUnknown: st['属性不可解析（跳过）'] += 1; continue
                    bs = [ST.blk('entry', [node, ST.node('ret', 'func.return', ins=[])], args=list(zip(ports, T)))]
                    try: ins_t = [TS.parse(tys[v]) for v in node['ins']]; outs_t = [TS.parse(t) for _, t in node['outs']]
                    except Exception: st['类型不可解析'] += 1; continue
                    seg = None; a_ = node['props'].get('_attrs', {}).get('operandSegmentSizes')
                    if a_: seg = I._segs(a_) or None
                    gi = G.bind(S, ins_t, S['ins'], seg); go = G.bind(S, outs_t, S['outs'])
                    try: at = A.from_core(node)
                    except Exception: at = None
                    res = {}
                    if gi is not None and go is not None and at is not None:
                        for k_, (f_, comp_) in rules.items():
                            try: res[k_] = f_(*(gi + go), at)
                            except NF.NFUnknown as e: res[k_] = None; note('未定', f"{n['op']} {k_[:40]}: {str(e)[:80]}")
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    ok = tr == 'ok'
                    if kind == '原类型' and not ok: st['原类型独立后被拒'] += 1; break
                    for k_, r_ in res.items():
                        summ = k_.split(': ', 1)[1].rstrip('′')
                        pd = tr != 'crash' and any(l == node['id'] and ('failed to verify that ' + summ) in m_ for l, m_ in diags)
                        st['P 比较'] += 1; st['P 诊断/complete'] += pd and rules[k_][1]; st['P 拒绝'] += r_ is False
                        if r_ is None:
                            if rules[k_][1]: st['P3 未定'] += 1; note('P3', f"{n['op']} {kind} {T} {mu}")
                        else:
                            if ok and r_ is False: st['P1 误拒'] += 1; note('P1 误拒', f"{n['op']} {k_[:40]} {kind} {T} {mu}")
                            if pd and rules[k_][1] and r_ is not False: st['P2 漏判'] += 1; note('P2 漏判', f"{n['op']} {kind} {T} {mu}")
                    if full:
                        old = C.ORACLE_ON; C.ORACLE_ON = False
                        try: full_rej = bool(C.typecheck(bs, '(' + ', '.join(T) + ') -> ()'))
                        except Exception as e: full_rej = None; note('完整模型异常', f"{n['op']} {type(e).__name__}: {str(e)[:80]}")
                        finally: C.ORACLE_ON = old
                        if full_rej is not None:
                            st['P4 比较'] += 1; st['P4 MLIR 拒绝'] += not ok
                            if ok and full_rej: st['P4 误拒'] += 1; note('P4 误拒', f"{n['op']} {kind} {T} {mu}")
                            if not ok and not full_rej: st['P4 误收'] += 1; note('P4 误收', f"{n['op']} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
    st['python 异常'] += CV.STATS['python 异常']
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/Linalg/*.mlir'))
    files = [f for f in files if re.search(r'vector\.(contract|deinterleave|scalable|print)|nvgpu\.|bufferization\.to_|linalg\.softmax', open(f, errors='ignore').read())]
    for path in files:
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:3]
    print('── 文件', len(files)); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:10]]
    V.check('P1 PredOpTrait 规则无误拒', tot['P 比较'] >= 200 and tot['P1 误拒'] == 0 and tot['文件超时/崩溃'] == 0, f"比较 {tot['P 比较']}，拒绝 {tot['P 拒绝']}，误拒 {tot['P1 误拒']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('P2 PredOpTrait 诊断必拒', tot['P 诊断/complete'] >= 20 and tot['P2 漏判'] == 0, f"诊断 {tot['P 诊断/complete']}，漏判 {tot['P2 漏判']}")
    V.check('P3 PredOpTrait 无未定', tot['P3 未定'] == 0 and tot['python 异常'] == 0, f"未定 {tot['P3 未定']}，解释器 Python 异常 {tot['python 异常']}")
    V.check('P4 残差为空 完整模型≡MLIR', tot['P4 比较'] >= 50 and tot['P4 误拒'] == 0 and tot['P4 误收'] == 0, f"比较 {tot['P4 比较']}（MLIR 拒绝 {tot['P4 MLIR 拒绝']}），误拒 {tot['P4 误拒']}，误收 {tot['P4 误收']}")
    V.finish()
