"""第 78 轮：C++ verify 解释执行（cppverify）的批量检验，对全部有源码的 op 一次进行（closure §9.18）。

在真实 IR 的单个顶层 op 上（端口 = 操作数），变异端口类型、结果类型、结果个数、整数数组/整数属性，逐例比较。判据（运行前写定）：
  C1 误拒 0：MLIR 接受 ⇒ 解释器不拒绝（全部 op，完备与否）。
  C2 漏判 0：静态完备的 op 上，MLIR 在该 op（loc = 节点 id）报出来自其 verify 的诊断（去掉 "'op' op " 前缀后以 verify 源码字面量开头）⇒ 解释器拒绝。
  变异属性在 MLIR Python 中不可解析的用例跳过（不在 IR 域内）。
  C3 未定 0：静态完备的 op，解释器运行时不得 CppUnknown。
  C5 完整模型≡MLIR：op 的残差（declared + unknown）为空时，core.typecheck（不开预言机）与 MLIR 一致：误拒 0、误收 0。
  非空：比较 ≥ 1000，完备 op 上的 verify 诊断 ≥ 100；文件崩溃 0。
第 80 轮（InferTypeOpInterface 的 C++ inferReturnTypes/isCompatibleReturnTypes 解释执行，运行前写定）：
  I1 误拒 0：MLIR 接受 ⇒ decide_infer 不拒绝。
  I2 漏判 0：infer 静态完备的 op 上，MLIR 在该 op 报 "failed to infer returned types" 或 "inferred type(s) … incompatible" ⇒ decide_infer 拒绝；此类诊断合计 ≥ 30。
  I3 未定 0：infer 静态完备的 op，运行时不得 CppUnknown。
  C5 同时覆盖因 infer 规则而残差为空的 op。
第 83 轮（接口 verify 的 C++ 源解释执行：ODS 接口记录的 verify 体 + 接口方法缺省体 + Interfaces/*.cpp，运行前写定）：
  IF1 误拒 0：MLIR 接受 ⇒ decide_iface 不拒绝（全部带接口 verify 的 op，完备与否）。
  IF2 漏判 0：(op, 接口) 静态完备时，MLIR 在该 op 报出来自该接口 verify 调用树的诊断（去前缀后以其源码字面量开头，且该字面量前缀
      不短于 op 自身 verify 及其他接口的最长匹配字面量——开发快检发现单独的 'expected' 会把 op verify 的诊断归给接口，正式运行前补写）⇒ decide_iface 拒绝；
      此类诊断合计 ≥ 20。
  IF3 未定 0：(op, 接口) 静态完备时，运行时不得 CppUnknown。
  C5 同时覆盖因接口规则而残差为空的 op。
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
                if lk is None: continue
                S0 = G.sig(dia, n['op'])
                has_v = 'C++ verify()' in S0['declared']
                has_i = any(k in S0['declared'] for k in ('C++ inferReturnTypes', '接口 verify: InferTypeOpInterface::Trait')) and CV.infer_source(dia, n['op'])[1] is not None
                ifs = [d_.split(': ', 1)[1] for d_ in S0['declared'] if d_.startswith('接口 verify: ') and CV.iface_source(dia, n['op'], d_.split(': ', 1)[1]) is not None
                       and 'InferTypeOpInterface' not in d_]
                if not has_v and not has_i and not ifs: continue
                if any(isinstance(v, str) and '@' in v for v in ((n.get('props') or {}).get('_attrs') or {}).values()): continue
                S = lk[1]
                if has_v and not CV.has_source(dia, n['op']): has_v = False
                if not has_v and not has_i and not ifs: continue
                complete_f = {t_: CV.completeness(dia, n['op'], 'iface:' + t_)[0] for t_ in ifs}
                lits_f = {t_: CV.iface_literals(dia, n['op'], t_) for t_ in ifs}
                complete = has_v and CV.completeness(dia, n['op'])[0]
                complete_i = has_i and CV.completeness(dia, n['op'], 'infer')[0]
                full = not S['declared'] and not S['unknown']
                lits = CV.verify_literals(dia, n['op']) if has_v else []
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
                    verdict = 'skip'; vi = 'skip'; vf = {}
                    if gi is not None and go is not None:
                        try: at = A.from_core(node)
                        except Exception: at = None
                        if at is not None and has_v:
                            try:
                                okv, msgv = CV.decide(dia, n['op'], S, gi + go, at); verdict = okv
                            except CV.CppUnknown as e: verdict = None; note(f'未定/{tag}', f"{n['op']}: {str(e)[:90]}")
                        if at is not None and has_i:
                            try: vi = CV.decide_infer(dia, n['op'], S, gi + go, at)[0]
                            except CV.CppUnknown as e: vi = None; note('infer 未定', f"{n['op']}: {str(e)[:90]}")
                        for t_ in (ifs if at is not None else []):
                            try: vf[t_] = CV.decide_iface(dia, n['op'], S, gi + go, at, t_)
                            except CV.CppUnknown as e: vf[t_] = None; note('接口 未定', f"{n['op']} {t_}: {str(e)[:90]}")
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    ok = tr == 'ok'
                    if kind == '原类型' and not ok: st['原类型独立后被拒'] += 1; note('原类型被拒', f"{n['op']} {' '.join(d[1] for d in diags)[:100]}"); break
                    # 诊断来自 verify：落在本节点，且去掉 "'op' op " 前缀后以 verify 源码里的某个字面量开头（ODS 生成的约束诊断不算）
                    strip = lambda m_: re.sub(r"^'[\w.]+' op ", '', m_)
                    vd = tr != 'crash' and any(l == node['id'] and any(strip(m_).startswith(s_[:40]) for s_ in lits) for l, m_ in diags)
                    if has_i and vi != 'skip':
                        idg = tr != 'crash' and any(l == node['id'] and ('failed to infer returned types' in m_ or 'inferred type' in m_) for l, m_ in diags)
                        st['I 比较'] += 1; st['I 诊断/complete'] += idg and complete_i
                        if vi is None:
                            if complete_i: st['I3 未定'] += 1; note('I3', f"{n['op']} {kind} {T} {mu}")
                        else:
                            st['I 拒绝'] += vi is False
                            if ok and vi is False: st['I1 误拒'] += 1; note('I1 误拒', f"{n['op']} {kind} {T} {mu}")
                            if idg and complete_i and vi is not False: st['I2 漏判'] += 1; note('I2 漏判', f"{n['op']} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
                    def _best(m_, ls): return max((min(len(s_), 40) for s_ in ls if strip(m_).startswith(s_[:40])), default=0)
                    for t_, r_ in vf.items():
                        # 归属：诊断以该接口字面量开头，且该前缀不短于 op 自身 verify 与其他接口的最长匹配字面量（短字面量如 'expected' 不得把别处的诊断归给接口）
                        fd = tr != 'crash' and any(l == node['id'] and _best(m_, lits_f[t_]) > 0 and _best(m_, lits_f[t_]) >= max([_best(m_, lits)] + [_best(m_, lits_f[u_]) for u_ in ifs if u_ != t_])
                                                   for l, m_ in diags)
                        st['IF 比较'] += 1; st['IF 诊断/complete'] += fd and complete_f[t_]
                        if r_ is None:
                            if complete_f[t_]: st['IF3 未定'] += 1; note('IF3', f"{n['op']} {t_} {kind} {T} {mu}")
                        else:
                            st['IF 拒绝'] += r_[0] is False
                            if ok and r_[0] is False: st['IF1 误拒'] += 1; note('IF1 误拒', f"{n['op']} {t_} {kind} {T} {mu} {r_[1]!r}")
                            if fd and complete_f[t_] and r_[0] is not False: st['IF2 漏判'] += 1; note('IF2 漏判', f"{n['op']} {t_} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
                    if not has_v:
                        verdict = 'skip'
                    st['比较'] += has_v; st[f'比较/{tag}'] += has_v; st['MLIR 接受'] += ok and has_v; st['MLIR 崩溃'] += tr == 'crash' and has_v
                    st[f'verify 诊断/{tag}'] += vd and has_v
                    if verdict == 'skip':
                        st['无法绑定/属性'] += has_v
                        if not full or has_v: continue          # 与第 78 轮相同：有 verify 的 op 绑定失败整例跳过；只有 infer 的 op 照常进入 C5
                    if verdict == 'skip': pass
                    elif verdict is None:
                        st[f'未定/{tag}'] += 1
                        if complete: st['C3 完备 op 未定'] += 1; note('C3', f"{n['op']} {kind} {T} {mu}")
                    else:
                        st[f'判定/{tag}'] += 1; st[f'拒绝/{tag}'] += (verdict is False)
                        if ok and verdict is False: st['C1 误拒'] += 1; note('C1 误拒', f"{n['op']} {kind} {T} {mu} {msgv!r}")
                        if vd and verdict is not False and complete: st['C2 漏判'] += 1; note('C2 漏判', f"{n['op']} {kind} {T} {mu} {' '.join(d[1] for d in diags)[:100]}")
                        if vd and verdict is False and msgv and not msgv.startswith('\x00'):   # \x00crash / \x00ub-div0（第 95 轮）
                            st['消息比较'] += 1; st['消息一致'] += any(msgv[:20] in m_ for _, m_ in diags)
                    if full:
                        old = C.ORACLE_ON; C.ORACLE_ON = False
                        try: full_rej = bool(C.typecheck(bs, '(' + ', '.join(T) + ') -> ()'))
                        except Exception as e: full_rej = None; note('完整模型异常', f"{n['op']} {type(e).__name__}: {str(e)[:80]}")
                        finally: C.ORACLE_ON = old
                        if full_rej is not None:
                            st['C5 比较'] += 1; st['C5 MLIR 拒绝'] += not ok
                            st['C5op ' + n['op']] += 1   # 第 92 轮：按 op 计 C5 覆盖
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
    import os
    files = (sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/Linalg/*.mlir')))[start:lim]
    if os.environ.get('CV_EXT'):   # 第 92 轮：主语料未覆盖的方言（corpus_ext/<方言>，MLIR 9bae84b0 mlir/test/Dialect/<方言>）
        files = [f for d in os.environ['CV_EXT'].split(',') for f in sorted(glob.glob(f'corpus_ext/{d}/*.mlir'))][start:lim]
    if os.environ.get('CV_FILTER'): files = [f for f in files if re.search(os.environ['CV_FILTER'], open(f, errors='ignore').read())]   # 第 90 轮：定向（只取含指定 op 的文件）
    for i, path in enumerate(files, start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:3]
    c5ops = {k[5:]: v for k, v in tot.items() if k.startswith('C5op ')}
    for k in list(tot):
        if k.startswith('C5op '): del tot[k]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    print('── C5 覆盖 op', len(c5ops), ' '.join(f'{k}:{v}' for k, v in sorted(c5ops.items())))
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:10]]
    V.check('C1 解释器无误拒', tot['比较'] >= 1000 and tot['C1 误拒'] == 0 and tot['文件超时/崩溃'] == 0,
            f"比较 {tot['比较']}（op 实例 {tot['op 实例']}），MLIR 接受 {tot['MLIR 接受']}，误拒 {tot['C1 误拒']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('C2 完备 op 无漏判', tot['verify 诊断/complete'] >= 100 and tot['C2 漏判'] == 0,
            f"完备 op 比较 {tot['比较/complete']}，verify 诊断 {tot['verify 诊断/complete']}，漏判 {tot['C2 漏判']}，拒绝 {tot['拒绝/complete']}")
    V.check('C3 完备 op 无未定', tot['C3 完备 op 未定'] == 0 and tot['python 异常'] == 0, f"未定 {tot['C3 完备 op 未定']}，解释器 Python 异常 {tot['python 异常']}")
    V.check('C5 残差为空 完整模型≡MLIR', tot['C5 比较'] >= 100 and tot['C5 误拒'] == 0 and tot['C5 误收'] == 0,
            f"比较 {tot['C5 比较']}（MLIR 拒绝 {tot['C5 MLIR 拒绝']}），误拒 {tot['C5 误拒']}，误收 {tot['C5 误收']}")
    V.check('I1 infer 无误拒', tot['I 比较'] >= 100 and tot['I1 误拒'] == 0, f"比较 {tot['I 比较']}，拒绝 {tot['I 拒绝']}，误拒 {tot['I1 误拒']}")
    V.check('I2 infer 诊断必拒', tot['I 诊断/complete'] >= 30 and tot['I2 漏判'] == 0, f"诊断 {tot['I 诊断/complete']}，漏判 {tot['I2 漏判']}")
    V.check('I3 infer 无未定', tot['I3 未定'] == 0, f"未定 {tot['I3 未定']}")
    V.check('IF1 接口 verify 无误拒', tot['IF 比较'] >= 100 and tot['IF1 误拒'] == 0, f"比较 {tot['IF 比较']}，拒绝 {tot['IF 拒绝']}，误拒 {tot['IF1 误拒']}")
    V.check('IF2 接口诊断必拒', tot['IF 诊断/complete'] >= 20 and tot['IF2 漏判'] == 0, f"诊断 {tot['IF 诊断/complete']}，漏判 {tot['IF2 漏判']}")
    V.check('IF3 接口无未定', tot['IF3 未定'] == 0, f"未定 {tot['IF3 未定']}")
    print(f"不完备 op：比较 {tot['比较/partial']}，判定 {tot['判定/partial']}，拒绝 {tot['拒绝/partial']}，未定 {tot['未定/partial']}；消息一致 {tot['消息一致']}/{tot['消息比较']}")
    V.finish()
