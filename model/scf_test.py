"""第 76 轮：scf 区域 op 的规范化规则（nfrules：scf.for/if/while/execute_region/index_switch）的检验（closure §9.16）。

规则覆盖这些 op 的全部 declared 条目（C++ verify、verifyRegions、LoopLikeOpInterface、RegionBranchOpInterface、InferTypeOpInterface）。
判据（运行前写定）：在真实 IR 的单个 scf op 上（端口 = 操作数与捕获值），变异端口类型、结果类型、块参数类型、
结果/块参数/终结子操作数的个数（去掉最后一个）、index_switch 的 cases，逐例比较：
  N2g-a 误拒 0：MLIR 接受 ⇒ scf 规则接受。
  N2g-b 漏判 0：MLIR 在该 scf op 上报出被覆盖校验的诊断（SCF_DIAG）⇒ scf 规则拒绝。MLIR 崩溃按拒绝计但不计入 N2g-b。
  N2g-d 完整模型≡MLIR：实例内全部 op（含区域内）都没有残差时，core.typecheck（不开预言机）与 MLIR 一致：误拒 0、误收 0。
  非空：比较 ≥ 300，MLIR 接受 ≥ 50，规则拒绝 ≥ 50，N2g-d 比较 ≥ 100；文件崩溃 0。
运行：PYTHONPATH=model python3 model/scf_test.py [语料文件上限] [起点]
"""
import sys, re, glob, random, collections, multiprocessing as mp, copy

SCF_OPS = ('scf.for', 'scf.if', 'scf.while', 'scf.execute_region', 'scf.index_switch')
SCF_DIAG = re.compile(r"mismatch in number of loop-carried|expected induction variable|mismatch in number of basic block args|types mismatch between|"
                      r"different number of|have different type|region control flow edge|along control flow edge|must have an else block|"
                      r"failed to infer returned types|inferred type|expects the 'before' region|expects the 'after' region|region needs to have at least one block|"
                      r"region cannot have any arguments|case regions but|duplicate case value|expected region to end with scf.yield|expected each region to return|"
                      r"of each region to be")
SCAL = {'index': ['i32', 'i64'], 'i32': ['i64', 'index', 'f32'], 'i64': ['i32', 'index'], 'i1': ['i32'], 'f32': ['f64', 'i32'], 'f64': ['f32'], 'f16': ['f32']}

def variants(t):
    if t in SCAL: return SCAL[t]
    m = re.fullmatch(r'(tensor|memref|vector)<(.*)>', t)
    if not m: return []
    k, body = m.groups()
    out = []
    mm = re.fullmatch(r'((?:[\d?]+x)*)(\w+)(.*)', body)
    if mm:
        dims, el, rest = mm.groups()
        el2 = 'f64' if el != 'f64' else 'f32'
        out.append(f'{k}<{dims}{el2}{rest}>')
        if k != 'vector':
            out.append(f'{k}<{"7x" + dims}{el}{rest}>' if not rest else f'{k}<7x{dims}{el}>')
            if k == 'tensor': out.append(f'memref<{dims}{el}>')
        else: out.append(f'vector<{"5x" + dims}{el}>')
    return out[:3]

def _nested(n):
    for r in n.get('regions') or []:
        for b in r:
            for m in b['ops']: yield m; yield from _nested(m)

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odssig as G, typesig as TS, nfrules as NF, attrsig as A, core as C
    from composite import instantiate
    from compcorpus_test import _funcs
    from regcomp_test import free_values, types_of
    st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    T_ = lambda t: True
    for ch in I.split_chunks(src):
        if not re.search(r'\S', I.strip_comments(ch)) or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for fn in _funcs(root):
            if len(fn['regions'][0]) != 1: continue
            body = fn['regions'][0][0]['ops']; tyof = types_of(fn)
            for n in body:
                if n['op'] not in SCF_OPS: continue
                if any(x.get('succs') or x['op'] in ('func.return', 'func.call') for x in _nested(n)): continue
                if any(isinstance(v, str) and '@' in v for x in [n] + list(_nested(n)) for v in ((x.get('props') or {}).get('_attrs') or {}).values()): continue
                lk = G.lookup(n['op']); S = lk[1]
                ports = free_values([n])
                if not ports or any(v not in tyof for v in ports): continue
                full = all((x['op'] in ('scf.yield', 'scf.condition')) or ((lambda l: l is not None and not l[1]['declared'] and not l[1]['unknown'])(G.lookup(x['op'])))
                           for x in [n] + list(_nested(n)))
                T0 = [tyof[v] for v in ports]
                cases = [('原类型', T0, None)]
                for j, t in enumerate(T0):
                    for t2 in variants(t): cases.append(('端口', T0[:j] + [t2] + T0[j + 1:], None))
                for k, (o, t) in enumerate(n['outs']):
                    for t2 in variants(t): cases.append(('结果', T0, ('res', k, t2)))
                if n['outs']: cases.append(('结果个数', T0, ('dropres',)))
                for ri, r in enumerate(n['regions']):
                    for bi, b in enumerate(r):
                        for ai, (a, t) in enumerate(b['args']):
                            for t2 in variants(t)[:2]: cases.append(('块参数', T0, ('arg', ri, bi, ai, t2)))
                        if b['args']: cases.append(('块参数个数', T0, ('droparg', ri, bi)))
                        if b['ops'] and b['ops'][-1]['op'] in ('scf.yield', 'scf.condition') and len(b['ops'][-1]['ins']) > (b['ops'][-1]['op'] == 'scf.condition'):
                            cases.append(('终结子个数', T0, ('dropterm', ri, bi)))
                at0 = (n.get('props') or {}).get('_attrs') or {}
                if n['op'] == 'scf.index_switch' and at0.get('cases'):
                    v0 = NF.ints(at0['cases'])
                    for v in ([v0[0]] * len(v0) if v0 else ['0'], v0[:-1], v0 + [99]):
                        cases.append(('cases', T0, ('attr', 'cases', 'array<i64' + (': ' + ', '.join(map(str, v)) if v else '') + '>')))
                st['op 实例'] += 1; st['op 实例/' + n['op']] += 1
                for kind, T, mu in cases:
                    bs = instantiate([n], ports, T)
                    if bs is None: st['不可实例化'] += 1; continue
                    node = bs[0]['ops'][0]
                    if mu is not None:
                        node['regions'] = copy.deepcopy(node['regions'])
                        if mu[0] == 'res': node['outs'] = [(o, mu[2] if i == mu[1] else t) for i, (o, t) in enumerate(node['outs'])]
                        elif mu[0] == 'dropres': node['outs'] = node['outs'][:-1]
                        elif mu[0] == 'arg':
                            b = node['regions'][mu[1]][mu[2]]; b['args'] = [(a, mu[4] if i == mu[3] else t) for i, (a, t) in enumerate(b['args'])]
                        elif mu[0] == 'droparg':
                            b = node['regions'][mu[1]][mu[2]]; a_ = b['args'][-1][0]
                            if any(a_ in x['ins'] for x in [y for y in _nested(node)]): st['跳过（被删参数有使用）'] += 1; continue
                            b['args'] = b['args'][:-1]
                        elif mu[0] == 'dropterm':
                            t_ = node['regions'][mu[1]][mu[2]]['ops'][-1]; t_['ins'] = t_['ins'][:-1]
                            if '_types' in (t_.get('props') or {}): t_['props'] = dict(t_['props']); t_['props']['_types'] = t_['props']['_types'][:-1]
                        elif mu[0] == 'attr':
                            node['props'] = dict(node['props']); node['props']['_attrs'] = dict(node['props']['_attrs']); node['props']['_attrs'][mu[1]] = mu[2]
                    tys = dict(zip(ports, T))
                    for x in [node] + list(_nested(node)):
                        for r in x.get('regions') or []:
                            for b in r:
                                for a, t in b['args']: tys[a] = t
                        for o, t in x['outs']: tys[o] = t
                    try:
                        ins_t = [TS.parse(tys[v]) for v in node['ins']]; outs_t = [TS.parse(t) for _, t in node['outs']]
                        gi = G.bind(S, ins_t, S['ins'], None); go = G.bind(S, outs_t, S['outs'])
                    except Exception: st['类型不可解析'] += 1; continue
                    if gi is None or go is None: st['无法绑定'] += 1; continue
                    try: at = NF.nf_attrs(node, A.from_core(node), {}, None, vty=tys.get)
                    except Exception as e: at = None; note('属性异常', str(e)[:80])
                    mini = dict(S, ins=[(q_, T_, mm, b) for q_, _, mm, b in S['ins']], outs=[(q_, T_, mm, b) for q_, _, mm, b in S['outs']], infer=None,
                                rel=[x for x in S['rel'] if x[0] == 'nf' and str(x[3]).startswith(n['op'])])
                    try: r_rej = bool(G.check(mini, gi, go, attrs=at, strict=True))
                    except NF.NFUnknown as e: r_rej = None; note('规则未定', f"{n['op']} {str(e)[:80]}")
                    if full:
                        old = C.ORACLE_ON; C.ORACLE_ON = False
                        try: full_rej = bool(C.typecheck(bs, '(' + ', '.join(T) + ') -> ()'))
                        except Exception as e: full_rej = None; note('完整模型异常', f"{n['op']} {type(e).__name__}: {str(e)[:80]}")
                        finally: C.ORACLE_ON = old
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    if tr == 'crash': st['MLIR 崩溃（按拒绝计）'] += 1
                    ok = tr == 'ok'; msg = ' '.join(d[1] for d in diags) if (diags and tr != 'crash') else ''
                    # 诊断须落在这个 scf op 本身（loc = 节点 id）；落在区域内同名 op 上的不算（第 76 轮首跑把它们误算进来，见 closure §9.16）
                    dg = tr != 'crash' and any(l == node['id'] and SCF_DIAG.search(m_) for l, m_ in diags)
                    st['scf 诊断在嵌套 op 上'] += (not dg) and bool(SCF_DIAG.search(msg))
                    if kind == '原类型' and not ok: st['原类型独立后被拒'] += 1; note('原类型被拒', msg[:120]); break
                    st['比较'] += 1; st['MLIR 接受'] += ok; st['变异/' + kind] += 1; st['scf 诊断'] += dg
                    if r_rej is None: st['规则未定'] += 1
                    else:
                        st['规则拒绝'] += r_rej
                        if ok and r_rej: st['N2g-a 误拒'] += 1; note('误拒', f"{n['op']} {kind} {T} {mu}")
                        if dg and not r_rej: st['N2g-b 漏判'] += 1; note('漏判', f"{n['op']} {kind} {T} {mu} {msg[:100]}")
                        if not ok and not dg and r_rej: st['规则拒绝/MLIR 别的诊断或崩溃'] += 1
                    if full and full_rej is not None:
                        st['N2g-d 比较'] += 1; st['N2g-d MLIR 拒绝'] += not ok
                        if ok and full_rej: st['N2g-d 误拒'] += 1; note('N2g-d 误拒', f"{n['op']} {kind} {T} {mu}")
                        if not ok and not full_rej: st['N2g-d 误收'] += 1; note('N2g-d 误收', f"{n['op']} {kind} {T} {mu} {msg[:100]}")
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    files = sorted(glob.glob('corpus/*/*.mlir'))[start:lim]
    for i, path in enumerate(files, start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; exs['文件超时/崩溃'].append(path); continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('N2g-a scf 规则无误拒', tot['比较'] >= 300 and tot['MLIR 接受'] >= 50 and tot['N2g-a 误拒'] == 0 and tot['文件超时/崩溃'] == 0,
            f"比较 {tot['比较']}（op 实例 {tot['op 实例']}），MLIR 接受 {tot['MLIR 接受']}，误拒 {tot['N2g-a 误拒']}，规则未定 {tot['规则未定']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('N2g-b scf 诊断必拒', tot['scf 诊断'] > 0 and tot['N2g-b 漏判'] == 0 and tot['规则拒绝'] >= 50,
            f"scf 诊断 {tot['scf 诊断']}，漏判 {tot['N2g-b 漏判']}，规则拒绝 {tot['规则拒绝']}（其中 MLIR 别的诊断或崩溃 {tot['规则拒绝/MLIR 别的诊断或崩溃']}），MLIR 崩溃 {tot['MLIR 崩溃（按拒绝计）']}")
    V.check('N2g-d 无残差实例 完整模型≡MLIR', tot['N2g-d 比较'] >= 100 and tot['N2g-d 误拒'] == 0 and tot['N2g-d 误收'] == 0,
            f"比较 {tot['N2g-d 比较']}（MLIR 拒绝 {tot['N2g-d MLIR 拒绝']}），误拒 {tot['N2g-d 误拒']}，误收 {tot['N2g-d 误收']}")
    V.finish()
