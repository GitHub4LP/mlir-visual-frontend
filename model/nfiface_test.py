"""第 69 轮：接口级规范化规则的检验（closure §9.9）。

接口规则只覆盖 op 的一部分 declared 条目（op 自身的 C++ verify 等可能仍在残差中），因此不能要求“规则 ≡ MLIR 整体判定”。
判据（运行前写定）：在真实 IR 的单个 op 上变异 init 端口类型与结果类型，逐例比较规则 r 与 MLIR：
  N2a 误拒 0：MLIR 接受 ⇒ r 接受（规则可靠，可以有拒绝权）。
  N2b 漏判 0：MLIR 报出该接口的诊断 ⇒ r 拒绝（规则不弱于该接口的 verify）。
      （MLIR 拒绝但报的是别的诊断时，接口 verify 可能没有运行，不计。）
  N2c 非空：MLIR 接受 ≥ 1，接口诊断 ≥ 1，r 拒绝 ≥ 1；文件工作进程崩溃 0。
运行：PYTHONPATH=model python3 model/nfiface_test.py [语料文件上限] [起点]
语料：corpus/*/*.mlir 加 corpus_ext/Linalg（MLIR 9bae84b0 mlir/test/Dialect/Linalg；主语料不含 Linalg，单独存放以免改变既有基线）。
"""
import sys, re, glob, random, collections, multiprocessing as mp

IFACE_DIAG = {'DPS': re.compile(r'is a tensor or a memref|to be equal to the number of output tensors|to match type of corresponding result'),
              'ConvolutionOpInterface': re.compile(r'expected op with 2 inputs and 1 output|unexpected input index map for convolutions|output/filter indexing maps to be projected|unexpected loop dimension for convolution|iterators used to access outputs to be parallel|iterators not used to access outputs to be reduction|convolved dim to be non-empty'),
              'ContractionOpInterface': re.compile(r'expected op with 2 inputs and 1 output|expected at least 1 reduction|indexing maps to be projected permutations|expected add/mul op in the body'),
              'LinalgHand': re.compile(r'arity of mapper|to match bbArg type|to match shape of output|same shapes|dimensions for reduction|after reduction|init dimensions|mismatching number of operands and block arguments|does not match corresponding block argument|permutation is not valid|does not match init rank|size of permutation|doesn.t match dim\(input|added dimensions|is out of range|should match init dim|Unexpected dim expression|Invalid broadcast|Unexpected result dim expression|exceeds 3|Invalid batch dimension|expects [23] dims|Invalid output map|must have 3 affine maps|not a projected permutation|ranks of shaped operand|non-shaped type|iteration spaces of provided|not used to access any operand|neither a contracting dim|contracting dimension|pure tensor or buffer semantics|indexing_map|non-invertible|shape-to-loops|1 region with 1 block|bb argument|operand rank'),
              'LinalgCxx': re.compile(r'Unexpected dim expression|Invalid broadcast|Unexpected result dim expression|exceeds 3|Invalid batch dimension|expects [23] dims|Invalid output map|must have 3 affine maps|not a projected permutation|ranks of shaped operand|non-shaped type|iteration spaces of provided|not used to access any operand|neither a contracting dim|contracting dimension'),
              'linalg.pack': re.compile(r'invalid zero tile factor|invalid inner_dims_pos|invalid outer_dims_perm|must be a permutation or empty|tiling factors must|packed rank !=|mismatch in inner tile sizes|for the packed domain value|expected padding_value has|Only full tiles are supported'),
              'linalg.unpack': re.compile(r'invalid zero tile factor|invalid inner_dims_pos|invalid outer_dims_perm|must be a permutation or empty|tiling factors must|packed rank !=|mismatch in inner tile sizes|for the packed domain value'),
              'LinalgOp': re.compile(r'pure tensor or buffer semantics|indexing_map|non-invertible|shape-to-loops|1 region with 1 block|non-induction variable region|bb argument|inferred input/output operand|unexpected result less than 0|operand rank')}
# 第 70 轮：LinalgOp 变异所有操作数（不只 init）；N2d：declared 已清空的 op 上，完整模型（无预言机）≡ MLIR。

def _variants(t):
    """一个类型的变异：tensor ↔ memref、换形状、无秩。"""
    m = re.fullmatch(r'(tensor|memref)<(.*)>', t)
    if not m: return []
    k, body = m.groups()
    if body.startswith('*x'): return [('memref' if k == 'tensor' else 'tensor') + '<' + body + '>']
    parts = body.split(',')[0]
    dims = re.match(r'((?:[0-9?]+x)*)(.*)', parts)
    d, e = dims.groups()
    other = 'memref' if k == 'tensor' else 'tensor'
    out = [f'{other}<{d}{e}>', f'{k}<*x{e}>' if k == 'tensor' else f'memref<*x{e}>', f'{k}<7x{d}{e}>']
    if e != 'f32': out.append(f'{k}<{d}f32>')
    ds = [x for x in d.split('x') if x]
    if ds:
        st = [i for i, x in enumerate(ds) if x != '?']
        if st: i = st[0]; out.append(f"{k}<{'x'.join(ds[:i] + [str(int(ds[i]) + 1)] + ds[i + 1:])}x{e}>")
        out.append(f"{k}<{'x'.join(['?'] + ds[1:])}x{e}>")
        out.append(f"{k}<{'x'.join(ds[:-1])}{'x' if len(ds) > 1 else ''}{e}>")
    return out

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, oracle as OR, odscomp as D, odssig as G, typesig as TS, structure as ST
    from composite import instantiate
    from compcorpus_test import _walk, _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
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
            body = fn['regions'][0][0]['ops']
            tyof = {a: t for a, t in fn['regions'][0][0]['args']} | {o: t for n in body for o, t in n['outs']}
            for n in body:
                at0 = (n.get('props') or {}).get('_attrs') or {}
                if 'linalg.memoized_indexing_maps' in at0:          # MLIR verify 时写入的缓存：去掉，检验由 yaml 推出的映射（前端的 op 不带它）
                    n = dict(n); n['props'] = dict(n['props']); n['props']['_attrs'] = {k: v for k, v in at0.items() if k != 'linalg.memoized_indexing_maps'}
                lk = G.lookup(n['op'])
                if not lk: continue
                S = lk[1]
                atoms = [(nm, x) for x in S['rel'] if x[0] == 'nf' for nm in IFACE_DIAG if str(x[3]).startswith(nm)]
                if not atoms: continue
                full = not S['declared'] and not S['unknown']
                pre = []                                                       # 第 73 轮：pack/unpack 的常量 tile 操作数连同其 arith.constant 一起实例化
                if n['op'] in ('linalg.pack', 'linalg.unpack'):
                    defs = {o: m for m in body for o, _ in m['outs']}
                    for v in n['ins']:
                        m = defs.get(v)
                        if m is not None and m['op'] == 'arith.constant' and m not in pre: pre.append(m)
                pouts = {o for m in pre for o, _ in m['outs']}
                ports = [v for v in dict.fromkeys(n['ins']) if v not in pouts]
                if not ports: continue
                hand = any(nm == 'LinalgHand' for nm, _ in atoms) and not any(nm == 'LinalgOp' for nm, _ in atoms)
                gname = 'outputs' if any(nm == 'LinalgOp' for nm, _ in atoms) else (names0 := [q for q, *_ in S['ins']])[-1] if hand else atoms[0][1][1][0][1]
                try:
                    gi0 = G.bind(S, [TS.parse(tyof[v]) for v in n['ins']], S['ins'], D._seg(n))
                except Exception: continue
                if gi0 is None: continue
                names = [nm for nm, *_ in S['ins']]
                k0 = sum(len(g) for g in gi0[:names.index(gname)]); kn = len(gi0[names.index(gname)])
                init_vals = n['ins'][k0:k0 + kn]
                T0 = [tyof[v] for v in ports]
                cases = [('原类型', T0, None)]
                mut_vals = n['ins'] if any(nm in ('LinalgOp', 'LinalgHand', 'linalg.pack', 'linalg.unpack') for nm, _ in atoms) else init_vals
                mut_vals = [v for v in mut_vals if v in ports]
                for k_, m in enumerate(pre):
                    for cv in (0, 1, 3, 5): cases.append(('const', T0, ('const', k_, cv)))
                for v in list(dict.fromkeys(mut_vals)):
                    j = ports.index(v)
                    for t in _variants(T0[j]): cases.append(('init', T0[:j] + [t] + T0[j + 1:], None))
                for k, (o, t) in enumerate(n['outs']):
                    for t2 in _variants(t): cases.append(('result', T0, (k, t2)))
                at1 = (n.get('props') or {}).get('_attrs') or {}
                for an in ('strides', 'dilations'):
                    k = None
                    if an in at1:
                        m_ = re.search(r'tensor<(\d+)xi64>', at1[an]); k = int(m_.group(1)) if m_ else None
                    elif any(a[0] == an for a in S['attrs']):
                        m_ = [a for a in S['attrs'] if a[0] == an]; m2 = re.search(r'\[(\d+)\]', str(m_[0][1])); k = int(m2.group(1)) if m2 else None
                    if k:
                        for v in (0, 2, 3, -1): cases.append(('attr', T0, ('attr', an, f'dense<{v}> : tensor<{k}xi64>')))
                if 'indexing_maps' in at1:
                    ms = re.findall(r'affine_map<.*?>(?=\s*[,\]])', at1['indexing_maps'])
                    if len(ms) >= 2: cases.append(('attr', T0, ('attr', 'indexing_maps', '[' + ', '.join(ms[::-1]) + ']')))
                    if hand or n['op'] in ('linalg.matmul', 'linalg.batch_matmul', 'linalg.batch_reduce_matmul', 'linalg.contract', 'linalg.elementwise'):
                        for k_, m_ in enumerate(ms):                       # 第 72 轮：逐个映射的结果变异（转置/去首/去尾/换成首映射）
                            mm = re.fullmatch(r'affine_map<(\(.*?\)(?:\[.*?\])?\s*->\s*)\((.*)\)>', m_.strip())
                            if not mm: continue
                            rs = [x.strip() for x in mm.group(2).split(',') if x.strip()]
                            for rs2 in (rs[::-1], rs[1:], rs[:-1]):
                                if rs2 == rs: continue
                                ms2 = list(ms); ms2[k_] = f'affine_map<{mm.group(1)}({", ".join(rs2)})>'
                                cases.append(('attr', T0, ('attr', 'indexing_maps', '[' + ', '.join(ms2) + ']')))
                            if k_: ms2 = list(ms); ms2[k_] = ms[0]; cases.append(('attr', T0, ('attr', 'indexing_maps', '[' + ', '.join(ms2) + ']')))
                    if 'iterator_types' in at1:
                        its = re.findall(r'#linalg\.iterator_type<\w+>', at1['iterator_types'])
                        if its: cases.append(('attr', T0, ('attr', 'iterator_types', '[' + ', '.join(its[:-1]) + ']')))
                for an in ('dimensions', 'permutation', 'inner_dims_pos', 'outer_dims_perm', 'static_inner_tiles'):                       # 第 72 轮：手写 op 的数组属性变异
                    if an in at1:
                        v0 = re.findall(r'-?\d+', at1[an].split(':', 1)[1]) if ':' in at1[an] else []
                        vs = [v0[::-1], v0[:-1], v0 + ['0'], ['5'] + v0[1:], [v0[0]] * len(v0) if v0 else ['0']]
                        for v in vs: cases.append(('attr', T0, ('attr', an, 'array<i64' + (': ' + ', '.join(v) if v else '') + '>')))
                if hand and n.get('regions') and len(n['regions'][0]) == 1 and n['regions'][0][0]['args']:
                    a0t = n['regions'][0][0]['args'][0][1]
                    cases.append(('bbarg', T0, ('bbarg', 'f64' if a0t != 'f64' else 'i32')))
                if n.get('regions') and len(n['regions'][0]) == 1:          # 第 71 轮：区域体变异（ContractionOpInterface 依赖区域体）
                    rops = {o['op'] for o in n['regions'][0][0]['ops']}
                    for a_, b_ in (('arith.mulf', 'arith.subf'), ('arith.addf', 'arith.subf'), ('arith.muli', 'arith.subi'), ('arith.addi', 'arith.subi'), ('arith.mulf', 'arith.addf')):
                        if a_ in rops: cases.append(('body', T0, ('body', a_, b_)))
                st['op 实例'] += 1
                for kind, T, rmut in cases:
                    bs = instantiate(pre + [n], ports, T)
                    if bs is None: st['不可实例化'] += 1; continue
                    node = bs[0]['ops'][len(pre)]
                    if rmut is not None and rmut[0] == 'const':
                        c_ = bs[0]['ops'][rmut[1]]; c_['props'] = dict(c_['props']); c_['props']['_attrs'] = dict(c_['props']['_attrs'])
                        c_['props']['_attrs']['value'] = f"{rmut[2]} : {c_['outs'][0][1]}"
                    elif rmut is not None and rmut[0] == 'bbarg':
                        import copy
                        node['regions'] = copy.deepcopy(node['regions'])
                        b0 = node['regions'][0][0]; b0['args'] = [(b0['args'][0][0], rmut[1])] + b0['args'][1:]
                    elif rmut is not None and rmut[0] == 'body':
                        import copy
                        node['regions'] = copy.deepcopy(node['regions'])
                        for o in node['regions'][0][0]['ops']:
                            if o['op'] == rmut[1]: o['op'] = rmut[2]; break
                    elif rmut is not None and rmut[0] == 'attr':
                        node['props'] = dict(node['props']); node['props']['_attrs'] = dict(node['props']['_attrs']); node['props']['_attrs'][rmut[1]] = rmut[2]
                    elif rmut is not None:
                        k, t2 = rmut; node['outs'] = [(o, t2 if i == k else t) for i, (o, t) in enumerate(node['outs'])]
                    try:
                        ins_t = [TS.parse(t) for t in node['props']['_types']]; outs_t = [TS.parse(t) for _, t in node['outs']]
                        gi = G.bind(S, ins_t, S['ins'], D._seg(n)); go = G.bind(S, outs_t, S['outs'])
                    except Exception: st['类型不可解析'] += 1; continue
                    if gi is None or go is None: st['无法绑定'] += 1; continue
                    import attrsig as A, nfrules as NF, composite as CP
                    consts = {}
                    for c_ in bs[0]['ops'][:len(pre)]: NF.note_const(c_, consts)
                    try: at = NF.nf_attrs(node, A.from_core(node), consts)
                    except Exception: at = None
                    rr = {}
                    for nm, x in atoms:
                        mini = dict(S, ins=[(q, T_, mm, b) for q, _, mm, b in S['ins']], outs=[(q, T_, mm, b) for q, _, mm, b in S['outs']], infer=None, rel=[x])
                        try: rr[nm] = bool(G.check(mini, gi, go, attrs=at, strict=True))
                        except NF.NFUnknown as e: rr[nm] = None; note('规则未定', f"{nm} {n['op']} {str(e)[:80]}")
                    if full:
                        try:
                            import core as C
                            old = C.ORACLE_ON; C.ORACLE_ON = False
                            full_rej = bool(C.typecheck(bs, '(' + ', '.join(T) + ') -> ()'))
                        except Exception as e: full_rej = None; note('完整模型异常', f"{n['op']} {type(e).__name__}: {str(e)[:80]}")
                        finally: C.ORACLE_ON = old
                    text = OR.emit_core(bs, '(' + ', '.join(T) + ') -> ()')
                    if text is None: st['不可发射'] += 1; continue
                    tr, diags = OR.get().verify(text)
                    if tr == 'parse': st['MLIR parse'] += 1; continue
                    if tr == 'crash': st['MLIR 崩溃（按拒绝计）'] += 1
                    ok = tr == 'ok'; msg = ' '.join(d[1] for d in diags) if (diags and tr != 'crash') else ''
                    st['比较'] += 1; st['MLIR 接受'] += ok; st['变异/' + kind] += 1
                    for nm, r_rej in rr.items():
                        dg = bool(IFACE_DIAG[nm].search(msg))
                        if r_rej is None: st[f'{nm}/规则未定'] += 1; continue
                        st[f'{nm}/比较'] += 1; st[f'{nm}/接口诊断'] += dg; st[f'{nm}/规则拒绝'] += r_rej; st[f'{nm}/MLIR 接受'] += ok
                        st['接口诊断'] += dg; st['规则拒绝'] += r_rej
                        if ok and r_rej: st['N2a 误拒'] += 1; note('误拒', f"{nm} {n['op']} {kind} {T} {rmut}")
                        if dg and not r_rej: st['N2b 漏判'] += 1; note('漏判', f"{nm} {n['op']} {kind} {T} {rmut} {msg[:90]}")
                        if not ok and not dg and r_rej: st['规则拒绝/MLIR 别的诊断'] += 1
                    if full and full_rej is not None:
                        st['N2d 比较'] += 1; st['N2d MLIR 拒绝'] += not ok
                        if ok and full_rej: st['N2d 误拒'] += 1; note('N2d 误拒', f"{n['op']} {kind} {T} {rmut}")
                        if not ok and not full_rej: st['N2d 误收'] += 1; note('N2d 误收', f"{n['op']} {kind} {T} {rmut} {msg[:90]}")
    q.put((st, dict(ex)))

if __name__ == '__main__':
    import verdict as V
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate((sorted(glob.glob('corpus/*/*.mlir')) + sorted(glob.glob('corpus_ext/*/*.mlir')))[start:lim], start):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q, i)); p.start()
        try: st, ex = q.get(timeout=600)
        except Exception: p.kill(); tot['文件超时/崩溃'] += 1; continue
        p.join(); tot.update(st)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── 合计'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
    V.check('N2a 接口规则无误拒', tot['比较'] > 0 and tot['N2a 误拒'] == 0 and tot['文件超时/崩溃'] == 0,
            f"比较 {tot['比较']}（op 实例 {tot['op 实例']}），MLIR 接受 {tot['MLIR 接受']}，误拒 {tot['N2a 误拒']}，文件崩溃 {tot['文件超时/崩溃']}")
    V.check('N2b 接口诊断必拒', tot['接口诊断'] > 0 and tot['N2b 漏判'] == 0 and tot['MLIR 接受'] > 0 and tot['规则拒绝'] > 0,
            f"接口诊断 {tot['接口诊断']}，漏判 {tot['N2b 漏判']}，规则拒绝 {tot['规则拒绝']}（其中 MLIR 报别的诊断 {tot['规则拒绝/MLIR 别的诊断']}），MLIR 崩溃 {tot['MLIR 崩溃（按拒绝计）']}")
    V.check('N2d 清空 op 完整模型≡MLIR', tot['N2d 比较'] > 0 and tot['N2d MLIR 拒绝'] > 0 and tot['N2d 误拒'] == 0 and tot['N2d 误收'] == 0,
            f"比较 {tot['N2d 比较']}（MLIR 拒绝 {tot['N2d MLIR 拒绝']}），误拒 {tot['N2d 误拒']}，误收 {tot['N2d 误收']}")
    V.finish()
