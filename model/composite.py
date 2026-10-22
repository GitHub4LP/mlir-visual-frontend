"""复合 op 的类型判定（第 22 轮，spec/closure.md）：以有拒绝权的层为承载，而不是第 8 轮的人工签名（typesig）。

复合 op = 端口（不标类型）+ 实现体（节点按 MLIR 的 op 名与属性原样保存）。结果类型的写法：
  ODS 有推断规则（odssig.sig 的 S['infer']，tblgen 自动生成的 inferReturnTypes）⇒ 随端口类型前推；否则是字面量。
实例 T 的判定 = instantiate(体, T) 后走 core.typecheck：结构层 + 自动签名 + 属性层（终判）→ 隔离单元预言机（declared/unknown 的残差）。
这就是 L25 的“Φ(T) ⇔ 内联后逐节点检查 ⇔ MLIR”，只是节点检查器换成了在真实 IR 上证明过不误拒的层（L33）。
compsig.py（typesig 上的符号 Φ）保留为呈现与端口投影（connectable、显示），不再有拒绝权。
"""
def infer_types(n, ins_t):
    """ODS 推断规则（odssig.sig 的 S['infer']）的具体求值；没有规则 ⇒ None（结果保持字面量）；规则求不出 ⇒ 'fail'。"""
    import odssig as G
    lk = G.lookup(n['op'])
    if lk is None or not lk[1].get('infer'): return None
    S = lk[1]; seg = None
    a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
    if a:
        import irimport as I; seg = I._segs(a) or None
    gi = G.bind(S, ins_t, S['ins'], seg)
    if gi is None: return 'fail'
    env = {nm: ts for (nm, *_), ts in zip(S['ins'], gi)}
    out = []
    for nm, *_ in S['outs']:
        rl = S['infer'][nm]
        if rl[0] == 'build': w = G.BUILD.get(rl[1].strip())
        elif rl[0] == 'copy': w = (env.get(rl[1]) or [None])[0]
        else:
            src = env.get(rl[1]) or []; w = rl[2](src[0]) if src else None
        if w is None or isinstance(w, list): return 'fail' if w is None and rl[0] != 'build' else None
        out.append(w)
    return out

def instantiate(sl, ports, T):
    """端口类型 T（字符串）→ 复合 op 体的 core 块（内部类型前推）。推断失败 ⇒ None。"""
    import typesig as TS, structure as ST, copy
    tys = dict(zip(ports, T)); nodes = []
    for n in sl:
        m = copy.deepcopy(n)
        try: ins_t = [TS.parse(tys[v]) for v in m['ins']]
        except Exception: return None
        inf = infer_types(m, ins_t)
        if inf == 'fail': return None
        if inf is not None:
            if len(inf) != len(m['outs']): return None
            m['outs'] = [(o, TS.show(t)) for (o, _), t in zip(m['outs'], inf)]
        m['props']['_types'] = [tys[v] for v in m['ins']]
        for o, t in m['outs']: tys[o] = t
        nodes.append(m)
    return [ST.blk('entry', nodes + [ST.node('ret', 'func.return', ins=[])], args=list(zip(ports, T)))]


def check(sl, ports, T, oracle=True):
    """复合 op 在端口类型 T 上的判定：[] 接受；否则诊断表。推断失败 ⇒ [('infer', …)]。"""
    import core as C
    bs = instantiate(sl, ports, T)
    if bs is None: return [('infer', '内部类型推断失败：该端口类型下不可实例化')]
    old = C.ORACLE_ON; C.ORACLE_ON = oracle
    try: return C.typecheck(bs, '(' + ', '.join(T) + ') -> ()')
    finally: C.ORACLE_ON = old

# ───────────────────────── L27 端口投影（第 24 轮，spec/closure.md §7）：建在有拒绝权的层上
SUBSET_REL = {'same', 'elt_eq', 'same_ops', 'elt_ops', 'samec_all', 'shape_ops', 'shape_all', 'elt_all', 'rank_eq', 'shape_eq', 'tdims'}
def project(sl, ports, port, t, others=None):
    """“端口 port 接类型 t”是否可能有补全。返回 ('no', 节点, 原因) 或 ('maybe',)。
    只用必要条件，故可靠（'no' ⇒ 任何补全都被有拒绝权的层拒绝 ⇒ MLIR 拒绝，L33）：
      (a) 已知类型的操作数满足自动签名的端口谓词（未登记原子取 ⊤，L29/L43）；
      (b) “全体相等/同形/同元素”一类关系（SUBSET_REL）限制在已知位置上——全体成立 ⇒ 子集成立；
      (c) 操作数全部已知的节点：完整走 odssig.check + 属性层（与 core.typecheck 同一判定）。
    已知 = 该端口、others 里给定的端口、字面量结果、由已知操作数经 ODS 推断出的结果。
    declared/unknown 的 C++ 部分、未知操作数参与的其余关系：不作为拒绝条件（不完备，L27）。"""
    import typesig as TS, odssig as G, attrsig as A, itertools
    from typesig import shaped, compat_shape, ranked, encoding
    known = {port: t} | dict(others or {})
    try: kt = {v: TS.parse(x) for v, x in known.items()}
    except Exception: return ('maybe',)
    E = lambda x: x[2] if shaped(x) else x
    for n in sl:
        ins = [kt.get(v) for v in n['ins']]
        lit = {}
        for o, ty in n['outs']:
            try: lit[o] = TS.parse(ty)
            except Exception: lit[o] = None
        lk = G.lookup(n['op'])
        if lk is None:
            for o, _ in n['outs']: kt[o] = lit[o]
            continue
        S = lk[1]; seg = None
        a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
        if a:
            import irimport as I; seg = I._segs(a) or None
        gidx = G.bind(S, list(range(len(ins))), S['ins'], seg)
        if gidx is None:
            for o, _ in n['outs']: kt[o] = None
            continue
        for (nm, p, m, _), g in zip(S['ins'], gidx):                                   # (a)
            for j in g:
                if ins[j] is not None and not p(ins[j]): return ('no', n['op'], f'{nm}: 端口约束')
        if all(x is not None for x in ins):                                            # (c)
            inf = infer_types(n, ins)
            if inf == 'fail': return ('no', n['op'], 'ODS 推断失败')
            outs = list(inf) if inf is not None else [lit[o] for o, _ in n['outs']]
            if any(x is None for x in outs):
                for o, _ in n['outs']: kt[o] = None
                continue
            gi = [[ins[j] for j in g] for g in gidx]; go = G.bind(S, outs, S['outs'])
            bad = G.check(S, gi, go)
            if bad: return ('no', n['op'], bad[0])
            nd = dict(n); nd['outs'] = [(o, TS.show(x)) for (o, _), x in zip(n['outs'], outs)]
            ab = A.node_check(nd, ins, outs)
            if ab: return ('no', n['op'], f'attr: {ab[0]}')
            for (o, _), x in zip(n['outs'], outs): kt[o] = x
            continue
        # (b) 部分已知：结果只有字面量（不随推断）的才算已知
        has_inf = bool(S.get('infer'))
        outs_k = [None if has_inf else lit[o] for o, _ in n['outs']]
        opd = [x for x in ins if x is not None]; res = [x for x in outs_k if x is not None]
        go = G.bind(S, outs_k, S['outs']) if not has_inf else None
        def grp(r):
            if r[0] == 'opd': return [ins[r[1]]] if r[1] < len(ins) else []
            if r[0] == 'res': return [outs_k[r[1]]] if r[1] < len(outs_k) else []
            for (nm, *_), g in zip(S['ins'], gidx):
                if nm == r[1]: return [ins[j] for j in g]
            for k, (nm, *_) in enumerate(S['outs']):
                if nm == r[1]: return go[k] if go is not None else []
            return []
        for x in S['rel']:
            k = x[0]
            if x[2] not in ('ods', 'cpred') or k not in SUBSET_REL: continue
            if k in ('same_ops', 'elt_ops', 'shape_ops'): ts = opd
            elif k in ('samec_all', 'shape_all', 'elt_all'): ts = opd + res
            else: ts = [y for r in x[1] for y in grp(r) if y is not None]
            if len(ts) < 2: continue
            pairs = list(itertools.combinations(ts, 2))
            if k in ('same', 'same_ops'): ok = all(u == v for u, v in pairs)
            elif k in ('elt_eq', 'elt_ops', 'elt_all'): ok = all(E(u) == E(v) for u, v in pairs)
            elif k == 'samec_all':                     # MLIR 逐个对照 result(0)：result(0) 未知时只有元素类型与编码可传递
                r0 = outs_k[0] if outs_k else None
                ok = all(E(u) == E(v) for u, v in pairs) and len({encoding(y) for y in ts if y[0] == 'tensor'}) <= 1
                if r0 is not None: ok = ok and all(compat_shape(y, r0) and y[0] == r0[0] for y in ts)
            elif k in ('shape_ops', 'shape_all'): ok = all(compat_shape(u, v) for u, v in pairs)
            elif k == 'rank_eq': ok = all(ranked(y) for y in ts) and len({len(y[1]) for y in ts}) <= 1
            elif k == 'shape_eq': ok = all(ranked(y) for y in ts) and len({y[1] for y in ts}) <= 1
            elif k == 'tdims':
                sh = lambda y: y[1] if y[0] == 'tensor' and y[1] is not None else (None if y[0] == 'tensor' else ())
                ok = all(sh(y) is not None for y in ts) and len({sh(y) for y in ts}) <= 1
            else: ok = True
            if not ok: return ('no', n['op'], f'{k}（已知位置）')
        for o, x in zip([o for o, _ in n['outs']], outs_k): kt[o] = x
    return ('maybe',)
