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
    import nfrules as _NFc
    if lk is None or not (lk[1].get('infer') or (lk[1].get('cppinfer') and _NFc.CPPINFER_FWD)): return None
    S = lk[1]; seg = None
    a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
    if a:
        import irimport as I; seg = I._segs(a) or None
    gi = G.bind(S, ins_t, S['ins'], seg)
    if gi is None: return 'fail'
    if not S.get('infer'):                                # 第 80 轮：C++ inferReturnTypes（等式语义的 op）解释执行；与 odscomp._cpp_result_exprs 同一函数
        import attrsig as A_
        try: at = A_.from_core(n)
        except Exception: return None
        r = S['cppinfer']([list(g) for g in gi], at)
        if r is G.XF_UNK: return None
        return 'fail' if r is None else r
    env = {nm: ts for (nm, *_), ts in zip(S['ins'], gi)}
    out = []
    for nm, *_ in S['outs']:
        rl = S['infer'][nm]
        if rl[0] == 'build': w = G.BUILD.get(rl[1].strip())
        elif rl[0] == 'copy': w = (env.get(rl[1]) or [None])[0]
        else:
            src = env.get(rl[1]) or []; w = rl[2](src[0]) if src else None
            if w is G.XF_UNK: return None                 # 第 79 轮：变换未建模 ⇒ 结果保持字面量
        if w is None or isinstance(w, list): return 'fail' if w is None and rl[0] != 'build' else None
        out.append(w)
    return out

def _retype_regions(m, tys):
    """区域内 op 的操作数类型表按当前值类型更新（块参数与区域内结果保持图上写定的类型）。第 77 轮起只在区域类型流关闭时使用。"""
    env = dict(tys)
    def walk(n):
        for r in n.get('regions') or []:
            for b in r:
                for a, t in b['args']: env[a] = t
                for k in b['ops']:
                    if 'props' in k and '_types' in k['props'] and all(v in env for v in k['ins']): k['props']['_types'] = [env[v] for v in k['ins']]
                    for o, t in k['outs']: env[o] = t
                    walk(k)
    walk(m)

class _Fail(Exception): pass

def _inst_node(m, tys, top):
    """第 77 轮（closure §9.17）：节点 m（已深拷贝）在值类型表 tys 下实例化，含区域。
    区域 op 按 regionflow 的入口边定义块参数与结果类型；区域内的 op 与顶层一样按 ODS 推断前推结果（无规则 ⇒ 字面量）。
    与 odscomp.derive 同一规则（Φ ≡ 逐点）。tys 原地更新为 m 的结果类型；兄弟区域的同名值各自作用域。"""
    import typesig as TS, regionflow as RF
    try: ins_t = [TS.parse(tys[v]) for v in m['ins']]
    except Exception: raise _Fail()
    if 'props' not in m: m['props'] = {}
    if 'props' in m: m['props']['_types'] = [tys[v] for v in m['ins']]
    fl = RF.flow(m) if m.get('regions') else None
    terms = {}
    for ri, r in enumerate(m.get('regions') or []):
        pre = dict(fl[0].get(ri, [])) if fl else {}
        env = dict(tys)                                   # 同区域的块共用（多块区域里后面的块可用前面块定义的值）
        for bi, b in enumerate(r):
            if bi == 0 and pre:
                args = []
                for k, (a, t) in enumerate(b['args']):
                    if k in pre:
                        x = _src_t(pre[k], ins_t, terms)
                        if x is None: raise _Fail()
                        t = TS.show(x)
                    args.append((a, t))
                b['args'] = args
            for a, t in b['args']: env[a] = t
        for bi, b in enumerate(r):
            for k_ in b['ops']: _inst_node(k_, env, False)
            if b['ops']:
                try: terms[(ri, bi)] = [TS.parse(env[v]) for v in b['ops'][-1]['ins']]
                except Exception: terms[(ri, bi)] = None
    if fl and fl[1]:
        outs = []
        for k, (o, t) in enumerate(m['outs']):
            src = dict(fl[1]).get(k)
            if src is not None:
                x = _src_t(src, ins_t, terms)
                if x is None: raise _Fail()
                t = TS.show(x)
            outs.append((o, t))
        m['outs'] = outs
    elif m['op'] not in ('func.return', 'scf.yield', 'scf.condition'):
        inf = infer_types(m, ins_t)
        if inf == 'fail': raise _Fail()
        if inf is not None:
            if len(inf) != len(m['outs']): raise _Fail()
            m['outs'] = [(o, TS.show(t)) for (o, _), t in zip(m['outs'], inf)]
    for o, t in m['outs']: tys[o] = t

def _src_t(src, ins_t, terms):
    import regionflow as RF
    if src[0] == 'elt': return RF.elt(_src_t(src[1], ins_t, terms))
    if src[0] == 'opd': return ins_t[src[1]] if src[1] < len(ins_t) else None
    t = terms.get((src[1], 0))
    return t[src[2]] if t is not None and src[2] < len(t) else None

def instantiate(sl, ports, T):
    """端口类型 T（字符串）→ 复合 op 体的 core 块（内部类型前推）。推断失败 ⇒ None。"""
    import typesig as TS, structure as ST, copy, regionflow as RF
    tys = dict(zip(ports, T)); nodes = []
    for n in sl:
        m = copy.deepcopy(n)
        if RF.ENABLED:
            try: _inst_node(m, tys, True)
            except _Fail: return None
            nodes.append(m); continue
        try: ins_t = [TS.parse(tys[v]) for v in m['ins']]
        except Exception: return None
        inf = infer_types(m, ins_t)
        if inf == 'fail': return None
        if inf is not None:
            if len(inf) != len(m['outs']): return None
            m['outs'] = [(o, TS.show(t)) for (o, _), t in zip(m['outs'], inf)]
        m['props']['_types'] = [tys[v] for v in m['ins']]
        _retype_regions(m, tys)                    # 第 75 轮：区域内用到端口（捕获）的 op，其操作数类型随端口改变
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
