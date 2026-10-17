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
