"""第 62 轮：复合 op 的显式签名 Φ（建在有拒绝权的自动签名 odssig + 属性层 attrsig 上）。

用户的核心需求：复合 op 的实现图定下来后，其签名允许的最大类型约束范围与端口间关系由图推出，且与 MLIR op 的签名同构。
此前（第 22/24 轮）只有逐点判定 composite.check（给定端口类型 T，内联后逐节点检查）与不完备的端口投影；
第 8 轮的符号签名 compsig 承载在人工签名 typesig 上，误拒 240/1014（L47），只作呈现。本模块补上：

  Φ_comp = ∃内部值类型. ⋀_节点 Φ_节点 。内部类型由 ODS 推断规则确定，∃ 消去 = 代换（第 8 轮 compsig 的思路），
  所以 Φ_comp 是端口类型变量上的原子合取，原子三类：
    ('pred', 约束名, 谓词, e)          端口约束（ODS TypeConstraint）作用于端口表达式 e
    ('rel',  关系, 节点信息, 各组表达式) ODS 关系（AllTypesMatch、TypesMatchWith、Same*…）
    ('attr', 节点, 输入表达式, 输出表达式) 属性层中与类型耦合的约束（不透明原子，逐点求值）
  表达式：('p', i) 端口 i | ('k', 类型) 字面量 | ('xf', 键, 函数, e) TypesMatchWith 变换 / ODS 推断。
  不能表达的部分作为 Unknown 残差显式列出（declared：C++ verify、接口 verify、C++ inferReturnTypes；unknown：未导入条目；
  无签名 op；端口组无法绑定）。残差不参与 Φ 的求值：Φ 对含残差的复合 op 是上近似，必须标 Unknown 并警告，交预言机终判。

有拒绝权的依据：每个原子都是 odssig/attrsig 的同一判定（L29/L34a，真实 IR 误拒 0），代换不改变判定（逐点等价由 odscomp_test S1/S2 检验）。
"""
import itertools
import typesig as TS, odssig as G, attrsig as A_

def ev(e, T):
    k = e[0]
    if k == 'p': return T[e[1]] if e[1] < len(T) else None
    if k == 'k': return e[1]
    x = ev(e[3], T)
    if x is None or isinstance(x, list): return None
    r = e[2](x)
    if r is G.XF_UNK:                       # 第 79 轮：解释器求值的 transformer 在此输入上未建模 ⇒ 该点 Unknown
        import nfrules as NF; raise NF.NFUnknown('TypesMatchWith 变换')
    return r
def ports_of(e):
    if e[0] == 'p': return {e[1]}
    if e[0] == 'k': return set()
    return ports_of(e[3])
def show(e, names):
    if e[0] == 'p': return names[e[1]]
    if e[0] == 'k': return TS.show(e[1])
    return f'{e[1]}({show(e[3], names)})'
def key(e): return (e[0], e[1]) if e[0] in ('p', 'k') else ('xf', e[1], key(e[3]))

XF_NAME = {}
for _k, _f in G.XF_OP.items():
    XF_NAME[id(_f)] = 'id' if '$_self' == _k or 'cast<Type>' in _k else 'i1same' if 'I1Same' in _k else 'celt' if 'ComplexType' in _k else 'elt' if 'getElementType' in _k else 'unpack'

class Phi:
    def __init__(self, nports):
        self.n = nports; self.atoms = []; self.residual = []; self.ground = []; self.bottom = None; self.outs = {}

def _seg(n):
    a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
    if not a: return None
    import irimport as I; return I._segs(a) or None

def _result_exprs(n, S, gi_e):
    """ODS 推断规则 → 结果表达式（与 composite.infer_types 同一规则）；None = 用字面量；'fail' = 不可实例化。"""
    if S is None or not S.get('infer'): return None
    env = {nm: es for (nm, *_), es in zip(S['ins'], gi_e)}
    out = []
    for nm, *_ in S['outs']:
        rl = S['infer'][nm]
        if rl[0] == 'build':
            w = G.BUILD.get(rl[1].strip())
            if w is None: return None
            out.append(('k', w))
        elif rl[0] == 'copy':
            src = env.get(rl[1]) or []
            if not src: return 'fail'
            out.append(src[0])
        else:
            src = env.get(rl[1]) or []
            if not src: return 'fail'
            out.append(('xf', XF_NAME.get(id(rl[2])) or getattr(rl[2], 'xfname', 'xf'), rl[2], src[0]))
    return out

def _resid(P, items, exprs):
    """节点的残差（C++ verify 等不透明校验）：若该节点全部操作数/结果类型都不含端口变量（定值），
    其校验结果与端口类型无关，是常量，在复合 op 定义时对实现图校验一次即可消去（第 64 轮）。记入 P.ground；否则记入 P.residual。"""
    if all(not ports_of(e) for e in exprs): P.ground += items
    else: P.residual += items

def captures(n):
    """第 75 轮：n 的区域内用到、但不在区域内定义的值（捕获）。父 op 的不透明校验（verifyRegions 等）可能读到它们的类型，
    所以判断残差是否定值时必须连同捕获一起看。"""
    defined, out = set(), []
    def walk(m):
        for r in m.get('regions') or []:
            for b in r:
                for a, _ in b['args']: defined.add(a)
                for k in b['ops']:
                    for v in k['ins']:
                        if v not in defined and v not in out: out.append(v)
                    walk(k)
                    for o, _ in k['outs']: defined.add(o)
    walk(n)
    return [v for v in out if v not in defined]

def derive(sl, ports, inner=None):
    """sl：节点表（irimport 形式）；ports：端口值名（按序）。inner：{op 名: (Phi, 端口数)}，复合 op 作节点（嵌套）。
    第 77 轮（closure §9.17）：区域 op 的块参数与结果按 regionflow 的入口边由端口表达式定义；区域内 op 与顶层一样按 ODS 推断前推结果；
    nf 规则读到的区域信息（块参数、块内结果、终结子操作数）记作表达式快照 '\x00inst'，在 holds 里按端口类型具体化。与 composite.instantiate 同一规则。"""
    import regionflow as RF
    P = Phi(len(ports)); ex = {v: ('p', i) for i, v in enumerate(ports)}
    import nfrules as _NF0; _consts = {}
    def _kt(t):
        try: return ('k', TS.parse(t))
        except Exception: return None
    def _src(src, ins_e, terms):
        if src[0] == 'elt':
            x = _src(src[1], ins_e, terms)
            return None if x is None else ('xf', 'rf_elt', RF.elt, x)
        if src[0] == 'opd': return ins_e[src[1]] if src[1] < len(ins_e) else None
        t = terms.get((src[1], 0))
        return t[src[2]] if t is not None and src[2] < len(t) and t[src[2]] is not None else None
    def _walk_regions(n, ins_e, idx):
        """走 n 的区域：块参数（入口边定义或写定）、区域内 op（递归 _node）；返回 (终结子表达式, 快照)。"""
        fl = RF.flow(n) if (RF.ENABLED and n.get('regions')) else None
        pinfo = _NF0.parent_info(n); terms = {}; snap = []
        for ri, r in enumerate(n.get('regions') or []):
            pre = dict(fl[0].get(ri, [])) if fl else {}
            rs_ = []
            for bi, b in enumerate(r):
                aes = []
                for k, (a, t) in enumerate(b['args']):
                    e = _src(pre[k], ins_e, terms) if (bi == 0 and k in pre) else _kt(t)
                    if e is None: e = _kt(t); P.bottom = P.bottom or f'{n["op"]}: 区域类型流缺源'
                    ex[a] = e; aes.append(e); _consts.pop(a, None)
                rs_.append(aes)
            for bi, b in enumerate(r):
                for m in b['ops']:
                    for o, _ in m['outs']: _consts.pop(o, None)
                    _NF0.note_const(m, _consts)
                    _node(m, idx, [ex[v] for v in m['ins']], pinfo, True)
                outs_e = {o: ex.get(o) for m in b['ops'] for o, _ in m['outs']}
                te = [ex.get(v) for v in b['ops'][-1]['ins']] if b['ops'] else []
                terms[(ri, bi)] = te
                rs_[bi] = (rs_[bi], outs_e, te)
            snap.append(rs_)
        return fl, terms, snap
    def _node(n, idx, ins_e, parent, nested):
        if nested and n['op'] in ('func.return', 'func.call', 'scf.yield'):      # 与 core.typecheck 相同：由父 op 的规则负责
            for o, t in n['outs']: ex[o] = _kt(t)
            return
        fl, terms, snap = _walk_regions(n, ins_e, idx) if n.get('regions') else (None, {}, None)
        lk = G.lookup(n['op'])
        if lk is None:
            for o, t in n['outs']: ex[o] = _kt(t)
            _resid(P, [f'{n["op"]}: 无自动签名（交预言机）'], ins_e + [ex[o] for o, _ in n['outs']] + [ex[v] for v in captures(n) if v in ex])
            return
        cls, S = lk
        gi = G.bind(S, ins_e, S['ins'], _seg(n))
        if fl and fl[1]:
            post = dict(fl[1]); outs_e = []
            for k, (o, t) in enumerate(n['outs']):
                e = _src(post[k], ins_e, terms) if k in post else _kt(t)
                if e is None: e = _kt(t); P.bottom = P.bottom or f'{n["op"]}: 区域类型流缺源'
                outs_e.append(e)
        else:
            re_ = (_result_exprs(n, S, gi) if gi is not None else None) if (RF.ENABLED or not nested) else None
            if re_ == 'fail': P.bottom = P.bottom or f'{n["op"]}: 推断规则缺输入'; re_ = None
            outs_e = re_ if re_ is not None and len(re_) == len(n['outs']) else [_kt(t) for _, t in n['outs']]
        for (o, _), e in zip(n['outs'], outs_e): ex[o] = e
        _resid(P, [f'{n["op"]}: {d}' for d in S['declared']] + [f'{n["op"]}: 未导入 {u}' for u in S['unknown']],
               ins_e + outs_e + [ex[v] for v in captures(n) if v in ex] + _snap_exprs(snap))   # 第 75 轮：捕获；第 77 轮：区域内随端口变的类型
        go = G.bind(S, outs_e, S['outs'])
        P.atoms.append(('attr', n, ins_e, outs_e))       # 属性层与绑定无关（与 core.typecheck 同）
        if gi is None or go is None: P.residual.append(f'{n["op"]}: 端口组无法绑定')
        if gi is None: return
        partial = go is None                          # 结果组无法绑定：与 odssig.check 的部分模式相同，只查操作数一侧
        for (nm, p, m, b), es in zip(S['ins'] + ([] if partial else S['outs']), gi + ([] if partial else go)):
            if m == 'one' and len(es) != 1 or m == 'optional' and len(es) > 1: P.bottom = P.bottom or f'{n["op"]}.{nm}: 个数'
            for e in es: P.atoms.append(('pred', b, p, e, (idx, nm)))
        mini = dict(S, ins=[(nm, _T, m, b) for nm, _, m, b in S['ins']], outs=[(nm, _T, m, b) for nm, _, m, b in S['outs']], infer=None)
        groups = gi + ([] if partial else go)
        for x in S['rel']:
            if x[2] not in ('ods', 'cpred', 'nf'): continue
            try: _at = A_.from_core(n)
            except Exception: _at = None
            import nfrules as _NF; _at = _NF.nf_attrs(n, _at, _consts, parent, inst=(n, snap) if snap is not None else None)
            P.atoms.append(('rel', x, (idx, n['op'], mini, partial, _at), groups, [e for g in gi for e in g], [] if partial else [e for g in go for e in g]))
    for idx, n in enumerate(sl):
        _NF0.note_const(n, _consts)
        ins_e = [ex[v] for v in n['ins']]
        if inner and n['op'] in inner:                  # 嵌套：内层 Φ 的原子经代换搬入（组合封闭）
            Q = inner[n['op']]
            sub = lambda e: ins_e[e[1]] if e[0] == 'p' else e if e[0] == 'k' else (e[0], e[1], e[2], sub(e[3]))
            for a in Q.atoms:
                if a[0] == 'pred': P.atoms.append(('pred', a[1], a[2], sub(a[3]), a[4]))
                elif a[0] == 'rel': P.atoms.append(('rel', a[1], _sub_at(a[2], sub), [[sub(e) for e in g] for g in a[3]], [sub(e) for e in a[4]], [sub(e) for e in a[5]]))
                else: P.atoms.append(('attr', a[1], [sub(e) for e in a[2]], [sub(e) for e in a[3]]))
            P.residual += [f'{n["op"]}⟨{r}⟩' for r in Q.residual]
            P.ground += [f'{n["op"]}⟨{r}⟩' for r in Q.ground]   # 内层的定值残差在外层仍是定值；内层非定值残差保守地留作残差
            if Q.bottom: P.bottom = P.bottom or Q.bottom
            for (o, _), e in zip(n['outs'], Q.outs_e): ex[o] = sub(e)
            continue
        _node(n, idx, ins_e, None, False)
    P.outs_e = []; P.env = ex
    return P

def _snap_exprs(snap):
    """快照里的全部表达式（块参数、块内结果、终结子操作数）。"""
    if not snap: return []
    out = []
    for r in snap:
        for aes, oes, te in r:
            out += [e for e in aes if e is not None] + [e for e in oes.values() if e is not None] + [e for e in te if e is not None]
    return out
_T = lambda t: True

def set_outs(P, outs):
    """复合 op 的输出端口（体内值名）→ 输出表达式（嵌套时使用者看到的结果类型）。"""
    P.outs_e = [P.env[o] for o in outs]

class _Unknown:
    """第 68 轮：三值中的“未定”。nf 规则在该点未建模（NFUnknown）时取此值；布尔上下文中为假（不当作接受）。"""
    def __bool__(self): return False
    def __repr__(self): return 'UNK'
UNK = _Unknown()

def _tterm_e(a):
    """第 77 轮：nf 规则读到的区域表达式（快照 '\x00inst'）。"""
    at = a[2][4] if a[0] == 'rel' else None
    if not at or '\x00inst' not in at: return []
    return _snap_exprs(at['\x00inst'][1])
def _sub_at(info, sub):
    at = info[4]
    if not at or '\x00inst' not in at: return info
    n, snap = at['\x00inst']
    s2 = [[([None if e is None else sub(e) for e in aes], {o: (None if e is None else sub(e)) for o, e in oes.items()}, [None if e is None else sub(e) for e in te])
           for aes, oes, te in r] for r in snap]
    at = dict(at); at['\x00inst'] = (n, s2)
    return info[:4] + (at,) + info[5:]
def _at_eval(at, T):
    """快照在端口类型 T 下具体化：区域各块的块参数、块内结果、终结子操作数取具体类型，再按 core 同一函数算区域信息。"""
    if not at or '\x00inst' not in at: return at
    import nfrules as NF
    n, snap = at['\x00inst']
    sh = lambda e: None if e is None else (lambda t: None if t is None else TS.show(t))(ev(e, T))
    regs, cap = [], {}
    for r, rs_ in zip(n['regions'], snap):
        rr = []
        for b, (aes, oes, te) in zip(r, rs_):
            args = [(a, sh(e) or t) for (a, t), e in zip(b['args'], aes)]
            ops = [dict(m, outs=[(o, sh(oes.get(o)) or t) for o, t in m['outs']]) for m in b['ops']]
            if b['ops']:
                for v, e in zip(b['ops'][-1]['ins'], te): cap.setdefault(v, sh(e))
            rr.append(dict(b, args=args, ops=ops))
        regs.append(rr)
    at = {k: v for k, v in at.items() if k != '\x00inst'}
    return NF.region_keys(dict(n, regions=regs), at, vty=cap.get)

def holds(a, T):
    """第 79 轮：表达式求值（解释器求值的 TypesMatchWith 变换）在该点未建模 ⇒ UNK。"""
    try: return _holds(a, T)
    except Exception as e:
        if type(e).__name__ == 'NFUnknown': return UNK
        raise

def _holds(a, T):
    """单个原子在端口类型 T 下：True / False / UNK（nf 规则在该点未建模）。表达式求值失败（变换无定义）⇒ 不成立（与 instantiate 返回 None 一致）。"""
    import attrsig as A
    if a[0] == 'pred':
        t = ev(a[3], T); return t is not None and bool(a[2](t))
    if a[0] == 'rel':
        x, (idx, op, mini, partial, at), groups = a[1], a[2], a[3]
        gt = [[ev(e, T) for e in g] for g in groups]
        if any(ev(e, T) is None for e in _aexprs(a)): return False      # 只要求被引用的表达式有值（投影时类外端口为 None）
        S1 = dict(mini, rel=[x]); ni = len(mini['ins'])
        try: return not G.check(S1, gt[:ni], None if partial else gt[ni:], attrs=_at_eval(at, T), strict=True)
        except Exception as e:
            if type(e).__name__ == 'NFUnknown': return UNK
            raise
    n, ie, oe = a[1], a[2], a[3]
    it = [ev(e, T) for e in ie]; ot = [ev(e, T) for e in oe]
    if any(t is None for t in it + ot): return False
    return not A.node_check(n, it, ot)

def evaluate(P, T):
    """Φ(T)：True 接受 / False 拒绝 / UNK 该点未定（有原子取 UNK 且无原子为假）；None = 类型文本不可解析。
    残差不参与（调用方据 P.residual 标 Unknown）。"""
    if P.bottom: return False
    try: Tp = [TS.parse(t) if isinstance(t, str) else t for t in T]
    except Exception: return None
    unk = False
    for a in P.atoms:
        h = holds(a, Tp)
        if h is False: return False
        if h is UNK: unk = True
    return UNK if unk else True

# ───────────────────────── 化简与呈现（只做逐点等价的变换；等价性由 odscomp_test S1 外部检验）
REFLEXIVE = {'same', 'elt_eq', 'samec_all', 'same_ops', 'elt_ops', 'elt_all', 'shape_ops', 'shape_all'}
def _relgroups(a):
    """关系实际引用的表达式组（命名端口组 / 第 k 个操作数或结果）；无参关系（Same* 等）引用全部操作数与结果。"""
    x, mini = a[1], a[2][2]
    if not x[1]: return [a[4], a[5]]
    names = [nm for nm, *_ in mini['ins'] + mini['outs']]; out = []
    for r in x[1]:
        if r[0] == 'name': out.append(a[3][names.index(r[1])] if r[1] in names and names.index(r[1]) < len(a[3]) else [])
        else: L = a[4] if r[0] == 'opd' else a[5]; out.append([L[r[1]]] if r[1] < len(L) else [])
    return out
def _aexprs(a):
    if a[0] == 'pred': return [a[3]]
    if a[0] == 'rel': return [e for g in _relgroups(a) for e in g] + _tterm_e(a)
    return a[2] + a[3]
def simplify(P):
    """返回新 Phi：常量原子求值（假 ⇒ ⊥）；同一表达式上的同名约束去重；全同表达式上的自反关系删去；属性原子不涉及端口的求值。"""
    Q = Phi(P.n); Q.residual = list(P.residual); Q.ground = list(P.ground); Q.bottom = P.bottom; Q.outs_e = getattr(P, 'outs_e', [])
    seen = set()
    for a in P.atoms:
        es = _aexprs(a)
        allx = es if a[0] != 'rel' else [e for g in a[3] for e in g] + a[4] + a[5] + _tterm_e(a)
        if not any(ports_of(e) for e in allx):               # 不依赖端口 ⇒ 常量（须连未引用的组一起不依赖端口，holds 才能在空 T 上求值）
            h = holds(a, [])
            if h is False: Q.bottom = Q.bottom or f'常量原子为假：{_label(a)}'
            elif h is UNK: Q.atoms.append(a)                 # 未定的常量原子保留，求值时仍给出 UNK
            continue
        if a[0] == 'pred':
            k = ('pred', a[1], key(a[3]))
            if k in seen: continue
            seen.add(k)
        elif a[0] == 'rel':
            if a[1][0] in REFLEXIVE and len({key(e) for e in es}) <= 1: continue
            k = ('rel', a[1][0], tuple(a[1][1]), id(a[1][3]) if a[1][0] == 'match' else id(a[1][4]) if a[1][0] == 'nf' else None, a[2][3],
                 tuple((k_, (id(v_[0]), tuple(key(e) for e in _snap_exprs(v_[1]))) if k_ == '\x00inst' else repr(v_)) for k_, v_ in sorted(a[2][4].items())) if a[1][0] == 'nf' and a[1][5] and a[2][4] else None, tuple(tuple(key(e) for e in g) for g in _relgroups(a)))
            if k in seen: continue
            seen.add(k)
        Q.atoms.append(a)
    return Q

def _label(a):
    if a[0] == 'pred': return a[1]
    if a[0] == 'rel': return a[1][0]
    return f'attr@{a[1]["op"]}'

def render(P, names):
    """端口约束（每个端口的一元约束）、端口间关系、Unknown 残差——前端显示用的签名。"""
    unary = {i: [] for i in range(P.n)}; rel = []
    for a in P.atoms:
        ps = set().union(*[ports_of(e) for e in _aexprs(a)])
        if a[0] == 'pred' and a[3][0] == 'p': unary[a[3][1]].append(a[1])
        elif a[0] == 'pred': rel.append(f'{a[1]}({show(a[3], names)})')
        elif a[0] == 'rel':
            es = _aexprs(a)
            if a[1][0] == 'nf': rel.append(f'nf[{a[1][3]}]({", ".join(show(e, names) for e in es)})')   # 第 66 轮：规范化规则带呈现文本
            else: rel.append(f'{a[1][0]}({", ".join(show(e, names) for e in es)})')
        else: rel.append(f'attr@{a[1]["op"]}({", ".join(names[i] for i in sorted(ps))})')
    return {'ports': {names[i]: sorted(set(v)) for i, v in unary.items()}, 'relations': list(dict.fromkeys(rel)),
            'unknown': P.residual, 'ground': P.ground, 'bottom': P.bottom}

# ───────────────────────── 单端口投影：端口 j 的最大类型范围 R_j(t) ≝ ∃其余端口. Φ（第 63 轮）
EQ = {'same', 'same_ops'}          # 精确相等（AllTypesMatch、SameTypeOperands）；samec_all 允许动态维兼容，不是相等
def project(Q):
    """返回 (cls, ranges)。cls：端口 → 等价类代表（由端口间精确相等关系合并）。
    ranges[j] = (atoms, exact)：atoms 是只涉及 j 所在等价类的原子（类内端口都取 t 后求值）；
    exact ⇔ 没有原子把该类与类外端口联系起来（此时 R_j 与其余端口独立：R_j(t) ⇔ atoms(t) ∧ 其余类可满足）。
    exact=False 时 atoms 只是必要条件（上近似），呈现须标“上近似”。"""
    par = list(range(Q.n))
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for a in Q.atoms:
        if a[0] == 'rel' and a[1][0] in EQ:
            es = _aexprs(a)
            if es and all(e[0] == 'p' for e in es):
                for e in es[1:]: par[f(e[1])] = f(es[0][1])
    cls = [f(i) for i in range(Q.n)]
    touch = {}
    for a in Q.atoms:
        cs = {cls[p] for e in _aexprs(a) for p in ports_of(e)}
        for c in cs: touch.setdefault(c, []).append((a, len(cs) == 1))
    ranges = {}
    for j in range(Q.n):
        lst = touch.get(cls[j], [])
        ranges[j] = ([a for a, solo in lst if solo], all(solo for _, solo in lst))
    return cls, ranges

def range_holds(Q, cls, ranges, j, t):
    """R_j(t) 的类内部分：类内端口都取 t，求值只涉及该类的原子。"""
    if Q.bottom: return False
    T = [t if cls[i] == cls[j] else None for i in range(Q.n)]
    hs = [holds(a, T) for a in ranges[j][0]]
    return False if any(h is False for h in hs) else UNK if any(h is UNK for h in hs) else True   # UNK 不构成“必要条件不成立”

def render_ranges(Q, names):
    cls, rg = project(Q)
    out = {}
    for j in range(Q.n):
        if cls[j] != j and names[cls[j]] in out: out[names[j]] = f'= {names[cls[j]]}'; continue
        cons = sorted({_label(a) if a[0] != 'pred' else a[1] for a, in [(x,) for x in rg[j][0]]})
        out[names[j]] = ('精确' if rg[j][1] else '上近似') + ': ' + ' ∧ '.join(cons)
    return out

def range3(Q, j, t, hints, maxw=1024):
    """三值的单端口范围：('in', 见证) 找到使 Φ 成立的完整端口赋值；('out', None) 必要条件（类内原子）已不成立；
    ('unknown', None) 必要条件成立但未找到见证。见证搜索：其余每个端口在 {hints[i]（如原类型）, t} 中取值（≤ maxw 种组合），
    这覆盖“其余端口保持原样”与“相关端口跟随 t”两类常见情形。'in' 与 'out' 都是确定判断；前端连线检查可据此给出 可连 / 不可连 / 未定。"""
    cls, rg = project(Q)
    try: tp = TS.parse(t) if isinstance(t, str) else t
    except Exception: return ('unknown', None)
    if range_holds(Q, cls, rg, j, tp) is False: return ('out', None)
    others = sorted({cls[i] for i in range(Q.n)} - {cls[j]})
    rep = {c: next(i for i in range(Q.n) if cls[i] == c) for c in others}
    for k, combo in enumerate(itertools.product((0, 1), repeat=len(others))):
        if k >= maxw: break
        asg = {c: (t if b else hints[rep[c]]) for c, b in zip(others, combo)} | {cls[j]: t}
        T = [asg[cls[i]] for i in range(Q.n)]
        if evaluate(Q, T) is True: return ('in', T)
    return ('unknown', None)
