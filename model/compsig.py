"""复合 op 的多态类型签名（第 8 轮）。
复合 op 的端口不标类型；体内每个值的类型是端口类型的**表达式**（由各 op 的推断规则得到），体内每个节点的每个签名条目
代换为端口表达式上的约束。结果 Φ(端口) 只提端口，内部变量已经消去（∃ 消去在这里是代换：内部类型被推断函数确定）。
推导出的签名与原生签名走同一个 typesig.check，所以复合 op 可以再作为节点嵌套（组合封闭）。

表达式：('p', 端口) | ('k', 类型) | ('elt', e) | ('i1s', e) | ('with', e, 元素类型)
  'with' 是“与 e 同形状、元素换成 t”：MLIR 的 IR 里没有这种写法（只在 C++ builder 里有），
  但没有它，trunci/extui/index_cast 这类“结果类型由用户选择”的 op 放进复合 op 后，复合 op 就会失去形状多态。
节点的结果类型写法：None（推断）| 类型字符串（字面量）| ('like', 值名, 元素类型字符串)
"""
import itertools
import typesig as TS

def ev(e, T):
    k = e[0]
    if k == 'p': return T[e[1]]
    if k == 'k': return e[1]
    x = ev(e[1], T)
    if x is None: return None
    if k == 'elt': return x[2] if TS.shaped(x) else None
    if k == 'i1s': return TS.i1same(x)
    if k == 'with': return (x[0], x[1], e[2]) if TS.shaped(x) else e[2]
    raise KeyError(k)
def norm(e):
    """表达式规范化：i1s(e) = with(e, i1)；with(with(e, a), b) = with(e, b)。两条都对任意 e 成立（标量时 with 取元素本身）。"""
    if e[0] in ('p', 'k'): return e
    x = norm(e[1])
    if e[0] == 'i1s': return norm(('with', x, ('i', 1, '')))
    if e[0] == 'with' and x[0] == 'with': return ('with', x[1], e[2])
    return (e[0], x, *e[2:])
def subst(e, m):
    if e[0] == 'p': return m[e[1]]
    if e[0] == 'k': return e
    return norm((e[0], subst(e[1], m), *e[2:]))
def fv(e): return {e[1]} if e[0] == 'p' else (set() if e[0] == 'k' else fv(e[1]))
def txt(e):
    k = e[0]
    if k == 'p': return e[1]
    if k == 'k': return TS.show(e[1])
    if k == 'with': return f'with({txt(e[1])}, {TS.show(e[2])})'
    return f'{k}({txt(e[1])})'

def infer_expr(op, args):
    """符号推断：与 typesig.infer 同一套规则，只是作用在表达式上。推不出 ⇒ None（须在体内给出结果类型）。"""
    if op in TS.COMPS:
        c = TS.COMPS[op]; return [subst(o, dict(zip(c.ins, args))) for o in c.out_exprs]
    sig = TS.load()[op]
    if len(sig['outs']) != 1: return None
    rn = sig['outs'][0][0]
    ei, _ = TS.bind(dict(sig, seg=None), args, [None])
    if ei is None: return None
    for r in sig['rel']:
        if r[0] in ('same', 'samec') and rn in r[1]:
            for p in r[1]:
                if p != rn and ei.get(p): return [ei[p][0]]
        if r[0] == 'match' and r[2] == rn and ei.get(r[1]):
            return [{'$_self': lambda e: e, 'getElementType': lambda e: ('elt', e), 'getI1SameShape': lambda e: ('i1s', e)}[r[3]](ei[r[1]][0])]
    return [('k', ('index',))] if sig['outs'][0][1] == 'Index' else None

class Comp:
    """name；ins：端口名；body：[(id, op, [实参值名], [结果类型写法])]；outs：结果值名。"""
    def __init__(s, name, ins, body, outs):
        s.name, s.ins, s.body_src, s.outs = name, list(ins), body, outs
        env = {p: ('p', p) for p in ins}; s.nodes = []
        for nid, op, args, rspec in body:
            a = [env[v] for v in args]
            inf = infer_expr(op, a)
            rs = []
            inf = None if inf is None else [norm(x) for x in inf]
            for i, sp in enumerate(rspec):
                if sp is None:
                    if inf is None: raise TypeError(f'{nid}: {op} 的结果类型推不出，须在体内给出')
                    rs.append(inf[i])
                elif isinstance(sp, tuple): rs.append(norm(('with', env[sp[1]], TS.parse(sp[2]))))
                else: rs.append(('k', TS.parse(sp)))
            for i, r in enumerate(rs): env[f'{nid}#{i}' if len(rs) > 1 else nid] = r
            s.nodes.append((nid, op, a, rs))
        s.out_exprs = [env[v] for v in outs]
        s.phi = s._flatten()
        TS.COMPS[name] = s

    # ── Φ：把每个节点的每个签名条目代换成端口表达式上的约束（嵌套的复合 op 递归展开）
    def _flatten(s):
        out = []
        for nid, op, a, rs in s.nodes:
            if op in TS.COMPS:
                c = TS.COMPS[op]; m = dict(zip(c.ins, a))
                out += [(subst_c(k, m)) for k in c.phi]
                continue
            sig = TS.load()[op]
            ei, eo = TS.bind(dict(sig, seg=None), a, rs)
            if ei is None or eo is None: raise TypeError(f'{nid}: 个数不符')
            for r in TS.items(sig):
                ports = [r[1]] if r[0] == 'port' else (r[1] if isinstance(r[1], list) else [r[1], r[2]] if r[0] == 'match' else [])
                if r[0] == 'elementwise': ports = [n for n, *_ in sig['ins'] + sig['outs']]
                env = {p: (ei | eo)[p] for p in ports if p in (ei | eo)}
                out.append((op, r, env))
        return simplify(out)

    def check(s, ins, outs=None, levels=TS.LEVELS):
        if len(ins) != len(s.ins): return [('ods', '操作数个数不符')]
        T = dict(zip(s.ins, ins))
        bad = [TS.label(r) for op, r, env in s.phi if src(r) in levels and not holds(op, r, env, T)]
        if outs is not None and [ev(e, T) for e in s.out_exprs] != list(outs): bad.append(('ods', '结果类型不符推导结果'))
        return bad
    def check_body(s, ins):
        """参照：按体逐节点检查（内联语义）。"""
        T = dict(zip(s.ins, ins))
        for nid, op, a, rs in s.nodes:
            at, rt = [ev(e, T) for e in a], [ev(e, T) for e in rs]
            if None in at or None in rt or TS.check(op, at, rt): return False
        return True
    def infer(s, ins): return [ev(e, dict(zip(s.ins, ins))) for e in s.out_exprs]
    def connectable(s, port, t):
        """端口投影：只看自由变量 ⊆ {port} 的约束。可靠（不会把可接受的拒绝），不完备（关系要等其余端口）。"""
        T = {port: t}
        return all(holds(op, r, env, T) for op, r, env in s.phi if set().union(*[fv(e) for es in env.values() for e in es]) <= {port})
    def show(s):
        ins = ', '.join(s.ins); outs = ', '.join(txt(e) for e in s.out_exprs)
        cs = sorted({render(op, r, env) for op, r, env in s.phi} - {''})
        return f'{s.name}: ∀ {ins}.  ({ins}) → ({outs})\n    Φ: ' + '\n       ∧ '.join(cs)

def src(r): return 'ods' if r[0] == 'port' else r[-1]
def subst_c(k, m):
    op, r, env = k
    return (op, r, {p: [subst(e, m) for e in es] for p, es in env.items()})
def holds(op, r, env, T):
    tenv = {p: [ev(e, T) for e in es] for p, es in env.items()}
    return TS.ok(TS.load()[op], r, tenv)
def render(op, r, env):
    if r[0] == 'port':
        if r[2] == 'AnyType': return ''
        return f'{r[2]}({", ".join(txt(e) for e in env[r[1]])})'
    es = list(dict.fromkeys(txt(e) for p in env for e in env[p]))     # 显示时去掉重复实参
    if r[0] == 'match':
        return f'{", ".join(txt(e) for e in env[r[2]])} = {r[3]}({txt(env[r[1]][0])})' if env.get(r[1]) else ''
    name = r[0] if r[0] not in ('pred', 'predv') else r[2].__name__
    return f'{name}({", ".join(es)})'

SYM = {'$_self': lambda e: e, 'getElementType': lambda e: ('elt', e), 'getI1SameShape': lambda e: norm(('i1s', e))}
def simplify(cs):
    """去重；去掉与端口无关且恒真的约束（常量表达式）；去掉 same/samec/shape 中表达式全相同的平凡关系；去掉 match 的恒等式 e = $_self(e)。"""
    out, seen = [], set()
    for op, r, env in cs:
        exprs = [e for es in env.values() for e in es]
        free = set().union(*[fv(e) for e in exprs]) if exprs else set()
        if not free and holds(op, r, env, {}): continue
        if r[0] in ('same', 'samec', 'shape') and len({repr(e) for e in exprs}) <= 1: continue
        if r[0] == 'match' and env.get(r[1]) and r[3] in SYM and env[r[2]] == [SYM[r[3]](env[r[1]][0])]: continue   # 推断留下的恒等式
        key = render(op, r, env) + '|' + src(r) + '|' + repr(env)
        if key in seen: continue
        seen.add(key); out.append((op, r, env))
    return out
