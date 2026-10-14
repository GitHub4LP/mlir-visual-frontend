"""编辑代数与增量查询 v0。

1 洞（hole）是一等的 op `edit.hole`：未连接的端口 = 连到一个洞。于是每个编辑状态都良定义（全性，参照 Hazel）：
  洞的值和毒值标志是自由变量，推导结果是“对洞参数化”的 Schema，与复合 op 对输入端口参数化是同一回事。
2 编辑操作（都是全函数，作用在定义的顶层块上）：
  insert / delete（使用者改连到新洞）/ rewire / disconnect（改连到新洞）/ fill / set_prop / set_op。
3 增量：Workspace 对每个定义按 Merkle 键缓存 Schema：键 = 定义体的规范序列化 + 各被调定义的**指纹**。
  指纹 = 摘要（输出项、UB、公理、效应、结构摘要）在端口规范改名后的序列化，所以改了体但摘要不变 ⇒ 依赖者不重算（早截断，L21）。
  semantic=True 时再加语义截断：指纹不同但 z3 判定等价 ⇒ 依赖者沿用旧指纹（回溯日期），定义自身仍用新 Schema。
4 连接洞 = 摘要里的项代换（L20），不用重推导；补全查询据此把候选分为 安全 / 有条件 / 必然 UB。
判据：增量 ≡ 全量（edit_test.py）。
"""
import copy, hashlib, re, z3
import core as C, structure as ST
from core import blk, node

ST.STRUCT['edit.hole'] = ST.StructSchema('edit.hole')
def _hole(p, a):
    hid = p['hid']; t = C.outty(p)
    return [(z3.Const(f'hole.{hid}', C.zsort(t)), z3.Bool(f'hole.Z.{hid}'))], C.F_
C.nat('edit.hole', [], [('r', 'T')], _hole)

def hole(hid, t): return node(hid, 'edit.hole', outs=[(hid, t)], props={'hid': hid})

# ───────────────────────── 规范序列化与指纹
def calls(blocks, defs):
    out = set()
    def sc(bs):
        for b in bs:
            for n in b['ops']:
                if n['op'] in defs: out.add(n['op'])
                for r in n['regions']: sc(r)
    sc(blocks); return out

def ser_def(d): return repr((d['ins'], d['outs'], d['mode'], _ser(d['blocks'])))
def _ser(bs):
    return [(b['name'], b['args'], [(n['id'], n['op'], n['ins'], n['outs'], sorted((k, repr(v)) for k, v in n.get('props', {}).items()),
                                     [_ser(r) for r in n['regions']]) for n in b['ops']]) for b in bs]

def fingerprint(s):
    pairs = []
    for i, (v, p) in enumerate(s.holes or []):
        if v is not None: pairs.append((v, z3.Const(f'H{i}', v.sort())))
        pairs.append((p, z3.Bool(f'HZ{i}')))
    for nm, c in s.pconsts.items(): pairs.append((c, z3.Const(f'P.{nm}', c.sort())))
    sub = lambda t: z3.substitute(t, pairs).sexpr() if pairs else t.sexpr()
    terms = [] if not s.value_known else [sub(v) if v is not None else 'None' for v, _ in s.outv] + [sub(p) for _, p in s.outv] + [sub(s.ubf), sub(s.ax)]
    txt = repr((s.ins, s.outs, s.value_known, s.exact, terms, sorted(map(repr, s.eff)) if s.eff != C.TOP else 'TOP',
                repr(s.summary), repr(list(s.struct_diag))))
    seen = {}                                                    # 新变量（ub.poison、load）按出现顺序改名：名字是任意的
    txt = re.sub(r'([A-Za-z_.]+)!(\d+)', lambda m: seen.setdefault(m.group(0), f'fresh{len(seen)}'), txt)
    return hashlib.sha1(txt.encode()).hexdigest()[:16]

# ───────────────────────── 工作区
class Workspace:
    def __init__(s, defs, semantic=False):
        s.defs = copy.deepcopy(defs); s.memo = {}; s.fpc = {}; s.last = {}; s.semantic = semantic
        s.stats = dict(推导=0, 命中=0, 语义截断=0)
        s.qmemo = {}; s.efpc = {}
    def efp(s, sch): return s.efpc.get(id(sch), s.fp(sch))          # 依赖者看到的指纹（语义截断后可能是旧版本的）
    def fp(s, sch):
        k = id(sch)
        if k not in s.fpc: s.fpc[k] = (sch, fingerprint(sch))
        return s.fpc[k][1]
    def schema(s, name):
        d = s.defs[name]
        fps = [(c, s.efp(s.schema(c))) for c in sorted(calls(d['blocks'], s.defs))]   # 先保证被调者在 REG 里是当前版本
        key = hashlib.sha1((name + ser_def(d) + repr(fps)).encode()).hexdigest()   # 名字进键：Schema 的端口变量带定义名
        if key in s.memo: sch = s.memo[key]; s.stats['命中'] += 1
        else:
            sch = C.derive(name, d['ins'], d['outs'], d['blocks'], d['mode']); s.stats['推导'] += 1
            old = s.last.get(name)
            if s.semantic and old is not None and s.efp(old) != s.fp(sch) and same(old, sch):
                s.efpc[id(sch)] = s.efp(old); s.stats['语义截断'] += 1
            s.memo[key] = sch
        C.REG[name] = sch; s.last[name] = sch
        return sch
    def all(s): return {n: s.schema(n) for n in s.defs}
    def query(s, name, q):
        """按指纹缓存的查询（caps、Pre）：摘要不变 ⇒ 查询结果不变。"""
        sch = s.schema(name); k = (name, s.fp(sch), q)     # 指纹与端口名无关，而查询结果（Pre）用的是本定义的端口变量 ⇒ 键里要有定义名
        if k not in s.qmemo: s.qmemo[k] = QUERIES[q](sch)
        return s.qmemo[k]

def same(a, b):
    """语义截断的条件：签名、效应、结构摘要、结构诊断、公理都相同，值层等价。"""
    return (a.ins == b.ins and a.outs == b.outs and a.value_known and b.value_known and a.eff == b.eff
            and repr(a.summary) == repr(b.summary) and a.struct_diag == b.struct_diag and a.exact == b.exact
            and a.ax.sexpr() == b.ax.sexpr() and C.equivalent(a, b))

QUERIES = {'caps': lambda s: C.caps(s), 'pre': lambda s: C.user_pre(s) if s.value_known else None}

# ───────────────────────── 编辑操作（顶层块；都是全函数）
def _ops(ws, name): return ws.defs[name]['blocks'][0]['ops']
def _idx(ops, nid): return next(i for i, n in enumerate(ops) if n['id'] == nid)
def _types(ws, name):
    b = ws.defs[name]['blocks'][0]; t = dict(b['args'])
    for n in b['ops']:
        for o, ty in n['outs']: t[o] = ty
    return t
def in_scope(ws, name, nid, ty):
    """在节点 nid 处可用、类型为 ty 的值（顶层：块参数与它之前的结果）。"""
    b = ws.defs[name]['blocks'][0]; ops = b['ops']; i = _idx(ops, nid)
    return [v for v, t in b['args'] if t == ty] + [o for n in ops[:i] for o, t in n['outs'] if t == ty]

def insert(ws, name, i, n): _ops(ws, name).insert(i, n)
def rewire(ws, name, nid, k, v): _ops(ws, name)[_idx(_ops(ws, name), nid)]['ins'][k] = v
def set_prop(ws, name, nid, key, val): _ops(ws, name)[_idx(_ops(ws, name), nid)].setdefault('props', {})[key] = val
def set_op(ws, name, nid, op): _ops(ws, name)[_idx(_ops(ws, name), nid)]['op'] = op
def disconnect(ws, name, nid, k):
    ops = _ops(ws, name); i = _idx(ops, nid); ty = _types(ws, name)[ops[i]['ins'][k]]
    hid = f'{name}.h{len(ops)}_{i}_{k}_{_uid()}'
    ops.insert(i, hole(hid, ty)); ops[i + 1]['ins'][k] = hid; return hid
def delete(ws, name, nid):
    """删除节点；它的结果的每个使用改连到一个新洞（而不是拒绝删除）。"""
    ops = _ops(ws, name); i = _idx(ops, nid); n = ops.pop(i); hs = []
    for o, ty in n['outs']:
        users = [m for m in ops if o in m['ins'] or _uses_in_regions(m, o)]
        if not users: continue
        hid = f'{name}.h{_uid()}'; ops.insert(i, hole(hid, ty)); hs.append(hid)
        for m in ops: _replace(m, o, hid)
    return hs
def fill(ws, name, hid, v):
    ops = _ops(ws, name); ops.pop(_idx(ops, hid))
    for m in ops: _replace(m, hid, v)
def _replace(n, a, b):
    n['ins'] = [b if x == a else x for x in n['ins']]
    for r in n['regions']:
        for bb in r:
            for m in bb['ops']: _replace(m, a, b)
def _uses_in_regions(n, v):
    return any(v in m['ins'] or _uses_in_regions(m, v) for r in n['regions'] for bb in r for m in bb['ops'])
_u = [0]
def _uid(): _u[0] += 1; return _u[0]

# ───────────────────────── L20：连接洞 = 项代换；补全
def subst_fill(sch, hid, v_term, p_term):
    """不重推导：把摘要里的洞代换为候选值的项（候选的项取自 sch.env）。"""
    h, hz = z3.Const(f'hole.{hid}', v_term.sort()), z3.Bool(f'hole.Z.{hid}')
    sub = lambda t: z3.substitute(t, [(h, v_term), (hz, p_term)])
    s2 = copy.copy(sch)
    s2.outv = [(sub(v) if v is not None else None, sub(p)) for v, p in sch.outv]
    s2.ubf, s2.ax = sub(sch.ubf), sub(sch.ax)
    return s2

def classify(s):
    """输入非毒值时：安全（一定不 UB）/ 必然 UB / 有条件。"""
    ctx = z3.And(s.ax, C.no_poison(s))
    if C.prove(z3.Implies(ctx, z3.Not(s.ubf))): return '安全'
    if C.prove(z3.Implies(ctx, s.ubf)): return '必然UB'
    return '有条件'

def complete(ws, name, hid):
    """洞 hid 的补全候选：作用域内、类型相符、不依赖这个洞（否则成环）的值，按 L20 代换后分类。"""
    sch = ws.schema(name); ops = _ops(ws, name)
    ty = _types(ws, name)[hid]
    users = [i for i, m in enumerate(ops) if hid in m['ins'] or _uses_in_regions(m, hid)]
    first = ops[min(users)]['id'] if users else ops[-1]['id']
    out = []
    for v in in_scope(ws, name, first, ty):
        if v == hid or v not in sch.env: continue
        vt, pt = sch.env[v]
        if f'hole.{hid}' in vt.sexpr() or f'hole.Z.{hid}' in pt.sexpr(): continue        # 依赖洞本身 ⇒ 连上会成环
        out.append((v, classify(subst_fill(sch, hid, vt, pt))))
    return out
