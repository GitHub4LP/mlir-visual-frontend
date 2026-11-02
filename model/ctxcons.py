"""第 50 轮（M1.2）：把 op 移到新的父上下文时的上下文约束（“提取为复合 op”、移动节点进出容器）。

capture_test 真实 IR 的 14 段上下文失败，没有一段来自 ODS 的 HasParent：
  · 子 op 的 C++ verify 看父 op（tensor.parallel_insert_slice：父须实现 InParallelOpInterface）；
  · 父 op 的 C++/接口校验看体内的 op（memref.generic_atomic_rmw：体内只能无副作用；linalg.matmul：体须是 add/mul）——
    把体内一段换成 func.call 就改变了它看到的东西。
所以只导入 HasParent 不够。三级判定（L33：只有已证明不误拒的才可拒绝）：
  reject  ODS 的 HasParent/ParentOneOf（MLIR 原生 trait 校验，新父不在允许集中必失败）——有拒绝权，须在真实 IR 上 0 误拒（capture_test C1）；
  unknown op 自己的已声明校验（C++ verify/verifyRegions、接口 verify，沿调用闭包）文本上读父 op ⇒ 移走后无法担保；
          祖先 op 的已声明校验文本上读体 ⇒ 体内换成 call 后无法担保；声明的校验在同一提交的源码里找不到 ⇒ 同样未知；
  ok      以上都不命中。
文本扫描是可审计的导入（与 odssig 的 cpred 同级）：源码 = inventory/cpp/（MLIR 9bae84b0 的 lib/Dialect/*/IR、lib/Interfaces、lib/IR 部分文件）。
它只产生“未知”，从不拒绝，所以扫描漏判只会让“ok”偏乐观——这一方向由 capture_test C2（14 段召回）与 C3（ok 段 MLIR 接受）在真实 IR 上量。
"""
import re, glob, os, functools
import odsjson as O, odssig as G

HERE = os.path.dirname(os.path.abspath(__file__))
CPP = os.path.join(HERE, '..', 'inventory', 'cpp')
PARENT_RE = re.compile(r'getParentOp\b|getParentOfType\b|getParentRegion\b|getParentWithTrait\b|->getParent\(\)|\.getParent\(\)')
BODY_RE = re.compile(r'getBody\(|getBlock\(|\bwalk\(|getOps<|getOperations\(|\bfront\(\)|getRegion\(|getRegions\(|getTerminator\(|without_terminator|'
                     r'isMemoryEffectFree|isPure\(|hasSingleEffect|getYieldedValues|getBodyRegion|getLoopRegions')
STOP = {'verify', 'verifyRegions', 'emitOpError', 'emitError', 'success', 'failure', 'isa', 'cast', 'dyn_cast', 'get', 'size', 'getType',
        'getOperation', 'llvm', 'if', 'for', 'while', 'switch', 'return', 'sizeof', 'static_cast', 'reinterpret_cast', 'emitWarning'}

def _strip(src):
    src = re.sub(r'//[^\n]*', '', src)
    src = re.sub(r'/\*.*?\*/', '', src, flags=re.S)
    return re.sub(r'"(?:\\.|[^"\\\n])*"', '""', src)

DEF_RE = re.compile(r'\b((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*\(([^;{}()]|\([^;{}()]*\))*\)\s*(?:const\s*)?(?:->\s*[\w:<>]+\s*)?\{')

@functools.lru_cache(None)
def index():
    """{(文件前缀, 限定名) → 体}，以及 {末段名 → [体]}。只收顶层（花括号深度 0 或命名空间内）的函数定义。"""
    full, short = {}, {}
    fs = sorted(glob.glob(os.path.join(CPP, '*.cpp')))
    if len(fs) < 80: raise RuntimeError(f'inventory/cpp 只有 {len(fs)} 个源文件：先运行 scripts/fetch_cpp.sh（缺源会把校验当成“未解析”，结论偏向未知）')
    for f in fs:
        s = _strip(open(f, errors='replace').read()); pre = os.path.basename(f).split('_')[0]
        depth, ns_depth, i = 0, 0, 0
        opens = []                                   # 栈：每层是否命名空间
        for m in re.finditer(r'[{}]|\bnamespace\b[^{;]*\{|' + DEF_RE.pattern, s):
            if m.start() < i: continue
            t = m.group(0)
            if t == '{': opens.append(False); continue
            if t == '}':
                if opens: opens.pop()
                continue
            if t.startswith('namespace'): opens.append(True); continue
            if any(not x for x in opens): continue   # 函数体内的调用/控制流，不是定义
            name = m.group(1)
            if name.split('::')[-1] in STOP - {'verify', 'verifyRegions'} or name in ('if', 'for', 'while', 'switch'): continue
            j, d = m.end(), 1
            while j < len(s) and d:
                d += {'{': 1, '}': -1}.get(s[j], 0); j += 1
            body = s[m.end():j]; i = j
            full.setdefault((pre, name), []).append(body); short.setdefault(name.split('::')[-1], []).append(body)
    return full, short

def _closure(bodies, depth=3):
    """沿调用闭包展开（按末段名；STOP 与定义过多的名字不跟）。返回（全部文本, 是否有未解析——此处只按体是否为空记）。"""
    _, short = index(); seen, out, frontier = set(), list(bodies), list(bodies)
    for _ in range(depth):
        nxt = []
        for b in frontier:
            for nm in set(re.findall(r'\b((?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)\s*(?:<[^<>;]*>)?\s*\(', b)):
                k = nm.split('::')[-1]
                if k in STOP or k in seen or k not in short or len(short[k]) > 3: continue
                seen.add(k); nxt += short[k]
        out += nxt; frontier = nxt
    return '\n'.join(out)

def _prefix(op):
    p = O.DIALECTS.get(op.split('.')[0])
    if p is None: return None
    return (p[0] if isinstance(p, list) else p).split('/')[0]

@functools.lru_cache(None)
def checks(op):
    """op 的已声明校验文本（含调用闭包）与未解析项。None ⇒ op 未导入。"""
    L = G.lookup(op)
    if L is None: return None
    S = L[1]; full, short = index(); pre = _prefix(op); cls = _cls(O.ops(op.split('.')[0])[op])
    bodies, unres = [], []
    for dcl in S['declared']:
        if dcl.startswith('C++ '):
            fn = 'verifyRegions' if 'verifyRegions' in dcl else 'verify'
            b = full.get((pre, f'{cls}::{fn}')) or [v for (p_, n), v in full.items() if n.endswith(f'::{cls}::{fn}') and p_ == pre]
            b = b[0] if b and isinstance(b[0], list) else b
            if b: bodies += b
            else: unres.append(f'{cls}::{fn}')
        elif dcl.startswith('接口 verify:'):
            it = dcl.split(':', 1)[1].strip().split('::')[0]
            code = _iface_verify(op, it)
            if code is None: unres.append(it); continue
            calls = [c.split('::')[-1] for c in re.findall(r'((?:\w+::)*verify\w*|(?:\w+::)*is\w+Impl)\s*\(', code)]
            if calls and not all(c in short for c in calls): unres.append(it)
            bodies.append(code)
    return _closure(bodies), unres, S['structural']

def _cls(rec):
    """ODS 记录名 → C++ 类名（tblgen 默认：去掉第一个 '_' 之前的方言前缀）。"""
    return rec.split('_', 1)[1] if '_' in rec else rec

def _iface_verify(op, it):
    d = O.records(op.split('.')[0])
    r = G._iface(d, it) or G._iface(d, it + 'Interface')
    if r is None:
        for n in d['!instanceof'].get('OpInterface', []):
            if n == it or d[n].get('cppInterfaceName') == it: r = d[n]; break
    return r.get('verify') if r else None

@functools.lru_cache(None)
def cpp2op():
    m = {}
    for dia in O.DIALECTS:
        try: ops = O.ops(dia)
        except Exception: continue
        for op, cls in ops.items(): m.setdefault(_cls(cls), []).append(op)
    return m

def allowed_parents(op):
    """HasParent/ParentOneOf ⇒ 允许的父 op 名集合；无此 trait ⇒ None；类名无法解析 ⇒ 'unresolved'。"""
    c = checks(op)
    if c is None: return None
    hp = [t for t in c[2] if t.startswith('HasParent<')]
    if not hp: return None
    names = [x.strip().split('::')[-1] for x in hp[0][len('HasParent<'):].rsplit('>::Impl', 1)[0].split(',')]
    m, dia, out = cpp2op(), op.split('.')[0], set()
    for cl in names:
        cand = m.get(cl, [])
        same = [o for o in cand if o.split('.')[0] == dia]
        pick = same or (cand if len(cand) == 1 else [])
        if not pick: return 'unresolved'
        out |= set(pick)
    return out

def _all_ops(n):
    yield n
    for r in n['regions']:
        for b in r:
            for x in b['ops']: yield from _all_ops(x)

def move_check(seg_ops, old_chain, new_parent='func.func'):
    """把 seg_ops（顶层 op 节点表）从祖先链 old_chain（直接父在前）里移到 new_parent 下、原处换成调用。
    返回 (结论 'reject'|'unknown'|'ok', [原因])。"""
    rej, unk = [], []
    for n in seg_ops:
        ap = allowed_parents(n['op'])
        if ap == 'unresolved': unk.append(f"{n['op']}：HasParent 类名无法解析")
        elif ap is not None and new_parent not in ap: rej.append(f"{n['op']} 只能在 {sorted(ap)} 内（ODS HasParent），新父 {new_parent}")
        for x in _all_ops(n):
            c = checks(x['op'])
            if c is None:
                if x is n and old_chain and old_chain[0] != new_parent: unk.append(f"{x['op']}：未导入，父上下文改变后无法担保")
                continue
            txt, unres, _ = c
            top = x is n
            pat = PARENT_RE if top else re.compile(r'getParentOfType\b|getParentWithTrait\b')
            if (top and old_chain and old_chain[0] != new_parent or not top) and pat.search(txt):
                unk.append(f"{x['op']}：已声明校验读取父/祖先 op（{pat.search(txt).group(0)}）")
            if top and unres and old_chain and old_chain[0] != new_parent: unk.append(f"{x['op']}：已声明校验在源码中未解析 {unres[:2]}")
    for a in old_chain:
        if a == new_parent and a == old_chain[-1]: break
        c = checks(a)
        if c is None: unk.append(f'祖先 {a}：未导入'); continue
        txt, unres, _ = c
        if BODY_RE.search(txt): unk.append(f'祖先 {a}：已声明校验读取体（{BODY_RE.search(txt).group(0)}），体内换成调用后无法担保')
        elif unres: unk.append(f'祖先 {a}：已声明校验未解析 {unres[:2]}')
    if rej: return 'reject', rej + unk
    return ('unknown' if unk else 'ok'), unk
