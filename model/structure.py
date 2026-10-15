"""结构元模型 v0：region / block / 后继 / symbol / 可见性 / 支配 / 捕获。

动机：Schema v3 只覆盖了端口、类型、语义、效应，没有覆盖 MLIR 的结构机制。
本模块把这些机制纳入同一个元模型，并给出三件东西：
  1. 结构 Schema（StructSchema）：每个 op 对区域、终结子、后继、符号、隔离性的声明，来源应为 ODS
     （RegionKindInterface / IsolatedFromAbove / Terminator / Symbol / SymbolTable），这里先人工登记。
  2. 判定：结构良构 Wf_struct，分解为互相独立的 8 条义务，每条都能定位到节点/边/块。
  3. 推导：复合 op 的结构摘要（捕获集、自由符号集、是否要求 SSACFG 区域、控制流边约束），
     使“推导出来的实体与原生实体地位相同”这一要求延伸到结构层。

判据（与 frontend-model.md §3 一致）：我们接受 ⇒ MLIR verifier 接受。
反方向允许不成立（我们更严），每一处都必须在 CONSERVATIVE 里登记理由。
运行：PYTHONPATH=model python3 model/structure.py
"""
import itertools
from dataclasses import dataclass, field

# ───────────────────────── 1 结构 Schema

@dataclass
class RegionSpec:
    kind: str = 'SSACFG'        # SSACFG | Graph
    isolated: bool = False      # IsolatedFromAbove：区域内不可引用外层定义的值
    term: str = None            # 要求的终结子 op 名；None 表示 Graph 区域无终结子
    single_block: bool = False
    args: str = ''              # 入口块参数的来源：'' | 'operands' | 'index+iter'
    table: bool = False         # 该区域是一张符号表

@dataclass
class StructSchema:
    name: str
    regions: list = field(default_factory=list)
    terminator: bool = False    # 本身是终结子
    succs: tuple = (0, 0)       # 后继数的下界和上界
    sym: bool = False           # 定义符号
    ref: str = None             # 引用符号：'flat' | 'nested'
    branch: str = None          # 控制流边规则：'if' | 'for' | None
    pure_graph_ok: bool = True  # 可否出现在 Graph 区域里
    term_op: str = None         # 第 15 轮：SingleBlockImplicitTerminator<X> 的 X（ODS）
    need_term: bool = None      # 第 15 轮：区域块须以终结子结尾（ODS：已导入、非 NoTerminator、无 RegionKindInterface）；None = 未知

def ssacfg(term, **kw): return RegionSpec(kind='SSACFG', term=term, **kw)

STRUCT = {
  'builtin.module': StructSchema('builtin.module',
      regions=[RegionSpec(kind='Graph', isolated=True, term=None, single_block=True, table=True)], sym=True),
  'func.func':  StructSchema('func.func',
      regions=[ssacfg('func.return', isolated=True, args='operands')], sym=True),
  'func.return': StructSchema('func.return', terminator=True),
  'func.call':  StructSchema('func.call', ref='flat'),
  'cf.br':      StructSchema('cf.br', terminator=True, succs=(1, 1), pure_graph_ok=False),
  'cf.cond_br': StructSchema('cf.cond_br', terminator=True, succs=(2, 2), pure_graph_ok=False),
  'scf.if':     StructSchema('scf.if',
      regions=[ssacfg('scf.yield', single_block=True), ssacfg('scf.yield', single_block=True)], branch='if'),
  'scf.for':    StructSchema('scf.for',
      regions=[ssacfg('scf.yield', single_block=True, args='index+iter')], branch='for'),
  'scf.yield':  StructSchema('scf.yield', terminator=True),
  'scf.execute_region': StructSchema('scf.execute_region', regions=[ssacfg('scf.yield')], branch='if'),
}
def st(op):
    """未登记的 op（含未注册方言）结构未知：区域个数、终结子都不约束，按 SSACFG、非隔离处理。"""
    if op in STRUCT: return STRUCT[op]
    # 第 12 轮：未手工登记的 op，终结子性质取自 ODS 的 IsTerminator；ODS 也没有的（未导入方言、未注册 op）⇒ 未知，不判 R2
    try:
        import odssig as G
        lk = G.lookup(op)
    except Exception: lk = None
    term = None if lk is None else any('IsTerminator' in t for t in lk[1]['structural'])
    need = None
    try:
        import symtab as SY
        it = SY.interfaces(op)
        if it is not None and not any(x in it for x in ('NoTerminator', 'RegionKindInterface')): need = True
        to = SY.implicit_terminator(op)
    except Exception: to = None
    return StructSchema(op, regions=None, terminator=term, succs=(0, 10 ** 9), need_term=need, term_op=to)
def region_specs(s, n): return s.regions if s.regions is not None else [RegionSpec(term='?')] * len(n['regions'])

# ───────────────────────── 2 图表示
# Blk  = dict(name, args=[(vid, ty)], ops=[Node])
# Node = dict(id, op, ins=[vid], outs=[(vid, ty)], regions=[[Blk]], succs=[(blk, [vid])],
#             sym=名字, vis='public'|'private'|'nested', callee='@f' 或 '@m::@f', props={})

def blk(name, ops, args=()): return dict(name=name, args=list(args), ops=ops)
def node(i, op, ins=(), outs=(), regions=(), succs=(), **kw):
    return dict(id=i, op=op, ins=list(ins), outs=list(outs), regions=[list(r) for r in regions],
                succs=list(succs), **kw)

# ───────────────────────── 3 判定：Wf_struct

import collections
CHECKED = collections.Counter()   # 第 15 轮：每条义务实际被求值的次数（防空检验：0 误拒必须连同触发数一起报告）

class Diag(list):
    def add(s, where, obligation, msg): s.append((where, obligation, msg))

def _defs_of(blocks):
    """返回 (vid → (块名, 序号))，序号 -1 表示块参数。"""
    d = {}
    for b in blocks:
        for v, _ in b['args']: d[v] = (b['name'], -1)
        for k, n in enumerate(b['ops']):
            for v, _ in n['outs']: d[v] = (b['name'], k)
    return d

def _types_of(blocks, out=None, deep=True):
    """值名 → 类型。deep=False 只收本区域（块参数与顶层 op 结果）。
    第 15 轮：真实 IR 中兄弟区域会复用值名（scf.while 的 before/after 都有 %1），深收集会互相覆盖 ⇒ 判定一律逐区域下行用 deep=False。"""
    t = {} if out is None else out
    for b in blocks:
        t.update(dict(b['args']))
        for n in b['ops']:
            t.update(dict(n['outs']))
            if deep:
                for r in n['regions']: _types_of(r, t)
    return t

def _reach(blocks):
    """从入口块出发的可达块。MLIR 的支配检查只作用在可达块上（已用真实 verifier 确认）。"""
    by = {b['name']: b for b in blocks}
    seen, stack = set(), [blocks[0]['name']] if blocks else []
    while stack:
        n = stack.pop()
        if n in seen or n not in by: continue
        seen.add(n)
        for o in by[n]['ops']:
            for s, _ in o['succs']: stack.append(s)
    return seen

def _idom(blocks):
    """块支配：返回 name → 支配它的块集合（含自身）。朴素不动点；只在可达块上计算（不可达前驱不参与）。"""
    reach = _reach(blocks)
    names = [b['name'] for b in blocks if b['name'] in reach]
    pred = {n: set() for n in names}
    for b in blocks:
        if b['name'] not in reach: continue
        for o in b['ops']:
            for s, _ in o['succs']:
                if s in pred: pred[s].add(b['name'])
    entry = names[0]
    dom = {n: set(names) for n in names}; dom[entry] = {entry}
    for _ in range(len(names) + 1):
        for n in names:
            if n == entry: continue
            ps = [dom[p] for p in pred[n] if p in dom]
            dom[n] = (set.intersection(*ps) if ps else set()) | {n}
    return dom

def check(root, diag=None):
    """root 是一个 builtin.module 节点。返回 Diag（空表示结构良构）。"""
    d = diag if diag is not None else Diag()
    _check_node(root, d, scope_vals=set(), tables=[], types=_types_of(root['regions'][0], deep=False) | dict(
        (v, t) for v, t in []), path='')
    _symbols(root, d)
    return d
TYPES = {}

def _visible_syms(table_node):
    """一张符号表里直接定义的符号：名字 → (可见性, 节点)。"""
    out = {}
    for r in table_node['regions']:
        for b in r:
            for n in b['ops']:
                if n.get('sym'): out[n['sym']] = (n.get('vis', 'public'), n)
    return out

def _same_ty(x, y):
    """两个类型文本是否同一类型：都能解析 ⇒ 比较 core 类型项（修饰单存）；否则比较文本；任一缺失 ⇒ None（不判）。"""
    if x is None or y is None: return None
    try:
        import typesig as TSg
        return TSg.parse(x) == TSg.parse(y)
    except Exception: return x.replace(' ', '') == y.replace(' ', '')

def _symbols(root, d):
    """义务 S1 重名、S2 解析、S3 可见性、S4 使用义务。
    第 14 轮：定义者 / 符号表 / 使用者 / 名字 / 可见性 / 引用全部经 symtab.py 从 ODS 的 trait/interface 与 _attrs 读出；
    手工字段（sym、vis、callee、ftype）仍可用（core 原生图）。第 12、13 轮真实 IR 上这些义务从未触发（见 spec/symbols.md）。
    已用真实 MLIR 确认的事实：扁平引用只在使用者的最近符号表里解析，既不向外层上溯，也看不见嵌套符号表内部；
    private 在本符号表内（含其中的非符号表区域）可见。"""
    import symtab as SY
    def table_of(n): return n['op'] in STRUCT and any(r.table for r in (STRUCT[n['op']].regions or [])) or SY.is_table(n['op'])
    def walk(tnode):
        syms = {}
        for r in tnode['regions']:
            for b in r:
                for n in b['ops']:
                    nm = SY.sym_of(n)
                    if nm and SY._has(n['op'], 'FunctionOpInterface'):
                        CHECKED['S5'] += 1
                        if not (n['regions'] and n['regions'][0]) and SY.vis_of(n) == 'public':
                            d.add(n['id'], 'S5', f"@{nm} 是声明（无函数体），不能是 public")
                    if nm:
                        CHECKED['S1'] += 1
                        if nm in syms: d.add(n['id'], 'S1', f"符号 @{nm} 在同一符号表内重名")
                        syms[nm] = (SY.vis_of(n), n)
        def inner_syms(t):
            out = {}
            for r in t['regions']:
                for b in r:
                    for n in b['ops']:
                        nm = SY.sym_of(n)
                        if nm: out[nm] = (SY.vis_of(n), n)
            return out
        def use(n, tys):
            ref = SY.ref_of(n)
            if not ref or not isinstance(ref, str) or not ref.startswith('@'): return
            parts = [p.strip('"') for p in ref.lstrip('@').split('::@')]; CHECKED['S2'] += 1
            tgt = None
            if len(parts) == 1:
                got = syms.get(parts[0])
                if not got: d.add(n['id'], 'S2', f"@{parts[0]} 在最近的符号表里不存在（不会上溯到外层）"); return
                tgt = got[1]
            else:
                got = syms.get(parts[0])
                if not got or not table_of(got[1]):
                    d.add(n['id'], 'S2', f"@{parts[0]} 不是可解析的嵌套符号表"); return
                inner = inner_syms(got[1]).get(parts[1])
                if not inner: d.add(n['id'], 'S2', f"@{parts[0]}::@{parts[1]} 不存在"); return
                if inner[0] in ('private', 'nested'):
                    d.add(n['id'], 'S3', f"@{parts[1]} 的可见性是 {inner[0]}，外部不可引用"); return
                tgt = inner[1]
            want_op, rel = (SY.USES[n['op']][1:] if n['op'] in SY.USES else ('func.func', 'call'))
            if tgt['op'] != want_op:
                d.add(n['id'], 'S2', f"@{parts[-1]} 是 {tgt['op']}，{n['op']} 需要 {want_op}"); return
            outs = [t for _, t in n['outs']]; CHECKED['S4'] += 1
            if rel == 'call':
                ft = SY.ftype_of(tgt)
                if ft is None: return
                wi, wo = _ftype(ft)
                gi = [tys.get(v) for v in n['ins']]
                if len(gi) != len(wi) or len(outs) != len(wo):
                    d.add(n['id'], 'S4', f"调用的操作数/结果个数 {len(gi)}/{len(outs)} 与 @{parts[-1]} 的 {len(wi)}/{len(wo)} 不符"); return
                for k, (x, y) in enumerate(zip(gi + outs, wi + wo)):
                    if _same_ty(x, y) is False:
                        d.add(n['id'], 'S4', f"调用的第 {k} 个端口类型 {x} 与 @{parts[-1]} 的 {y} 不符"); return
            elif rel == 'fn':
                if outs and _same_ty(outs[0], SY.ftype_of(tgt)) is False:
                    d.add(n['id'], 'S4', f"结果类型 {outs[0]} 与 @{parts[-1]} 的函数类型 {SY.ftype_of(tgt)} 不符")
            elif rel == 'ty':
                gt = (tgt.get('props', {}).get('_attrs') or {}).get('type')
                if outs and _same_ty(outs[0], gt) is False:
                    d.add(n['id'], 'S4', f"结果类型 {outs[0]} 与全局 @{parts[-1]} 的类型 {gt} 不符")
        def refs(ops, tys):
            # 值名只在隔离区域内唯一（真实 IR 每个函数都有 %arg0）⇒ 类型表随区域下行，不能用模块级的全局表
            for n in ops:
                use(n, tys)
                if table_of(n): walk(n); continue
                for r in n['regions']:
                    t2 = _types_of(r, {} if n['op'] == 'func.func' or 'IsIsolatedFromAbove' in (SY.interfaces(n['op']) or ()) else dict(tys), deep=False)
                    for b in r: refs(b['ops'], t2)
        for r in tnode['regions']:
            t0 = _types_of(r, deep=False)
            for b in r: refs(b['ops'], t0)
    walk(root)

def _check_node(n, d, scope_vals, tables, types, path):
    """义务 R1 区域个数、R2 终结子、R3 后继合法、R4 后继参数类型、R5 支配、R6 捕获/隔离、
    R7 Graph 区域里不允许 CFG、R8 控制流边类型。"""
    s = st(n['op'])
    specs = region_specs(s, n)
    if len(n['regions']) != len(specs):
        d.add(n['id'], 'R1', f"{n['op']} 需要 {len(specs)} 个区域，给了 {len(n['regions'])} 个")
    if n['regions'] or s.regions: CHECKED['R1'] += 1
    rc = _ods_regions(n['op'], len(n['regions']))      # 第 12 轮：区域块数取自 ODS 的区域约束（真实 IR：scf.if 的 else 可为空）
    for ri, (blocks, spec) in enumerate(zip(n['regions'], specs)):
        lo = rc[ri][0] if rc else (1 if s.regions is not None else 0)   # 第 14 轮：未知 op 的空区域不判（真实 IR：transform 方言）
        if not blocks:
            if lo > 0: d.add(n['id'], 'R1', f"区域 #{ri} 为空")
            continue
        if rc and rc[ri][1] is not None and len(blocks) > rc[ri][1]:
            d.add(n['id'], 'R1', f"区域 #{ri} 至多 {rc[ri][1]} 个块")
        elif not rc and spec.single_block and len(blocks) > 1:
            d.add(n['id'], 'R1', f"区域 #{ri} 只允许一个块")
        names = {b['name'] for b in blocks}
        inner_defs = _defs_of(blocks)
        inner_types = dict(types); _types_of(blocks, inner_types, deep=False)
        visible = set(inner_defs) if spec.isolated else scope_vals | set(inner_defs)
        reach = _reach(blocks)
        dom = _idom(blocks)
        for b in blocks:
            if spec.term == '?' and s.need_term:            # 第 15 轮：MLIR 核心规则——非 NoTerminator 的已注册 op，其区域的块必须以终结子结尾
                CHECKED['R2'] += 1
                if not b['ops']: d.add(f"{n['id']}/{b['name']}", 'R2', "空块：至少要有终结子")
                elif s.term_op not in (None, '?') and b['ops'][-1]['op'] != s.term_op:   # 名字比较：末 op 未导入也成立
                    d.add(f"{n['id']}/{b['name']}", 'R2', f"区域须以 {s.term_op} 结尾，实为 {b['ops'][-1]['op']}")
                elif st(b['ops'][-1]['op']).terminator is False:   # 末 op 未知（未导入方言）⇒ 不判
                    d.add(f"{n['id']}/{b['name']}", 'R2', f"块没有终结子（末 op {b['ops'][-1]['op']} 不是终结子）")
            if spec.term is None or spec.term == '?': pass     # NoTerminator / 父 op 未知
            elif CHECKED.update(['R2']): pass
            elif not b['ops'] or st(b['ops'][-1]['op']).terminator is False:
                d.add(f"{n['id']}/{b['name']}", 'R2', "块没有终结子")
            elif spec.term and b['ops'][-1]['op'] != spec.term and spec.single_block:
                d.add(f"{n['id']}/{b['name']}", 'R2', f"终结子应为 {spec.term}，实为 {b['ops'][-1]['op']}")
            for k, o in enumerate(b['ops']):
                os_ = st(o['op'])
                if os_.terminator and k != len(b['ops']) - 1:
                    d.add(o['id'], 'R2', "终结子不在块尾")
                if not (os_.succs[0] <= len(o['succs']) <= os_.succs[1]):
                    d.add(o['id'], 'R3', f"{o['op']} 需要 {os_.succs} 个后继，给了 {len(o['succs'])}")
                if o['succs']: CHECKED['R3'] += 1
                if o['succs'] and spec.kind != 'SSACFG':
                    d.add(o['id'], 'R7', "Graph 区域里不允许块间跳转")
                for si, (sn, sargs) in enumerate(o['succs']):
                    if sn not in names:
                        d.add(o['id'], 'R3', f"后继 ^{sn} 不在同一区域"); continue
                    tgt = next(b2 for b2 in blocks if b2['name'] == sn)
                    if len(sargs) != len(tgt['args']):
                        if o.get('props', {}).get('_succ_opaque'): continue      # 未知 op 的后继实参划分未知
                        d.add(o['id'], 'R4', f"后继 #{si} 参数个数 {len(sargs)} ≠ ^{sn} 的 {len(tgt['args'])}")
                    else:
                        CHECKED['R4'] += 1
                        for ai, (v, (_, ty)) in enumerate(zip(sargs, tgt['args'])):
                            if inner_types.get(v) != ty:
                                d.add(o['id'], 'R4', f"后继 #{si} 第 {ai} 个参数类型 {inner_types.get(v)} ≠ {ty}")
                for v in o['ins'] + [x for _, sa in o['succs'] for x in sa]:   # 后继操作数也是使用
                    CHECKED['R6'] += 1
                    if v not in visible:
                        d.add(o['id'], 'R6', f"{v} 在此处不可见"
                              + ("（跨越了 IsolatedFromAbove 边界）" if v in scope_vals else ""))
                    elif v in inner_defs and b['name'] in reach and spec.kind == 'SSACFG':
                        CHECKED['R5'] += 1
                        db, di = inner_defs[v]
                        if db == b['name']:
                            if di >= k: d.add(o['id'], 'R5', f"{v} 的定义不支配这次使用（同块靠后）")
                        elif db not in dom[b['name']]:
                            d.add(o['id'], 'R5', f"{v} 的定义块 ^{db} 不支配 ^{b['name']}")
                _check_node(o, d, visible, tables, inner_types, f"{path}/{o['id']}")
        # R8 控制流边类型：终结子操作数 ↔ op 结果 / 迭代参数
        if s.branch == 'if':
            for b in blocks:
                if b['ops'] and b['ops'][-1]['op'] == (spec.term or ''):
                    CHECKED['R8'] += 1
                    got = [inner_types.get(v) for v in b['ops'][-1]['ins']]
                    want = [t for _, t in n['outs']]
                    if got != want:
                        d.add(b['ops'][-1]['id'], 'R8', f"沿控制流边到父结果：{got} ≠ {want}")
        if n['op'] == 'func.func':                   # 父 op ↔ 入口块参数、ReturnLike ↔ 父结果
            import symtab as SY                       # 第 14 轮：导入节点的函数类型在 _attrs['function_type']
            ins_t, outs_t = _ftype(SY.ftype_of(n) or '() -> ()'); CHECKED['R8'] += 1
            if [t for _, t in blocks[0]['args']] != ins_t:
                d.add(n['id'], 'R8', f"入口块参数 {[t for _, t in blocks[0]['args']]} ≠ 函数输入 {ins_t}")
            for b in blocks:
                if b['ops'] and b['ops'][-1]['op'] == 'func.return':
                    got = [inner_types.get(v) for v in b['ops'][-1]['ins']]
                    if got != outs_t: d.add(b['ops'][-1]['id'], 'R8', f"return {got} ≠ 函数结果 {outs_t}")
        if s.branch == 'for':
            iters = [t for _, t in n['outs']]
            want = [types.get(n['ins'][0], 'index')] + iters    # 第 12 轮：归纳变量的类型 = 边界的类型（index 或无符号整数），不恒为 index
            got = [t for _, t in blocks[0]['args']]; CHECKED['R8'] += 1
            if got != want: d.add(n['id'], 'R8', f"循环入口块参数 {got} ≠ {want}")
            for b in blocks:
                if b['ops'] and b['ops'][-1]['op'] == 'scf.yield':
                    y = [inner_types.get(v) for v in b['ops'][-1]['ins']]
                    if y != iters: d.add(b['ops'][-1]['id'], 'R8', f"yield 类型 {y} ≠ 迭代参数 {iters}")

def _split_top(s, sep):
    out, d, cur, i = [], 0, '', 0
    while i < len(s):
        ch = s[i]
        if ch in '<([{': d += 1
        elif ch in '>)]}' and not (ch == '>' and i > 0 and s[i - 1] == '-'): d -= 1
        if d == 0 and s.startswith(sep, i):
            out.append(cur); cur = ''; i += len(sep); continue
        cur += ch; i += 1
    out.append(cur); return out

def _ftype(f):
    """函数类型 → (输入, 结果)。按括号深度切分（第 12 轮：真实 IR 的类型里有逗号与 ->，如 tensor<8xf32, "foo">、(i8) -> i8 作参数）。"""
    a, r = _split_top(f, '->')[:2] if len(_split_top(f, '->')) >= 2 else (f, '()')
    def sp(x):
        x = x.strip()
        if x.startswith('(') and x.endswith(')') and len(_split_top(x[1:-1], '->')) == 1 or (x.startswith('(') and _close(x) == len(x) - 1): x = x[1:-1]
        return [t.strip() for t in _split_top(x, ',') if t.strip()]
    return sp(a), sp(r)

def _close(x):
    d = 0
    for i, ch in enumerate(x):
        if ch == '(': d += 1
        elif ch == ')':
            d -= 1
            if d == 0: return i
    return -1

def _ods_regions(op, k):
    try:
        import odssig as G
        rc = G.region_cons(op)
    except Exception: return None
    if not rc: return None
    if any(v for *_, v in rc):                            # 可变区域：按位置展开（ODS 规定可变区域只能在最后）
        fixed = [r for r in rc if not r[3]]; var = [r for r in rc if r[3]][0]
        return [(lo, hi) for _, lo, hi, _ in fixed] + [(var[1], var[2])] * max(0, k - len(fixed))
    return [(lo, hi) for _, lo, hi, _ in rc] if len(rc) == k else None

# ───────────────────────── 4 推导：复合 op 的结构摘要

@dataclass
class StructSummary:
    captures: list        # 自由值（跨边界捕获）：内联到隔离上下文时必须提升为端口
    free_syms: list       # 自由符号：对宿主符号表的作用域义务
    needs_ssacfg: bool    # 体内有块间跳转 ⇒ 不能内联进 Graph 区域
    multi_block: bool
    isolated_ok: bool     # 可否声明 IsolatedFromAbove
    regions: int

def derive_struct(blocks):
    """从实现图推导结构摘要。与 core.derive 对应：声明只能收窄它，且必须验证。"""
    defs, caps, syms, jumps, nreg = set(), [], [], False, 0
    def walk(bs, bound):
        nonlocal jumps, nreg
        here = set(bound)
        for b in bs:
            here |= {v for v, _ in b['args']}
            for n in b['ops']: here |= {v for v, _ in n['outs']}
        for b in bs:
            for n in b['ops']:
                if n['succs']: jumps = True
                for v in n['ins']:
                    if v not in here and v not in caps: caps.append(v)
                import symtab as SY
                r_ = SY.ref_of(n)
                if r_ and r_ not in syms: syms.append(r_)
                nreg += len(n['regions'])
                for r in n['regions']: walk(r, set() if any(x.isolated for x in (st(n['op']).regions or [])) else here)
    walk(blocks, set())
    return StructSummary(caps, syms, jumps, len(blocks) > 1, not caps, nreg)

# ───────────────────────── 5 发射 MLIR（通用形式，便于差分测试）

def _mt(t): return t
def emit(root):
    def val(v): return '%' + v.replace('.', '_')
    def body(blocks, ind):
        out = []
        for b in blocks:
            if b['args'] or b['name'] != 'entry' or not b['ops']:     # 空入口块必须显式写出，否则成了“无区域体”的声明
                out.append(f"{ind}^{b['name']}(" + ', '.join(f"{val(v)}: {t}" for v, t in b['args']) + '):'
                           if b['args'] else f"{ind}^{b['name']}:")
            for n in b['ops']: out.append(op(n, ind + '  '))
        return '\n'.join(out)
    def op(n, ind):
        s = st(n['op'])
        res = ', '.join(val(v) for v, _ in n['outs'])
        head = (res + ' = ') if res else ''
        allins = n['ins'] + [v for _, sa in n['succs'] for v in sa]     # 通用形式：后继参数并入操作数表
        ops_ = '(' + ', '.join(val(v) for v in allins) + ')'
        sc = ('[' + ', '.join('^' + sn for sn, _ in n['succs']) + ']') if n['succs'] else ''
        rg = ''
        if n['regions']:
            rg = ' (' + ', '.join('{\n' + body(r, ind + '  ') + f'\n{ind}}}' for r in n['regions']) + ')'
        props = []
        if n.get('sym'): props.append(f'sym_name = "{n["sym"]}"')
        if n['op'] == 'func.func': props.append('function_type = ' + n.get('ftype', '() -> ()'))
        if n.get('callee'): props.append(f'callee = {n["callee"]}')
        if n.get('value') is not None: props.append(f'value = {n["value"]}')
        if n['op'] == 'cf.cond_br':
            props.append('operandSegmentSizes = array<i32: ' + ', '.join(
                str(x) for x in [len(n['ins'])] + [len(sa) for _, sa in n['succs']]) + '>')
        pr = ' <{' + ', '.join(props) + '}>' if props else ''
        attrs = ''
        if n.get('vis') and n['vis'] != 'public': attrs = f' {{sym_visibility = "{n["vis"]}"}}'
        sig = '(' + ', '.join(_tyof(v) for v in allins) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
        return f'{ind}{head}"{n["op"]}"{ops_}{sc}{pr}{rg}{attrs} : {sig}'
    global _TY; _TY = _types_of(root['regions'][0])
    return op(root, '')
_TY = {}
def _tyof(v): return _TY.get(v, 'i32')

# ───────────────────────── 6 差分测试语料

def M(*ops): return node('m', 'builtin.module', regions=[[blk('entry', list(ops))]])
def F(name, body, ftype='() -> ()', vis='public', args=(), nid=None, **kw):
    if args: body[0]['args'] = list(args)
    return node(nid or name, 'func.func', sym=name, vis=vis, ftype=ftype, regions=[body], **kw)
def ret(i='ret', ins=()): return node(i, 'func.return', ins=ins)
def cst(i, v, t='i32'): return node(i, 'arith.constant', outs=[(i, t)], value=f'{v} : {t}')

CASES = []
def case(name, g, expect, why=''): CASES.append((name, g, expect, why))   # 负例的 why = 期望触发的义务

case('最小函数', M(F('f', [blk('entry', [ret()])])), True)
case('private 同表调用',
     M(F('f', [blk('entry', [ret()])], vis='private'),
       F('g', [blk('entry', [node('c1', 'func.call', callee='@f'), ret()])])), True)
case('cf 后继参数匹配',
     M(F('f', [blk('entry', [node('b1', 'cf.br', succs=[('x', ['a0'])])]),
               blk('x', [ret()], args=[('a', 'i32')])], ftype='(i32) -> ()', args=[('a0', 'i32')])), True)
case('不可达块里的先用后定义（MLIR 接受）',
     M(F('f', [blk('entry', [ret()]),
               blk('d', [node('x1', 'arith.addi', ins=['y1', 'y1'], outs=[('x1', 'i32')]),
                         node('y1', 'arith.addi', ins=['x1', 'x1'], outs=[('y1', 'i32')]),
                         node('bd', 'cf.br', succs=[('d', [])])])])), True)
case('func 隔离：引用外层值',
     M(F('g', [blk('entry', [node('w', 'test.wrap', regions=[[blk('entry', [
           F('h', [blk('entry', [ret(ins=['a'])])], ftype='() -> i32')])]]), ret()])],
         ftype='(i32) -> ()', args=[('a', 'i32')])), False, 'R6')
case('支配违例：跨分支使用',
     M(F('f', [blk('entry', [node('cb', 'cf.cond_br', ins=['p'], succs=[('x', []), ('y', [])])]),
               blk('x', [cst('c', 1), node('bx', 'cf.br', succs=[('z', [])])]),
               blk('y', [node('by', 'cf.br', succs=[('z', [])])]),
               blk('z', [ret(ins=['c'])])], ftype='(i1) -> i32', args=[('p', 'i1')])), False, 'R5')
case('return 与函数结果不符', M(F('f', [blk('entry', [cst('c', 1, 'i64'), ret(ins=['c'])])], ftype='() -> i32')), False, 'R8')
case('后继参数类型不符',
     M(F('f', [blk('entry', [cst('c', 1, 'i64'), node('b1', 'cf.br', succs=[('x', ['c'])])]),
               blk('x', [ret()], args=[('a', 'i32')])])), False, 'R4')
case('scf.if 两支 yield 类型不一致',
     M(F('f', [blk('entry', [
            cst('p', 1, 'i1'),
            node('if', 'scf.if', ins=['p'], outs=[('r', 'i32')], regions=[
                [blk('t', [cst('a', 1, 'i64'), node('y1', 'scf.yield', ins=['a'])])],
                [blk('e', [cst('b', 2, 'i32'), node('y2', 'scf.yield', ins=['b'])])]]),
            ret()])])), False, 'R8')
case('符号重名', M(F('f', [blk('entry', [ret()])], nid='f1'),
                   F('f', [blk('entry', [ret()])], nid='f2')), False, 'S1')
case('平坦引用看不见嵌套表里的符号',
     M(node('m2', 'builtin.module', sym='m', regions=[[blk('entry', [F('f', [blk('entry', [ret()])])])]]),
       F('g', [blk('entry', [node('c1', 'func.call', callee='@f'), ret()])])), False, 'S2')
case('嵌套表里不能上溯到外层',
     M(F('f', [blk('entry', [ret()])]),
       node('m2', 'builtin.module', sym='m', regions=[[blk('entry', [
           F('g', [blk('entry', [node('c1', 'func.call', callee='@f'), ret()])])])]])), False, 'S2')
case('终结子缺失', M(F('f', [blk('entry', [cst('c', 1)])])), False, 'R2')
case('后继不在本区域',
     M(F('f', [blk('entry', [node('b1', 'cf.br', succs=[('nowhere', [])])])])), False, 'R3')
case('调用签名不符',
     M(F('f', [blk('entry', [ret()])], ftype='(i64) -> ()', args=[('x', 'i64')]),
       F('g', [blk('entry', [cst('c', 1, 'i32'),
                             node('c1', 'func.call', callee='@f', ins=['c']), ret()])])), False, 'S4')

def verdict_model(g):
    d = check(g)
    return (not d), d

def verdict_mlir(g):
    import mlir.ir as ir
    src = emit(g)
    with ir.Context() as c:
        c.allow_unregistered_dialects = True
        try:
            m = ir.Module.parse(src)
            return bool(m.operation.verify()), src
        except Exception as e:
            msg = [l for l in str(e).split('\n') if 'error:' in l]
            return False, (msg[0].split('error:')[1].strip() if msg else str(e))[:110]

CONSERVATIVE = {
  # 模型拒绝、MLIR 接受的已知来源（随机差分中观察到），都是有意的更严：
  'R2/未注册 op 收尾': 'MLIR 不知道未注册 op 是否终结子，于是放行；模型要求终结子必须由 Schema 声明',
}

if __name__ == '__main__':
    print('── 结构判定 × 真实 MLIR 差分')
    bad = 0
    for name, g, expect, why in CASES:
        ok_m, d = verdict_model(g)
        try: ok_r, src = verdict_mlir(g)
        except Exception as e: ok_r, src = False, f'(发射失败) {e}'
        tag = '一致' if ok_m == ok_r else ('不可靠！' if ok_m and not ok_r else '保守')
        if not ok_m and why and why not in {o for _, o, _ in d}: tag += f'（未触发期望义务 {why}）'; bad += 1
        if ok_m and not ok_r and name not in CONSERVATIVE: bad += 1
        print(f'{name:28s} 模型={"接受" if ok_m else "拒绝"} MLIR={"接受" if ok_r else "拒绝"} {tag}')
        if d: print('      模型: ' + '; '.join(f'[{o}@{w}] {m}' for w, o, m in d[:3]))
        if not ok_r: print('      MLIR: ' + src)
    print(f'\n不可靠方向的反例：{bad} 个（必须为 0）')

    print('\n── 结构摘要推导')
    body = [blk('entry', [cst('c', 1), node('cb', 'cf.cond_br', ins=['p'], succs=[('x', []), ('y', [])])]),
            blk('x', [node('c1', 'func.call', callee='@h'), node('bx', 'cf.br', succs=[('y', [])])]),
            blk('y', [ret(ins=['c'])])]
    s = derive_struct(body)
    print(f'捕获={s.captures} 自由符号={s.free_syms} 需要SSACFG={s.needs_ssacfg} 多块={s.multi_block} 可声明隔离={s.isolated_ok}')
