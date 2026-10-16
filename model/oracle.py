"""预言机层（第 10 轮）：在具体实例上调用同一提交的 MLIR verifier，作为 declared/unknown 账目的最终权威（import.md §6）。
设计：
  - 隔离：verifier 在常驻子进程里运行；子进程因断言崩溃时记为 'crash'（按拒绝处理）并自动重启。前端的崩溃不会传染给编辑器。
  - 诊断：每个 op 带 loc("节点 id")，MLIR 的诊断按节点归位，所以预言机给的不只是是/否，还有“哪个节点、MLIR 原话”。
  - 缓存：按文本哈希记忆。编辑期同一个定义体反复查询时只算一次（与 edit.md 的 Merkle 键对接，留作后续）。
接口：verify(text) → (状态 'ok'|'reject'|'crash'|'parse', [(节点 id 或 None, 消息)])；'parse' 是发射器的缺陷，不是类型结论
"""
import subprocess, sys, json, hashlib, re, time

_WORKER = r'''
import sys, json, os
import mlir.ir as ir
ctx = ir.Context(); ctx.emit_error_diagnostics = True
for line in sys.stdin:
    text, unreg = json.loads(line)
    ctx.allow_unregistered_dialects = unreg
    diags = []
    def h(d):
        diags.append([str(d.location), str(d.message)]); return True
    hid = ctx.attach_diagnostic_handler(h)
    try:
        m = ir.Module.parse(text, ctx)
    except Exception as e:
        for d in getattr(e, 'error_diagnostics', []): diags.append([str(d.location), str(d.message)])
        sys.stdout.write(json.dumps(['parse', diags or [['?', str(e).split(chr(10))[0]]]]) + '\n'); sys.stdout.flush(); hid.detach(); continue
    try:
        ok = m.operation.verify()
    except Exception as e:
        ok = False
        for d in getattr(e, 'error_diagnostics', []): diags.append([str(d.location), str(d.message)])
        if not diags: diags.append(['?', str(e).split('\n')[0]])
    hid.detach()
    sys.stdout.write(json.dumps([bool(ok), diags]) + '\n'); sys.stdout.flush()
'''

class Oracle:
    def __init__(s):
        s.p = None; s.cache = {}; s.stats = {'查询': 0, '缓存命中': 0, '崩溃': 0, '重启': 0, '耗时 s': 0.0}
    def _start(s):
        s.p = subprocess.Popen([sys.executable, '-c', _WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def verify(s, text, unreg=False):
        """unreg：允许未注册方言（第 16 轮）。未注册 op 对 MLIR 不透明——它自身的义务不查（上近似、不误拒），其余已注册 op 照查。
        只在模块上下文本就允许未注册方言时打开（导入语料：源文件带 --allow-unregistered-dialect；前端：用户自定义而未降级的 op）。"""
        k = hashlib.sha1((('U' if unreg else '') + text).encode()).hexdigest()
        if k in s.cache: s.stats['缓存命中'] += 1; return s.cache[k]
        s.stats['查询'] += 1; t0 = time.perf_counter()
        if s.p is None or s.p.poll() is not None:
            if s.p is not None: s.stats['重启'] += 1
            s._start()
        try:
            s.p.stdin.write(json.dumps([text, unreg]) + '\n'); s.p.stdin.flush()
            line = s.p.stdout.readline()
        except BrokenPipeError: line = ''
        if not line:                                   # 子进程死了：MLIR 断言崩溃
            s.p.wait(); s.stats['崩溃'] += 1; res = ('crash', [(None, 'MLIR verifier 崩溃（断言失败），按拒绝处理')])
        else:
            ok, diags = json.loads(line)
            ds = [(_loc(l), m) for l, m in diags]
            if ok == 'parse' and ds and all(x is not None for x, _ in ds): ok = False   # Module.parse 解析后即校验：诊断都落在 loc("节点") 上 ⇒ 是校验拒绝
            res = ('parse' if ok == 'parse' else 'ok' if ok else 'reject', ds)
        s.stats['耗时 s'] += time.perf_counter() - t0
        s.cache[k] = res; return res
    def close(s):
        if s.p and s.p.poll() is None: s.p.stdin.close(); s.p.wait()

def _loc(l):
    m = re.search(r'loc\("([^"]+)"\)', l)            # 只认 loc("节点")；文件行列位置（"-":2:30）是语法错误
    return m.group(1) if m else None

# ── core 图 → MLIR 通用形式（每个 op 带 loc("节点 id")）。第 11 轮：区域、后继、多块、透明复合 op 内联、切片简写降级。
# 发射不忠实时返回 None（预言机不可用），原因计入 WHY_NONE。
KDYN = -9223372036854775808          # ShapedType::kDynamic
def _why(k): WHY_NONE[k] = WHY_NONE.get(k, 0) + 1

_REF = re.compile(r'@("(?:[^"\\]|\\.)*"|[\w$.\-]+)((?:::@(?:"(?:[^"\\]|\\.)*"|[\w$.\-]+))*)')
_STR = re.compile(r'"(?:[^"\\]|\\.)*"')

def _refs_in(v):
    """属性文本里的符号引用（去掉字符串字面量里的 @，再找 @name 与 @a::@b）。字符串里的 @ 不算；无法确定的情况由调用者保守处理。"""
    if v.startswith('"') and v.endswith('"') and _STR.fullmatch(v): return []          # 纯字符串属性
    return [(m.group(1).strip('"'), bool(m.group(2))) for m in _REF.finditer(v)]

def emit_core(blocks, ftype, name='f', symbols=None, mod_attrs=None):
    """core 函数体 → MLIR 模块文本（交预言机）。
    第 16 轮（modular.md）：
      · 导入节点（props 带 _attrs）逐字发射（属性原样、类型取当前值类型），不再经 ODS 守卫：MLIR 自己认识的方言（linalg、affine…）都可送检；
      · 模块级符号上下文：体内引用的符号（func.call 的 callee、memref.get_global 的 name、属性里的任何 @ref）从 symbols（名字 → 定义节点）补成私有声明；
        解析不了的引用 / 嵌套引用 / dense_resource ⇒ 不发射（否则 MLIR 会以缺符号、缺资源误拒——预言机有拒绝权，发射必须忠实）；
      · 模块属性（gpu.container_module 等）原样带上。"""
    import odssig as G, typesig as TSg, core as C
    try: blocks = C.inline_all(blocks)                     # 透明复合 op 内联；不透明的留下，在 op() 里被拒发
    except Exception as e: _why('内联失败'); return None
    tys, lines, cnt = {}, [], [0]
    refs = set()
    def val(v): return '%' + re.sub(r'[^\w]', '_', v)
    def bname(b): return '^' + re.sub(r'[^\w]', '_', b)
    def fresh(): cnt[0] += 1; return f'%_o{cnt[0]}'
    def attrs(n):
        """core 的参数 → MLIR 属性：统一由 attrsig.from_core 翻译（第 13 轮）。翻不了 ⇒ 不发射。"""
        import attrsig as A
        try: at = A.from_core(n)
        except (A.NotMLIR, A.Symbolic) as e: raise LookupError(str(e))
        return [k if v == 'unit' else f'{k} = {v}' for k, v in at.items()]
    def line(ind, outs, op, ins, props, regions, succs, nid):
        res = ', '.join(val(o) for o, _ in outs)
        sc = ('[' + ', '.join(bname(b) for b, _ in succs) + ']') if succs else ''
        rg = (' (' + ', '.join(regions) + ')') if regions else ''
        lines.append(f'{ind}' + (res + ' = ' if res else '') + f'"{op}"(' + ', '.join(ins) + ')' + sc
                     + (' <{' + ', '.join(props) + '}>' if props else '') + rg
                     + ' : (' + ', '.join(tys[v] for v in ins) + ') -> (' + ', '.join(t for _, t in outs) + f') loc("{nid}")')
    def lower_slice(n, ind):
        """core 简写 → MLIR 形式：步长恒为 1；insert_slice 的尺寸取源的 extent（静态维直接写，动态维用 tensor.dim 取）。"""
        if n['op'] == 'tensor.extract_slice':
            r = (len(n['ins']) - 1) // 2
            props = ['operandSegmentSizes = array<i32: 1, %d, %d, 0>' % (r, r), 'static_offsets = array<i64: %s>' % ', '.join([str(KDYN)] * r),
                     'static_sizes = array<i64: %s>' % ', '.join([str(KDYN)] * r), 'static_strides = array<i64: %s>' % ', '.join(['1'] * r)]
            ins = [val(v) for v in n['ins']]
            for v, x in zip(ins, n['ins']): tys[v] = tys[x]
            return ins, props
        src = n['ins'][0]; r = len(n['ins']) - 2; st = TSg.parse(tys[src])
        if not TSg.shaped(st) or st[1] is None or len(st[1]) != r: raise LookupError('insert_slice 的源秩与偏移数不符')
        sizes, dyn = [], []
        for d, e in enumerate(st[1]):
            if e is None or e == '?':
                k, m = fresh(), fresh(); tys[k] = 'index'; tys[m] = 'index'
                lines.append(f'{ind}{k} = "arith.constant"() <{{value = {d} : index}}> : () -> (index) loc("{n["id"]}")')
                tys[val(src)] = tys[src]
                lines.append(f'{ind}{m} = "tensor.dim"({val(src)}, {k}) : ({tys[src]}, index) -> (index) loc("{n["id"]}")')
                sizes.append(str(KDYN)); dyn.append(m)
            else: sizes.append(str(e))
        ins = [val(v) for v in n['ins']] + dyn
        for v, x in zip(ins, n['ins']): tys[v] = tys[x]
        props = ['operandSegmentSizes = array<i32: 1, 1, %d, %d, 0>' % (r, len(dyn)), 'static_offsets = array<i64: %s>' % ', '.join([str(KDYN)] * r),
                 'static_sizes = array<i64: %s>' % ', '.join(sizes), 'static_strides = array<i64: %s>' % ', '.join(['1'] * r)]
        return ins, props
    def region(bs, ind, top, verbatim_blocks=False):
        start = len(lines)
        for k, b in enumerate(bs):
            for v, t in b['args']: tys[v] = t; tys[val(v)] = t
            if k > 0 or (b["args"] and not top) or (verbatim_blocks and not top):   # 导入区域：入口块也写块标签（空块 ⇒ 否则区域变成 0 块）
                lines.append(f'{ind[:-2]}{bname(b["name"])}(' + ', '.join(f'{val(v)}: {t}' for v, t in b['args']) + '):' if b['args']
                             else f'{ind[:-2]}{bname(b["name"])}:')
            for n in b['ops']: op(n, ind)
        body = lines[start:]; del lines[start:]
        return body
    def verbatim(n, ind):
        a = n['props']['_attrs']
        for k, v in a.items():
            if 'dense_resource' in v: raise LookupError('dense_resource（方言资源表未发射）')
            for r, nested in _refs_in(v):
                if nested: raise LookupError('嵌套符号引用')
                refs.add(r)
        for x, t in n['outs']: tys[x] = t; tys[val(x)] = t
        ins = list(n['ins'])
        if n['succs'] and not n['props'].get('_succ_opaque'): ins += [x for _, sa in n['succs'] for x in sa]
        for x in ins: tys[val(x)] = tys[x]
        regs = []
        for r in n['regions']:
            body = region(r, ind + '    ', False, verbatim_blocks=True)
            regs.append('{\n' + '\n'.join(body) + '\n' + ind + '}')
        sc = ('[' + ', '.join(bname(b) for b, _ in n['succs']) + ']') if n['succs'] else ''
        res = ', '.join(val(o) for o, _ in n['outs'])
        lines.append(f'{ind}' + (res + ' = ' if res else '') + f'"{n["op"]}"(' + ', '.join(val(x) for x in ins) + ')' + sc
                     + ((' (' + ', '.join(regs) + ')') if regs else '')
                     + ((' {' + ', '.join(f'{k} = {v}' for k, v in a.items()) + '}') if a else '')
                     + ' : (' + ', '.join(tys[x] for x in ins) + ') -> (' + ', '.join(t for _, t in n['outs']) + f') loc("{n["id"]}")')
    def op(n, ind):
        o = n['op']
        if '_attrs' in (n.get('props') or {}): return verbatim(n, ind)
        if o == 'func.call':
            c = (n.get('callee') or '').lstrip('@')
            if not c or '::' in c: raise LookupError('func.call 缺被调符号')
            refs.add(c)
            for x, t in n['outs']: tys[x] = t; tys[val(x)] = t
            ins = [val(v) for v in n['ins']]
            for v, x in zip(ins, n['ins']): tys[v] = tys[x]
            return line(ind, n['outs'], o, ins, [f'callee = @{c}'], [], [], n['id'])
        if o not in ('func.return', 'scf.yield') and G.lookup(o) is None: raise LookupError(f'MLIR 不认识 {o}（不透明复合 op 或未导入方言）')
        for x, t in n['outs']: tys[x] = t; tys[val(x)] = t
        if o in ('tensor.extract_slice', 'tensor.insert_slice'):
            ins, props = lower_slice(n, ind)
            return line(ind, n['outs'], o, ins, props, [], [], n['id'])
        props = attrs(n)
        ins = [val(v) for v in n['ins']]
        for v, x in zip(ins, n['ins']): tys[v] = tys[x]
        succs = n['succs']
        for _, sa in succs:
            for x in sa: tys[val(x)] = tys[x]
        ins += [val(x) for _, sa in succs for x in sa]
        seg = n.get('seg')
        if o == 'cf.cond_br': seg = [len(n['ins'])] + [len(sa) for _, sa in succs]
        lk = G.lookup(o)
        if lk:                                           # 忠实性守卫：必需属性都发射了、分段 op 带了分段
            S = lk[1]; have = {p.split(' = ')[0].strip() for p in props}
            miss = [nm for nm, _, kind, req in S['attrs'] if req and nm not in have]
            if miss: raise LookupError(f'{o} 缺必需属性 {miss}（core 的简写形式，须先降级）')
            if any('AttrSizedOperandSegments' in t for t in S['structural']) and not seg: raise LookupError('缺分段')
        if seg: props.append('operandSegmentSizes = array<i32: ' + ', '.join(map(str, seg)) + '>')
        regs = []
        for r in n['regions']:
            body = region(r, ind + '    ', False)
            regs.append('{\n' + '\n'.join(body) + '\n' + ind + '}')
        line(ind, n['outs'], o, ins, props, regs, succs, n['id'])
    try:
        args = blocks[0]['args']
        for v, t in args: tys[v] = t; tys[val(v)] = t
        body = region(blocks, '  ', True)
    except (LookupError, KeyError) as e:
        _why(str(e)[:60]); return None
    for t in set(tys.values()):
        try: TSg.parse(t)
        except (ValueError, AttributeError): _why('core 的抽象类型'); return None
    head = f'func.func @{_q(name)}(' + ', '.join(f'{val(v)}: {t}' for v, t in args) + ') -> ' + _res(ftype) + ' {'
    fn = head + '\n' + '\n'.join(body) + '\n}'
    decls = []
    for r in sorted(refs - {name}):
        d = (symbols or {}).get(r)
        if d is None: _why('未解析的符号引用（无模块上下文）'); return None
        dec = _decl(d, r)
        if dec is None: _why(f'{d["op"]} 定义者无法声明'); return None
        decls.append(dec)
    if not decls and not mod_attrs: return fn
    ma = (' attributes {' + ', '.join(f'{k} = {v}' for k, v in mod_attrs.items()) + '}') if mod_attrs else ''
    return f'module{ma} {{\n' + '\n'.join(decls + [fn]) + '\n}'

def _q(nm): return nm if re.fullmatch(r'[\w$.]+', nm) else '"' + nm + '"'

def _res(ftype):
    import structure as ST
    parts = ST._split_top(ftype, '->')
    return '->'.join(parts[1:]).strip() if len(parts) >= 2 else '()'

def _decl(d, r):
    """被引用符号的私有声明：函数类 ⇒ 无体的 private func.func；无区域的定义者（memref.global 等）⇒ 原样（去掉初值以外不改）。"""
    import symtab as SY
    a = (d.get('props') or {}).get('_attrs') or {}
    ft = SY.ftype_of(d)
    if d['op'] == 'func.func' and ft:                     # func.call 的 verifySymbolUses 要求目标是 func.func
        return f'"func.func"() ({{}}) {{function_type = {ft}, sym_name = "{r}", sym_visibility = "private"}} : () -> ()'
    if not d['regions'] and a:
        if any('dense_resource' in v or _refs_in(v) for k, v in a.items() if k != 'sym_name'): return None
        return f'"{d["op"]}"() {{' + ', '.join(f'{k} = {v}' for k, v in a.items()) + '} : () -> ()'
    return None

WHY_NONE = {}
ORACLE = None
def get():
    global ORACLE
    if ORACLE is None: ORACLE = Oracle()
    return ORACLE
