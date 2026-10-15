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
    text = json.loads(line)
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
    def verify(s, text):
        k = hashlib.sha1(text.encode()).hexdigest()
        if k in s.cache: s.stats['缓存命中'] += 1; return s.cache[k]
        s.stats['查询'] += 1; t0 = time.perf_counter()
        if s.p is None or s.p.poll() is not None:
            if s.p is not None: s.stats['重启'] += 1
            s._start()
        try:
            s.p.stdin.write(json.dumps(text) + '\n'); s.p.stdin.flush()
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

def emit_core(blocks, ftype, name='f'):
    import odssig as G, typesig as TSg, core as C
    try: blocks = C.inline_all(blocks)                     # 透明复合 op 内联；不透明的留下，在 op() 里被拒发
    except Exception as e: _why('内联失败'); return None
    tys, lines, cnt = {}, [], [0]
    def val(v): return '%' + re.sub(r'[^\w]', '_', v)
    def bname(b): return '^' + re.sub(r'[^\w]', '_', b)
    def fresh(): cnt[0] += 1; return f'%_o{cnt[0]}'
    def attrs(n):
        """core 的参数 → MLIR 属性。只翻译登记过的参数；有未登记的参数 ⇒ 不发射（宁可预言机不可用，也不发出错误的文本）。"""
        p = dict(n.get('props', {})); out = []
        lk = G.lookup(n['op']); names = {nm for nm, *_ in lk[1]['attrs']} if lk else set()
        def own(a):                                        # core 的参数只有在 MLIR 这个 op 也有对应属性时才能翻译
            if a not in names: raise LookupError(f'core 参数在 MLIR {n["op"]} 上没有对应属性（{a}）')
        if 'value' in p and not (isinstance(p['value'], (int, bool)) or re.fullmatch(r'-?\d+|true|false', str(p['value']))):
            raise LookupError('符号参数（如模式变量、复合 op 参数）不是具体常量')
        if 'value' in p: own('value'); out.append(f'value = {p.pop("value")} : {p.get("type") or n["outs"][0][1]}')
        p.pop('type', None)
        fl = [k for k in ('nsw', 'nuw') if p.pop(k, False) is True]
        if fl: own('overflowFlags'); out.append(f'overflowFlags = #arith.overflow<{", ".join(fl)}>')
        if 'predicate' in p and isinstance(p['predicate'], int): own('predicate'); out.append(f'predicate = {p.pop("predicate")} : i64')
        if 'exact' in p:
            if p.pop('exact') is True: own('isExact'); out.append('isExact')
        if p: raise LookupError(f'未登记的参数 {sorted(p)}')
        return out
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
    def region(bs, ind, top):
        start = len(lines)
        for k, b in enumerate(bs):
            for v, t in b['args']: tys[v] = t; tys[val(v)] = t
            if k > 0 or (b['args'] and not top):
                lines.append(f'{ind[:-2]}{bname(b["name"])}(' + ', '.join(f'{val(v)}: {t}' for v, t in b['args']) + '):' if b['args']
                             else f'{ind[:-2]}{bname(b["name"])}:')
            for n in b['ops']: op(n, ind)
        body = lines[start:]; del lines[start:]
        return body
    def op(n, ind):
        o = n['op']
        if o == 'func.call': raise LookupError('func.call 需要模块级符号')
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
    head = f'func.func @{name}(' + ', '.join(f'{val(v)}: {t}' for v, t in args) + ') -> ' + ftype.split('->')[1].strip() + ' {'
    return head + '\n' + '\n'.join(body) + '\n}'

WHY_NONE = {}
ORACLE = None
def get():
    global ORACLE
    if ORACLE is None: ORACLE = Oracle()
    return ORACLE
