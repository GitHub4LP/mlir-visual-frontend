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

# ── core 图 → MLIR 通用形式（每个 op 带 loc("节点 id")）。图里有 MLIR 不认识的 op（复合 op、未导入方言）⇒ None
def emit_core(blocks, ftype, name='f'):
    import odssig as G
    tys = {}
    def known(op): return op in ('func.return', 'func.call', 'scf.yield') or G.lookup(op) is not None
    def val(v): return '%' + re.sub(r'[^\w]', '_', v)
    def attrs(n):
        """core 的参数 → MLIR 属性。只翻译登记过的参数；有未登记的参数 ⇒ 不发射（宁可预言机不可用，也不发出错误的文本）。"""
        p = dict(n.get('props', {})); out = []
        if 'value' in p and not (isinstance(p['value'], (int, bool)) or re.fullmatch(r'-?\d+|true|false', str(p['value']))):
            raise LookupError('符号参数（如模式变量、复合 op 参数）不是具体常量')
        if 'value' in p: out.append(f'value = {p.pop("value")} : {p.get("type") or n["outs"][0][1]}')
        p.pop('type', None)
        fl = [k for k in ('nsw', 'nuw') if p.pop(k, False) is True]
        if fl: out.append(f'overflowFlags = #arith.overflow<{", ".join(fl)}>')
        if 'predicate' in p and isinstance(p['predicate'], int): out.append(f'predicate = {p.pop("predicate")} : i64')
        if 'exact' in p:
            if p.pop('exact') is True: out.append('isExact')
        if p: raise LookupError(f'未登记的参数 {sorted(p)}')
        return out
    lines = []
    def body(bs, ind):
        for b in bs:
            for v, t in b['args']: tys[v] = t
            if b is not bs[0] or b['args'] and ind != '  ':
                lines.append(f'{ind}^{b["name"]}(' + ', '.join(f'{val(v)}: {t}' for v, t in b['args']) + '):')
            for n in b['ops']:
                if not known(n['op']) or n['regions'] or n['succs']: raise LookupError(n['op'])
                for o, t in n['outs']: tys[o] = t
                props = attrs(n)
                lk = G.lookup(n['op'])
                if lk:                                           # 忠实性守卫：必需属性都发射了、分段 op 带了分段，才算忠实的 MLIR 形式
                    S = lk[1]; have = {p.split(' = ')[0].strip() for p in props}
                    miss = [nm for nm, _, kind, req in S['attrs'] if req and nm not in have]
                    if miss: raise LookupError(f'{n["op"]} 缺必需属性 {miss}（core 的简写形式，须先降级）')
                    if any('AttrSizedOperandSegments' in t for t in S['structural']) and not n.get('seg'): raise LookupError('缺分段')
                if n.get('seg'): props.append('operandSegmentSizes = array<i32: ' + ', '.join(map(str, n['seg'])) + '>')
                res = ', '.join(val(o) for o, _ in n['outs'])
                lines.append(f'{ind}' + (res + ' = ' if res else '') + f'"{n["op"]}"(' + ', '.join(val(v) for v in n['ins']) + ')'
                             + (' <{' + ', '.join(props) + '}>' if props else '')
                             + ' : (' + ', '.join(tys[v] for v in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + f') loc("{n["id"]}")')
    if len(blocks) != 1: WHY_NONE['多块'] = WHY_NONE.get('多块', 0) + 1; return None   # 多块体：留待结构层接入
    try:
        args = blocks[0]['args']
        for v, t in args: tys[v] = t
        body(blocks, '  ')
    except (LookupError, KeyError) as e:
        WHY_NONE[str(e)[:60]] = WHY_NONE.get(str(e)[:60], 0) + 1; return None
    import typesig as TSg
    for t in tys.values():
        try: TSg.parse(t)
        except (ValueError, AttributeError):
            WHY_NONE['core 的抽象类型'] = WHY_NONE.get('core 的抽象类型', 0) + 1; return None
    if not lines or not lines[-1].lstrip().startswith('"func.return"'):
        WHY_NONE['体不以 func.return 结尾'] = WHY_NONE.get('体不以 func.return 结尾', 0) + 1; return None
    head = f'func.func @{name}(' + ', '.join(f'{val(v)}: {t}' for v, t in args) + ') -> ' + ftype.split('->')[1].strip() + ' {'
    return head + '\n' + '\n'.join(lines) + '\n}'

WHY_NONE = {}
ORACLE = None
def get():
    global ORACLE
    if ORACLE is None: ORACLE = Oracle()
    return ORACLE
