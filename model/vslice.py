"""纵向切片 v0：可视化图 → 增量推导（类型、效应、能力）→ 编辑期查询 → 生成 MLIR → lowering → 执行。
覆盖：多态复合 op、region（scf.if/for）、递归函数（symbol）、局部可变状态（memref.alloca）、诊断回映到节点。
复合 op 与原生 op 同构：都有 Schema（∀ 类型变量:类. 输入→输出, 效应, 能力），都出现在补全里，都可以作为节点使用。
"""
import itertools, time, ctypes, re

# ───────────── 类型项与合一（带类约束的 HM）
class TV:
    _c = itertools.count()
    def __init__(s, cls=()): s.id = next(TV._c); s.ref = None; s.cls = set(cls)
    def __repr__(s): t = find(s); return show(t)
def find(t):
    r = t
    while isinstance(r, TV) and r.ref is not None: r = r.ref
    while isinstance(t, TV) and t.ref is not None and t.ref is not r: t.ref, t = r, t.ref     # 路径压缩
    return r
def show(t):
    t = find(t)
    if isinstance(t, TV): return f"T{t.id}" + (':' + '|'.join(sorted(t.cls)) if t.cls else '')
    if isinstance(t, tuple): return f"memref<{t[2]}x{show(t[1])}>"
    return t
CLS = {'Float': lambda t: t in ('f32', 'f64'), 'Int': lambda t: t in ('i1', 'i32', 'i64', 'index'),
       'Num': lambda t: t in ('f32', 'f64', 'i32', 'i64', 'index')}
class TErr(Exception): pass
def unify(a, b, trail=None):
    a, b = find(a), find(b)
    if a is b: return
    if isinstance(b, TV) and not isinstance(a, TV): a, b = b, a
    if isinstance(a, TV):
        if isinstance(b, TV): b.cls |= a.cls
        else:
            for c in a.cls:
                if not CLS[c](b) if not isinstance(b, tuple) else True: raise TErr(f"{show(b)} 不满足 {c}")
        if trail is not None: trail.append(a)
        a.ref = b; return
    if isinstance(a, tuple) and isinstance(b, tuple) and a[2] == b[2]: return unify(a[1], b[1], trail)
    if a != b: raise TErr(f"{show(a)} ≠ {show(b)}")
def fresh_copy(t, m):
    t = find(t)
    if isinstance(t, TV): return m.setdefault(t.id, TV(t.cls))
    if isinstance(t, tuple): return ('memref', fresh_copy(t[1], m), t[2])
    return t

# ───────────── 原生 op 表（生产中由 ODS 导入 + 语义覆盖层给出；见 ods_import.py / semantic-overlay.md）
def same(c, n):
    def f(at): T = TV({c}); return [T] * n, [T]
    return f
def bin_(c, spec=True): return dict(sig=same(c, 2), spec=spec)
OPS = {
 'arith.addf': bin_('Float'), 'arith.mulf': bin_('Float'), 'arith.subf': bin_('Float'),
 'arith.addi': bin_('Int'), 'arith.subi': bin_('Int'), 'arith.muli': bin_('Int'),
 'arith.divsi': bin_('Int', spec=False),                                    # UB：除零
 'arith.cmpi': dict(sig=lambda at: (lambda T: ([T, T], ['i1']))(TV({'Int'})), spec=True),
 'arith.select': dict(sig=lambda at: (lambda T: (['i1', T, T], [T]))(TV()), spec=True),
 'arith.constant': dict(sig=lambda at: ([], [at.get('type') or TV({'Num'})]), spec=True),   # 类型可由使用处推出
 'arith.index_cast': dict(sig=lambda at: ([TV({'Int'})], [at.get('type') or TV({'Int'})]), spec=True),
 'memref.alloca': dict(sig=lambda at: ([], [('memref', at.get('elem') or TV({'Num'}), at['size'])]), spec=True, eff=[('Alloc', 'r0')]),
 'memref.load': dict(sig=lambda at: (lambda E: ([('memref', E, at['size']), 'index'], [E]))(TV()), spec=False, eff=[('Read', 'i0')]),
 'memref.store': dict(sig=lambda at: (lambda E: ([E, ('memref', E, at['size']), 'index'], []))(TV()), spec=False, eff=[('Write', 'i1')]),
}

# ───────────── 定义（函数 / 复合 op）与推导出的 Schema
class Schema:
    def __init__(s, ins, outs, eff, spec, recursive):
        s.ins, s.outs, s.eff, s.spec, s.recursive = ins, outs, eff, spec, recursive
    def inst(s):
        m = {}; return [fresh_copy(t, m) for t in s.ins], [fresh_copy(t, m) for t in s.outs]
    def __repr__(s):
        m = {}; vs = []
        def nm(t):
            t = find(t)
            if isinstance(t, TV):
                if t.id not in m: m[t.id] = 'TUVW'[len(m)]; vs.append(f"{m[t.id]}:{'|'.join(sorted(t.cls)) or 'Any'}")
                return m[t.id]
            if isinstance(t, tuple): return f"memref<{t[2]}x{nm(t[1])}>"
            return t
        sig = f"({', '.join(map(nm, s.ins))}) → ({', '.join(map(nm, s.outs))})"
        caps = 'Pure' if s.spec and not s.eff else ('Speculatable' if s.spec else 'NotSpeculatable')
        return (f"∀{', '.join(vs)}. " if vs else '') + sig + f"  效应={sorted(s.eff) or '∅'}  {caps}" + ('（递归：终止性未知）' if s.recursive else '')

class Program:
    def __init__(p): p.defs, p.cache = {}, {}           # cache: 名字 → (版本, Schema)；增量的单位是定义
    def define(p, name, params, body, results, ptypes=None, mode='opaque'):
        p.defs[name] = dict(name=name, params=params, body=body, results=results, ptypes=ptypes or {}, mode=mode,
                            ver=p.defs.get(name, {}).get('ver', -1) + 1)
        for k in list(p.cache):                         # 依赖它的定义一并失效
            if k == name or name in p.callees(k): p.cache.pop(k, None)
    def callees(p, name, seen=None):
        seen = seen or set()
        def walk(g):
            for n in g['nodes']:
                if n['op'] in p.defs and n['op'] not in seen: seen.add(n['op']); p.callees(n['op'], seen)
                for r in n.get('regions', []): walk(r)
        walk(p.defs[name]['body']); return seen
    def schema(p, name, stack=()):
        if name in p.cache: return p.cache[name][1]
        d = p.defs[name]; env = {}; errs = []
        for x in d['params']: env[x] = d['ptypes'].get(x) or TV()
        outs = [TV() for _ in d['results']]
        self_sig = (list(env[x] for x in d['params']), outs)   # 单态递归
        info = dict(eff=[], spec=True, recursive=False, alloc=set(), escape=set())
        p.infer(d['body'], env, errs, info, stack + ((name, self_sig),))
        for o, r in zip(outs, d['results']):
            try: unify(o, env[r])
            except TErr as e: errs.append((r, f"结果 {r}：{e}"))
        if errs: raise DiagErr(name, errs)
        esc = info['escape'] | set(d['results'])
        local = {a for a in info['alloc'] if a not in esc}
        pidx = {x: i for i, x in enumerate(d['params'])}
        eff = {(k, f"arg{pidx[v]}") for k, v in info['eff'] if v not in local and v in pidx}
        eff |= {(k, 'result') for k, v in info['eff'] if v in d['results'] and k == 'Alloc'}
        eff |= {(k, '?') for k, v in info['eff'] if v not in local and v not in pidx and v not in d['results']}
        s = Schema(self_sig[0], outs, eff, info['spec'] and not info['recursive'], info['recursive'])
        p.cache[name] = (d['ver'], s); return s

    def node_sig(p, n, errs, stack):
        op = n['op']
        if op in OPS: return OPS[op]['sig'](n.get('attrs', {})), OPS[op]
        for nm, sig in stack:
            if nm == op: return sig, dict(spec=True, recursive=True)
        if op in p.defs:
            s = p.schema(op, stack); return s.inst(), dict(spec=s.spec, eff_schema=s.eff)
        raise TErr(f"未知 op {op}")

    def infer(p, g, env, errs, info, stack):
        for n in g['nodes']:
            nid, ins = n['id'], n.get('in', [])
            try:
                if n['op'] == 'scf.if':
                    p.edge(env, ins[0], 'i1', nid, errs)
                    outs = [TV() for _ in n['regions'][0]['yield']]
                    for r in n['regions']:
                        p.infer(r, env, errs, info, stack)
                        for o, y in zip(outs, r['yield']): p.edge(env, y, o, nid, errs)
                elif n['op'] == 'scf.for':
                    for x in ins[:3]: p.edge(env, x, 'index', nid, errs)
                    r = n['regions'][0]; outs = [TV() for _ in ins[3:]]
                    env[r['args'][0]] = 'index'
                    for o, x, a in zip(outs, ins[3:], r['args'][1:]): p.edge(env, x, o, nid, errs); env[a] = o
                    p.infer(r, env, errs, info, stack)
                    for o, y in zip(outs, r['yield']): p.edge(env, y, o, nid, errs)
                    info['escape'] |= set(ins[3:]) | set(r['yield'])
                else:
                    (ti, to), meta = p.node_sig(n, errs, stack)
                    if len(ti) != len(ins): raise TErr(f"需要 {len(ti)} 个输入，连了 {len(ins)} 个")
                    for k, (t, x) in enumerate(zip(ti, ins)): p.edge(env, x, t, nid, errs, k)
                    outs = to
                    info['spec'] &= meta.get('spec', True); info['recursive'] |= meta.get('recursive', False)
                    for k, w in meta.get('eff', []):
                        v = ins[int(w[1:])] if w[0] == 'i' else nid
                        info['eff'].append((k, v)); 
                        if k == 'Alloc': info['alloc'].add(nid)
                    for k, w in meta.get('eff_schema', ()):
                        if w.startswith('arg'): info['eff'].append((k, ins[int(w[3:])]))
                        elif w == 'result': info['eff'].append((k, nid)); info['alloc'].add(nid)
                        else: info['eff'].append((k, '?'))
                    if n['op'] in p.defs or any(nm == n['op'] for nm, _ in stack): info['escape'] |= set(ins)
                for k, o in enumerate(outs): env[nid if k == 0 else f"{nid}#{k}"] = o
                n['_types'] = outs
            except TErr as e:
                errs.append((nid, str(e)))
                for k in range(8): env.setdefault(nid if k == 0 else f"{nid}#{k}", TV())

    def edge(p, env, src, t, nid, errs, k=None):
        if src not in env: errs.append((nid, f"输入 {src} 未定义（悬空端口）")); return
        try: unify(env[src], t)
        except TErr as e:
            errs.append((nid, f"边 {src}→{nid}{'' if k is None else f'.in{k}'}：{e}"))

    # ── 编辑期查询：端口可以接哪些 op（原生与复合同等对待）
    def completions(p, t):
        res = []
        for name in list(OPS) + list(p.defs):
            try:
                ti = OPS[name]['sig']({'size': t[2]} if isinstance(t, tuple) else {})[0] if name in OPS else p.schema(name).inst()[0]
            except Exception: continue
            for k, x in enumerate(ti):
                trail = []
                try: unify(x, t, trail); ok = True
                except TErr: ok = False
                for v in trail: v.ref = None
                if ok: res.append(f"{name}.in{k}"); break
        return res

class DiagErr(Exception):
    def __init__(s, d, errs): s.d, s.errs = d, errs; super().__init__('; '.join(f'[{d}:{n}] {m}' for n, m in errs))

# ───────────── 生成 MLIR（单态化：多态复合 op 按实例类型各生成一份）
def mt(t):
    t = find(t)
    if isinstance(t, TV):                              # 缺省规则：只受数值类约束的变量取 i64/f64（编辑器以提示形式告知用户）
        if not t.cls: raise TErr(f"类型未定：{show(t)}")
        t.ref = 'f64' if 'Float' in t.cls else 'i64'; DEFAULTED.append(t.id); return find(t)
    return f"memref<{t[2]}x{mt(t[1])}>" if isinstance(t, tuple) else t

DEFAULTED = []
class Emitter:
    def __init__(e, prog): e.p, e.funcs, e.done = prog, [], set()
    def mangle(e, name, ts): return name + '__' + '_'.join(mt(t).replace('<', '').replace('>', '') for t in ts)
    def func(e, name, ptypes, export=False):
        key = e.mangle(name, ptypes)
        if key in e.done: return key
        e.done.add(key); d = e.p.defs[name]
        env = {x: t for x, t in zip(d['params'], ptypes)}; errs = []
        info = dict(eff=[], spec=True, recursive=False, alloc=set(), escape=set())
        sig = (list(ptypes), [TV() for _ in d['results']])
        e.p.infer(d['body'], env, errs, info, ((name, sig),))
        for o, r in zip(sig[1], d['results']): unify(o, env[r])
        if errs: raise DiagErr(name, errs)
        body = []; e.region(d['body'], env, body, '  ', name)
        rt = [mt(env[r]) for r in d['results']]
        args = ', '.join(f"%{x}: {mt(t)}" for x, t in zip(d['params'], ptypes))
        attrs = ' attributes {llvm.emit_c_interface}' if export else ''
        ret = f"  func.return {', '.join(e.v(r) for r in d['results'])} : {', '.join(rt)}"
        e.funcs.append(f"func.func @{key}({args}) -> ({', '.join(rt)}){attrs} {{\n" + '\n'.join(body) + f"\n{ret}\n}}")
        return key
    def v(e, r): return '%' + r.replace('#', '_r')
    def lhs(e, n):
        k = len(n['_types']); return '' if k == 0 else ', '.join(e.v(n['id'] if i == 0 else f"{n['id']}#{i}") for i in range(k)) + ' = '
    def region(e, g, env, out, ind, fn):
        for n in g['nodes']:
            op, ins, at, loc = n['op'], n.get('in', []), n.get('attrs', {}), f' loc("{fn}:{n["id"]}")'
            ts = [mt(t) for t in n['_types']]; L = e.lhs(n)
            if op == 'scf.if':
                out.append(f"{ind}{L}scf.if {e.v(ins[0])} -> ({', '.join(ts)}) {{")
                for i, r in enumerate(n['regions']):
                    if i: out.append(f"{ind}}} else {{")
                    e.region(r, env, out, ind + '  ', fn)
                    out.append(f"{ind}  scf.yield {', '.join(map(e.v, r['yield']))} : {', '.join(ts)}")
                out.append(f"{ind}}}{loc}")
            elif op == 'scf.for':
                r = n['regions'][0]; it = ', '.join(f"{e.v(a)} = {e.v(x)}" for a, x in zip(r['args'][1:], ins[3:]))
                out.append(f"{ind}{L}scf.for {e.v(r['args'][0])} = {e.v(ins[0])} to {e.v(ins[1])} step {e.v(ins[2])}"
                           + (f" iter_args({it}) -> ({', '.join(ts)})" if it else '') + ' {')
                e.region(r, env, out, ind + '  ', fn)
                if r['yield']: out.append(f"{ind}  scf.yield {', '.join(map(e.v, r['yield']))} : {', '.join(ts)}")
                out.append(f"{ind}}}{loc}")
            elif op == 'arith.constant':
                val = at['value']; val = f"{float(val)!r}" if ts[0].startswith('f') else str(int(val))
                out.append(f"{ind}{L}arith.constant {val} : {ts[0]}{loc}")
            elif op == 'arith.cmpi':
                out.append(f"{ind}{L}arith.cmpi {at['pred']}, {e.v(ins[0])}, {e.v(ins[1])} : {mt(env[ins[0]])}{loc}")
            elif op == 'arith.select':
                out.append(f"{ind}{L}arith.select {', '.join(map(e.v, ins))} : {ts[0]}{loc}")
            elif op == 'arith.index_cast':
                out.append(f"{ind}{L}arith.index_cast {e.v(ins[0])} : {mt(env[ins[0]])} to {ts[0]}{loc}")
            elif op == 'memref.alloca': out.append(f"{ind}{L}memref.alloca() : {ts[0]}{loc}")
            elif op == 'memref.load': out.append(f"{ind}{L}memref.load {e.v(ins[0])}[{e.v(ins[1])}] : {mt(env[ins[0]])}{loc}")
            elif op == 'memref.store': out.append(f"{ind}memref.store {e.v(ins[0])}, {e.v(ins[1])}[{e.v(ins[2])}] : {mt(env[ins[1]])}{loc}")
            elif op in OPS: out.append(f"{ind}{L}{op} {', '.join(map(e.v, ins))} : {ts[0]}{loc}")
            else:                                        # 复合 op / 函数调用：不透明模式生成 func.call
                callee = e.func(op, [env[x] for x in ins]) if op in e.p.defs else op
                out.append(f"{ind}{L}func.call @{callee}({', '.join(map(e.v, ins))}) : ({', '.join(mt(env[x]) for x in ins)}) -> ({', '.join(ts)}){loc}")
    def module(e): return 'module {\n' + '\n'.join(e.funcs) + '\n}'

PIPE = ('builtin.module(convert-scf-to-cf,finalize-memref-to-llvm,convert-arith-to-llvm,'
        'convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)')
def compile_and_load(src):
    from mlir.ir import Context, Module
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    diags = []
    ctx = Context(); ctx.__enter__()
    ctx.attach_diagnostic_handler(lambda d: (diags.append((str(d.location), str(d))), True)[1])
    try:
        m = Module.parse(src); PassManager.parse(PIPE).run(m.operation)
        return ExecutionEngine(m, opt_level=2), []
    except Exception as ex:
        for d in getattr(ex, 'error_diagnostics', []): diags.append((str(d.location), d.message))
        return None, [(re.sub(r'loc\("([^"]+)"\)', r'\1', l), msg) for l, msg in diags] or [('?', str(ex)[:200])]

def call(ee, fn, args, rtype):
    C = {'i64': ctypes.c_int64, 'f64': ctypes.c_double, 'index': ctypes.c_int64}
    a = [(C[t] * 1)(v) for v, t in args]; r = (C[rtype] * 1)(0)
    ee.invoke(fn, *a, r); return r[0]
