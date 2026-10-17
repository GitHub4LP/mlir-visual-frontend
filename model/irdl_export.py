"""第 22 轮：复合 op 的不透明身份——导出为 IRDL 动态方言里的 op，交 MLIR 的 IRDL verifier（同一提交）。

IRDL 对 builtin 类型能表达的（本提交实测，spec/closure.md §3）：
  irdl.any；irdl.is <具体类型>；同一约束变量出现在多个端口 ⇒ 类型相等；irdl.base "!builtin.integer"/"!builtin.tensor" 单独使用。
不能表达的：builtin 容器的参数（元素类型、形状、秩）——irdl.parametric 只对 IRDL 定义的类型可用；
  端口间除“相等”外的任何关系（getElementType、getI1SameShape、with(x,t)、形状相容）；C++ verify。
  另：any_of(base, base) 在加载时触发 MLIR 的 UNREACHABLE（IRDLLoading.cpp:541），base "!builtin.ranked_tensor" 加载失败。
所以导出是**上近似**：只放 any / is / 相等；被丢掉的约束逐条记账（导出缺口），它们留在前端（透明判定或前端自己的 verifier）。
"""
import json, subprocess, sys

def expr_of(sl, ports):
    """各值的符号类型表达式（ODS 推断规则的符号求值）：('p', j) | ('k', 类型文本) | ('xf', 名)。"""
    import odssig as G, typesig as TS
    E = {v: ('p', j) for j, v in enumerate(ports)}; eqs, gaps = [], []
    for n in sl:
        lk = G.lookup(n['op']); S = lk[1] if lk else None
        env = {}
        if S is not None:
            gi = G.bind(S, n['ins'], S['ins'], _seg(n))
            if gi is not None: env = {nm: vs for (nm, *_), vs in zip(S['ins'], gi)}
        rule = (S or {}).get('infer') or {}
        outs_names = [nm for nm, *_ in S['outs']] if S else []
        for i, (o, t) in enumerate(n['outs']):
            nm = outs_names[i] if i < len(outs_names) else None
            rl = rule.get(nm)
            if rl is None: E[o] = ('k', t)
            elif rl[0] == 'copy': E[o] = E[(env.get(rl[1]) or [None])[0]] if env.get(rl[1]) else ('xf', '?')
            elif rl[0] == 'build':
                b = G.BUILD.get(rl[1].strip()); E[o] = ('k', TS.show(b)) if b else ('xf', 'build')
            else: E[o] = ('xf', 'TypesMatchWith')
            if rl is not None and rl[0] == 'xf': gaps.append(f'{n["op"]}: 结果 = 变换(操作数)')
        if S is None: gaps.append(f'{n["op"]}: 无 ODS'); continue
        for x in S['rel']:
            if x[0] in ('same', 'same_ops'):
                vs = [v for a in x[1] if a[0] == 'name' for v in env.get(a[1], [])] if x[0] == 'same' else list(n['ins'])
                es = [E[v] for v in vs if v in E]
                if len(es) > 1: eqs.append(es)
            else: gaps.append(f'{n["op"]}: 关系 {x[0]}')
        for nm, p, m, b in S['ins']:
            if b not in ('AnyType',): gaps.append(f'{n["op"]}: 端口约束 {b}')
        if S['declared']: gaps.append(f'{n["op"]}: C++ ' + '/'.join(S['declared']))
    return E, eqs, gaps

def _seg(n):
    a = (n['props'].get('_attrs') or {}).get('operandSegmentSizes')
    if not a: return None
    import irimport as I; return I._segs(a) or None

def export(name, sl, ports):
    """返回 (IRDL 方言文本, 结果表达式, 缺口表)。"""
    E, eqs, gaps = expr_of(sl, ports)
    par = list(range(len(ports))); pin = {}
    def f(j):
        while par[j] != j: j = par[j]
        return j
    for es in eqs:
        ps = [e[1] for e in es if e[0] == 'p']; ks = [e[1] for e in es if e[0] == 'k']
        for a in ps[1:]: par[f(a)] = f(ps[0])
        if ps and ks: pin.setdefault(f(ps[0]), ks[0])
    lines, var = [], {}
    for j in range(len(ports)):
        r = f(j)
        if r not in var:
            var[r] = f'%v{r}'
            lines.append(f'    %v{r} = irdl.is {pin[r]}' if r in pin else f'    %v{r} = irdl.any')
    outs = sl[-1]['outs']; rv = []
    for i, (o, _) in enumerate(outs):
        e = E[o]
        if e[0] == 'p': rv.append(var[f(e[1])])
        else:
            lines.append(f'    %r{i} = ' + (f'irdl.is {e[1]}' if e[0] == 'k' else 'irdl.any')); rv.append(f'%r{i}')
            if e[0] == 'xf': gaps.append('结果类型是端口类型的变换（IRDL 只能表达相等）')
    ops = ', '.join(f'p{j}: {var[f(j)]}' for j in range(len(ports)))
    res = ', '.join(f'r{i}: {v}' for i, v in enumerate(rv))
    txt = (f'irdl.dialect @vf {{\n  irdl.operation @{name} {{\n' + '\n'.join(lines) +
           f'\n    irdl.operands({ops})\n    irdl.results({res})\n  }}\n}}')
    return txt, gaps

_W = r'''
import sys, json
import mlir.ir as ir, mlir.dialects.irdl as I
from mlir.passmanager import PassManager
for line in sys.stdin:
    d, f = json.loads(line)
    with ir.Context() as c:
        try: I.load_dialects(ir.Module.parse(d))
        except Exception as e: print(json.dumps(['load', str(e)[:200]]), flush=True); continue
        try:
            m = ir.Module.parse(f)
            PassManager.parse('builtin.module(canonicalize,cse)').run(m.operation)      # 身份：普通 pass 能把它当作 op 处理
            print(json.dumps(['ok', '']), flush=True)
        except Exception as e: print(json.dumps(['reject', str(e)[:300]]), flush=True)
'''
class Verifier:
    def __init__(s): s.p = None; s.stats = {'调用': 0, '崩溃': 0}
    def verify(s, dtext, ftext):
        if s.p is None or s.p.poll() is not None:
            s.p = subprocess.Popen([sys.executable, '-c', _W], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        s.stats['调用'] += 1
        try:
            s.p.stdin.write(json.dumps([dtext, ftext]) + '\n'); s.p.stdin.flush(); line = s.p.stdout.readline()
        except BrokenPipeError: line = ''
        if not line: s.p.wait(); s.stats['崩溃'] += 1; return 'crash', ''
        return tuple(json.loads(line))
_V = []
def get():
    if not _V: _V.append(Verifier())
    return _V[0]

def instance(name, T, R):
    args = ', '.join(f'%a{j}: {t}' for j, t in enumerate(T))
    res = ', '.join(f'%r{i}' for i in range(len(R)))
    call = f'"vf.{name}"(' + ', '.join(f'%a{j}' for j in range(len(T))) + ') : (' + ', '.join(T) + ') -> (' + ', '.join(R) + ')'
    return f'func.func @f({args}) {{\n  ' + (f'{res} = ' if R else '') + call + '\n  return\n}'
