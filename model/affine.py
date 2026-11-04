"""第 70 轮：仿射映射（affine_map）的解析、构造期化简与求值，供 LinalgOp / IndexingMapOpInterface 规则使用（closure §9.10）。

表示：表达式为
  ('lin', {('d', i) 或 ('s', j): 系数}, 常数)       线性部分（MLIR 构造 AffineExpr 时对线性式做常数折叠与合并）
  ('nl', 种类, a, b)                                 其余（floordiv / ceildiv / mod / 两个非常数相乘），按原样求值
化简只模仿 MLIR 构造期的、能把结果变成单个维的那部分（x*1、x floordiv 1、系数整除）——判定只依赖“结果是否恰为一个维”与求值。
"""
import re

class AffineError(Exception):
    pass

def _lin(d=None, c=0):
    return ('lin', {k: v for k, v in (d or {}).items() if v != 0}, c)

def _const(e):
    return e[0] == 'lin' and not e[1]

def _add(a, b):
    if a[0] == 'lin' and b[0] == 'lin':
        d = dict(a[1])
        for k, v in b[1].items(): d[k] = d.get(k, 0) + v
        return _lin(d, a[2] + b[2])
    if _const(b) and b[2] == 0: return a
    if _const(a) and a[2] == 0: return b
    return ('nl', '+', a, b)

def _mul(a, b):
    if _const(b): a, b = b, a
    if _const(a):
        c = a[2]
        if b[0] == 'lin': return _lin({k: v * c for k, v in b[1].items()}, b[2] * c)
        if c == 1: return b
        if c == 0: return _lin()
        return ('nl', '*', a, b)
    return ('nl', '*', a, b)

def _div(kind, a, b):
    if not _const(b): return ('nl', kind, a, b)
    c = b[2]
    if c == 0: raise AffineError('除以 0')
    if a[0] == 'lin':
        if _const(a):
            x = a[2]
            if kind == 'floordiv': return _lin(c=x // c)
            if kind == 'ceildiv': return _lin(c=-(-x // c))
            return _lin(c=x % c)
        if c == 1: return a if kind != 'mod' else _lin()
        if c > 0 and all(v % c == 0 for v in a[1].values()) and a[2] % c == 0:
            if kind == 'mod': return _lin()
            return _lin({k: v // c for k, v in a[1].items()}, a[2] // c)
    return ('nl', kind, a, b)

_TOK = re.compile(r'\s*(floordiv|ceildiv|mod|[A-Za-z_][\w$.]*|\d+|[-+*()])')

def parse_expr(txt, dims, syms):
    toks = []; pos = 0; txt = txt.strip()
    while pos < len(txt):
        m = _TOK.match(txt, pos)
        if not m: raise AffineError(f'无法解析：{txt[pos:]}')
        toks.append(m.group(1)); pos = m.end()
        while pos < len(txt) and txt[pos].isspace(): pos += 1
    i = [0]
    def peek(): return toks[i[0]] if i[0] < len(toks) else None
    def take(): t = toks[i[0]]; i[0] += 1; return t
    def factor():
        t = take()
        if t == '-': return _mul(_lin(c=-1), factor())
        if t == '(':
            e = expr()
            if take() != ')': raise AffineError('缺少 )')
            return e
        if t.isdigit(): return _lin(c=int(t))
        if t in dims: return _lin({('d', dims.index(t)): 1})
        if t in syms: return _lin({('s', syms.index(t)): 1})
        raise AffineError(f'未知标识符 {t}')
    def term():
        e = factor()
        while peek() in ('*', 'floordiv', 'ceildiv', 'mod'):
            op = take(); f = factor()
            e = _mul(e, f) if op == '*' else _div(op, e, f)
        return e
    def expr():
        e = term()
        while peek() in ('+', '-'):
            op = take(); t = term()
            e = _add(e, t if op == '+' else _mul(_lin(c=-1), t))
        return e
    e = expr()
    if peek() is not None: raise AffineError(f'多余记号 {peek()}')
    return e

def _split_top(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == ',' and depth == 0: out.append(cur); cur = ''
        else: cur += ch
    if cur.strip(): out.append(cur)
    return out

def parse_map(txt):
    """'affine_map<(d0, d1)[s0] -> (d1, d0 * s0)>' 或去掉外壳的形式 → (维数, 符号数, [表达式])。"""
    t = ' '.join(txt.split())
    m = re.fullmatch(r'(?:affine_map<)?\s*\(([^()]*)\)\s*(?:\[([^\]]*)\])?\s*->\s*\((.*)\)\s*>?', t)
    if not m: raise AffineError(f'不是 affine_map：{txt[:80]}')
    dims = [x.strip() for x in m.group(1).split(',') if x.strip()]
    syms = [x.strip() for x in (m.group(2) or '').split(',') if x.strip()]
    body = m.group(3).strip()
    res = [parse_expr(e, dims, syms) for e in _split_top(body)] if body else []
    return len(dims), len(syms), res

def subst_syms(e, vals):
    """把符号 s_j 换成常数 vals[j]（vals[j] 为 None 时保留符号），并重新化简。"""
    if e[0] == 'lin':
        r = _lin(c=e[2])
        for k, v in e[1].items():
            if k[0] == 's' and vals.get(k[1]) is not None: r = _add(r, _lin(c=v * vals[k[1]]))
            else: r = _add(r, _lin({k: v}))
        return r
    _, kind, a, b = e
    a, b = subst_syms(a, vals), subst_syms(b, vals)
    if kind == '+': return _add(a, b)
    if kind == '*': return _mul(a, b)
    return _div(kind, a, b)

def uses_syms(e):
    if e[0] == 'lin': return {k[1] for k in e[1] if k[0] == 's'}
    return uses_syms(e[2]) | uses_syms(e[3])

def is_dim(e):
    """结果恰为一个维（AffineDimExpr）时返回其下标。"""
    if e[0] == 'lin' and e[2] == 0 and len(e[1]) == 1:
        (k, v), = e[1].items()
        if k[0] == 'd' and v == 1: return k[1]
    return None

def evaluate(e, dv):
    if e[0] == 'lin':
        s = e[2]
        for k, v in e[1].items():
            if k[0] != 'd': raise AffineError('求值时仍有符号')
            s += v * dv[k[1]]
        return s
    _, kind, a, b = e
    x, y = evaluate(a, dv), evaluate(b, dv)
    if kind == '+': return x + y
    if kind == '*': return x * y
    if y == 0: raise AffineError('除以 0')
    if kind == 'floordiv': return x // y
    if kind == 'ceildiv': return -(-x // y)
    return x % y          # MLIR mod：结果非负（除数为正时）；Python % 在除数为正时同为非负
