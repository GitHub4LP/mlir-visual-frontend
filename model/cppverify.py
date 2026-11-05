"""第 78 轮：C++ verify 的批量规范化——在模型的类型项上直接解释 9bae84b0 的 verify 源码（closure §9.18）。

为什么这样做：剩余残差的主体是各 op 的 C++ verify()（153 个 op）。逐个手写规则既慢又是“逐 op”的；而这些函数体用的是一套很小的词汇：
取操作数/结果的类型、getRank/getShape/getElementType/getDimSize、个数比较、isa/cast、早返回 emitOpError。
所以把它们当作一门小语言：解析 C++ 子集，在 typesig 的类型项与属性值上求值。一次实现覆盖全部 op，覆盖面随词汇增长而不是随 op 增长。

语义（三值）：
  执行到 `return emitOpError(...)` / `return failure()` ⇒ 拒绝（False），并带出消息字面量；
  执行到 `return success()` ⇒ 接受（True）；
  执行路径上遇到不支持的构造 ⇒ CppUnknown（交上层作 Unknown，不猜）。
  cast<T> 失败在 MLIR 是断言崩溃，按拒绝计（与检验口径一致）。
静态完备（complete）：函数体及其调用到的方法/辅助函数全部可解析、全部名字都在词汇表里 ⇒ 该 op 的 'C++ verify()' 可移出残差（须经 cppverify_test 的 V 判据）。
源：inventory/cpp/<方言>_IR_*.cpp（scripts/fetch_cpp.sh）与 ODS 的 extraClassDeclaration/extraClassDefinition。
"""
import re, glob, os, functools
import typesig as TS

class CppUnknown(Exception): pass
class _Ret(Exception):
    def __init__(self, v): self.v = v
class _Break(Exception): pass
class _Continue(Exception): pass

DYN = -(2 ** 63)                    # ShapedType::kDynamic
_HERE = os.path.dirname(os.path.abspath(__file__))
CPP_DIR = os.path.join(_HERE, '..', 'inventory', 'cpp')

# ───────────────────────── 词法
_TOK = re.compile(r'''
   (?P<ws>\s+|//[^\n]*|/\*.*?\*/)
 | (?P<str>"(?:\\.|[^"\\])*")
 | (?P<chr>'(?:\\.|[^'\\])')
 | (?P<num>0[xX][0-9a-fA-F]+[uUlL]*|\d+(?:\.\d+)?[uUlLfF]*)
 | (?P<id>(?:::)?[A-Za-z_]\w*(?:::~?[A-Za-z_]\w*)*)
 | (?P<op>->|\+\+|--|<<=|>>=|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|::|<<|[-+*/%<>=!&|^~?:;,.(){}\[\]])
''', re.S | re.X)

def lex(s):
    out = []; i = 0
    while i < len(s):
        m = _TOK.match(s, i)
        if not m: raise CppUnknown(f'lex@{s[i:i+20]!r}')
        i = m.end(); k = m.lastgroup
        if k == 'ws': continue
        out.append((k, m.group(k)))
    out.append(('eof', ''))
    return out

# ───────────────────────── 语法（C++ 子集）
TEMPLATE_FNS = {'cast', 'dyn_cast', 'isa', 'dyn_cast_or_null', 'cast_or_null', 'dyn_cast_if_present', 'isa_and_nonnull', 'isa_and_present',
                'static_cast', 'reinterpret_cast', 'const_cast', 'getParentOfType', 'getParentWithTrait', 'SmallVector', 'ArrayRef',
                'std::optional', 'llvm::SmallVector', 'llvm::ArrayRef', 'getDefiningOp', 'hasTrait', 'get', 'getAs', 'getDialect'}
def _base(n): return n.split('::')[-1]

class P:
    def __init__(self, toks): self.t = toks; self.i = 0
    def pk(self, k=0): return self.t[self.i + k]
    def nx(self): x = self.t[self.i]; self.i += 1; return x
    def isop(self, v, k=0): t = self.pk(k); return t[0] == 'op' and t[1] == v
    def isid(self, v, k=0): t = self.pk(k); return t[0] == 'id' and t[1] == v
    def ex(self, v):
        t = self.nx()
        if t[1] != v: raise CppUnknown(f'parse: 期望 {v!r} 得 {t[1]!r}')
        return t
    # 类型（宽松）：限定名 + 模板实参 + const/&/*；返回字符串
    def skip_targs(self):
        d = 0; out = []
        while True:
            t = self.nx(); out.append(t[1])
            if t[1] == '<': d += 1
            elif t[1] == '>': d -= 1
            elif t[1] == '>>': d -= 2
            if d <= 0: return ''.join(out)
            if t[0] == 'eof': raise CppUnknown('parse: 模板实参未闭合')
    def typ(self):
        parts = []
        while self.isid('const') or self.isid('constexpr') or self.isid('static') or self.isid('unsigned') or self.isid('signed') or self.isid('struct'):
            parts.append(self.nx()[1])
        if parts and parts[-1] in ('unsigned', 'signed') and not (self.pk()[0] == 'id' and self.pk()[1] in ('int', 'long', 'short', 'char')):
            while self.isid('const') or self.isop('&') or self.isop('*'): self.nx()
            return parts[-1]
        t = self.nx()
        if t[0] != 'id': raise CppUnknown(f'parse: 类型 {t[1]!r}')
        name = t[1]
        if t[1] in ('long', 'unsigned', 'int', 'short', 'char'):
            while self.pk()[0] == 'id' and self.pk()[1] in ('long', 'int', 'short', 'char'): name += ' ' + self.nx()[1]
        if self.isop('<'): name += self.skip_targs()
        while self.isop('::') and self.pk(1)[0] == 'id': self.nx(); name += '::' + self.nx()[1]
        while self.isid('const') or self.isop('&') or self.isop('*') or self.isop('&&'): self.nx()
        return name
    def looks_decl(self):
        """语句开头是否为声明：类型 名字 (= | ; | { | ( | :) 或 auto [ ... ]"""
        j = self.i
        try:
            if self.isid('auto') or self.isid('const'):
                if self.isid('auto') and (self.isop('[', 1) or (self.isop('&', 1) and self.isop('[', 2))): return True
            self.typ()
            ok = self.pk()[0] == 'id' and (self.pk(1)[1] in ('=', ';', '{', '(', ':', ',', '['))
            return ok
        except CppUnknown: return False
        finally: self.i = j
    # 语句
    def block(self):
        self.ex('{'); out = []
        while not self.isop('}'): out.append(self.stmt())
        self.ex('}'); return ('block', out)
    def stmt(self):
        t = self.pk()
        if t == ('op', '{'): return self.block()
        if t == ('op', ';'): self.nx(); return ('block', [])
        if t[0] == 'id':
            w = t[1]
            if w == 'if':
                self.nx(); self.ex('(')
                init = None
                if self.looks_decl():
                    init = self.decl()
                    if self.isop(';'): self.nx(); c = self.expr()
                    else: c = ('declcond', init); init = None
                else: c = self.expr()
                self.ex(')'); th = self.stmt(); el = None
                if self.isid('else'): self.nx(); el = self.stmt()
                return ('if', init, c, th, el)
            if w == 'return':
                self.nx(); e = None if self.isop(';') else self.expr(); self.ex(';'); return ('return', e)
            if w == 'for':
                self.nx(); self.ex('(')
                if self.looks_decl():
                    j = self.i
                    d = self.decl_head()
                    if self.isop(':'):                          # range-for
                        self.nx(); rng = self.expr(); self.ex(')'); body = self.stmt(); return ('rfor', d, rng, body)
                    self.i = j; init = self.decl()
                elif self.isop(';'): init = None
                else: init = ('expr', self.expr())
                self.ex(';'); c = None if self.isop(';') else self.expr(); self.ex(';')
                inc = [] 
                while not self.isop(')'):
                    inc.append(self.expr())
                    if self.isop(','): self.nx()
                self.ex(')'); body = self.stmt(); return ('for', init, c, inc, body)
            if w == 'while':
                self.nx(); self.ex('('); c = self.expr(); self.ex(')'); return ('while', c, self.stmt())
            if w in ('break', 'continue'): self.nx(); self.ex(';'); return (w,)
            if w in ('switch', 'do', 'goto', 'try', 'case', 'default', 'using', 'typedef'):
                raise CppUnknown(f'parse: 不支持 {w}')
            if w in ('assert', 'LLVM_DEBUG', 'llvm_unreachable'):
                self.nx(); self.ex('('); d = 1
                while d:
                    x = self.nx()[1]; d += (x == '(') - (x == ')')
                self.ex(';'); return ('unreachable',) if w == 'llvm_unreachable' else ('block', [])
            if self.looks_decl():
                d = self.decl(); self.ex(';'); return d
        e = self.expr(); self.ex(';'); return ('expr', e)
    def decl_head(self):
        if self.isid('auto') and (self.isop('[', 1) or (self.isop('&', 1) and self.isop('[', 2))):
            self.nx()
            if self.isop('&'): self.nx()
            self.ex('['); names = []
            while not self.isop(']'):
                names.append(self.nx()[1])
                if self.isop(','): self.nx()
            self.ex(']'); return ('bind', names)
        ty = self.typ(); nm = self.nx()[1]; return ('var', nm, ty)
    def decl(self):
        out = []
        h = self.decl_head()
        while True:
            init = None
            if self.isop('=') and self.isop('{', 1): self.nx(); self.nx(); init = ('ctor', 'SmallVector', self.args('}'))
            elif self.isop('='): self.nx(); init = self.expr(no_comma=True)
            elif self.isop('('): self.nx(); args = self.args(')'); init = ('ctor', h[2] if h[0] == 'var' else None, args)
            elif self.isop('{'): self.nx(); args = self.args('}'); init = ('ctor', h[2] if h[0] == 'var' else None, args)
            elif self.isop('['): raise CppUnknown('parse: C 数组')
            out.append(('decl', h, init))
            if self.isop(',') and h[0] == 'var':
                self.nx(); nm = self.nx()[1]; h = ('var', nm, h[2]); continue
            break
        return out[0] if len(out) == 1 else ('block', out)
    def args(self, close):
        out = []
        while not self.isop(close):
            out.append(self.expr(no_comma=True))
            if self.isop(','): self.nx()
        self.ex(close); return out
    # 表达式（Pratt）
    BIN = [('||',), ('&&',), ('|',), ('^',), ('&',), ('==', '!='), ('<', '>', '<=', '>='), ('<<', '>>'), ('+', '-'), ('*', '/', '%')]
    def expr(self, no_comma=False):
        e = self.assign()
        return e
    def assign(self):
        l = self.ternary()
        if self.pk()[0] == 'op' and self.pk()[1] in ('=', '+=', '-=', '*=', '/='):
            op = self.nx()[1]; r = self.assign(); return ('assign', op, l, r)
        return l
    def ternary(self):
        c = self.bin(0)
        if self.isop('?'):
            self.nx(); a = self.assign(); self.ex(':'); b = self.assign(); return ('?', c, a, b)
        return c
    def bin(self, lv):
        if lv == len(self.BIN): return self.unary()
        l = self.bin(lv + 1)
        while self.pk()[0] == 'op' and self.pk()[1] in self.BIN[lv]:
            op = self.nx()[1]; r = self.bin(lv + 1); l = ('bin', op, l, r)
        return l
    def unary(self):
        t = self.pk()
        if t[0] == 'op' and t[1] in ('!', '-', '+', '~', '*', '&', '++', '--'):
            self.nx(); return ('un', t[1], self.unary())
        if t == ('op', '(') and self.pk(1)[0] == 'id' and self._is_cstyle_cast(): 
            self.nx(); ty = self.typ(); self.ex(')'); return ('cast_c', ty, self.unary())
        return self.postfix(self.primary())
    def _is_cstyle_cast(self):
        j = self.i
        try:
            self.nx(); self.typ(); r = self.isop(')') and self.pk(1)[0] in ('id', 'num') or (self.isop(')') and self.isop('(', 1))
            nm = self.t[j + 1][1]
            return r and _base(nm) in ('int64_t', 'int', 'unsigned', 'size_t', 'int32_t', 'uint64_t', 'bool')
        except CppUnknown: return False
        finally: self.i = j
    def primary(self):
        t = self.nx()
        if t[0] == 'num':
            s = t[1].rstrip('uUlLfF')
            return ('lit', int(s, 16) if s.lower().startswith('0x') else (float(s) if '.' in s else int(s)))
        if t[0] == 'str':
            s = t[1][1:-1]
            while self.pk()[0] == 'str': s += self.nx()[1][1:-1]
            return ('lit', s)
        if t[0] == 'chr': return ('lit', t[1][1:-1])
        if t == ('op', '('):
            e = self.expr(); self.ex(')'); return e
        if t == ('op', '{'): return ('ctor', 'SmallVector', self.args('}'))     # 花括号初始化列表
        if t == ('op', '['):                            # lambda
            cap = []
            while not self.isop(']'): cap.append(self.nx()[1])
            self.ex(']'); params = []
            if self.isop('('):
                self.nx()
                while not self.isop(')'):
                    self.typ(); params.append(self.nx()[1])
                    if self.isop(','): self.nx()
                self.ex(')')
            while self.isid('mutable') or self.isop('->'):
                if self.isop('->'): self.nx(); self.typ()
                else: self.nx()
            body = self.block(); return ('lambda', params, body)
        if t[0] == 'id':
            n = t[1]
            if n in ('true', 'false'): return ('lit', n == 'true')
            if n in ('nullptr', 'std::nullopt', 'llvm::None'): return ('lit', None)
            if n == 'this': return ('this',)
            if self.isop('<') and (_base(n) in TEMPLATE_FNS or n.startswith('std::') or n.endswith('Type') or n.endswith('Attr') or n.endswith('Op')):
                j = self.i
                try:
                    targs = self._targs()
                except CppUnknown:
                    self.i = j; return ('id', n)
                return ('tid', n, targs)
            return ('id', n)
        raise CppUnknown(f'parse: 表达式 {t[1]!r}')
    def _targs(self):
        self.ex('<'); out = []
        while True:
            out.append(self.typ())
            if self.isop(','): self.nx(); continue
            if self.isop('>'): self.nx(); return out
            if self.isop('>>'): raise CppUnknown('parse: >>')
            raise CppUnknown('parse: 模板实参')
    def postfix(self, e):
        while True:
            if self.isop('('): self.nx(); e = ('call', e, self.args(')'))
            elif self.isop('.') or self.isop('->'):
                self.nx(); nm = self.nx()[1]
                if self.isop('<') and nm in TEMPLATE_FNS: targs = self._targs(); e = ('mem', e, nm, targs)
                else: e = ('mem', e, nm, None)
            elif self.isop('['): self.nx(); i = self.expr(); self.ex(']'); e = ('idx', e, i)
            elif self.isop('++') or self.isop('--'): self.nx(); e = ('post', e)
            elif self.isop('{') and e[0] in ('id', 'tid'): self.nx(); e = ('ctor', e[1], self.args('}'))
            else: return e

@functools.lru_cache(4096)
def parse_body(src):
    p = P(lex('{' + src + '}')); b = p.block()
    if p.pk()[0] != 'eof': raise CppUnknown('parse: 尾部')
    return b

# ───────────────────────── 源码定位
_FILE_PREFIX = {}
def _dialect_files(dia):
    import odsjson as O
    ds = O.DIALECTS[dia]; ds = ds if isinstance(ds, list) else [ds]
    out = []
    for d in ds:
        a, b = d.split('/')[:2]
        out += sorted(glob.glob(os.path.join(CPP_DIR, f'{a}_{b}_*.cpp')))
    return sorted(set(out))

@functools.lru_cache(None)
def _file(path): return re.sub(r'^\s*#.*$', '', open(path, errors='ignore').read(), flags=re.M)

def _body_at(s, i):
    d = 1; j = i
    while d and j < len(s):
        c = s[j]
        if c == '"':
            j += 1
            while s[j] != '"': j += 2 if s[j] == '\\' else 1
        elif c == '{': d += 1
        elif c == '}': d -= 1
        j += 1
    return s[i:j - 1]

@functools.lru_cache(None)
def find_method(dia, cls, meth):
    """Class::meth 的定义（out-of-line，在方言的 .cpp 里）→ (参数表文本, 函数体) 或 None。"""
    for f in _dialect_files(dia):
        s = _file(f)
        m = re.search(r'(?:^|\n)[\w:<>,\s\*&]*?\b' + re.escape(cls) + r'::' + re.escape(meth) + r'\s*\(([^)]*)\)\s*(?:const\s*)?\{', s)
        if m: return (m.group(1), _body_at(s, m.end()))
    return None

@functools.lru_cache(None)
def inline_methods(dia, opname):
    """ODS extraClassDeclaration/extraClassDefinition 里的内联方法：{名字: (参数表, 体)}。"""
    import odsjson as O
    r = O.records(dia)[O.ops(dia)[opname]]
    out = {}
    for fld in ('extraClassDeclaration', 'extraClassDefinition'):
        s = r.get(fld) or ''
        if not isinstance(s, str): continue
        s = s.replace('$cppClass::', '')
        for m in re.finditer(r'(?:^|\n|;|\})\s*(?:static\s+|inline\s+|virtual\s+)*[\w:<>,\s\*&]+?\b([A-Za-z_]\w*)\s*\(([^)]*)\)\s*(?:const\s*)?\{', s):
            out.setdefault(m.group(1), (m.group(2), _body_at(s, m.end())))
    return out

@functools.lru_cache(None)
def find_function(dia, name, extra=()):
    """方言 .cpp 里的自由函数（static/匿名命名空间/template）→ (模板形参, 参数表, 体)。"""
    files = list(_dialect_files(dia)) + [os.path.join(CPP_DIR, x) for x in extra]
    for f in files:
        if not os.path.exists(f): continue
        s = _file(f)
        for m in re.finditer(r'(template\s*<([^>]*)>\s*)?(?:static\s+|inline\s+)*(?:[\w:]+(?:<[^;{}()]*>)?[\s\*&]+)\b' + re.escape(name) + r'\s*\(((?:[^()]|\([^()]*\))*)\)\s*\{', s):
            tps = [x.split()[-1] for x in (m.group(2) or '').split(',') if x.strip()]
            return (tps, m.group(3), _body_at(s, m.end()))
    return None

@functools.lru_cache(None)
def td_const(dia, name):
    """方言 .td（extraClassDeclaration）里的 `static constexpr <整型> NAME = <整数>;`（如 Vector 方言的 kPoisonIndex）。"""
    import odsjson as O
    ds = O.DIALECTS[dia]; ds = ds if isinstance(ds, list) else [ds]
    for d in ds:
        for f in glob.glob(os.path.join(O.SITE, 'mlir', 'Dialect', *d.split('/')[:2], '*.td')):
            m = re.search(r'static\s+constexpr\s+(?:u?int\d*_t|int|unsigned)\s+' + re.escape(name) + r'\s*=\s*(-?\d+)\s*;', open(f, errors='ignore').read())
            if m: return int(m.group(1))
    return None

@functools.lru_cache(None)
def verify_source(dia, opname):
    import odsjson as O
    rn = O.ops(dia)[opname]; r = O.records(dia)[rn]
    cls = rn.split('_', 1)[-1] if '_' in rn else rn
    cls = (r.get('cppClassName') or cls)
    got = find_method(dia, cls, 'verify')
    return (cls, got[1]) if got else (cls, None)

# ───────────────────────── 值
class Op:
    """被校验的 op：ODS 端口组（已绑定的类型）与属性。"""
    def __init__(self, dia, opname, S, groups, attrs, cls):
        self.dia, self.opname, self.S, self.cls = dia, opname, S, cls
        names = [nm for nm, *_ in S['ins']] + [nm for nm, *_ in S['outs']]
        self.kinds = {nm: m for nm, _, m, _ in S['ins'] + S['outs']}
        self.isres = {nm for nm, *_ in S['outs']}
        self.groups = dict(zip(names, groups))
        self.ins = [t for (nm, *_), g in zip(S['ins'], groups) for t in g]
        self.outs = [t for (nm, *_), g in zip(S['outs'], groups[len(S['ins']):]) for t in g]
        self.attrs = attrs                     # None：查询不带属性信息，读到属性即 CppUnknown
        self.attr_decl = {nm: (c, k, req) for nm, c, k, req in S['attrs']}

def camel(nm):
    if nm is None: return '\x00none'
    parts = nm.split('_'); return parts[0][:1].upper() + parts[0][1:] + ''.join(p[:1].upper() + p[1:] for p in parts[1:])

class V:                 # mlir::Value
    def __init__(self, t): self.t = t
class Rng(list): pass    # ValueRange / OperandRange
class Diag:              # InFlightDiagnostic（隐式转换为 failure）
    def __init__(self, msg): self.msg = msg
class LR:                # LogicalResult
    def __init__(self, ok): self.ok = ok
OK, FAIL = LR(True), LR(False)
class Layout:
    def __init__(self, txt): self.txt = txt
class Space:
    def __init__(self, txt): self.txt = txt
    def __eq__(self, o): return isinstance(o, Space) and _norm_space(self.txt) == _norm_space(o.txt)
    def __hash__(self): return 0
def _norm_space(t):
    if t is None: return None
    m = re.fullmatch(r'\s*(\d+)\s*(?::\s*i64)?\s*', t)
    if m and int(m.group(1)) == 0: return None
    return re.sub(r'\s+', '', t)
class Lambda:
    def __init__(self, params, body, env): self.params, self.body, self.env = params, body, env
class Self: pass
class Attr:
    def __init__(self, a): self.a = a

TYPE_CLASSES = {
    'Type': lambda t: True, 'ShapedType': TS.shaped, 'TensorType': lambda t: t[0] == 'tensor', 'RankedTensorType': lambda t: t[0] == 'tensor' and t[1] is not None,
    'UnrankedTensorType': lambda t: t[0] == 'tensor' and t[1] is None, 'VectorType': lambda t: t[0] == 'vector', 'MemRefType': lambda t: t[0] == 'memref' and t[1] is not None,
    'BaseMemRefType': lambda t: t[0] == 'memref', 'UnrankedMemRefType': lambda t: t[0] == 'memref' and t[1] is None,
    'IntegerType': lambda t: t[0] == 'i', 'FloatType': lambda t: t[0] == 'f', 'IndexType': lambda t: t[0] == 'index', 'ComplexType': lambda t: t[0] == 'complex',
    'NoneType': lambda t: t[0] == 'none', 'TupleType': lambda t: t[0] == 'tuple',
    'ScalableVectorType': lambda t: t[0] == 'vector' and any(_scal(t)), 'FixedVectorType': lambda t: t[0] == 'vector' and not any(_scal(t)),
}
def _mods(t): return dict(t[3][1]) if len(t) > 3 and t[3] and t[3][0] == 'mods' else {}
def _scal(t): return tuple(_mods(t).get('scalable', (False,) * len(t[1] or ())))
def _isa(cls, x):
    b = _base(cls)
    if isinstance(x, V): x = x.t
    if isinstance(x, tuple):
        if b in TYPE_CLASSES: return TYPE_CLASSES[b](x)
        raise CppUnknown(f'isa<{b}> 于类型')
    if isinstance(x, Attr): return _attr_isa(b, x.a, cls)
    if x is None: return False
    raise CppUnknown(f'isa<{b}> 于 {type(x).__name__}')

def _attr_isa(b, a, full=None):
    import mlir.ir as ir
    extra = {'IntegerSetAttr': lambda: str(a).startswith('affine_set<'), 'SymbolRefAttr': lambda: str(a).startswith('@'),
             'FlatSymbolRefAttr': lambda: re.fullmatch(r'@[\w$."-]+', str(a)) is not None and '::' not in str(a),
             'DenseBoolArrayAttr': lambda: str(a).startswith('array<i1'), 'DenseI16ArrayAttr': lambda: str(a).startswith('array<i16'),
             'DenseI32ArrayAttr': lambda: str(a).startswith('array<i32'), 'DenseI8ArrayAttr': lambda: str(a).startswith('array<i8'),
             'DenseF32ArrayAttr': lambda: str(a).startswith('array<f32'), 'DenseF64ArrayAttr': lambda: str(a).startswith('array<f64'),
             'DenseArrayAttr': lambda: str(a).startswith('array<'), 'DictionaryAttr': lambda: ir.DictAttr.isinstance(a)}
    if b in extra: return bool(extra[b]())
    if full and '::' in full and b not in ('IntegerAttr', 'FloatAttr', 'ArrayAttr', 'UnitAttr', 'StringAttr', 'TypeAttr', 'DenseElementsAttr', 'AffineMapAttr',
                                           'DenseI64ArrayAttr', 'BoolAttr', 'DenseIntElementsAttr', 'SplatElementsAttr', 'ElementsAttr', 'DenseFPElementsAttr', 'TypedAttr'):
        import attrsig as AS, odsjson as O
        nm = full if full.startswith('::') else '::' + full
        for k in (nm, nm[2:]):
            try: return bool(AS._dialect_isa(k)(a))
            except O.Unknown: pass
        raise CppUnknown(f'isa<{full}>')
    m = {'IntegerAttr': ir.IntegerAttr, 'FloatAttr': ir.FloatAttr, 'ArrayAttr': ir.ArrayAttr, 'UnitAttr': ir.UnitAttr,
         'StringAttr': ir.StringAttr, 'TypeAttr': ir.TypeAttr, 'DenseElementsAttr': ir.DenseElementsAttr, 'AffineMapAttr': ir.AffineMapAttr,
         'DenseI64ArrayAttr': ir.DenseI64ArrayAttr, 'BoolAttr': ir.BoolAttr, 'DenseIntElementsAttr': ir.DenseIntElementsAttr,
         'SplatElementsAttr': None, 'ElementsAttr': None, 'TypedAttr': None, 'DenseFPElementsAttr': ir.DenseFPElementsAttr}
    if b not in m: raise CppUnknown(f'isa<{b}> 于属性')
    if b == 'TypedAttr': return any(k.isinstance(a) for k in (ir.IntegerAttr, ir.FloatAttr, ir.DenseElementsAttr)) or 'dense_resource' in str(a)[:15] or str(a).startswith('sparse')
    if b == 'SplatElementsAttr': return ir.DenseElementsAttr.isinstance(a) and ir.DenseElementsAttr(a).is_splat
    if m[b] is None:
        if b == 'ElementsAttr': return ir.DenseElementsAttr.isinstance(a) or 'dense_resource' in str(a) or 'sparse' in str(a)[:7]
        raise CppUnknown(f'isa<{b}>')
    return m[b].isinstance(a)

# ───────────────────────── 解释器
TYPE_METHODS = {'getRank', 'getShape', 'getElementType', 'getDimSize', 'isDynamicDim', 'getNumDynamicDims', 'hasStaticShape', 'hasRank',
                'getNumElements', 'getMemorySpace', 'getLayout', 'isIndex', 'isInteger', 'isSignlessInteger', 'isSignless', 'isUnsigned', 'isSigned',
                'isIntOrFloat', 'isIntOrIndex', 'isIntOrIndexOrFloat', 'isF16', 'isF32', 'isF64', 'isBF16', 'getIntOrFloatBitWidth', 'getWidth',
                'getScalableDims', 'isScalable', 'getElementTypeBitWidth', 'isSignedInteger', 'isUnsignedInteger', 'getNumScalableDims', 'isSignlessIntOrIndex',
                'isSignlessIntOrIndexOrFloat', 'isSignlessIntOrFloat', 'getDynamicDimIndex'}
FREE_FNS = {'success', 'failure', 'failed', 'succeeded', 'isa', 'cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'isa_and_nonnull', 'dyn_cast_if_present',
            'static_cast', 'size', 'any_of', 'all_of', 'none_of', 'count_if', 'count', 'enumerate', 'zip', 'zip_equal', 'is_contained', 'isPowerOf2_32',
            'isPowerOf2_64', 'max', 'min', 'abs', 'isDynamic', 'isStatic', 'emitOpError', 'emitError', 'getOperation', 'getNumOperands', 'getNumResults',
            'getOperand', 'getResult', 'getType', 'getOperandTypes', 'getResultTypes', 'getOperands', 'getResults', 'seq', 'product', 
            'verifyCompatibleShape', 'getElementTypeOrSelf', 'drop_begin', 'drop_end', 'reverse', 'to_vector', 'SmallVector', 'ArrayRef', 'getConstantIntValue',
            'isPermutationVector', 'accumulate', 'is_sorted', 'getTensorTypeFromMemRefType'}

class Interp:
    def __init__(self, op, depth=0):
        self.op = op; self.depth = depth
    # 入口
    def run_verify(self, body_src):
        b = parse_body(body_src)
        try: self.exec(b, {})
        except _Ret as r: return self.lr(r.v)
        raise CppUnknown('verify 未 return')
    def lr(self, v):
        if isinstance(v, Diag): return (False, v.msg)
        if isinstance(v, LR): return (v.ok, None)
        if isinstance(v, bool): return (v, None)
        raise CppUnknown(f'返回值 {type(v).__name__}')
    # 语句
    def exec(self, s, env):
        k = s[0]
        if k == 'block':
            env2 = dict(env); env2['\x00parent'] = env
            for x in s[1]: self.exec(x, env2)
            self._writeback(env2, env)
        elif k == 'decl':
            h, init = s[1], s[2]
            v = None if init is None else self.ev(init, env)
            if h[0] == 'var':
                if isinstance(v, Diag) and _base(h[2]) in ('LogicalResult',): v = FAIL
                if init is None: v = _default(h[2])
                if isinstance(v, V) and _base(h[2]).endswith('Type'): v = v.t
                env[h[1]] = _conv(h[2], v)
            else:
                vs = list(v) if not isinstance(v, tuple) or not v or not isinstance(v[0], str) else None
                if vs is None or len(vs) != len(h[1]): raise CppUnknown('结构化绑定')
                for nm, x in zip(h[1], vs): env[nm] = x
        elif k == 'expr': self.ev(s[1], env)
        elif k == 'if':
            env2 = dict(env); env2['\x00parent'] = env
            if s[1] is not None: self.exec(s[1], env2)
            if s[2][0] == 'declcond':
                self.exec(s[2][1], env2); nm = s[2][1][1][1]; c = self.truth(env2[nm])
            else: c = self.truth(self.ev(s[2], env2))
            if c: self.exec(s[3], env2)
            elif s[4] is not None: self.exec(s[4], env2)
            self._writeback(env2, env)
        elif k == 'return':
            raise _Ret(None if s[1] is None else self.ev(s[1], env))
        elif k == 'rfor':
            rng = self.ev(s[2], env); it = self.iter(rng); n = 0
            for x in it:
                n += 1
                if n > 4096: raise CppUnknown('循环过长')
                env2 = dict(env); env2['\x00parent'] = env
                h = s[1]
                if h[0] == 'var': env2[h[1]] = x
                else:
                    xs = list(x)
                    if len(xs) != len(h[1]): raise CppUnknown('结构化绑定')
                    for nm, y in zip(h[1], xs): env2[nm] = y
                try: self.exec(s[3], env2)
                except _Break: self._writeback(env2, env); break
                except _Continue: pass
                self._writeback(env2, env)
        elif k == 'for':
            env2 = dict(env); env2['\x00parent'] = env
            if s[1] is not None: self.exec(s[1], env2)
            n = 0
            while s[2] is None or self.truth(self.ev(s[2], env2)):
                n += 1
                if n > 4096: raise CppUnknown('循环过长')
                try: self.exec(s[4], env2)
                except _Break: break
                except _Continue: pass
                for e in s[3]: self.ev(e, env2)
            self._writeback(env2, env)
        elif k == 'while':
            n = 0
            while self.truth(self.ev(s[1], env)):
                n += 1
                if n > 4096: raise CppUnknown('循环过长')
                try: self.exec(s[2], env)
                except _Break: break
                except _Continue: pass
        elif k == 'break': raise _Break()
        elif k == 'continue': raise _Continue()
        elif k == 'unreachable': raise CppUnknown('llvm_unreachable')
        else: raise CppUnknown(f'语句 {k}')
    def _writeback(self, inner, outer):
        for k_ in list(inner):
            if k_ != '\x00parent' and k_ in outer: outer[k_] = inner[k_]
    def lookup(self, env, n):
        e = env
        while e is not None:
            if n in e: return e, e[n]
            e = e.get('\x00parent')
        raise KeyError(n)
    def truth(self, v):
        if isinstance(v, V): return True
        if isinstance(v, LR): return v.ok
        if isinstance(v, Diag): return True
        if isinstance(v, (list, Rng)): raise CppUnknown('范围的真值')
        if isinstance(v, Attr): return v.a is not None
        if isinstance(v, (Layout, Space, Self, tuple)): return True
        return bool(v)
    def iter(self, r):
        if isinstance(r, (list, tuple, Rng)) and not (isinstance(r, tuple) and r and isinstance(r[0], str)): return list(r)
        if isinstance(r, range): return list(r)
        if isinstance(r, Attr): return [Attr(x) for x in _attr_list(r.a)]
        raise CppUnknown(f'迭代 {type(r).__name__}')
    # 表达式
    def ev(self, e, env):
        k = e[0]
        if k == 'lit' or k == '\x00val': return e[1]
        if k == 'id': return self.name(e[1], env)
        if k == 'this': return Self()
        if k == 'bin': return self.binop(e[1], e[2], e[3], env)
        if k == 'un':
            if e[1] in ('++', '--'):
                v = self.ev(e[2], env); nv = v + (1 if e[1] == '++' else -1); self.store(e[2], nv, env); return nv
            v = self.ev(e[2], env)
            if e[1] == '!': return not self.truth(v)
            if e[1] == '-': return -_num(v)
            if e[1] == '+': return _num(v)
            if e[1] == '~': return ~_num(v)
            if e[1] == '*':
                if isinstance(v, Self) or v is None and False: return v
                if isinstance(v, Self): return v
                if isinstance(v, (int, tuple, V, list, Attr)) and not isinstance(v, bool): return v      # *optional / *this
                raise CppUnknown('解引用')
            if e[1] == '&': return v
        if k == 'post':
            v = self.ev(e[1], env); self.store(e[1], v + 1, env); return v
        if k == '?': return self.ev(e[2], env) if self.truth(self.ev(e[1], env)) else self.ev(e[3], env)
        if k == 'assign':
            if e[1] == '=': v = self.ev(e[3], env)
            else:
                a, b = _num(self.ev(e[2], env)), _num(self.ev(e[3], env))
                v = {'+=': a + b, '-=': a - b, '*=': a * b, '/=': (int(a / b) if b else _crash())}[e[1]]
            self.store(e[2], v, env); return v
        if k == 'cast_c': return _conv(e[1], self.ev(e[2], env))
        if k == 'lambda': return Lambda(e[1], e[2], env)
        if k == 'idx':
            c, i = self.ev(e[1], env), _num(self.ev(e[2], env))
            seq = self.iter(c)
            if not 0 <= i < len(seq): _crash()
            return seq[i]
        if k == 'mem': return ('\x00bound', self.ev(e[1], env), e[2], e[3])
        if k == 'call': return self.call(e[1], e[2], env)
        if k == 'tid': return ('\x00tfn', e[1], e[2])
        if k == 'ctor':
            args = [self.ev(a, env) for a in e[2]]
            b = _base(e[1] or '')
            if b.startswith('SmallVector') or b.startswith('ArrayRef') or b.startswith('std::vector') or b.startswith('SmallVector<'):
                if len(args) == 1 and isinstance(args[0], (list, Rng)): return list(args[0])
                if len(args) == 2 and isinstance(args[0], int): return [args[1]] * args[0]
                return list(args)
            if len(args) == 1: return _conv(e[1], args[0])
            if not args: return _default(e[1])
            raise CppUnknown(f'构造 {e[1]}')
        raise CppUnknown(f'表达式 {k}')
    def store(self, tgt, v, env):
        if tgt[0] == 'id':
            try: e, _ = self.lookup(env, tgt[1])
            except KeyError: raise CppUnknown(f'赋值未声明 {tgt[1]}')
            e[tgt[1]] = v; return
        raise CppUnknown('复杂赋值')
    def binop(self, op, a_, b_, env):
        if op == '&&':
            return self.truth(self.ev(a_, env)) and self.truth(self.ev(b_, env))
        if op == '||':
            return self.truth(self.ev(a_, env)) or self.truth(self.ev(b_, env))
        a, b = self.ev(a_, env), self.ev(b_, env)
        if op == '<<':
            if isinstance(a, Diag): return a                    # 诊断流
            return _num(a) << _num(b)
        if op in ('==', '!='):
            r = _eq(a, b); return r if op == '==' else not r
        if op in ('<', '>', '<=', '>='):
            a, b = _num(a), _num(b)
            return {'<': a < b, '>': a > b, '<=': a <= b, '>=': a >= b}[op]
        a, b = _num(a), _num(b)
        if op == '+': return a + b
        if op == '-': return a - b
        if op == '*': return a * b
        if op == '/': return int(a / b) if b else _crash()
        if op == '%': return (a - int(a / b) * b) if b else _crash()
        if op == '&': return a & b
        if op == '|': return a | b
        if op == '^': return a ^ b
        if op == '>>': return a >> b
        raise CppUnknown(f'运算 {op}')
    def name(self, n, env):
        try: return self.lookup(env, n)[1]
        except KeyError: pass
        b = _base(n)
        if b == 'kDynamic': return DYN
        c = td_const(self.op.dia, b)
        if c is not None: return c
        if n in ('std::numeric_limits<int64_t>::max',): raise CppUnknown(n)
        if b in FREE_FNS or b.startswith('get') or b.startswith('verify') or b.startswith('is') or b.startswith('compute') or b.startswith('infer'):
            return ('\x00fn', n)
        if re.fullmatch(r'(?:\w+::)*\w+Kind::\w+', n) or re.fullmatch(r'\w+::\w+', n): raise CppUnknown(f'枚举/限定名 {n}')
        raise CppUnknown(f'未知名字 {n}')
    # 调用
    def call(self, f, args_e, env):
        if f[0] == 'mem':
            recv = self.ev(f[1], env); return self.method(recv, f[2], f[3], args_e, env)
        if f[0] == 'tid':
            return self.tcall(_base(f[1]), f[2], args_e, env)
        if f[0] == 'id':
            try:
                v = self.lookup(env, f[1])[1]
                if isinstance(v, Lambda): return self.apply(v, [self.ev(a, env) for a in args_e])
            except KeyError: pass
            return self.fcall(f[1], args_e, env)
        fv = self.ev(f, env)
        if isinstance(fv, Lambda): return self.apply(fv, [self.ev(a, env) for a in args_e])
        raise CppUnknown('调用目标')
    def apply(self, lam, args):
        if isinstance(lam, tuple) and lam and lam[0] == '\x00fn':     # 以名字传入的函数（如 ShapedType::isDynamic）
            return self.fcall(lam[1], [('lit', a) if not isinstance(a, (list, tuple, V, Attr)) else ('\x00val', a) for a in args], {})
        if not isinstance(lam, Lambda): raise CppUnknown('调用非函数值')
        env = dict(zip(lam.params, args)); env['\x00parent'] = lam.env
        try: self.exec(lam.body, env)
        except _Ret as r: return r.v
        return None
    def tcall(self, n, targs, args_e, env):
        args = [self.ev(a, env) for a in args_e]
        if n in ('isa', 'isa_and_nonnull', 'isa_and_present'):
            x = args[0]
            if x is None: 
                if n == 'isa': _crash()
                return False
            return any(_isa(_resolve_t(c, env), x) for c in targs)
        if n in ('cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'dyn_cast_if_present'):
            x = args[0]
            if x is None:
                if n in ('cast', 'dyn_cast'): _crash()
                return None
            ok = _isa(_resolve_t(targs[0], env), x)
            if not ok:
                if n in ('cast', 'cast_or_null'): _crash()
                return None
            return x.t if isinstance(x, V) and _base(targs[0]) not in ('Value', 'OpResult', 'BlockArgument') else x
        if n == 'static_cast': return _conv(targs[0], args[0])
        if n in ('SmallVector', 'ArrayRef'):
            if len(args) == 1 and isinstance(args[0], (list, Rng)): return list(args[0])
            return list(args)
        raise CppUnknown(f'模板函数 {n}')
    def fcall(self, n, args_e, env):
        b = _base(n)
        q = n.split('::')
        if len(q) >= 2 and q[-1] == 'get' and q[-2] in CTORS:
            return CTORS[q[-2]]([self.ev(a, env) for a in args_e])
        if b == 'emitOpError' or b == 'emitError':
            args = [self.ev(a, env) for a in args_e]
            return Diag(args[0] if args and isinstance(args[0], str) else '')
        args = [self.ev(a, env) for a in args_e]
        if b == 'success': return OK if not args or self.truth(args[0]) else FAIL
        if b == 'failure': return FAIL if not args or self.truth(args[0]) else OK
        if b == 'failed':
            x = args[0]; return isinstance(x, Diag) or (isinstance(x, LR) and not x.ok) or (x is False)
        if b == 'succeeded':
            x = args[0]; return isinstance(x, LR) and x.ok or x is True
        if b == 'size': return len(self.iter(args[0]))
        if b in ('any_of', 'all_of', 'none_of', 'count_if'):
            xs = self.iter(args[0]); fn = args[1]
            r = [self.truth(self.apply(fn, [x])) for x in xs]
            return {'any_of': any(r), 'all_of': all(r), 'none_of': not any(r), 'count_if': sum(r)}[b]
        if b == 'count': return sum(1 for x in self.iter(args[0]) if _eq(x, args[1]))
        if b == 'is_sorted': xs = [_num(x) for x in self.iter(args[0])]; return xs == sorted(xs)
        if b == 'is_contained': return any(_eq(x, args[1]) for x in self.iter(args[0]))
        if b == 'enumerate': return [(i, x) for i, x in enumerate(self.iter(args[0]))]
        if b in ('zip', 'zip_equal'):
            ls = [self.iter(a) for a in args]
            if b == 'zip_equal' and len({len(l) for l in ls}) > 1: _crash()
            return [tuple(t) for t in zip(*ls)]
        if b in ('isPowerOf2_32', 'isPowerOf2_64'): x = _num(args[0]); return x > 0 and x & (x - 1) == 0
        if b == 'max': return max(_num(x) for x in args)
        if b == 'min': return min(_num(x) for x in args)
        if b == 'abs': return abs(_num(args[0]))
        if b == 'isDynamic': return _num(args[0]) == DYN
        if b == 'isStatic': return _num(args[0]) != DYN
        if b == 'getElementTypeOrSelf':
            t = _ty(args[0]); return TS.elt(t)
        if b == 'verifyCompatibleShape':
            if len(args) == 2:
                x, y = args
                if isinstance(x, (V, tuple)) and isinstance(y, (V, tuple)):
                    x, y = _ty(x), _ty(y)
                    if not (TS.shaped(x) and TS.shaped(y)): return OK if not TS.shaped(x) and not TS.shaped(y) else FAIL
                    if x[1] is None or y[1] is None: return OK
                    x, y = _shape(x), _shape(y)
                if len(x) != len(y): return FAIL
                return OK if all(a == b or a == DYN or b == DYN for a, b in zip(x, y)) else FAIL
            raise CppUnknown('verifyCompatibleShape 参数')
        if b == 'isPermutationVector':
            xs = [_num(x) for x in self.iter(args[0])]; return sorted(xs) == list(range(len(xs)))
        if b == 'getConstantIntValue': raise CppUnknown('getConstantIntValue（值前提）')
        if b in ('drop_begin',): xs = self.iter(args[0]); return xs[(_num(args[1]) if len(args) > 1 else 1):]
        if b in ('to_vector', 'SmallVector', 'ArrayRef'): return list(self.iter(args[0])) if args else []
        if b == 'product':
            p = 1
            for x in self.iter(args[0]): p *= _num(x)
            return p
        if b == 'seq': return list(range(_num(args[0]), _num(args[1])))
        # op 自身的方法（隐式 this）
        return self.method(Self(), n, None, args_e, env, pre=args)
    # 方法
    def method(self, recv, m, targs, args_e, env, pre=None):
        args = pre if pre is not None else [self.ev(a, env) for a in args_e]
        if isinstance(recv, tuple) and recv and recv[0] == '\x00bound': recv = self.method(recv[1], recv[2], recv[3], [], env, pre=[])
        if isinstance(recv, Self): return self.op_method(m, args, env, targs)
        if isinstance(recv, V):
            if m == 'getType': return recv.t
            if m in ('getDefiningOp',): raise CppUnknown('getDefiningOp（值前提）')
            raise CppUnknown(f'Value.{m}')
        if isinstance(recv, tuple) and recv and isinstance(recv[0], str) and recv[0][:1] != '\x00':
            return self.type_method(recv, m, args, targs)
        if isinstance(recv, (list, Rng)):
            if m == 'size': return len(recv)
            if m == 'empty': return len(recv) == 0
            if m == 'front':
                if not recv: _crash()
                return recv[0]
            if m == 'back':
                if not recv: _crash()
                return recv[-1]
            if m == 'getTypes': return [x.t if isinstance(x, V) else x for x in recv]
            if m == 'getType' and isinstance(recv, Rng): raise CppUnknown('Range.getType')
            if m == 'push_back' or m == 'emplace_back': recv.append(args[0]); return None
            if m == 'append': recv.extend(self.iter(args[0])); return None
            if m == 'drop_front': return recv[(_num(args[0]) if args else 1):]
            if m == 'drop_back': return recv[:len(recv) - (_num(args[0]) if args else 1)]
            if m == 'take_front': return recv[:(_num(args[0]) if args else 1)]
            if m == 'slice': return recv[_num(args[0]):_num(args[0]) + _num(args[1])]
            if m == 'contains' or m == 'count': return any(_eq(x, args[0]) for x in recv)
            if m in ('begin', 'end'): raise CppUnknown('迭代器')
            raise CppUnknown(f'range.{m}')
        if isinstance(recv, Layout):
            if m == 'isIdentity': return _layout_identity(recv.txt)
            raise CppUnknown(f'layout.{m}')
        if isinstance(recv, Diag): return recv
        if isinstance(recv, Attr): return self.attr_method(recv, m, args)
        if recv is None and m in ('has_value',): return False
        if m in ('has_value',): return True
        if m == 'value':
            if recv is None: _crash()
            return recv
        if isinstance(recv, int) and not isinstance(recv, bool):
            if m == 'isNegative': return recv < 0
            if m in ('getZExtValue', 'getSExtValue', 'getLimitedValue'): return recv
            if m == 'isZero': return recv == 0
            if m == 'isOne': return recv == 1
            if m == 'isStrictlyPositive': return recv > 0
            if m == 'isNonNegative': return recv >= 0
        if isinstance(recv, LR):
            if m == 'failed': return not recv.ok
            if m == 'succeeded': return recv.ok
        raise CppUnknown(f'方法 {type(recv).__name__}.{m}')
    def attr_method(self, a, m, args):
        x = a.a
        import mlir.ir as ir
        if m in ('getInt', 'getSInt', 'getUInt') and ir.IntegerAttr.isinstance(x): return ir.IntegerAttr(x).value
        if m == 'getValue' and ir.IntegerAttr.isinstance(x): return ir.IntegerAttr(x).value
        if m in ('size', 'getNumElements', 'empty'):
            n_ = len(_attr_list(x)); return n_ == 0 if m == 'empty' else n_
        if m == 'getType':
            for k in (ir.IntegerAttr, ir.FloatAttr, ir.DenseElementsAttr):
                if k.isinstance(x): return TS.parse(str(k(x).type))
            raise CppUnknown('attr.getType')
        if m == 'getElementType': raise CppUnknown('attr.getElementType')
        if m == 'getValue' and ir.ArrayAttr.isinstance(x): return [Attr(y) for y in ir.ArrayAttr(x)]
        if m == 'asArrayRef' and ir.DenseI64ArrayAttr.isinstance(x): return list(ir.DenseI64ArrayAttr(x))
        if m == 'asArrayRef' and re.fullmatch(r'array<i(?:8|16|32)(?::.*)?>', str(x), re.S): return [int(v) for v in _attr_list(x)]
        raise CppUnknown(f'attr.{m}')
    def type_method(self, t, m, args, targs):
        if m in ('getRank',):
            if not TS.ranked(t): _crash()
            return len(t[1])
        if m == 'hasRank': return TS.ranked(t)
        if m == 'getShape':
            if not TS.ranked(t): _crash()
            return _shape(t)
        if m == 'getElementType':
            if t[0] == 'complex': return t[1]
            if not TS.shaped(t): _crash()
            return t[2]
        if m == 'getDimSize':
            if not TS.ranked(t): _crash()
            i = _num(args[0])
            if not 0 <= i < len(t[1]): _crash()
            return _shape(t)[i]
        if m == 'isDynamicDim':
            if not TS.ranked(t): _crash()
            return _shape(t)[_num(args[0])] == DYN
        if m == 'getNumDynamicDims':
            if not TS.ranked(t): _crash()
            return sum(1 for x in t[1] if x is None)
        if m == 'hasStaticShape': return TS.ranked(t) and all(x is not None for x in t[1])
        if m == 'getNumElements':
            if not TS.ranked(t) or any(x is None for x in t[1]): _crash()
            p = 1
            for x in t[1]: p *= x
            return p
        if m == 'getMemorySpace':
            if t[0] != 'memref': _crash()
            return Space(_mods(t).get('space'))
        if m == 'getLayout':
            if t[0] != 'memref' or t[1] is None: _crash()
            return Layout(_mods(t).get('layout'))
        if m == 'getScalableDims':
            if t[0] != 'vector': _crash()
            return list(_scal(t))
        if m == 'isScalable':
            if t[0] != 'vector': _crash()
            return any(_scal(t))
        if m == 'getNumScalableDims': return sum(_scal(t))
        if m == 'isIndex': return t[0] == 'index'
        if m == 'isInteger':
            if not args: return t[0] == 'i'
            return t[0] == 'i' and t[1] == _num(args[0])
        if m == 'isSignlessInteger':
            if not args: return t[0] == 'i' and t[2] == ''
            return t[0] == 'i' and t[2] == '' and t[1] == _num(args[0])
        if m == 'isSignedInteger': return t[0] == 'i' and t[2] == 's' and (not args or t[1] == _num(args[0]))
        if m == 'isUnsignedInteger': return t[0] == 'i' and t[2] == 'u' and (not args or t[1] == _num(args[0]))
        if m in ('isSignless', 'isUnsigned', 'isSigned'):
            if t[0] != 'i': _crash()
            return t[2] == {'isSignless': '', 'isUnsigned': 'u', 'isSigned': 's'}[m]
        if m == 'getWidth':
            if t[0] == 'i': return t[1]
            if t[0] == 'f': return _fwidth(t)
            _crash()
        if m == 'isIntOrFloat': return t[0] in ('i', 'f')
        if m == 'isIntOrIndex': return t[0] in ('i', 'index')
        if m == 'isIntOrIndexOrFloat': return t[0] in ('i', 'index', 'f')
        if m == 'isSignlessIntOrIndex': return (t[0] == 'i' and t[2] == '') or t[0] == 'index'
        if m == 'isSignlessIntOrIndexOrFloat': return (t[0] == 'i' and t[2] == '') or t[0] in ('index', 'f')
        if m == 'isSignlessIntOrFloat': return (t[0] == 'i' and t[2] == '') or t[0] == 'f'
        if m in ('isF16', 'isF32', 'isF64', 'isBF16'): return t == ('f', m[2:].lower())
        if m == 'getIntOrFloatBitWidth':
            if t[0] == 'i': return t[1]
            if t[0] == 'f': return _fwidth(t)
            _crash()
        if m == 'getElementTypeBitWidth':
            if not TS.shaped(t): _crash()
            e_ = t[2]
            if e_[0] == 'i': return e_[1]
            if e_[0] == 'f': return _fwidth(e_)
            raise CppUnknown('getElementTypeBitWidth')
        if m == 'getDynamicDimIndex':
            i = _num(args[0]); return sum(1 for x in t[1][:i] if x is None)
        if m == 'cast' or m == 'dyn_cast' or m == 'isa':
            ok = _isa(targs[0], t)
            if m == 'isa': return ok
            if not ok:
                if m == 'cast': _crash()
                return None
            return t
        if m == 'cloneWith':                       # ShapedType::cloneWith(optional shape, elt)：保留编码/布局/内存空间/可伸缩标记
            if not TS.shaped(t): _crash()
            sh, el = args
            if sh is not None:
                sh = tuple(None if x == DYN else _num(x) for x in self.iter(sh))
                if t[0] == 'vector' and len(sh) != len(t[1] or ()): raise CppUnknown('cloneWith 改秩')
            else: sh = t[1]
            return (t[0], sh, _ty(el)) + tuple(t[3:])
        if m == 'getContext': return None
        if m == 'getTypeID': raise CppUnknown('getTypeID')
        raise CppUnknown(f'type.{m}')
    def op_method(self, m, args, env, targs):
        op = self.op
        if m in ('getOperation',): return Self()
        if m == 'emitOpError' or m == 'emitError': return Diag(args[0] if args and isinstance(args[0], str) else '')
        if m == 'getNumOperands': return len(op.ins)
        if m == 'getNumResults': return len(op.outs)
        if m == 'getOperand':
            i = _num(args[0]) if args else 0
            if not 0 <= i < len(op.ins): _crash()
            return V(op.ins[i])
        if m == 'getOperands': return Rng(V(t) for t in op.ins)
        if m == 'getResults': return Rng(V(t) for t in op.outs)
        if m == 'getOperandTypes': return list(op.ins)
        if m == 'getResultTypes': return list(op.outs)
        if m == 'getType' and not args:
            if len(op.outs) != 1: raise CppUnknown('getType 多结果')
            return op.outs[0]
        if m == 'getResult' and not args:
            if len(op.outs) != 1: raise CppUnknown('getResult 多结果')
            return V(op.outs[0])
        if m == 'getResult' and args:
            i = _num(args[0])
            if not 0 <= i < len(op.outs): _crash()
            return V(op.outs[i])
        if m in ('getParentOp', 'getParentOfType', 'getParentWithTrait', 'getBlock', 'getRegion', 'getBody', 'getRegions', 'getLoc', 'getContext', 'walk'):
            raise CppUnknown(f'op.{m}（父/区域）')
        # ODS 端口访问器 getFoo()
        for nm in op.groups:
            if m == 'get' + camel(nm):
                g = op.groups[nm]; kind = op.kinds[nm]
                if kind == 'one': 
                    if len(g) != 1: _crash()
                    return V(g[0])
                if kind == 'optional': return V(g[0]) if g else None
                return Rng(V(t) for t in g)
            if m == 'getNum' + camel(nm) and op.kinds[nm] != 'one': return len(op.groups[nm])
        # ODS 属性访问器
        for nm, (c, kind, req) in op.attr_decl.items():
            cm = camel(nm)
            if m in ('get' + cm, 'get' + cm + 'Attr'):
                return self.attr_get(nm, c, kind, req, raw=m.endswith('Attr') and not cm.endswith('Attr') or m == 'get' + cm + 'Attr')
            if m == 'get' + cm + 'AttrName' or m == 'get' + cm + 'AttrStrName': return nm
        # 内联/外部方法
        im = inline_methods(op.dia, op.opname).get(m)
        if im is not None: return self.invoke(im[0], im[1], args, ())
        fm = find_method(op.dia, op.cls, m)
        if fm is not None: return self.invoke(fm[0], fm[1], args, ())
        ff = find_function(op.dia, _base(m))
        if ff is not None: return self.invoke(ff[1], ff[2], args, ff[0], targs=targs)
        raise CppUnknown(f'op 方法 {m}')
    def attr_get(self, nm, c, kind, req, raw):
        """ODS 属性访问器：getFooAttr() 返回属性本身；getFoo() 按 ODS 的 returnType 转换（与 tblgen 生成的 convertFromStorage 对应）。"""
        import odsjson as O
        rec = O.records(self.op.dia).get(c, {}) if isinstance(c, str) else {}
        rt = re.sub(r'\s+', '', rec.get('returnType') or '')
        if self.op.attrs is None: raise CppUnknown('查询未提供属性')
        txt = self.op.attrs.get(nm)
        opt = rt.startswith('::std::optional<') or rt.startswith('std::optional<')
        if opt: rt = rt[rt.index('<') + 1:-1]
        if txt is None:
            if raw: return None
            if opt: return None
            if rt == 'bool' and (kind in ('UnitAttr', 'UnitProp') or c in ('UnitAttr', 'UnitProp')): return False
            raise CppUnknown(f'属性 {nm} 缺省值')
        a = _parse_attr(txt)
        if raw: return Attr(a)
        import mlir.ir as ir
        if rt in ('bool',):
            if ir.UnitAttr.isinstance(a): return True
            if ir.BoolAttr.isinstance(a): return ir.BoolAttr(a).value
        if rt.endswith('ArrayRef<int64_t>') or rt.endswith('ArrayRef<int32_t>'):
            if ir.DenseI64ArrayAttr.isinstance(a): return list(ir.DenseI64ArrayAttr(a))
            if hasattr(ir, 'DenseI32ArrayAttr') and ir.DenseI32ArrayAttr.isinstance(a): return list(ir.DenseI32ArrayAttr(a))
        if rt.endswith('ArrayRef<bool>') and hasattr(ir, 'DenseBoolArrayAttr') and ir.DenseBoolArrayAttr.isinstance(a): return list(ir.DenseBoolArrayAttr(a))
        if rt in ('uint64_t', 'int64_t', 'uint32_t', 'int32_t', 'unsigned', 'int', 'uint16_t', 'int16_t', 'uint8_t', 'int8_t', '::llvm::APInt', 'llvm::APInt') and ir.IntegerAttr.isinstance(a):
            v = ir.IntegerAttr(a).value
            if rt.startswith('uint') or rt == 'unsigned':
                w = int(re.sub(r'\D', '', rt) or 32); v &= (1 << w) - 1
            return v
        if rt in ('::mlir::Type', 'mlir::Type', 'Type') and ir.TypeAttr.isinstance(a): return TS.parse(str(ir.TypeAttr(a).value))
        if rt in ('::llvm::StringRef', 'llvm::StringRef', 'StringRef') and ir.StringAttr.isinstance(a): return ir.StringAttr(a).value
        if rt.startswith('::mlir::') and rt.endswith('Attr') or rt in ('::mlir::TypedAttr', '::mlir::Attribute', '::mlir::ElementsAttr', '::mlir::ArrayAttr', '::mlir::DenseElementsAttr'):
            return Attr(a)
        raise CppUnknown(f'属性 {nm} 的返回类型 {rt or kind}')
    def invoke(self, params_txt, body, args, tparams, targs=None):
        if self.depth > 6: raise CppUnknown('调用过深')
        ps = _params(params_txt)
        if len(args) > len(ps): raise CppUnknown('实参过多')
        env = {}
        for (ty, nm, dflt), a in zip(ps, args + [None] * (len(ps) - len(args))):
            if a is None and dflt is not None: a = self.ev(P(lex(dflt)).expr(), {})
            if isinstance(a, Self) and _base(ty).endswith('Op') is False and ty not in ('Operation*', 'Operation *'): pass
            env[nm] = _conv(ty, a) if a is not None else None
        for tp, ta in zip(tparams, targs or []): env['\x00T:' + tp] = ta
        sub = Interp(self.op, self.depth + 1)
        b = parse_body(body)
        try: sub.exec(b, env)
        except _Ret as r: return r.v
        return None

def _resolve_t(c, env):
    e = env
    while e is not None:
        if '\x00T:' + c in e: return e['\x00T:' + c]
        e = e.get('\x00parent')
    return c

def _params(txt):
    txt = txt.strip()
    if not txt or txt == 'void': return []
    out = []; d = 0; cur = ''
    for ch in txt + ',':
        if ch in '<(': d += 1
        if ch in '>)': d -= 1
        if ch == ',' and d == 0:
            p = cur.strip(); cur = ''
            dflt = None
            if '=' in p: p, dflt = p.split('=', 1); p = p.strip(); dflt = dflt.strip()
            m = re.match(r'(.*?)([A-Za-z_]\w*)\s*$', p)
            if not m: raise CppUnknown('形参')
            out.append((m.group(1).strip(), m.group(2), dflt))
        else: cur += ch
    return out

def _crash(): raise _Ret(Diag('\x00crash'))
def _num(v):
    if isinstance(v, bool): return int(v)
    if isinstance(v, int): return v
    if isinstance(v, float): return v
    raise CppUnknown(f'数值 {type(v).__name__}')
def _ty(x):
    if isinstance(x, V): return x.t
    if isinstance(x, tuple) and x and isinstance(x[0], str): return x
    raise CppUnknown('期望类型')
def _shape(t): return [DYN if x is None else x for x in t[1]]
def _fwidth(t):
    n = t[1]
    m = re.search(r'(\d+)', n)
    if n == 'bf16': return 16
    if n == 'tf32': return 19
    if m: return int(m.group(1))
    raise CppUnknown('浮点位宽')
def _eq(a, b):
    if isinstance(a, V): a = a.t
    if isinstance(b, V): b = b.t
    if isinstance(a, Self) or isinstance(b, Self): raise CppUnknown('op 比较')
    if isinstance(a, Attr) and isinstance(b, Attr): return str(a.a) == str(b.a)
    if isinstance(a, (list, Rng)) and isinstance(b, (list, Rng)):
        return len(a) == len(b) and all(_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, LR) or isinstance(b, LR): raise CppUnknown('LogicalResult 比较')
    if isinstance(a, bool) or isinstance(b, bool): return bool(a) == bool(b) if a is not None and b is not None else a is b
    if isinstance(a, tuple) and isinstance(b, tuple): return _tnorm(a) == _tnorm(b)
    return a == b
def _tnorm(t):
    if t[0] == 'memref' and len(t) > 3:
        md = dict(_mods(t)); sp = _norm_space(md.get('space'))
        if sp is None: md.pop('space', None)
        else: md['space'] = sp
        if 'layout' in md and _layout_identity(md['layout']): md.pop('layout')
        return t[:3] + ((('mods', tuple(sorted(md.items()))),) if md else ())
    return t
def _layout_identity(txt):
    if txt is None: return True
    m = re.fullmatch(r'affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>', txt.replace(' ', '').replace('affine_map<(', 'affine_map<(').strip()) if False else None
    s = re.sub(r'\s+', '', txt)
    m = re.fullmatch(r'affine_map<\(([^)]*)\)->\(([^)]*)\)>', s)
    if m: return m.group(1) == m.group(2)
    return False
def _default(ty):
    b = _base(ty or '')
    if b in ('int', 'int64_t', 'unsigned', 'size_t', 'int32_t', 'uint64_t', 'uint32_t', 'long'): return 0
    if b == 'bool': return False
    if b.startswith('SmallVector') or b.startswith('std::vector') or b.startswith('SmallVector<'): return []
    if b in ('Type', 'Value', 'Attribute') or b.endswith('Type'): return None
    raise CppUnknown(f'缺省值 {ty}')
def _conv(ty, v):
    b = _base(ty or 'auto')
    if b in ('int', 'int64_t', 'unsigned', 'size_t', 'int32_t', 'uint64_t', 'uint32_t', 'long', 'unsigned int', 'unsigned long'):
        if isinstance(v, bool): return int(v)
        if isinstance(v, (int, float)): return int(v)
        if isinstance(v, Attr): raise CppUnknown('属性转整数')
        return v
    if b == 'bool': return v if isinstance(v, (bool, Attr, type(None))) else (bool(v) if isinstance(v, int) else v)
    if b in ('LogicalResult',) and isinstance(v, Diag): return FAIL
    if b in ('Value', 'OpResult', 'BlockArgument', 'TypedValue'): return v
    return v

@functools.lru_cache(None)
def _ctx():
    import mlir.ir as ir
    c = ir.Context(); c.allow_unregistered_dialects = True; return c
@functools.lru_cache(4096)
def _parse_attr(txt):
    import mlir.ir as ir
    try: return ir.Attribute.parse(txt, _ctx())
    except Exception: raise CppUnknown(f'属性解析 {txt[:40]}')
def _attr_list(a):
    import mlir.ir as ir
    if ir.ArrayAttr.isinstance(a): return list(ir.ArrayAttr(a))
    if ir.DenseI64ArrayAttr.isinstance(a): return list(ir.DenseI64ArrayAttr(a))
    m = re.fullmatch(r'array<\w+(?::\s*(.*))?>', str(a), re.S)
    if m: return [x.strip() for x in (m.group(1) or '').split(',') if x.strip()]
    if ir.DenseElementsAttr.isinstance(a): return list(range(len(ir.DenseElementsAttr(a))))
    raise CppUnknown('属性列表')

# ───────────────────────── 对外：判定与静态完备
def _rt_get(a):
    sh, el = a[0], a[1]
    if len(a) > 2 and a[2] is not None: raise CppUnknown('RankedTensorType::get 带编码')
    return ('tensor', tuple(None if x == DYN else _num(x) for x in sh), _ty(el))
def _vt_get(a):
    sh, el = a[0], a[1]
    t = ('vector', tuple(_num(x) for x in sh), _ty(el))
    if len(a) > 2 and any(a[2]): t = t + (('mods', (('scalable', tuple(bool(x) for x in a[2])),)),)
    return t
CTORS = {'IntegerType': lambda a: ('i', _num(a[1]), ''), 'IndexType': lambda a: ('index',),
         'UnrankedTensorType': lambda a: ('tensor', None, _ty(a[0])), 'RankedTensorType': _rt_get, 'VectorType': _vt_get}
STATS = {'python 异常': 0}
def decide(dia, opname, S, groups, attrs):
    """→ (True | False, 消息)；不支持 ⇒ CppUnknown。解释器自身的 Python 异常也作 CppUnknown（不猜），并计数。"""
    cls, body = verify_source(dia, opname)
    if body is None: raise CppUnknown('未找到 verify 源码')
    it = Interp(Op(dia, opname, S, groups, None if attrs is None else {k: v for k, v in attrs.items() if not k.startswith('\x00')}, cls))
    try: return it.run_verify(body)
    except CppUnknown: raise
    except RecursionError: raise CppUnknown('递归')
    except Exception as e:
        STATS['python 异常'] += 1; STATS.setdefault('例', []).append(f'{opname}: {type(e).__name__}: {e}'[:160])
        raise CppUnknown(f'解释器异常 {type(e).__name__}: {e}')

def has_source(dia, opname):
    try: return verify_source(dia, opname)[1] is not None
    except Exception: return False

def rule(S):
    """nf 原子：op 的 C++ verify 按解释执行。→ (rel, 完备?) 或 None。"""
    dia = S['op'].split('.')[0]
    if not has_source(dia, S['op']): return None
    import nfrules as NF
    refs = [('name', nm) for nm, *_ in S['ins']] + [('name', nm) if nm is not None else ('res', k) for k, (nm, *_) in enumerate(S['outs'])]
    def fn(*a):
        groups, attrs = list(a[:-1]), a[-1]
        try: ok, msg = decide(dia, S['op'], S, groups, attrs)
        except CppUnknown as e: raise NF.NFUnknown(f'cppverify: {e}')
        return ok
    ok, _ = completeness(dia, S['op'])
    return ('nf', refs, 'nf', 'C++ verify（9bae84b0 源码解释执行）', fn, 'opt'), ok

def _calls(node, out):
    if isinstance(node, tuple):
        if node and node[0] == 'call':
            f = node[1]
            if f[0] == 'id': out.add(('fn', f[1]))
            elif f[0] == 'mem': out.add(('m', f[2]))
            elif f[0] == 'tid': out.add(('t', _base(f[1])))
        if node and node[0] == 'mem': out.add(('m', node[2]))
        for x in node: _calls(x, out)
    elif isinstance(node, list):
        for x in node: _calls(x, out)

KNOWN_METHODS = TYPE_METHODS | {'cloneWith', 'getContext', 'getType', 'size', 'empty', 'front', 'back', 'getTypes', 'push_back', 'emplace_back', 'append', 'drop_front', 'drop_back',
                                'take_front', 'slice', 'contains', 'isIdentity', 'has_value', 'value', 'getInt', 'getValue', 'getSInt', 'getUInt',
                                'asArrayRef', 'getNumElements', 'failed', 'succeeded', 'getOperation', 'getNumOperands', 'getNumResults', 'getOperand',
                                'getOperands', 'getResults', 'getOperandTypes', 'getResultTypes', 'getResult', 'emitOpError', 'emitError', 'isa', 'cast', 'dyn_cast'}
@functools.lru_cache(None)
def completeness(dia, opname):
    """静态完备性：→ (完备?, 不支持的名字集合)。递归展开 op 方法与方言内自由函数。"""
    import odssig as G
    cls, body = verify_source(dia, opname)
    if body is None: return (False, frozenset({'<无源码>'}))
    S = G.sig(dia, opname)
    acc = {'get' + camel(nm) for nm, *_ in S['ins'] + S['outs']} | {'getNum' + camel(nm) for nm, *_ in S['ins'] + S['outs']}
    for nm, *_ in S['attrs']:
        cm = camel(nm); acc |= {'get' + cm, 'get' + cm + 'Attr', 'get' + cm + 'AttrName', 'get' + cm + 'AttrStrName'}
    bad = set(); seen = set()
    def walk(src, tag):
        if tag in seen: return
        seen.add(tag)
        try: ast = parse_body(src)
        except CppUnknown as e: bad.add(f'{tag}: {e}'); return
        _scan_stmts(ast, bad, tag)
        cs = set(); _calls(ast, cs)
        for k, n in cs:
            b = _base(n)
            if k == 't':
                if b not in ('isa', 'cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'isa_and_nonnull', 'dyn_cast_if_present', 'static_cast', 'SmallVector', 'ArrayRef'): bad.add(b)
                continue
            if b in KNOWN_METHODS or b in FREE_FNS - {'getConstantIntValue'} or b in acc: continue
            if td_const(dia, b) is not None: continue
            if k == 'fn' and len(n.split('::')) >= 2 and n.split('::')[-1] == 'get' and n.split('::')[-2] in CTORS: continue
            im = inline_methods(dia, opname).get(b)
            if im is not None: walk(im[1], b); continue
            fm = find_method(dia, cls, b)
            if fm is not None: walk(fm[1], b); continue
            ff = find_function(dia, b)
            if ff is not None: walk(ff[2], b); continue
            bad.add(b)
    walk(body, 'verify')
    return (not bad, frozenset(bad))

def _scan_stmts(node, bad, tag):
    if isinstance(node, tuple):
        if node and node[0] in ('unreachable',): bad.add(f'{tag}: llvm_unreachable')
        if node and node[0] == 'id' and re.fullmatch(r'(?:\w+::)+\w+', node[1]) and _base(node[1]) not in FREE_FNS and _base(node[1]) != 'kDynamic' \
                and not (node[1].split('::')[-1] == 'get' and node[1].split('::')[-2] in CTORS) \
                and not _base(node[1]).startswith(('get', 'verify', 'is', 'compute', 'infer')):
            bad.add(f'限定名 {node[1]}')
        if node and node[0] == 'tid' and _base(node[1]) not in ('isa', 'cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'isa_and_nonnull', 'dyn_cast_if_present', 'static_cast', 'SmallVector', 'ArrayRef'):
            bad.add(f'模板 {node[1]}')
        for x in node: _scan_stmts(x, bad, tag)
    elif isinstance(node, list):
        for x in node: _scan_stmts(x, bad, tag)

@functools.lru_cache(None)
def verify_literals(dia, opname):
    """verify 及其调用到的方法/辅助函数里的字符串字面量（诊断消息片段），供检验判断“MLIR 的诊断来自这段 verify”。"""
    cls, body = verify_source(dia, opname)
    if body is None: return frozenset()
    out, seen = set(), set()
    def walk(src, tag):
        if tag in seen: return
        seen.add(tag)
        for m in re.finditer(r'"((?:\\.|[^"\\])*)"', src):
            s = m.group(1).strip()
            if len(s) >= 8: out.add(s)
        try: ast = parse_body(src)
        except CppUnknown: return
        cs = set(); _calls(ast, cs)
        for k, n in cs:
            b = _base(n)
            im = inline_methods(dia, opname).get(b)
            if im is not None: walk(im[1], b); continue
            fm = find_method(dia, cls, b)
            if fm is not None: walk(fm[1], b); continue
            ff = find_function(dia, b)
            if ff is not None: walk(ff[2], b)
    walk(body, 'verify')
    return frozenset(out)


# ───────────────────────── 第 78 轮：ODS 属性谓词原子（C++ 表达式，$_self 为属性）用同一个解释器求值
ATOM_METHODS = {'getType', 'getElementType', 'isSignlessInteger', 'isInteger', 'size', 'empty', 'asArrayRef', 'getValue', 'isNegative', 'getInt',
                'getZExtValue', 'getSExtValue', 'getNumElements', 'getRank', 'getShape', 'isIndex', 'isZero', 'isOne', 'isStrictlyPositive', 'isNonNegative',
                'getWidth', 'isa', 'cast', 'dyn_cast', 'isSignedInteger', 'isUnsignedInteger', 'isF32', 'isF64', 'isF16', 'isBF16', 'isIntOrFloat', 'hasRank'}
ATOM_FNS = {'ArrayRef', 'SmallVector', 'isa', 'cast', 'dyn_cast', 'all_of', 'any_of', 'none_of', 'is_sorted', 'size', 'count', 'is_contained', 'isPermutationVector', 'isPowerOf2_32', 'isPowerOf2_64'}

@functools.lru_cache(None)
def compile_attr_atom(txt):
    """→ f(属性) -> bool，或 None（静态不完备：出现词汇表外的名字）。运行时不支持 ⇒ CppUnknown。"""
    src = txt.replace('$_self', '__self')
    try:
        p = P(lex(src)); e = p.expr()
        if p.pk()[0] != 'eof': return None
    except CppUnknown: return None
    cs = set(); _calls(e, cs)
    for k, n in cs:
        b = _base(n)
        if k == 'm' and b not in ATOM_METHODS: return None
        if k in ('fn', 't') and b not in ATOM_FNS: return None
    def f(a):
        it = Interp(Op.__new__(Op), 0)
        v = it.ev(e, {'__self': Attr(a)})
        if isinstance(v, (int, bool)): return bool(v)
        raise CppUnknown('原子值')
    return f
