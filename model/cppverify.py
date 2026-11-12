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
class _Crash(_Ret): pass   # 第 82 轮：断言崩溃穿过被调函数直达入口（原先被调函数把它当返回值 Diag 吞掉）
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
TEMPLATE_FNS = {'getAttrOfType', 'IsaPred', 'cast', 'dyn_cast', 'isa', 'dyn_cast_or_null', 'cast_or_null', 'dyn_cast_if_present', 'isa_and_nonnull', 'isa_and_present',
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
            if self._bind_ahead() is not None: return True
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
            if w == 'switch':
                self.nx(); self.ex('('); e = self.expr(); self.ex(')'); self.ex('{')
                arms = []; dflt = None                       # arms：[(标签表达式列表, 语句列表)]；dflt：默认分支下标
                while not self.isop('}'):
                    if self.isid('case') or self.isid('default'):
                        labels = []
                        while self.isid('case') or self.isid('default'):
                            if self.nx()[1] == 'case': labels.append(self.ternary())
                            else: dflt = len(arms); labels.append(None)
                            self.ex(':')
                        arms.append((labels, []))
                    else:
                        if not arms: raise CppUnknown('parse: switch 首语句')
                        arms[-1][1].append(self.stmt())
                self.ex('}'); return ('switch', e, arms, dflt)
            if w in ('do', 'goto', 'try', 'case', 'default', 'using', 'typedef'):
                raise CppUnknown(f'parse: 不支持 {w}')
            if w == 'assert':                                     # 第 82 轮：本 MLIR 构建开启断言 ⇒ assert(c) 求值，c 为假 ⇒ 崩溃（按拒绝）；原先整句跳过
                self.nx(); self.ex('('); c_ = self.expr(); self.ex(')'); self.ex(';'); return ('assert', c_)
            if w in ('LLVM_DEBUG', 'llvm_unreachable', 'static_assert'):
                self.nx(); self.ex('('); d = 1
                while d:
                    x = self.nx()[1]; d += (x == '(') - (x == ')')
                self.ex(';'); return ('unreachable',) if w == 'llvm_unreachable' else ('block', [])
            if self.looks_decl():
                d = self.decl(); self.ex(';'); return d
        e = self.expr(); self.ex(';'); return ('expr', e)
    def _bind_ahead(self):
        k = 0
        if self.isid('const', k): k += 1
        if not self.isid('auto', k): return None
        k += 1
        while self.isop('&', k) or self.isop('&&', k): k += 1
        return k if self.isop('[', k) else None
    def decl_head(self):
        k = self._bind_ahead()
        if k is not None:
            for _ in range(k): self.nx()
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
            if self.isop('=') and self.isop('{', 1): self.nx(); self.nx(); init = ('brace', h[2] if h[0] == 'var' else None, self.args('}'))
            elif self.isop('='): self.nx(); init = self.expr(no_comma=True)
            elif self.isop('('): self.nx(); args = self.args(')'); init = ('ctor', h[2] if h[0] == 'var' else None, args)
            elif self.isop('{'): self.nx(); args = self.args('}'); init = ('brace', h[2] if h[0] == 'var' else None, args)
            elif self.isop('[') and self.isop(']', 1) and self.isop('=', 2) and self.isop('{', 3):      # T a[] = {…}
                self.nx(); self.nx(); self.nx(); self.nx(); init = ('brace', 'std::array', self.args('}'))
            elif self.isop('['): raise CppUnknown('parse: C 数组')
            out.append(('decl', h, init))
            if self.isop(',') and h[0] == 'var':
                self.nx(); nm = self.nx()[1]; h = ('var', nm, h[2]); continue
            break
        return out[0] if len(out) == 1 else ('seq', out)
    def args(self, close):
        out = []
        while not self.isop(close):
            e_ = self.expr(no_comma=True)
            if self.isop('.') and self.isop('.', 1) and self.isop('.', 2):     # 包展开 rs...
                self.nx(); self.nx(); self.nx(); e_ = ('pack', e_)
            out.append(e_)
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
            return r and _base(nm) in ('int64_t', 'int', 'unsigned', 'size_t', 'int32_t', 'uint64_t', 'bool', 'void')   # 第 82 轮：(void)x;
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
        if t == ('op', '{'): return ('brace', None, self.args('}'))     # 花括号初始化列表（initializer_list / 聚合：pair 等）
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
            if self.isop('<'):
                j = self.i
                try:
                    targs = self._targs()
                    if _base(n) in TEMPLATE_FNS or n.startswith('std::') or n.endswith('Type') or n.endswith('Attr') or n.endswith('Op') or self.isop('('):
                        return ('tid', n, targs)
                except CppUnknown: pass
                self.i = j; return ('id', n)
            return ('id', n)
        raise CppUnknown(f'parse: 表达式 {t[1]!r}')
    def _targs(self):
        self.ex('<'); out = []
        while True:
            if self.t[self.i][0] == 'num': out.append(self.nx()[1])     # 非类型模板实参：std::get<2>
            else: out.append(self.typ())
            if self.isop('.') and self.isop('.', 1) and self.isop('.', 2):   # 模板包展开 Ty...
                self.nx(); self.nx(); self.nx(); out[-1] = str(out[-1]) + '...'
            if self.isop(','): self.nx(); continue
            if self.isop('>'): self.nx(); return out
            if self.isop('>>'): raise CppUnknown('parse: >>')
            raise CppUnknown('parse: 模板实参')
    def postfix(self, e):
        while True:
            if self.isop('('): self.nx(); e = ('call', e, self.args(')'))
            elif self.isop('.') and self.isop('.', 1): return e     # 包展开 x...（由 args 处理）
            elif self.isop('.') or self.isop('->'):
                self.nx(); nm = self.nx()[1]
                if self.isop('<') and nm in TEMPLATE_FNS: targs = self._targs(); e = ('mem', e, nm, targs)
                else: e = ('mem', e, nm, None)
            elif self.isop('['): self.nx(); i = self.expr(); self.ex(']'); e = ('idx', e, i)
            elif self.isop('++') or self.isop('--'): d_ = 1 if self.isop('++') else -1; self.nx(); e = ('post', e, d_)   # 第 82 轮：记下 ++/--（原先后置 -- 按 ++ 求值）
            elif self.isop('{') and e[0] in ('id', 'tid'): self.nx(); e = ('brace', e[1], self.args('}'))
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
    if dia not in O.DIALECTS: return []
    ds = O.DIALECTS[dia]; ds = ds if isinstance(ds, list) else [ds]
    out = []
    for d in ds:
        a, b = d.split('/')[:2]
        out += sorted(glob.glob(os.path.join(CPP_DIR, f'{a}_{b}_*.cpp')))
    # 第 82 轮：方言共用的工具源（Dialect/Utils/ReshapeOpsUtils.{h,cpp}）排在方言自身文件之后，同名时方言文件优先
    return sorted(set(out)) + [os.path.join(CPP_DIR, x) for x in SHARED_SRC if os.path.exists(os.path.join(CPP_DIR, x))]

SHARED_SRC = ('Utils_ReshapeOpsUtils.h', 'Utils_ReshapeOpsUtils.cpp')

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
def find_methods(dia, cls, meth):
    """第 82 轮：Class::meth 的全部 out-of-line 重载 → [(参数表文本, 函数体)]。"""
    out = []
    for f in _dialect_files(dia):
        s = _file(f)
        for m in re.finditer(r'(?:^|\n)[\w:<>,\s\*&]*?\b' + re.escape(cls) + r'::' + re.escape(meth) + r'\s*\(([^)]*)\)\s*(?:const\s*)?\{', s):
            out.append((m.group(1), _body_at(s, m.end())))
    return out

def _arg_fits(ptype, v):
    """重载选择的最小类型判别（第 82 轮）：只区分 ArrayRef<AffineMap> 与 ReassociationIndices 一类的形参；其余视为相容。"""
    pt = ptype or ''
    if isinstance(v, TList) and v.ety:
        e_ = _base(v.ety)
        if 'AffineMap' in pt and 'AffineMapAttr' not in pt: return e_ == 'AffineMap'
        if 'ReassociationIndices' in pt: return e_ in ('ReassociationIndices',)
    if 'AffineMap' in pt and 'AffineMapAttr' not in pt:
        return isinstance(v, AMap) or isinstance(v, list) and all(isinstance(x, AMap) for x in v)
    if 'ReassociationIndices' in pt:
        return isinstance(v, list) and all(isinstance(x, list) and all(isinstance(y, int) for y in x) for x in v)
    return True

def pick_overload(cands, args):
    """按实参个数与 _arg_fits 选唯一重载；不唯一 ⇒ CppUnknown。"""
    if len(cands) == 1: return cands[0]
    ok = []
    for c in cands:
        ps = _params(c[0])
        req = sum(1 for _, _, d in ps if d is None)
        if not (req <= len(args) <= len(ps)): continue
        if all(_arg_fits(t, a) for (t, _, _), a in zip(ps, args)): ok.append(c)
    if len(ok) != 1: raise CppUnknown(f'重载不唯一（{len(ok)}）')
    return ok[0]

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
    if opname not in O.ops(dia): return {}
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
def find_functions(dia, name, extra=()):
    """方言 .cpp 里的自由函数（static/匿名命名空间/template）的全部重载 → [(模板形参, 参数表, 体)]。
    第 80 轮：形参包 `typename... Ty` 记作 'Ty...'。"""
    files = list(_dialect_files(dia)) + [os.path.join(CPP_DIR, x) for x in extra]
    out = []
    for f in files:
        if not os.path.exists(f): continue
        s = _file(f)
        for m in re.finditer(r'(template\s*<([^>]*)>\s*)?(?:static\s+|inline\s+)*(?:[\w:]+(?:<[^;{}()]*>)?[\s\*&]+)(?:\w+::)*\b' + re.escape(name) + r'\s*\(((?:[^()]|\([^()]*\))*)\)\s*\{', s):
            tps = [x.split()[-1] + ('...' if '...' in x else '') for x in (m.group(2) or '').split(',') if x.strip()]
            out.append((tps, m.group(3), _body_at(s, m.end())))
    return out

def find_function(dia, name, extra=(), nargs=None):
    fs = find_functions(dia, name, extra)
    if not fs: return None
    if nargs is not None:
        for f in fs:                                       # 先找形参个数恰好相等、无形参包的重载，再找带形参包的
            ps = _params(f[1])
            if len(ps) == nargs and not any('...' in t for t, _, _ in ps): return f
        for f in fs:
            ps = _params(f[1])
            if any('...' in t for t, _, _ in ps) and nargs >= len(ps) - 1: return f
    return fs[0]

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
class TList(list):        # 第 82 轮：带声明元素类型的 SmallVector（仅供重载选择；空表也能区分 ArrayRef<AffineMap> 与 ReassociationIndices）
    ety = ''
def _tlist(xs, ety):
    t = TList(xs); t.ety = ety; return t
class Diag:              # InFlightDiagnostic（隐式转换为 failure）
    def __init__(self, msg): self.msg = msg
class LR:                # LogicalResult
    def __init__(self, ok): self.ok = ok
OK, FAIL = LR(True), LR(False)
class Layout:
    def __init__(self, txt, rank=None): self.txt, self.rank = txt, rank
class Space:
    def __init__(self, txt): self.txt = txt
    def __eq__(self, o): return isinstance(o, Space) and _norm_space(self.txt) == _norm_space(o.txt)
    def __hash__(self): return 0
def _norm_space(t):
    if t is None: return None
    m = re.fullmatch(r'\s*(\d+)\s*(?::\s*i64)?\s*', t)
    if m and int(m.group(1)) == 0: return None
    return re.sub(r'\s+', '', t)
class InvariantFail(Exception):
    """ODS 不变式（verifyInvariants）不满足：MLIR 不会运行 C++ verify，整体拒绝。"""

class EnumV:
    """枚举值：类名 + 符号。"""
    def __init__(self, cls, sym): self.cls, self.sym = cls, sym
    def __eq__(self, o): return isinstance(o, EnumV) and o.sym == self.sym and (not o.cls or not self.cls or o.cls == self.cls)
    def __hash__(self): return hash(self.sym)

@functools.lru_cache(None)
def enum_classes():
    """全部方言 ODS 的枚举：类名 → {值: 符号}, {串: 符号}。"""
    import odsjson as O
    out = {}
    for dia in O.DIALECTS:
        d = O.records(dia)
        for k, r in d.items():
            if not isinstance(r, dict) or 'EnumInfo' not in r.get('!superclasses', []) or not r.get('className') or not r.get('enumerants'): continue
            vals, strs = {}, {}
            for e in r['enumerants']:
                c = d.get(O.ref(e), {})
                if 'symbol' in c:
                    if c.get('value') is not None: vals[c['value']] = c['symbol']
                    strs[c.get('str') or c['symbol']] = c['symbol']
            out.setdefault(r['className'], (vals, strs, 'BitEnum' in str(r.get('!superclasses'))))
    return out

class AExpr:
    """第 82 轮：mlir::AffineExpr（经 MLIR Python 的 AffineExpr）。"""
    def __init__(self, e): self.e = e
    def __eq__(self, o): return isinstance(o, AExpr) and str(o.e) == str(self.e)
    def __hash__(self): return hash(str(self.e))
def _aexpr_kind(e):
    import mlir.ir as ir
    for k, K in (('AffineDimExpr', ir.AffineDimExpr), ('AffineSymbolExpr', ir.AffineSymbolExpr), ('AffineConstantExpr', ir.AffineConstantExpr)):
        if K.isinstance(e): return k, K(e)
    if ir.AffineBinaryExpr.isinstance(e): return 'AffineBinaryOpExpr', ir.AffineBinaryExpr(e)
    raise CppUnknown('AffineExpr 种类')
class AMap:
    """mlir::AffineMap（经 MLIR Python 的 AffineMap）。"""
    def __init__(self, m): self.m = m
    def __eq__(self, o): return isinstance(o, AMap) and str(o.m) == str(self.m)
    def __hash__(self): return hash(str(self.m))

class Lambda:
    def __init__(self, params, body, env): self.params, self.body, self.env = params, body, env
class Self: pass
class EnumPair(tuple):    # 第 82 轮：llvm::enumerate 的 result_pair（index()/value()；结构化绑定照常）
    pass
class Sat:                # 第 82 轮：mlir/Dialect/Utils/StaticValueUtils.h 的 SaturatedInteger（逐行照抄语义）
    __slots__ = ('saturated', 'v')
    def __init__(self, saturated=True, v=0): self.saturated, self.v = saturated, v
    @staticmethod
    def wrap(v): return Sat(True, 0) if v == DYN else Sat(False, v)
    def asInteger(self): return DYN if self.saturated else self.v
    def desaturate(self, o):
        if self.saturated and not o.saturated: return o
        if not self.saturated and not o.saturated and self.v != o.v: return FAIL
        return self
    def __eq__(self, o): return isinstance(o, Sat) and ((self.saturated and o.saturated) or (not self.saturated and not o.saturated and self.v == o.v))
    def __hash__(self): return hash((self.saturated, self.v))
    def add(self, o): return Sat(True, 0) if self.saturated or o.saturated else Sat(False, o.v + self.v)
    def mul(self, o):
        if not o.saturated and o.v == 0: return Sat(False, 0)
        if not self.saturated and self.v == 0: return Sat(False, 0)
        if self.saturated or o.saturated: return Sat(True, 0)
        return Sat(False, o.v * self.v)
class Attr:
    def __init__(self, a): self.a = a

@functools.lru_cache(None)
def cpp_type_spellings():
    """C++ 手写类型（无 ODS TypeDef）的拼写：解释方言 printType 里 `.Case<Cls>([&](…) { os << 表达式; })` 的表达式。→ {类名: '!方言.拼写'}。"""
    import odsjson as O
    out = {}; saved = CUR['dia']
    try: _spellings(out)
    finally: CUR['dia'] = saved
    return out

def _spellings(out):
    import odsjson as O
    for dia in O.DIALECTS:
        for f in _dialect_files(dia):
            src = open(f).read()
            m = re.search(r'void\s+\w+Dialect::printType\(Type \w+, DialectAsmPrinter &(\w+)\) const \{', src)
            if not m: continue
            os_ = m.group(1); body = src[m.end():m.end() + 6000]
            for c in re.finditer(r'\.Case<(\w+)>\(\s*\[&\]\([^)]*\)\s*\{\s*' + os_ + r'\s*<<\s*([^;{}]+);\s*\}\s*\)', body):
                CUR['dia'] = dia; CUR['loose_enum'] = True
                try:
                    it = Interp(Op(dia, '', {'ins': [], 'outs': [], 'attrs': []}, [], None, '\x00none'), 0)
                    v = it.ev(P(lex(c.group(2))).expr(), {})
                    if isinstance(v, str): out[c.group(1)] = '!' + dia + '.' + v
                except CppUnknown: pass
                finally: CUR['loose_enum'] = False

_MMA_RE = re.compile(r'!gpu\.mma_matrix<((?:\d+x)+)([^,>]+),\s*"(\w+)"\s*>')
def _mma(t):
    """gpu::MMAMatrixType（C++ 手写类型，无 ODS 拼写）：!gpu.mma_matrix<16x16xf16, "AOp"> → (形状, 元素类型文本, operand)。"""
    if t[0] != 'opaque': return None
    m = _MMA_RE.fullmatch(re.sub(r'\s+', ' ', t[1].strip()))
    return (tuple(int(x) for x in m.group(1).split('x') if x), m.group(2).strip(), m.group(3)) if m else None

TYPE_CLASSES = {
    'MMAMatrixType': lambda t: _mma(t) is not None,
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
    if isinstance(x, V):
        if b in ('Value', 'OpFoldResult'): return True
        if b == 'Attribute' or b.endswith('Attr'): return False        # OpFoldResult 的动态分量
        x = x.t
    if isinstance(x, Attr) and b in ('Attribute', 'OpFoldResult'): return True
    if isinstance(x, tuple):
        if b in TYPE_CLASSES: return TYPE_CLASSES[b](x)
        import odsjson as O                      # 方言 ODS 类型（!shape.size 等）：按 TypeDef 的拼写判定
        m_ = re.fullmatch(r'(?:::)?(?:mlir::)?(\w+)::(\w+)', cls)
        ns = m_.group(1) if m_ else CUR['dia']
        xo = x if x[0] == 'opaque' else ('opaque', TS.show(x))          # typesig 给部分方言类型专门的项（('shape.shape',)）：按打印文本比较
        f = O._typedef_isa(ns, b) if ns else None
        if f is not None: return f(xo)
        sp = cpp_type_spellings().get(b)
        if sp is not None: return xo[1].strip() == sp
        raise CppUnknown(f'isa<{b}> 于类型')
    if isinstance(x, Attr): return _attr_isa(b, x.a, cls)
    if isinstance(x, AExpr):                                              # 第 82 轮
        if b == 'AffineExpr': return True
        if b in ('AffineDimExpr', 'AffineSymbolExpr', 'AffineConstantExpr', 'AffineBinaryOpExpr'): return _aexpr_kind(x.e)[0] == b
        raise CppUnknown(f'isa<{b}> 于 AffineExpr')
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
            'isPermutationVector', 'accumulate', 'is_sorted', 'zip_first', 'swap', 'tie', 'equal', 'getMixedValues', 'emitOptionalError', 'verifyCompatibleShapes', 'append_range', 'back_inserter', 'copy', 'getTensorTypeFromMemRefType', 'map_range', 'to_vector', 'to_string', 'string', 'APInt', 'getAffineDimExpr', 'getAffineSymbolExpr', 'getAffineConstantExpr', 'move', 'Twine'}

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
        if k == 'seq':                                  # 多声明符 T a = …, b = …;：同一作用域
            for x in s[1]: self.exec(x, env)
        elif k == 'block':
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
                v = _conv(h[2], v)
                if isinstance(v, list) and '&' not in (h[2] or ''):                # 值语义：SmallVector/auto 拷贝
                    m_ = re.match(r'(?:llvm::)?SmallVector\s*<\s*([\w:]+)', h[2] or '')
                    v = _tlist(v, m_.group(1)) if m_ else _tlist(v, v.ety) if isinstance(v, TList) else list(v)
                env[h[1]] = v
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
        elif k == 'switch':
            v = self.ev(s[1], env); start = None
            for i_, (labels, _) in enumerate(s[2]):
                if any(l is not None and _eq(v, self.ev(l, env)) for l in labels): start = i_; break
            if start is None: start = s[3]
            if start is not None:
                try:
                    for _, body in s[2][start:]:
                        for x in body: self.exec(x, env)
                except _Break: pass
        elif k == 'break': raise _Break()
        elif k == 'continue': raise _Continue()
        elif k == 'unreachable': raise CppUnknown('llvm_unreachable')
        elif k == 'assert':
            if not self.truth(self.ev(s[1], env)): _crash()
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
        if isinstance(v, tuple) and v and v[0] == '\x00it': raise CppUnknown('迭代器的真值')
        if isinstance(v, tuple) and v and v[0] == '\x00bound': raise CppUnknown(f'未解析的成员 {v[2]}')   # 第 82 轮：数据成员不可当真值（原先 tuple ⇒ 恒真）
        if isinstance(v, V): return True
        if isinstance(v, LR): return v.ok
        if isinstance(v, Diag): return True
        if isinstance(v, (list, Rng)): raise CppUnknown('范围的真值')
        if isinstance(v, Attr): return v.a is not None
        if isinstance(v, (Layout, Space, Self, tuple, EnumV)): return True
        return bool(v)
    def iter(self, r):
        if isinstance(r, (list, tuple, Rng)) and not (isinstance(r, tuple) and r and isinstance(r[0], str)): return list(r)
        if isinstance(r, range): return list(r)
        if isinstance(r, Attr): return [x if isinstance(x, (int, float, str)) else Attr(x) for x in _attr_list(r.a)]   # 第 81 轮：密集数组的元素是数值，不包成属性
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
                if isinstance(v, (int, tuple, V, list, Attr, Sat, Layout)) and not isinstance(v, bool): return v      # *optional / *this
                if isinstance(v, LR) and not v.ok: _crash()                  # 第 82 轮：*FailureOr 在 failure 上断言
                raise CppUnknown('解引用')
            if e[1] == '&': return v
        if k == 'post':
            v = self.ev(e[1], env); self.store(e[1], _num(v) + (e[2] if len(e) > 2 else 1), env); return v
        if k == '?': return self.ev(e[2], env) if self.truth(self.ev(e[1], env)) else self.ev(e[3], env)
        if k == 'assign':
            if e[1] == '=': v = self.ev(e[3], env)
            else:
                a, b = _num(self.ev(e[2], env)), _num(self.ev(e[3], env))
                v = {'+=': a + b, '-=': a - b, '*=': a * b, '/=': (int(a / b) if b else _crash())}[e[1]]
            self.store(e[2], v, env); return v
        if k == 'cast_c':
            v_ = self.ev(e[2], env)
            return None if _base(e[1]) == 'void' else _conv(e[1], v_)
        if k == 'lambda': return Lambda(e[1], e[2], env)
        if k == 'idx':
            c, i = self.ev(e[1], env), _num(self.ev(e[2], env))
            seq = self.iter(c)
            if not 0 <= i < len(seq): _crash()
            return seq[i]
        if k == 'mem':
            r_ = self.ev(e[1], env)
            if isinstance(r_, Props): return self.op_method('get' + camel(e[2]) + 'Attr', [], env, None)
            if isinstance(r_, Sat) and e[2] in ('saturated', 'v'): return getattr(r_, e[2])     # 第 82 轮：SaturatedInteger 的数据成员
            return ('\x00bound', r_, e[2], e[3])
        if k == 'call': return self.call(e[1], e[2], env)
        if k == 'tid': return ('\x00tfn', e[1], e[2])
        if k == 'brace':                               # T x{a, b} / = {a, b} / return {a, b}：列表初始化，无 (n, v) 计数语义
            args = [self.ev(a, env) for a in e[2]]
            b = _base(e[1] or '')
            if e[1] is None or b == 'auto' or re.match(r'(llvm::|std::)?(SmallVector|ArrayRef|vector|pair|tuple|array)\b', e[1].strip()):
                return list(args)
            if len(args) == 1: return _conv(e[1], args[0])
            if not args: return _default(e[1])
            raise CppUnknown(f'列表初始化 {e[1]}')
        if k == 'ctor':
            args = [self.ev(a, env) for a in e[2]]
            b = _base(e[1] or '')
            if b.startswith('SmallVector') or b.startswith('ArrayRef') or b.startswith('std::vector') or b.startswith('SmallVector<'):
                if len(args) == 1 and isinstance(args[0], (list, Rng)): return list(args[0])
                if len(args) == 2 and isinstance(args[0], int): return [args[1]] * args[0]
                if len(args) == 1 and isinstance(args[0], int) and not isinstance(args[0], bool) and not b.startswith('ArrayRef'):
                    # 第 82 轮：SmallVector<T> v(n) = n 个值初始化元素（原先误作单元素 [n]）
                    m_ = re.search(r'<\s*([\w:]+)', e[1]); et = m_.group(1) if m_ else ''
                    if _base(et) in ('int', 'int64_t', 'unsigned', 'size_t', 'int32_t', 'uint64_t', 'uint32_t', 'long', 'int8_t', 'uint8_t', 'int16_t'): return [0] * args[0]
                    if _base(et) == 'bool': return [False] * args[0]
                    raise CppUnknown(f'SmallVector 计数构造 {e[1]}')
                return list(args)
            if re.sub(r'^(::)?(mlir::)?', '', e[1].strip()) in ('Builder', 'OpBuilder'): return BuilderV()   # 不含 VectorType::Builder
            if b.endswith('Adaptor'): return Self()                   # XxxOpAdaptor adaptor(operands, attributes, properties, regions)
            if len(args) == 1: return _conv(e[1], args[0])
            if not args: return _default(e[1])
            raise CppUnknown(f'构造 {e[1]}')
        raise CppUnknown(f'表达式 {k}')
    def store(self, tgt, v, env):
        if tgt[0] == 'call' and tgt[1][0] == 'id' and _base(tgt[1][1]) == 'tie':       # std::tie(a, b) = pair
            vs = list(v) if isinstance(v, (list, tuple)) else None
            if vs is None or len(vs) != len(tgt[2]): raise CppUnknown('std::tie')
            for t_, x_ in zip(tgt[2], vs): self.store(t_, x_, env)
            return
        if tgt[0] == 'id':
            try: e, _ = self.lookup(env, tgt[1])
            except KeyError: raise CppUnknown(f'赋值未声明 {tgt[1]}')
            e[tgt[1]] = list(v) if isinstance(v, list) else v; return
        if tgt[0] == 'idx':                              # v[i] = x（SmallVector 元素赋值）
            c = self.ev(tgt[1], env); i = _num(self.ev(tgt[2], env))
            if isinstance(c, list):
                if not 0 <= i < len(c): _crash()
                c[i] = v; return
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
        if isinstance(a, Sat) or isinstance(b, Sat):                      # 第 82 轮：SaturatedInteger 的 + 与 *
            if not (isinstance(a, Sat) and isinstance(b, Sat)) or op not in ('+', '*'): raise CppUnknown(f'SaturatedInteger 运算 {op}')
            return a.add(b) if op == '+' else a.mul(b)
        if op == '+' and isinstance(a, str) and isinstance(b, str): return a + b
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
        q = n.split('::')
        if len(q) >= 2 and q[-2] in enum_classes() and not enum_classes()[q[-2]][2]:
            if q[-1] in enum_classes()[q[-2]][1].values(): return EnumV(q[-2], q[-1])
        if CUR.get('loose_enum') and len(q) >= 2 and q[-2][:1].isupper() and q[-1][:1].isupper(): return EnumV(q[-2], q[-1])   # 方言 .cpp 内的 C++ enum class
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
        if isinstance(lam, tuple) and lam and lam[0] == '\x00tfn' and _base(lam[1]) == 'IsaPred':     # llvm::IsaPred<A, B…> 作为谓词值
            return any(_isa(_resolve_t(c, {}), args[0]) for c in lam[2])
        if isinstance(lam, tuple) and lam and lam[0] == '\x00fn':     # 以名字传入的函数（如 ShapedType::isDynamic）
            return self.fcall(lam[1], [('lit', a) if not isinstance(a, (list, tuple, V, Attr)) else ('\x00val', a) for a in args], {})
        if not isinstance(lam, Lambda): raise CppUnknown('调用非函数值')
        env = dict(zip(lam.params, args)); env['\x00parent'] = lam.env
        try: self.exec(lam.body, env)
        except _Crash: raise
        except _Ret as r: return r.v
        return None
    def evargs(self, args_e, env):
        out = []
        for a in args_e:
            if a[0] == 'pack':
                v = self.ev(a[1], env)
                if not isinstance(v, Pack): raise CppUnknown('包展开')
                out.extend(v.xs)
            else: out.append(self.ev(a, env))
        return out
    def tcall(self, n, targs, args_e, env):
        targs = _expand_targs(targs, env)
        args = self.evargs(args_e, env)
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
        if n == 'get' and len(args) == 1 and isinstance(args[0], tuple) and targs and str(targs[0]).strip().isdigit():   # std::get<k>(pair)
            return args[0][int(str(targs[0]).strip())]
        if n == 'to_vector': return list(self.iter(args[0]))                 # 第 82 轮：llvm::to_vector<N>(range)
        if n == 'static_cast': return _conv(targs[0], args[0])
        if n in ('min', 'max'): return (min if n == 'min' else max)(_num(x) for x in args)
        if n == 'seq': return list(range(_num(args[0]), _num(args[1]))) if len(args) == 2 else list(range(_num(args[0])))
        if n in ('SmallVector', 'ArrayRef'):
            if len(args) == 1 and isinstance(args[0], (list, Rng)): return list(args[0])
            return list(args)
        ff = find_function(self.op.dia, n, nargs=len(args))
        if ff is not None: return self.invoke(ff[1], ff[2], args, ff[0], targs=targs)
        raise CppUnknown(f'模板函数 {n}')
    def fcall(self, n, args_e, env):
        b = _base(n)
        q = n.split('::')
        if len(q) >= 2 and q[-1] == 'get' and q[-2] not in CTORS and _nullary_type(q[-2], self.op.dia) is not None:
            return _nullary_type(q[-2], self.op.dia)          # 无参数的方言类型 X::get(ctx)：!shape.size 等
        if len(q) >= 2 and q[-1] == 'get' and q[-2] in CTORS:
            return CTORS[q[-2]]([self.ev(a, env) for a in args_e])
        if len(q) == 2 and q[0].endswith('Op') and not q[0].startswith('std'):
            fms = find_methods(self.op.dia, q[0], q[1])
            if not fms and q[0] == getattr(self.op, 'cls', None) and not inline_methods(self.op.dia, self.op.opname).get(q[1]):   # 第 82 轮：XxxOp::getFooAttrName() 等本 op 的静态访问器
                return self.op_method(q[1], [self.ev(a, env) for a in args_e], env, None)
            if fms:
                av_ = [self.ev(a, env) for a in args_e]; fm = pick_overload(fms, av_)
                return self.invoke(fm[0], fm[1], av_, ())
        if b.endswith('Type') and not args_e and b[:1].isupper(): return None          # RankedTensorType() 等：空类型
        if b == 'getMixedValues':                       # 静态数组中的 kDynamic 依次换成动态值（OpFoldResult）
            st_, dy_ = self.iter(self.ev(args_e[0], env)), self.iter(self.ev(args_e[1], env)); out_ = []; k_ = 0
            for x_ in st_:
                if _num(x_) == DYN:
                    if k_ >= len(dy_): _crash()
                    out_.append(dy_[k_]); k_ += 1
                else:
                    import mlir.ir as ir
                    out_.append(Attr(ir.IntegerAttr.get(ir.IntegerType.get_signless(64, _ctx()), _num(x_))))   # b.getI64IntegerAttr
            return out_
        if b == 'equal':
            a_, c_ = self.iter(self.ev(args_e[0], env)), self.iter(self.ev(args_e[1], env)); return len(a_) == len(c_) and all(_eq(x, y) for x, y in zip(a_, c_))
        if b == 'swap' and len(args_e) == 2 and all(a[0] == 'id' for a in args_e):     # std::swap(a, b)
            x_, y_ = self.ev(args_e[0], env), self.ev(args_e[1], env); self.store(args_e[0], y_, env); self.store(args_e[1], x_, env); return None
        if re.fullmatch(r'stringify\w+', b):
            v_ = self.ev(args_e[0], env)
            if isinstance(v_, EnumV): return v_.sym
            raise CppUnknown('stringify 非枚举')
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
        if b == 'enumerate':
            if len(args) == 1: return [EnumPair((i, x)) for i, x in enumerate(self.iter(args[0]))]
            ls = [self.iter(a) for a in args]                            # 第 82 轮：多范围 enumerate（STLExtras 断言等长）
            if len({len(l) for l in ls}) > 1: _crash()                  # 断言 "Ranges have different length"（本 MLIR 构建开启断言，同 zip_equal）
            return [EnumPair((i,) + tuple(t)) for i, t in enumerate(zip(*ls))]
        if b in ('zip', 'zip_equal', 'zip_first'):
            ls = [self.iter(a) for a in args]
            if b == 'zip_first' and any(len(l) < len(ls[0]) for l in ls): _crash()
            if b == 'zip_equal' and len({len(l) for l in ls}) > 1: _crash()
            return [tuple(t) for t in zip(*ls)]
        if b in ('isPowerOf2_32', 'isPowerOf2_64'): x = _num(args[0]); return x > 0 and x & (x - 1) == 0
        if b == 'emitOptionalError': return Diag(args[1] if len(args) > 1 and isinstance(args[1], str) else '')
        if b == 'verifyCompatibleShapes' and len(args) == 1:     # mlir/lib/IR/TypeUtilities.cpp verifyCompatibleShapes(TypeRange)
            ts = [_ty(x) for x in self.iter(args[0])]
            sh = [t for t in ts if TS.shaped(t)]
            if len(sh) != len(ts): return FAIL if sh else OK
            sc = [t[0] == 'vector' and any(_scal(t)) for t in ts]
            if any(sc) and not all(sc): return FAIL
            rk = [t for t in ts if t[1] is not None]
            if not rk: return OK
            if len({len(t[1]) for t in rk}) > 1: return FAIL
            for i_ in range(len(rk[0][1])):
                st_ = {t[1][i_] for t in rk if t[1][i_] is not None}
                if len(st_) > 1: return FAIL
            return OK
        if b == 'APInt' and len(args) == 2:                              # APInt(64, v)：值即 v（IntegerAttr::get 只取其数值）
            w_, v_ = _num(args[0]), _num(args[1])
            if not -(1 << (w_ - 1)) <= v_ < (1 << w_): raise CppUnknown('APInt 截断')
            return v_
        if b in ('getAffineDimExpr', 'getAffineSymbolExpr', 'getAffineConstantExpr'):   # 第 82 轮：mlir::getAffine*Expr(位置/值, ctx)
            import mlir.ir as ir
            K = {'getAffineDimExpr': ir.AffineDimExpr, 'getAffineSymbolExpr': ir.AffineSymbolExpr, 'getAffineConstantExpr': ir.AffineConstantExpr}[b]
            return AExpr(K.get(_num(args[0]), _ctx()))
        if n in ('std::move', 'move'): return args[0]
        if b == 'Twine' and len(args) == 1: return args[0] if isinstance(args[0], str) else str(_num(args[0]))
        if b == 'map_range': return [self.apply(args[1], [x]) for x in self.iter(args[0])]     # 第 82 轮
        if b == 'to_vector': return list(self.iter(args[0]))
        if n in ('std::to_string', 'to_string'): return str(_num(args[0]))
        if n in ('std::string', 'string'): return ''.join(args) if all(isinstance(x, str) for x in args) else _crash()
        if n == 'SaturatedInteger::wrap': return Sat.wrap(_num(args[0]))
        if b == 'append_range':
            if not isinstance(args[0], list): raise CppUnknown('append_range 目标')
            args[0].extend(self.iter(args[1])); return None
        if b == 'back_inserter': return BackIns(args[0])
        if b == 'copy' and len(args) == 2 and isinstance(args[1], BackIns):
            args[1].dst.extend(self.iter(args[0])); return None
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
        if b == 'getConstantIntValue':
            v_ = self.ev(args_e[0], env)
            import mlir.ir as ir
            if isinstance(v_, Attr) and ir.IntegerAttr.isinstance(v_.a): return ir.IntegerAttr(v_.a).value
            if isinstance(v_, int) and not isinstance(v_, bool): return v_
            raise CppUnknown('getConstantIntValue（值前提）')
        if b in ('drop_begin',): xs = self.iter(args[0]); return xs[(_num(args[1]) if len(args) > 1 else 1):]
        if b == 'drop_end': xs = self.iter(args[0]); return xs[:len(xs) - (_num(args[1]) if len(args) > 1 else 1)]   # 第 82 轮
        if b == 'reverse': return list(reversed(self.iter(args[0])))
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
        if isinstance(recv, AExpr):                                       # 第 82 轮：AffineExpr 方法
            k_, e_ = _aexpr_kind(recv.e)
            if m == 'getPosition' and k_ in ('AffineDimExpr', 'AffineSymbolExpr'): return e_.position
            if m == 'getValue' and k_ == 'AffineConstantExpr': return e_.value
            if m == 'getContext': return None
            if m == 'walk':                                                   # 后序遍历（AffineExprVisitor walkPostOrder）
                def w_(x):
                    kx, ex = _aexpr_kind(x)
                    if kx == 'AffineBinaryOpExpr': w_(ex.lhs); w_(ex.rhs)
                    self.apply(args[0], [AExpr(x)])
                w_(recv.e); return None
            raise CppUnknown(f'AffineExpr.{m}')
        if isinstance(recv, EnumPair):                                    # 第 82 轮
            if m == 'index': return recv[0]
            if m == 'value': return recv[1]
            raise CppUnknown(f'enumerate.{m}')
        if isinstance(recv, Sat):
            if m == 'asInteger': return recv.asInteger()
            if m == 'desaturate': return recv.desaturate(args[0]) if isinstance(args[0], Sat) else _crash()
            raise CppUnknown(f'SaturatedInteger.{m}')
        if m == 'getStridesAndOffset' and isinstance(recv, tuple) and recv and isinstance(recv[0], str):   # 第 82 轮：BuiltinTypes.cpp
            t_ = recv
            if t_[0] != 'memref': _crash()
            try: st_ = _strides(t_)
            except CppUnknown: raise
            if not args: 
                if st_ is False: _crash()
                return [list(st_[0]), st_[1]]                           # std::pair（结构化绑定）
            if len(args) == 2:
                if st_ is False: return FAIL
                if not isinstance(args[0], list): raise CppUnknown('getStridesAndOffset 输出形参')
                args[0][:] = st_[0]; self.store(args_e[1], st_[1], env); return OK
            raise CppUnknown('getStridesAndOffset 形参')
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
            if m == 'equals':
                o_ = self.iter(args[0]); return len(o_) == len(recv) and all(_eq(x, y) for x, y in zip(recv, o_))
            if m == 'assign':                                         # SmallVectorImpl::assign({…}) / assign(n, v)
                xs = list(args[0]) if len(args) == 1 and isinstance(args[0], (list, Rng)) else [args[1]] * _num(args[0]) if len(args) == 2 else None
                if xs is None: raise CppUnknown('assign')
                recv[:] = xs; return None
            if m == 'reserve': return None
            if m == 'resize':                                             # 第 82 轮：resize(n[, v])；增长且无 v 时元素类型未知 ⇒ 不猜
                n_ = _num(args[0])
                if n_ < 0: _crash()
                if n_ <= len(recv): del recv[n_:]; return None
                if len(args) < 2: raise CppUnknown('resize 增长无初值')
                recv.extend([args[1]] * (n_ - len(recv))); return None
            if m in ('begin', 'end') and not args: return ('\x00it', recv, 0 if m == 'begin' else len(recv))   # 第 82 轮：仅供 append(b, e)
            if m == 'append' and len(args) == 2 and all(isinstance(x, tuple) and x and x[0] == '\x00it' for x in args):
                if args[0][1] is not args[1][1] or args[0][2] > args[1][2]: raise CppUnknown('append 迭代器')
                recv.extend(list(args[0][1][args[0][2]:args[1][2]])); return None
            if m == 'append': recv.extend([args[1]] * _num(args[0]) if len(args) == 2 else self.iter(args[0])); return None   # append(n, v) / append(range)
            if m == 'drop_front': return recv[(_num(args[0]) if args else 1):]
            if m == 'drop_back': return recv[:len(recv) - (_num(args[0]) if args else 1)]
            if m == 'take_front': return recv[:(_num(args[0]) if args else 1)]
            if m == 'slice': return recv[_num(args[0]):_num(args[0]) + _num(args[1])]
            if m == 'contains' or m == 'count': return any(_eq(x, args[0]) for x in recv)
            raise CppUnknown(f'range.{m}')
        if isinstance(recv, BuilderV):
            if m in BUILDER_TYPES: return BUILDER_TYPES[m]
            raise CppUnknown(f'Builder.{m}')
        if isinstance(recv, EnumV):
            if m == 'getValue': return recv
            raise CppUnknown(f'枚举.{m}')
        if isinstance(recv, AMap):
            am = recv.m
            if m == 'getNumDims': return am.n_dims
            if m == 'getNumSymbols': return am.n_symbols
            if m == 'getNumResults': return len(am.results)
            if m == 'getResults': return [AExpr(r) for r in am.results]          # 第 82 轮
            if m == 'getNumInputs': return am.n_inputs
            if m == 'isIdentity': return am.is_identity if hasattr(am, 'is_identity') else str(am) == str(type(am).get_identity(am.n_dims, _ctx()))
            if m == 'isPermutation': return am.is_permutation
            if m == 'isProjectedPermutation': return am.is_projected_permutation
            if m == 'isEmpty': return am.n_dims == 0 and am.n_symbols == 0 and len(am.results) == 0
            raise CppUnknown(f'AffineMap.{m}')
        if isinstance(recv, Layout):
            if m == 'isIdentity':                                       # 接口缺省实现 getAffineMap().isIdentity()（第 82 轮：strided<[1]> 等也算恒等）
                if recv.txt is not None and recv.txt.strip().startswith('strided<'):
                    am_ = _layout_map(recv).m; return str(am_) == str(type(am_).get_identity(am_.n_dims, _ctx())) and am_.n_symbols == 0
                return _layout_identity(recv.txt)
            if m == 'getAffineMap': return _layout_map(recv)
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
        if m in ('getValue', 'getAffineMap') and ir.AffineMapAttr.isinstance(x): return AMap(ir.AffineMapAttr(x).value)
        if m == 'asArrayRef' and ir.DenseI64ArrayAttr.isinstance(x): return list(ir.DenseI64ArrayAttr(x))
        if m == 'asArrayRef' and re.fullmatch(r'array<i(?:8|16|32)(?::.*)?>', str(x), re.S): return [int(v) for v in _attr_list(x)]
        raise CppUnknown(f'attr.{m}')
    def type_method(self, t, m, args, targs):
        mm = _mma(t)
        if mm is not None:
            if m == 'getOperand': return mm[2]
            if m == 'getShape': return list(mm[0])
            if m == 'getNumDims': return len(mm[0])
            if m == 'getElementType':
                try: return TS.parse(mm[1])
                except Exception: raise CppUnknown('mma 元素类型')
            raise CppUnknown(f'MMAMatrixType.{m}')
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
            return Layout(_mods(t).get('layout'), len(t[1]))
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
        if m in ('setDim', 'setShape') and t[0] == 'vector':     # VectorType::Builder（对类型值的函数式修改）
            mods = dict(t[3][1]) if len(t) > 3 else {}
            if m == 'setDim':
                i, n = _num(args[0]), _num(args[1]); sh = list(t[1])
                if not 0 <= i < len(sh): _crash()
                sh[i] = n
            else:
                sh = [_num(x) for x in self.iter(args[0])]
                if len(args) > 1: raise CppUnknown('setShape 可伸缩')
                mods.pop('scalable', None)
            mods = tuple((k, v) for k, v in mods.items() if not (k == 'scalable' and not any(v)))
            if 'scalable' in dict(mods) and len(dict(mods)['scalable']) != len(sh): raise CppUnknown('setDim 可伸缩')
            return ('vector', tuple(sh), t[2]) + ((('mods', mods),) if mods else ())
        if m == 'cloneWith':                       # ShapedType::cloneWith(optional shape, elt)：保留编码/布局/内存空间/可伸缩标记
            if not TS.shaped(t): _crash()
            sh, el = args
            if sh is not None:
                sh = tuple(None if x == DYN else _num(x) for x in self.iter(sh))
                if t[0] == 'vector' and len(sh) != len(t[1] or ()): raise CppUnknown('cloneWith 改秩')
            else: sh = t[1]
            return (t[0], sh, _ty(el)) + tuple(t[3:])
        if m == 'isLastDimUnitStride':
            st_ = _strides(t)
            return st_ is not False and (not st_[0] or st_[0][-1] == 1)
        if m == 'isStrided': return _strides(t) is not False
        if m == 'getContext': return None
        if m == 'getTypeID': raise CppUnknown('getTypeID')
        raise CppUnknown(f'type.{m}')
    def op_method(self, m, args, env, targs):
        op = self.op
        if m in ('getOperation',): return Self()
        if m == 'getContext': return None
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
        if m == 'getProperties': return Props()
        if m == 'getAttrOfType' and args and isinstance(args[0], str):
            if op.attrs is None: raise CppUnknown('查询未提供属性')
            txt = op.attrs.get(args[0])
            if txt is None: return None
            a_ = Attr(_parse_attr(txt))
            return a_ if not targs or _isa(targs[0], a_) else None
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
        fms = find_methods(op.dia, op.cls, m)
        if fms: fm = pick_overload(fms, args); return self.invoke(fm[0], fm[1], args, ())
        ff = find_function(op.dia, _base(m), nargs=len(args))
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
        ec0 = enum_classes().get(_base(rt))
        if txt is None and (rec.get('defaultValue') or '').strip() and ec0 is not None and not ec0[2]:
            # DefaultValuedAttr：注册 op 创建时 populateDefaultAttrs 物化缺省值；枚举缺省值是限定的枚举常量
            q_ = rec['defaultValue'].strip().split('::')
            if len(q_) >= 2 and q_[-1] in ec0[1].values(): return EnumV(_base(rt), q_[-1])
        if txt is None:
            if raw: return None
            if opt: return None
            if rt == 'bool' and (kind in ('UnitAttr', 'UnitProp') or c in ('UnitAttr', 'UnitProp')): return False
            raise CppUnknown(f'属性 {nm} 缺省值')
        a = _parse_attr(txt)
        import mlir.ir as ir
        st_ = _base(rec.get('storageType') or '')
        if raw and not (ec0 is not None and not ec0[2] and st_.endswith('Attr') and st_ != 'IntegerAttr'):
            return Attr(a)                                  # 特化的枚举属性类（XxxAttr）：getXAttr().getValue() 即枚举值，下面按枚举转换
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
        if rt in ('::mlir::AffineMap', 'mlir::AffineMap', 'AffineMap') and ir.AffineMapAttr.isinstance(a): return AMap(ir.AffineMapAttr(a).value)
        ec = enum_classes().get(_base(rt))
        if ec is not None and not ec[2]:
            vals, strs, _ = ec
            if ir.IntegerAttr.isinstance(a) and ir.IntegerAttr(a).value in vals: return EnumV(_base(rt), vals[ir.IntegerAttr(a).value])
            m_ = re.fullmatch(r'#[\w.]+<\s*(?:[\w.]+\s+)?([\w.]+)\s*>', str(a))       # #d.m<sym> 或 #d<m sym>
            if m_ and m_.group(1) in strs: return EnumV(_base(rt), strs[m_.group(1)])
            if ir.IntegerAttr.isinstance(a): raise InvariantFail(f'枚举 {_base(rt)} 无此值 {str(a)[:30]}')   # I32/I64EnumAttr 谓词先拒（verifyInvariants 先于 verify）
            raise CppUnknown(f'枚举属性值 {str(a)[:30]}')
        if rt in ('::mlir::Type', 'mlir::Type', 'Type') and ir.TypeAttr.isinstance(a): return TS.parse(str(ir.TypeAttr(a).value))
        if rt in ('::llvm::StringRef', 'llvm::StringRef', 'StringRef') and ir.StringAttr.isinstance(a): return ir.StringAttr(a).value
        if rt.startswith('::mlir::') and rt.endswith('Attr') or rt in ('::mlir::TypedAttr', '::mlir::Attribute', '::mlir::ElementsAttr', '::mlir::ArrayAttr', '::mlir::DenseElementsAttr'):
            return Attr(a)
        raise CppUnknown(f'属性 {nm} 的返回类型 {rt or kind}')
    def invoke(self, params_txt, body, args, tparams, targs=None):
        if self.depth > 6: raise CppUnknown('调用过深')
        ps = _params(params_txt)
        env = {}
        if ps and '...' in ps[-1][0]:                       # 函数形参包：其余实参
            env[ps[-1][1]] = Pack(list(args[len(ps) - 1:])); args = list(args[:len(ps) - 1]); ps = ps[:-1]
        if len(args) > len(ps): raise CppUnknown('实参过多')
        for (ty, nm, dflt), a in zip(ps, args + [None] * (len(ps) - len(args))):
            if a is None and dflt is not None: a = self.ev(P(lex(dflt)).expr(), {})
            if isinstance(a, Self) and _base(ty).endswith('Op') is False and ty not in ('Operation*', 'Operation *'): pass
            env[nm] = _conv(ty, a) if a is not None else None
        ta_ = list(targs or [])
        for i_, tp in enumerate(tparams):
            if tp.endswith('...'): env['\x00T:' + tp[:-3]] = Pack(ta_[i_:]); break    # 模板形参包
            if i_ < len(ta_): env['\x00T:' + tp] = ta_[i_]
        sub = Interp(self.op, self.depth + 1)
        b = parse_body(body)
        try: sub.exec(b, env)
        except _Crash: raise
        except _Ret as r: return r.v
        return None

def _nullary_type(cls, dia):
    """无参数类型的 X::get(context)：ODS TypeDef（parameters 为空）或 C++ 手写类型（printType 拼写）→ typesig 项。"""
    import odsjson as O
    sp = cpp_type_spellings().get(cls)
    if sp is None:
        for (ns, c), nm in O.typedefs().items():
            if c == cls and ns.split('::')[-1] == dia:
                rec = None
                for d_ in O.DIALECTS:
                    r_ = O.records(d_).get(next((k for k, v in O.records(d_).items() if isinstance(v, dict) and v.get('cppClassName') == cls and v.get('typeName') == nm), ''), None)
                    if r_: rec = r_; break
                if rec is not None and not O.dag_args(rec.get('parameters') or {'args': []}): sp = '!' + nm
                break
    if sp is None: return None
    try: return TS.parse(sp)
    except Exception: return None

class BackIns:
    def __init__(self, dst): self.dst = dst
class Props:
    """adaptor.getProperties()：成员即属性存储。"""
class BuilderV:
    """mlir::Builder：只用到其类型工厂。"""
BUILDER_TYPES = {'getIndexType': ('index',), 'getI1Type': ('i', 1, ''), 'getI8Type': ('i', 8, ''), 'getI16Type': ('i', 16, ''), 'getI32Type': ('i', 32, ''),
                 'getI64Type': ('i', 64, ''), 'getF16Type': ('f', 'f16'), 'getBF16Type': ('f', 'bf16'), 'getF32Type': ('f', 'f32'), 'getF64Type': ('f', 'f64')}

class Pack:
    """形参包（模板类型包或函数实参包）。"""
    def __init__(self, xs): self.xs = list(xs)

def _expand_targs(targs, env):
    if not targs: return targs
    out = []
    for c in targs:
        if isinstance(c, str) and c.strip().endswith('...'):
            v = _resolve_t(c.strip()[:-3].strip(), env)
            if not isinstance(v, Pack): raise CppUnknown('模板包展开')
            out.extend(v.xs)
        else: out.append(c)
    return out

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
            if not m or not m.group(1).strip() or m.group(1).rstrip().endswith('::'):   # 无名形参（MLIRContext *, std::optional<Location>）
                out.append((p, f'\x00arg{len(out)}', dflt))
            else: out.append((m.group(1).strip(), m.group(2), dflt))
        else: cur += ch
    return out

def _crash(): raise _Crash(Diag('\x00crash'))
def _num(v):
    if isinstance(v, bool): return int(v)
    if isinstance(v, int): return v
    if isinstance(v, float): return v
    raise CppUnknown(f'数值 {type(v).__name__}')
def _ty(x):
    if isinstance(x, V): return x.t
    if isinstance(x, tuple) and x and isinstance(x[0], str) and x[0][:1] != '\x00': return x
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
    if any(isinstance(x, tuple) and x and x[0] == '\x00bound' for x in (a, b)): raise CppUnknown('未解析的成员')   # 第 82 轮
    if isinstance(a, V): a = a.t
    if isinstance(b, V): b = b.t
    if isinstance(a, Self) and b is None or isinstance(b, Self) and a is None: return False   # 第 82 轮：op != nullptr（被校验 op 非空）
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
def _strides(t):
    """MemRefType::getStridesAndOffset：恒等布局按行主序；strided<[…], offset: …> 按文本；其余仿射布局未建模。→ (strides, offset) 或 False（无步长）。"""
    if t[0] != 'memref' or t[1] is None: _crash()
    lay = _mods(t).get('layout')
    if lay is None or _layout_identity(lay):
        st, acc = [], 1
        for d in reversed(t[1]):
            st.append(acc); acc = DYN if (acc == DYN or d is None) else acc * d
        return (list(reversed(st)), 0)
    m = re.fullmatch(r'strided<\[([^\]]*)\](?:,\s*offset:\s*(\?|-?\d+))?>', re.sub(r'\s+', ' ', lay).strip())
    if m:
        st = [DYN if x.strip() == '?' else int(x) for x in m.group(1).split(',') if x.strip()]
        off = m.group(2); return (st, DYN if off == '?' else int(off or 0))
    # 第 82 轮：一般仿射布局交给 MLIR 自身的 getStridesAndOffset（Python 绑定 MemRefType.get_strides_and_offset）；提取失败 ⇒ 无步长
    import mlir.ir as ir
    try:
        with _ctx(): mt = ir.MemRefType(ir.Type.parse(TS.show(t)))
    except Exception: raise CppUnknown('仿射布局的步长（类型不可解析）')
    try: st, off = mt.get_strides_and_offset()
    except Exception as e:
        if 'Failed to extract strides' in str(e): return False
        raise CppUnknown('仿射布局的步长')
    return (list(st), off)

def _layout_map(lay):
    import mlir.ir as ir
    if lay.txt is None or _layout_identity(lay.txt):
        if lay.rank is None: raise CppUnknown('布局秩')
        return AMap(ir.AffineMap.get_identity(lay.rank, _ctx()))
    if lay.txt.strip().startswith('affine_map'):
        return AMap(ir.AffineMapAttr(_parse_attr(lay.txt)).value)
    m = re.fullmatch(r'strided<\[([^\]]*)\](?:,\s*offset:\s*(\?|-?\d+))?>', re.sub(r'\s+', ' ', lay.txt).strip())
    if m:                                         # makeStridedLinearLayoutMap：动态偏移在前、动态步长随后各占一个符号
        st = [x.strip() for x in m.group(1).split(',') if x.strip()]; off = m.group(2) or '0'
        syms = []; terms = []
        if off == '?': syms.append('s%d' % len(syms)); terms.append(syms[-1])
        elif off != '0': terms.append(off)
        for i_, x in enumerate(st):
            if x == '?': syms.append('s%d' % len(syms)); terms.append(f'd{i_} * {syms[-1]}')
            else: terms.append(f'd{i_} * {x}')
        dims = ', '.join(f'd{i_}' for i_ in range(len(st)))
        txt = f"affine_map<({dims})[{', '.join(syms)}] -> ({' + '.join(terms) or '0'})>"
        return AMap(ir.AffineMapAttr(_parse_attr(txt)).value)
    raise CppUnknown('布局的仿射映射')

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
    if b.endswith('AttrInterface') or b.endswith('TypeInterface'): return None   # 第 81 轮：接口句柄缺省构造 = 空（C++ 语义）
    if b == 'SaturatedInteger': return Sat()                                     # 第 82 轮：成员缺省 saturated=true, v=0
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
def _mt_get(a):
    """MemRefType::get(shape, elt, layout = 恒等, memorySpace = 缺省)；整数 0 内存空间规范化为缺省（与 MLIR 打印一致）。"""
    sh, el = a[0], a[1]
    lay = a[2] if len(a) > 2 else None; sp = a[3] if len(a) > 3 else None
    mods = []
    if lay is not None and isinstance(lay, Layout) and lay.txt is not None and re.match(r'strided<', lay.txt): mods.append(('layout', lay.txt))   # 第 82 轮：StridedLayoutAttr
    elif lay is not None and not (isinstance(lay, Layout) and (lay.txt is None or _layout_identity(lay.txt))): raise CppUnknown('MemRefType::get 带布局')
    if isinstance(sp, Space) and _norm_space(sp.txt) is not None: mods.append(('space', sp.txt))
    elif sp is not None and not isinstance(sp, Space): raise CppUnknown('MemRefType::get 内存空间')
    t = ('memref', tuple(None if _num(x) == DYN else _num(x) for x in sh), _ty(el))
    return t + ((('mods', tuple(mods)),) if mods else ())
CTORS = {'MemRefType': _mt_get, 'IntegerType': lambda a: ('i', _num(a[1]), ''), 'IndexType': lambda a: ('index',),
         'UnrankedTensorType': lambda a: ('tensor', None, _ty(a[0])), 'RankedTensorType': _rt_get, 'VectorType': _vt_get,
         'IntegerAttr': lambda a: _int_attr(a), 'ArrayAttr': lambda a: _array_attr(a), 'StridedLayoutAttr': lambda a: _strided_attr(a), 'AffineMap': lambda a: _amap_get(a)}
def _amap_get(a):                    # 第 82 轮：AffineMap::get(dimCount, symbolCount, results, ctx)
    if len(a) != 4 or not isinstance(a[2], list) or not all(isinstance(x, AExpr) for x in a[2]): raise CppUnknown('AffineMap::get 形参')
    import mlir.ir as ir
    return AMap(ir.AffineMap.get(_num(a[0]), _num(a[1]), [x.e for x in a[2]], _ctx()))
def _strided_attr(a):                # 第 82 轮：StridedLayoutAttr::get(ctx, offset, strides)；打印文本由 MLIR 自身给出（与 typesig 读入的布局同一规范形）
    off, st = _num(a[-2]), [_num(x) for x in a[-1]]
    import mlir.ir as ir
    return Layout(str(ir.StridedLayoutAttr.get(off, st, _ctx())), len(st))
def _int_attr(a):                    # 第 82 轮：IntegerAttr::get(IntegerType 宽 w, APInt/整数)，仅无符号性为空的整数类型
    t, v = _ty(a[0]), _num(a[1])
    if t[0] != 'i' or t[2] != '': raise CppUnknown('IntegerAttr::get 类型')
    import mlir.ir as ir
    return Attr(ir.IntegerAttr.get(ir.IntegerType.get_signless(t[1], _ctx()), v))
def _array_attr(a):                  # 第 82 轮：ArrayAttr::get(ctx, ArrayRef<Attribute>)
    xs = a[-1]
    if not isinstance(xs, list) or not all(isinstance(x, Attr) for x in xs): raise CppUnknown('ArrayAttr::get 元素')
    import mlir.ir as ir
    return Attr(ir.ArrayAttr.get([x.a for x in xs], _ctx()))
STATS = {'python 异常': 0}
CUR = {'dia': None}
def decide(dia, opname, S, groups, attrs):
    """→ (True | False, 消息)；不支持 ⇒ CppUnknown。解释器自身的 Python 异常也作 CppUnknown（不猜），并计数。"""
    cls, body = verify_source(dia, opname)
    if body is None: raise CppUnknown('未找到 verify 源码')
    CUR['dia'] = dia
    it = Interp(Op(dia, opname, S, groups, None if attrs is None else {k: v for k, v in attrs.items() if not k.startswith('\x00')}, cls))
    try: return it.run_verify(body)
    except CppUnknown: raise
    except InvariantFail as e: return (False, f'ODS 不变式不满足（verify 不可达）：{e}')
    except RecursionError: raise CppUnknown('递归')
    except Exception as e:
        STATS['python 异常'] += 1; STATS.setdefault('例', []).append(f'{opname}: {type(e).__name__}: {e}'[:160])
        raise CppUnknown(f'解释器异常 {type(e).__name__}: {e}')

def infer_source(dia, opname):
    """第 80 轮：op 类外定义的 inferReturnTypes 与 isCompatibleReturnTypes → (类名, (参数表, 体) | None, (参数表, 体) | None)。"""
    cls, _ = verify_source(dia, opname)
    return cls, find_method(dia, cls, 'inferReturnTypes'), find_method(dia, cls, 'isCompatibleReturnTypes')

def _run_fn(it, params_txt, body, argmap):
    """按形参类型给实参（argmap(类型, 名字) → 值），执行函数体；→ (返回值, 函数结束时的环境)。"""
    env = {}
    for ty, nm, dflt in _params(params_txt): env[nm] = argmap(ty, nm)
    b = parse_body(body); rv = None
    try: it.exec(b, env)
    except _Ret as r: rv = r.v
    return rv, env

def decide_infer(dia, opname, S, groups, attrs):
    """InferTypeOpInterface::verifyTrait（mlir/lib/Interfaces/InferTypeOpInterface.cpp verifyInferredResultTypes）：
    inferReturnTypes 失败 ⇒ 拒绝（"failed to infer returned types"）；否则 isCompatibleReturnTypes(推断, 实际)（op 未覆盖时为逐个相等）。"""
    cls, inf, comp = infer_source(dia, opname)
    if inf is None: raise CppUnknown('未找到 inferReturnTypes 源码')
    CUR['dia'] = dia
    op = Op(dia, opname, S, groups, None if attrs is None else {k: v for k, v in attrs.items() if not k.startswith('\x00')}, cls)
    it = Interp(op)
    out = []
    def am(ty, nm):
        b = _base(re.sub(r'[&*]', '', ty).strip())
        if 'SmallVectorImpl' in ty: return out
        if b.endswith('Adaptor') or b in ('DictionaryAttr', 'OpaqueProperties', 'RegionRange', 'PropertyRef'): return Self()
        if b == 'ValueRange': return [V(t) for t in op.ins]
        return None                                       # MLIRContext*、std::optional<Location>
    try:
        rv, env = _run_fn(it, inf[0], inf[1], am)
        res = [nm_ for ty, nm_, _ in _params(inf[0]) if 'SmallVectorImpl' in ty]
        got = env.get(res[0]) if res else out             # 输出形参可能被整体重新赋值（inferredReturnTypes = SmallVector…）
        if isinstance(rv, Diag) or (isinstance(rv, LR) and not rv.ok): return (False, 'failed to infer returned types')
        got = [x.t if isinstance(x, V) else x for x in (got if isinstance(got, list) else [])]
        actual = list(op.outs)
        if comp is not None:
            ok = Interp(op).invoke(comp[0], comp[1], [got, actual], ())
            if not isinstance(ok, bool): raise CppUnknown('isCompatibleReturnTypes 返回值')
        else:
            ok = len(got) == len(actual) and all(_eq(a, b) for a, b in zip(got, actual))
        return (ok, None if ok else 'inferred type(s) are incompatible with return type(s) of operation')
    except CppUnknown: raise
    except InvariantFail as e: return (False, f'ODS 不变式不满足：{e}')
    except _Crash: return (False, '\x00crash')                         # 第 82 轮：断言崩溃 ⇒ 拒绝
    except RecursionError: raise CppUnknown('递归')
    except Exception as e:
        STATS['python 异常'] += 1; STATS.setdefault('例', []).append(f'{opname} infer: {type(e).__name__}: {e}'[:160])
        raise CppUnknown(f'解释器异常 {type(e).__name__}: {e}')

def infer_types(dia, opname, S, in_groups, attrs):
    """第 80 轮：只求 inferReturnTypes（不比较）。→ 类型表 | None（推断失败或断言）；运行时未建模 ⇒ CppUnknown。"""
    cls, inf, _ = infer_source(dia, opname)
    CUR['dia'] = dia
    groups = list(in_groups) + [[] for _ in S['outs']]
    op = Op(dia, opname, S, groups, None if attrs is None else {k: v for k, v in attrs.items() if not k.startswith('\x00')}, cls)
    out = []
    def am(ty, nm):
        b = _base(re.sub(r'[&*]', '', ty).strip())
        if 'SmallVectorImpl' in ty: return out
        if b.endswith('Adaptor') or b in ('DictionaryAttr', 'OpaqueProperties', 'RegionRange', 'PropertyRef'): return Self()
        if b == 'ValueRange': return [V(t) for t in op.ins]
        return None
    try:
        rv, env = _run_fn(Interp(op), inf[0], inf[1], am)
    except CppUnknown: raise
    except InvariantFail: return None
    except _Crash: return None                                            # 第 82 轮：推断中断言崩溃（同 InvariantFail：该点 op 不合法）
    except RecursionError: raise CppUnknown('递归')
    except Exception as e:
        STATS['python 异常'] += 1; raise CppUnknown(f'解释器异常 {type(e).__name__}: {e}')
    if isinstance(rv, Diag) or (isinstance(rv, LR) and not rv.ok): return None
    res = [nm_ for ty, nm_, _ in _params(inf[0]) if 'SmallVectorImpl' in ty]
    got = env.get(res[0]) if res else out
    if not isinstance(got, list): return None
    got = [x.t if isinstance(x, V) else x for x in got]
    return got if all(isinstance(x, tuple) for x in got) else None

def infer_fn(S):
    """S['cppinfer']：(操作数组类型表, 属性) → 结果类型表 | None | odssig.XF_UNK。"""
    import odssig as G
    dia = S['op'].split('.')[0]
    def f(groups, attrs):
        try: return infer_types(dia, S['op'], S, groups, attrs)
        except CppUnknown: return G.XF_UNK
    return f

def rule_infer(S):
    """nf 原子：InferTypeOpInterface 的结果类型校验，按 C++ inferReturnTypes/isCompatibleReturnTypes 解释执行。→ (rel, 完备?) 或 None。"""
    dia = S['op'].split('.')[0]
    try:
        if infer_source(dia, S['op'])[1] is None: return None
    except Exception: return None
    import nfrules as NF
    refs = [('name', nm) for nm, *_ in S['ins']] + [('name', nm) if nm is not None else ('res', k) for k, (nm, *_) in enumerate(S['outs'])]
    def fn(*a):
        groups, attrs = list(a[:-1]), a[-1]
        try: ok, msg = decide_infer(dia, S['op'], S, groups, attrs)
        except CppUnknown as e: raise NF.NFUnknown(f'cppverify infer: {e}')
        return ok
    ok, _ = completeness(dia, S['op'], 'infer')
    return ('nf', refs, 'nf', 'InferTypeOpInterface（9bae84b0 inferReturnTypes 解释执行）', fn, 'opt'), ok

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

KNOWN_METHODS = TYPE_METHODS | {'assign', 'equals', 'getProperties', *BUILDER_TYPES, 'getNumDims', 'getNumSymbols', 'getNumInputs', 'isPermutation', 'isProjectedPermutation', 'getAffineMap', 'getAttrOfType', 'isEmpty', 'cloneWith', 'getContext', 'isLastDimUnitStride', 'isStrided', 'reserve', 'getType', 'size', 'empty', 'front', 'back', 'getTypes', 'push_back', 'emplace_back', 'append', 'drop_front', 'drop_back',
                                'take_front', 'slice', 'contains', 'isIdentity', 'has_value', 'value', 'getInt', 'getValue', 'getSInt', 'getUInt',
                                'asArrayRef', 'getNumElements', 'failed', 'succeeded', 'getOperation', 'getNumOperands', 'getNumResults', 'getOperand',
                                'getOperands', 'getResults', 'getOperandTypes', 'getResultTypes', 'getResult', 'emitOpError', 'emitError', 'isa', 'cast', 'dyn_cast',
                                'index', 'asInteger', 'desaturate', 'saturated', 'getStridesAndOffset', 'resize', 'begin', 'end', 'getPosition', 'walk', 'getResults'}   # 第 82 轮
@functools.lru_cache(None)
def completeness(dia, opname, which='verify'):
    """静态完备性：→ (完备?, 不支持的名字集合)。递归展开 op 方法与方言内自由函数。
    which='infer'（第 80 轮）：inferReturnTypes 与（若有）isCompatibleReturnTypes 的体。"""
    import odssig as G
    if which == 'infer':
        cls, inf, comp = infer_source(dia, opname)
        body = None if inf is None else inf[1]
        try: [_params(x[0]) for x in (inf, comp) if x]
        except CppUnknown as e: return (False, frozenset({f'形参: {e}'}))
        extra = [comp[1]] if comp else []
    else:
        cls, body = verify_source(dia, opname); extra = []
    CUR['dia'] = dia
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
        lam = set(); _local_lambdas(ast, lam)              # 局部 lambda 变量（auto f = [](…){…}）：体已在 AST 内一并扫描
        for k, n in cs:
            b = _base(n)
            if k == 't':
                if b in ('isa', 'cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'isa_and_nonnull', 'dyn_cast_if_present', 'static_cast', 'SmallVector', 'ArrayRef', 'IsaPred', 'seq', 'min', 'max', 'get', 'to_vector'): continue
                ff = find_function(dia, b)
                if ff is not None: walk(ff[2], b); continue
                bad.add(b); continue
            qn = n.split('::')
            if k == 'fn' and len(qn) == 2 and qn[0].endswith('Op'):
                fms = find_methods(dia, qn[0], qn[1])          # 第 82 轮：全部重载都要完备
                if fms:
                    for i_, fm in enumerate(fms): walk(fm[1], f'{n}#{i_}')
                    continue
            if k == 'fn' and n in lam: continue
            if k == 'fn' and n == 'SaturatedInteger::wrap': continue
            if b in KNOWN_METHODS or b in FREE_FNS - {'getConstantIntValue'} or b in acc: continue
            if td_const(dia, b) is not None: continue
            if re.fullmatch(r'stringify\w+', b): continue
            if k == 'fn' and b.endswith('Type') and b[:1].isupper(): continue
            if k == 'fn' and len(n.split('::')) >= 2 and n.split('::')[-1] == 'get' and (n.split('::')[-2] in CTORS or _nullary_type(n.split('::')[-2], dia) is not None): continue
            if k == 'm' and 'get' + camel(b) + 'Attr' in acc: continue          # adaptor.getProperties().属性
            im = inline_methods(dia, opname).get(b)
            if im is not None: walk(im[1], b); continue
            fms = find_methods(dia, cls, b)                    # 第 82 轮：全部重载
            if fms:
                for i_, fm in enumerate(fms): walk(fm[1], f'{b}#{i_}')
                continue
            ff = find_function(dia, b)
            if ff is not None: walk(ff[2], b); continue
            bad.add(b)
    walk(body, which)
    for i_, b_ in enumerate(extra): walk(b_, f'isCompatibleReturnTypes')
    return (not bad, frozenset(bad))

def _local_lambdas(node, out):
    if isinstance(node, tuple):
        if node and node[0] == 'decl' and node[1][0] == 'var' and isinstance(node[2], tuple) and node[2] and node[2][0] == 'lambda': out.add(node[1][1])
        for x in node: _local_lambdas(x, out)
    elif isinstance(node, list):
        for x in node: _local_lambdas(x, out)

def _scan_stmts(node, bad, tag):
    if isinstance(node, tuple):
        if node and node[0] in ('unreachable',): bad.add(f'{tag}: llvm_unreachable')
        if node and node[0] == 'id' and re.fullmatch(r'(?:\w+::)+\w+', node[1]) and _base(node[1]) not in FREE_FNS and _base(node[1]) != 'kDynamic' \
                and not (node[1].split('::')[-1] == 'get' and (node[1].split('::')[-2] in CTORS or _nullary_type(node[1].split('::')[-2], CUR['dia'] or '') is not None)) \
                and not (node[1].split('::')[-2] in enum_classes() and not enum_classes()[node[1].split('::')[-2]][2]) \
                and not re.fullmatch(r'stringify\w+', _base(node[1])) and _base(node[1]) not in ('tie', 'equal', 'swap', 'zip_first') and node[1] != 'SaturatedInteger::wrap' \
                and not _base(node[1]).startswith(('get', 'verify', 'is', 'compute', 'infer')):
            bad.add(f'限定名 {node[1]}')
        if node and node[0] == 'tid' and _base(node[1]) not in ('isa', 'cast', 'dyn_cast', 'dyn_cast_or_null', 'cast_or_null', 'isa_and_nonnull', 'dyn_cast_if_present', 'static_cast', 'SmallVector', 'ArrayRef', 'IsaPred', 'seq', 'min', 'max', 'get', 'to_vector') \
                and find_function(CUR['dia'] or '', _base(node[1])) is None:
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
            fms = find_methods(dia, cls, b)                    # 第 82 轮：全部重载
            if fms:
                for i_, fm in enumerate(fms): walk(fm[1], f'{b}#{i_}')
                continue
            ff = find_function(dia, b)
            if ff is not None: walk(ff[2], b)
    walk(body, 'verify')
    return frozenset(out)


# ───────────────────────── 第 78 轮：ODS 属性谓词原子（C++ 表达式，$_self 为属性）用同一个解释器求值
ATOM_METHODS = {'getType', 'getElementType', 'isSignlessInteger', 'isInteger', 'size', 'empty', 'asArrayRef', 'getValue', 'isNegative', 'getInt',
                'getZExtValue', 'getSExtValue', 'getNumElements', 'getRank', 'getShape', 'isIndex', 'isZero', 'isOne', 'isStrictlyPositive', 'isNonNegative',
                'getWidth', 'isa', 'cast', 'dyn_cast', 'isSignedInteger', 'isUnsignedInteger', 'isF32', 'isF64', 'isF16', 'isBF16', 'isIntOrFloat', 'hasRank'}
ATOM_FNS = {'ArrayRef', 'SmallVector', 'isa', 'cast', 'dyn_cast', 'all_of', 'any_of', 'none_of', 'is_sorted', 'size', 'count', 'is_contained', 'isPermutationVector', 'isPowerOf2_32', 'isPowerOf2_64'}

TYPE_XF_METHODS = {'getElementType', 'getRank', 'getDimSize', 'getShape', 'setDim', 'setShape', 'getOperand', 'getNumElements', 'size'}
@functools.lru_cache(None)
def compile_type_xform(txt):
    """ODS TypesMatchWith 的 transformer（C++ 表达式，$_self 为类型）→ f(类型) -> 类型/类型表；静态不完备 ⇒ None。运行时不支持 ⇒ CppUnknown。"""
    src = txt.replace('$_self', '__self')
    try:
        p = P(lex(src)); e = p.expr()
        if p.pk()[0] != 'eof': return None
    except CppUnknown: return None
    cs = set(); _calls(e, cs)
    for k, n in cs:
        b = _base(n)
        if k == 'm' and b not in TYPE_XF_METHODS: return None
        if k == 'fn' and b not in ('Builder',): return None
        if k == 't' and b not in ('cast',): return None
    def f(t):
        it = Interp(Op.__new__(Op), 0)
        v = it.ev(e, {'__self': t})
        if isinstance(v, V): v = v.t
        if isinstance(v, tuple) and v and isinstance(v[0], str): return v
        raise CppUnknown('变换值')
    return f

def _tids(node, out):
    if isinstance(node, tuple):
        if node and node[0] == 'tid' and _base(node[1]) in ('isa', 'cast', 'dyn_cast', 'isa_and_nonnull', 'dyn_cast_or_null'): out.extend(node[2])
        for x in node: _tids(x, out)
    elif isinstance(node, list):
        for x in node: _tids(x, out)

def _type_class_known(cls):
    """类型类是封闭且可判定的：内建类、ODS TypeDef、C++ 手写类型（printType 拼写）、MMAMatrixType。接口不算。"""
    import odsjson as O
    b = _base(cls)
    if 'Interface' in b: return False
    if b in TYPE_CLASSES or b in cpp_type_spellings(): return True
    m_ = re.fullmatch(r'(?:::)?(?:mlir::)?(\w+)::(\w+)', str(cls).strip())
    return bool(m_) and O._typedef_isa(m_.group(1), b) is not None

@functools.lru_cache(None)
def compile_type_atom(txt):
    """ODS 类型约束的 CPred 原子（$_self 为类型）→ f(类型) -> bool；静态不完备 ⇒ None。
    运行时 CppUnknown ⇒ True（与未登记原子同一上近似方向：不据此拒绝），并计数 STATS['类型原子运行时未定']。"""
    src = txt.replace('$_self', '__self')
    try:
        p = P(lex(src)); e = p.expr()
        if p.pk()[0] != 'eof': return None
    except CppUnknown: return None
    cs = set(); _calls(e, cs)
    tids = []; _tids(e, tids)
    if any(not _type_class_known(c) for c in tids): return None      # 开放接口（TypeInterface）等：留作 unknown，不据此拒绝
    for k, n in cs:
        b = _base(n)
        if k == 'm' and b not in ATOM_METHODS | TYPE_METHODS: return None
        if k in ('fn', 't') and b not in ATOM_FNS: return None
    def f(t):
        try:
            v = Interp(Op.__new__(Op), 0).ev(e, {'__self': t})
            if isinstance(v, (int, bool)): return bool(v)
            raise CppUnknown('原子值')
        except CppUnknown:
            STATS['类型原子运行时未定'] = STATS.get('类型原子运行时未定', 0) + 1; return True
        except _Ret: return False                     # cast 失败（断言崩溃）⇒ 拒绝
    return f

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
