"""第 14 轮：符号层 —— 谁定义符号、谁是符号表、谁使用符号，全部取自 ODS 的 trait/interface（含接口继承），
不再依赖手工字段（node 的 sym / vis / callee / ftype）。

复盘（spec/symbols.md §1）：structure.py 的 S1–S4 只认手工字段；irimport 导入的真实 IR 把属性原样放在 props['_attrs']，
从不设置这些字段 ⇒ 第 12、13 轮 C2 “结构层误拒 0” 在符号义务上是空检验（真实 IR 上 S1–S4 从未触发）。

分层：
  MLIR 核心机制（SymbolTable.cpp，与方言无关）：
    定义者 = 实现 SymbolOpInterface 的 op（FunctionOpInterface 等经 baseInterfaces 继承）；名字 = sym_name，可见性 = sym_visibility；
    符号表 = 带 SymbolTable trait 的 op（builtin.module 不在 12 方言的 .td 中，按 MLIR 内建登记）；
    S1 同表重名；扁平引用在“使用者的最近符号表”里解析，嵌套引用 @a::@b 逐表下行。
  方言的使用义务（SymbolUserOpInterface::verifySymbolUses，C++）：declared 账，见 USES。
    使用者的集合取自 ODS（实现 SymbolUserOpInterface 的 op）；USES 未登记的使用者 ⇒ 不判（上近似），记账。
"""
import functools, re
import odsjson as O

BUILTIN_TABLES = {'builtin.module'}          # 内建：ModuleOp 带 SymbolTable、Symbol（可选名）
BUILTIN_SYMBOLS = {'builtin.module'}

@functools.lru_cache(None)
def interfaces(op):
    """op 的全部 trait / interface 名（cppInterfaceName 或 trait 名），含接口继承闭包。未导入 ⇒ None。"""
    dia = op.split('.')[0]
    if dia not in O.DIALECTS or op not in O.ops(dia): return None
    d = O.records(dia); out = set()
    def add(name):
        x = d.get(name)
        if not isinstance(x, dict): out.add(name); return
        if 'TraitList' in (x.get('!superclasses') or []):        # 第 15 轮：TraitList（如 SingleBlockImplicitTerminator）展开成员
            for t in x.get('traits') or []: add(O.ref(t))
            return
        nm = x.get('cppInterfaceName') or x.get('trait') or name
        if nm in out: return
        out.add(nm)
        for b in x.get('baseInterfaces') or []: add(O.ref(b))
    for t in d[O.ops(dia)[op]].get('traits', []): add(O.ref(t))
    return frozenset(out)

def _has(op, k):
    s = interfaces(op)
    return None if s is None else any(k == x or x.endswith('::' + k) for x in s)

def is_symbol(op): return op in BUILTIN_SYMBOLS or bool(_has(op, 'SymbolOpInterface'))
def is_table(op):  return op in BUILTIN_TABLES or bool(_has(op, 'SymbolTable'))
def is_user(op):   return bool(_has(op, 'SymbolUserOpInterface'))

def _unq(s): return s[1:-1] if s and len(s) >= 2 and s[0] == s[-1] == '"' else s

def sym_of(n):
    if n.get('sym'): return n['sym']
    a = n.get('props', {}).get('_attrs')
    if a and 'sym_name' in a and is_symbol(n['op']): return _unq(a['sym_name'])
    return None

def vis_of(n):
    if n.get('vis'): return n['vis']
    a = n.get('props', {}).get('_attrs') or {}
    return _unq(a.get('sym_visibility', '"public"'))

def ftype_of(n):
    if n.get('ftype'): return n['ftype']
    a = n.get('props', {}).get('_attrs') or {}
    return a.get('function_type')

# 使用义务（declared：MLIR 的 C++ verifySymbolUses，按其诊断登记；由 symtab_test 的 Y1 不误拒 / Y2 变异差分审计）
#   (引用属性, 目标 op, 关系)   关系：'call' 操作数/结果类型 = 目标函数类型；'fn' 结果类型 = 目标函数类型；'ty' 结果类型 = 目标的 type 属性
USES = {
    'func.call':         ('callee', 'func.func', 'call'),
    'func.constant':     ('value', 'func.func', 'fn'),
    'memref.get_global': ('name', 'memref.global', 'ty'),
}

def ref_of(n):
    """使用者 → (义务名, 引用文本 '@a' 或 '@a::@b') ；手工字段 callee 视同 func.call。"""
    if n.get('callee'): return n['callee']
    if n['op'] in USES:
        a = n.get('props', {}).get('_attrs') or {}
        return a.get(USES[n['op']][0])
    return None

def ledger():
    """12 方言中的符号定义者 / 符号表 / 使用者，以及使用者中 USES 未登记的（上近似）。"""
    out = {'定义者': [], '符号表': [], '使用者': [], '使用者未登记义务': []}
    for dia in O.DIALECTS:
        for op in O.ops(dia):
            if is_symbol(op): out['定义者'].append(op)
            if is_table(op): out['符号表'].append(op)
            if is_user(op):
                out['使用者'].append(op)
                if op not in USES: out['使用者未登记义务'].append(op)
    return out

@functools.lru_cache(None)
def _cls_index():
    """C++ 类名（去掉方言前缀的 def 名，如 SCF_YieldOp → YieldOp）+ 命名空间 → op 名。"""
    idx = {}
    for dia in O.DIALECTS:
        d = O.records(dia)
        for op, rec in O.ops(dia).items():
            cls = rec.split('_', 1)[1] if '_' in rec else rec
            ns = (d[rec].get('cppNamespace') or '').strip(':').split('::')[-1]
            idx[(ns, cls)] = op; idx.setdefault(('', cls), op)
    return idx

def implicit_terminator(op):
    """SingleBlockImplicitTerminator<X> 的 X 对应的 op 名；没有该 trait ⇒ None；有但解析不出 ⇒ '?'。"""
    for t in interfaces(op) or ():
        m = re.match(r'SingleBlockImplicitTerminator<(.+)>::Impl$', t)
        if m:
            full = m.group(1).split('::'); cls = full[-1]; ns = full[-2] if len(full) > 1 else ''
            ix = _cls_index()
            if not ns:                                  # 未写命名空间 ⇒ 与该 op 同一 C++ 命名空间（shape.reduce 的 YieldOp 是 shape.yield）
                dia = op.split('.')[0]; d = O.records(dia)
                ns = (d[O.ops(dia)[op]].get('cppNamespace') or '').strip(':').split('::')[-1]
            return ix.get((ns, cls)) or '?'
    return None

def func_scopes(g):
    """第 16 轮：每个有体的 func.func → (函数节点, 最近符号表的符号 名字→定义者, 最近 builtin.module 的属性)。
    预言机按此为函数体补模块级符号上下文（oracle.emit_core 的 symbols / mod_attrs）。"""
    out = []
    def rec(n, syms, mattrs):
        if is_table(n['op']):
            syms = {}
            for r in n['regions']:
                for b in r:
                    for m in b['ops']:
                        nm = sym_of(m)
                        if nm is None and ((m.get('props') or {}).get('_attrs') or {}).get('sym_name'):   # 未导入 ODS 的方言（shard.grid…）：
                            nm = _unq(m['props']['_attrs']['sym_name'])  # 只用于给预言机补声明，不用于结构层的拒绝（真值在 MLIR）
                        if nm: syms.setdefault(nm, m)
        if n['op'] == 'builtin.module':
            mattrs = dict((n.get('props') or {}).get('_attrs') or {}); mattrs.pop('sym_name', None); mattrs.pop('sym_visibility', None)
        if n['op'] == 'func.func' and n['regions'] and n['regions'][0]: out.append((n, syms, mattrs))
        for r in n['regions']:
            for b in r:
                for m in b['ops']: rec(m, syms, mattrs)
    rec(g, {}, {})
    return out
