"""ODS 导入（第 9 轮）：用与被测 MLIR 同一提交的 llvm-tblgen -dump-json 读取 ODS 记录，替代第 1–8 轮的正则解析。
原因：(1) 正则解析器只能处理见过的写法；(2) 之前的 .td 来自 llvm main，而被测 verifier 来自 wheel 的提交 9bae84b0，
两者有版本偏差（例：main 中 arith.addi 用 Arith_SignlessIntegerOrIndexLike，wheel 中是 SignlessIntegerOrIndexLike）。
wheel 自带它编译时用的 .td（mlir/_mlir_libs/include），tblgen 取自同一提交的 mlir_native_tools。

本模块只做两件事：
  load(dialect)            → 该方言全部 op 的 ODS 记录（tblgen 求值后的结构，不是文本）
  pred(record) / Unknown   → 把 Pred 树（CPred / And / Or / Neg / Concat / SubstLeaves）解释为 Python 谓词；
                             遇到原子表 ATOM / 变换表 XFORM 里没有的 C++ 片段 ⇒ 抛 Unknown(片段)，由调用者计入“须人工登记”。
"""
import json, os, re, subprocess, functools, glob

SITE = os.path.expanduser('~/.local/lib/python3.12/site-packages/mlir/_mlir_libs/include')
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = glob.glob(f'{_ROOT}/tools/mlir_native_tools-*.data/data/bin') + glob.glob('/tmp/nt/mlir_native_tools-*.data/data/bin')   # scripts/fetch_tools.sh
CACHE = '/tmp/odsjson'
DIALECTS = {'arith': 'Arith/IR/ArithOps', 'math': 'Math/IR/MathOps', 'index': 'Index/IR/IndexOps', 'complex': 'Complex/IR/ComplexOps',
            'tensor': 'Tensor/IR/TensorOps', 'memref': 'MemRef/IR/MemRefOps', 'scf': 'SCF/IR/SCFOps', 'func': 'Func/IR/FuncOps',
            'cf': 'ControlFlow/IR/ControlFlowOps', 'vector': 'Vector/IR/VectorOps', 'shape': 'Shape/IR/ShapeOps', 'ub': 'UB/IR/UBOps'}

@functools.lru_cache(None)
def records(dialect):
    os.makedirs(CACHE, exist_ok=True); out = f'{CACHE}/{dialect}.json'
    if not os.path.exists(out):
        subprocess.run([TOOLS[0] + '/llvm-tblgen', '-dump-json', '-I', SITE, f'{SITE}/mlir/Dialect/{DIALECTS[dialect]}.td', '-o', out], check=True)
    return json.load(open(out))

def ref(v): return v['def'] if isinstance(v, dict) and v.get('kind') == 'def' else None

def ops(dialect):
    d = records(dialect); res = {}
    for n in d['!instanceof']['Op']:
        r = d[n]; dia = d[ref(r['opDialect'])]['name']
        if dia != dialect: continue
        res[f'{dia}.{r["opName"]}'] = n
    return res

def dag_args(v):
    """(ins …)/(outs …)/(region …) dag → [(约束记录名, 名字)]"""
    return [(ref(c) if ref(c) else c, nm) for c, nm in (v or {}).get('args', [])]

# ── Pred 树解释 ──────────────────────────────────────────────────────────────
class Unknown(Exception): pass
WS = re.compile(r'\s+')
def norm(s): return WS.sub('', s)

def interp(d, name, atom, xform):
    """返回 f(env) → bool；env 把 C++ 标识符（'$_self'、lambda 形参）映射到类型值。"""
    r = d[name]; sc = r['!superclasses']
    if 'CPred' in sc:
        k = norm(r['predExpr'])
        if k not in atom: raise Unknown(('atom', r['predExpr'].strip()))
        a = atom[k]; return lambda env: a(env['$_self'])
    kids = lambda: [interp(d, ref(c), atom, xform) for c in r['children']]
    if 'And' in sc:
        cs = kids(); return lambda env: all(c(env) for c in cs)
    if 'Or' in sc:
        cs = kids(); return lambda env: any(c(env) for c in cs)
    if 'Neg' in sc:
        c = interp(d, ref((r.get('children') or [r.get('child')])[0]), atom, xform); return lambda env: not c(env)
    if 'SubstLeaves' in sc:
        c = interp(d, ref((r.get('children') or [r.get('child')])[0]), atom, xform); pat, rep = r['pattern'], r['replacement'].strip()
        if pat != '$_self': raise Unknown(('subst', pat))
        if re.fullmatch(r'[A-Za-z_]\w*', rep): return lambda env: c({**env, '$_self': env[rep]})
        k = norm(rep)
        if k not in xform: raise Unknown(('xform', rep))
        x = xform[k]; return lambda env: (lambda v: v is not None and c({**env, '$_self': v}))(x(env['$_self']))
    if 'Concat' in sc:
        m = re.fullmatch(r'\[\]\(::mlir::Type(\w+)\)\{return', norm(r['prefix']))
        n = re.fullmatch(r';\}\((.*)\)', norm(r['suffix']))
        if not (m and n): raise Unknown(('concat', r['prefix'].strip() + ' … ' + r['suffix'].strip()))
        if n.group(1) not in xform: raise Unknown(('xform', n.group(1)))
        c = interp(d, ref((r.get('children') or [r.get('child')])[0]), atom, xform); var, x = m.group(1), xform[n.group(1)]
        return lambda env: (lambda v: v is not None and c({**env, var: v}))(x(env['$_self']))
    if 'TruePred' in sc or name == 'TruePred': return lambda env: True
    raise Unknown(('pred', name))

def type_pred(dialect, cname, atom, xform):
    """类型约束记录 → t → bool。"""
    d = records(dialect); r = d[cname]
    p = ref(r.get('predicate'))
    if p is None: raise Unknown(('noconstraint', cname))
    f = interp(d, p, atom, xform); return lambda t: f({'$_self': t})

def leaves(d, name, out):
    """收集 Pred 树中全部 CPred 文本与 Concat 形状（只做统计，不解释）。"""
    r = d[name]; sc = r['!superclasses']
    if 'CPred' in sc: out.add(('atom', r['predExpr'].strip())); return
    if 'Concat' in sc: out.add(('concat-xform', re.sub(r'\s+', ' ', r['suffix']).strip()))
    if 'SubstLeaves' in sc and not re.fullmatch(r'[A-Za-z_]\w*', r['replacement'].strip()): out.add(('subst-xform', r['replacement'].strip()))
    for k in ('child',):
        if ref(r.get(k)): leaves(d, ref(r[k]), out)
    for c in r.get('children') or []: leaves(d, ref(c), out)

# ── 内建类型模型的原子（与方言无关）：C++ 片段 → 类型项上的谓词 ────────────────────
# 11 个方言、305 个 op 的全部端口类型约束只用到下面这些原子和 2 个变换（odsjson_test L0 统计）。
# 它们是内建类型系统的查询，不是任何方言的语义；登记一次，所有方言共用。
from typesig import shaped, ranked, elt as _elt
_I = lambda t: t[0] == 'i'
ATOM_RE = [   # (正则, 由匹配组生成谓词)；作用于规范化（去空白）后的文本
    (r'\(true\)', lambda: lambda t: True),
    (r'\(\$_self\.hasTrait<::mlir::ValueSemantics>\(\)\)', lambda: lambda t: t[0] in ('tensor', 'vector')),
    (r'\(::llvm::isa<::mlir::FloatType>\(\$_self\)\)', lambda: lambda t: t[0] == 'f'),
    (r'\(::llvm::isa<::mlir::IndexType>\(\$_self\)\)', lambda: lambda t: t[0] == 'index'),
    (r'\(\$_self\.isSignlessIntOrIndex\(\)\)', lambda: lambda t: (_I(t) and t[2] == '') or t[0] == 'index'),
    (r'\(::llvm::isa<::mlir::VectorType>\(\$_self\)\)', lambda: lambda t: t[0] == 'vector'),
    (r'\(::llvm::isa<::mlir::FixedVectorType>\(\$_self\)\)', lambda: lambda t: t[0] == 'vector'),
    (r'\(::llvm::isa<::mlir::ScalableVectorType>\(\$_self\)\)', lambda: lambda t: False),          # 类型池不含可伸缩向量
    (r'\(::llvm::isa<::mlir::RankedTensorType>\(\$_self\)\)', lambda: lambda t: t[0] == 'tensor' and t[1] is not None),
    (r'\(::llvm::isa<::mlir::UnrankedTensorType>\(\$_self\)\)', lambda: lambda t: t[0] == 'tensor' and t[1] is None),
    (r'\(::llvm::isa<::mlir::TensorType>\(\$_self\)\)', lambda: lambda t: t[0] == 'tensor'),
    (r'\(::llvm::isa<::mlir::MemRefType>\(\$_self\)\)', lambda: lambda t: t[0] == 'memref' and t[1] is not None),
    (r'\(::llvm::isa<::mlir::UnrankedMemRefType>\(\$_self\)\)', lambda: lambda t: t[0] == 'memref' and t[1] is None),
    (r'\(::llvm::isa<::mlir::BaseMemRefType>\(\$_self\)\)', lambda: lambda t: t[0] == 'memref'),
    (r'\(::llvm::isa<::mlir::ShapedType>\(\$_self\)\)', lambda: shaped),
    (r'\(::llvm::isa<::mlir::ComplexType>\(\$_self\)\)', lambda: lambda t: t[0] == 'complex'),
    (r'\(::llvm::isa<::mlir::IntegerType>\(\$_self\)\)', lambda: _I),
    (r'\(::llvm::isa<::mlir::FunctionType>\(\$_self\)\)', lambda: lambda t: t[0] == 'fn'),
    (r'\(::llvm::isa<::mlir::shape::(\w+)Type>\(\$_self\)\)', lambda k: lambda t: t[0] == 'shape.' + {'Shape': 'shape', 'Size': 'size', 'Witness': 'witness', 'ValueShape': 'value_shape'}.get(k, k)),
    (r'\(\$_self\.isSignlessInteger\(\)\)', lambda: lambda t: _I(t) and t[2] == ''),
    (r'\(\$_self\.isSignlessInteger\((\d+)\)\)', lambda n: lambda t: t == ('i', int(n), '')),
    (r'\(\$_self\.isUnsignedInteger\(\)\)', lambda: lambda t: _I(t) and t[2] == 'u'),
    (r'\(\$_self\.isSignedInteger\(\)\)', lambda: lambda t: _I(t) and t[2] == 's'),
    (r'\(::llvm::cast<::mlir::ShapedType>\(\$_self\)\.hasRank\(\)\)', lambda: ranked),
    (r'\(::llvm::cast<::mlir::ShapedType>\(\$_self\)\.hasStaticShape\(\)\)', lambda: lambda t: ranked(t) and None not in t[1]),
    (r'\(::llvm::cast<::mlir::(?:Shaped|Vector)Type>\(\$_self\)\.getRank\(\)(==|>=|>)(\d+)\)',
     lambda op, n: lambda t: ranked(t) and {'==': len(t[1]) == int(n), '>=': len(t[1]) >= int(n), '>': len(t[1]) > int(n)}[op]),
    (r'\(::llvm::cast<::mlir::MemRefType>\(\$_self\)\.isStrided\(\)\)', lambda: lambda t: True),       # 类型池的 memref 都是恒等布局
]
XFORM = {norm('::llvm::cast<::mlir::ShapedType>($_self).getElementType()'): lambda t: t[2] if shaped(t) else None,
         norm('::llvm::cast<::mlir::ComplexType>($_self).getElementType()'): lambda t: t[1] if t[0] == 'complex' else None}
class _AtomTable(dict):
    def __contains__(self, k): return self.get(k) is not None
    def get(self, k, default=None):
        for rx, mk in ATOM_RE:
            m = re.fullmatch(rx, k)
            if m: return mk(*m.groups())
        return default
    def __getitem__(self, k):
        v = self.get(k)
        if v is None: raise KeyError(k)
        return v
ATOM = _AtomTable()
