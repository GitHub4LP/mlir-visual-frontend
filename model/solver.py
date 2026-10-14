"""最小推导原型：合一 + 谓词归约 + ∃ 消去 + 蕴含查询。
维度采用"运行期 extent"建模：静态类型中的 ? 只是 extent 的投影，约束一律施加在 extent 上。
"""
import itertools
_c = itertools.count()

class V:
    def __init__(s, sort, hint=''): s.sort, s.hint, s.id = sort, hint, next(_c)
    def __repr__(s): return f'{s.hint or s.sort}{s.id}'
class C:
    def __init__(s, head, *args): s.head, s.args = head, list(args)
    def __repr__(s): return f'{s.head}({", ".join(map(repr, s.args))})' if s.args else s.head

class Store:
    def __init__(s): s.sub, s.preds = {}, []
    def find(s, t):
        while isinstance(t, V) and t in s.sub: t = s.sub[t]
        return t
    def res(s, t):
        t = s.find(t)
        return C(t.head, *[s.res(a) for a in t.args]) if isinstance(t, C) else t
    def occurs(s, v, t):
        t = s.find(t)
        return t is v or (isinstance(t, C) and any(s.occurs(v, a) for a in t.args))
    def unify(s, a, b, why):
        a, b = s.find(a), s.find(b)
        if a is b: return
        if isinstance(a, V):
            if s.occurs(a, b): raise TypeError(f'occurs: {why}')
            s.sub[a] = b; return
        if isinstance(b, V): return s.unify(b, a, why)
        if isinstance(a, int) or isinstance(b, int):
            if a != b: raise TypeError(f'dim 冲突 {a}≠{b}: {why}')
            return
        if a.head != b.head or len(a.args) != len(b.args):
            raise TypeError(f'构造子冲突 {a.head}≠{b.head}: {why}')
        for x, y in zip(a.args, b.args): s.unify(x, y, why)

# ---- 谓词：规范化原子（type-algebra.md §2）。返回 True / False / 残余谓词列表
FLOATS = {'f16', 'bf16', 'f32', 'f64'}
SHAPED_VALSEM = {'rtensor', 'vector'}           # 内置 valsem（另有 utensor）
def reduce(st, p):
    name, t = p[0], st.find(p[1])
    if isinstance(t, V): return [(name, t)]      # 未定：保留为残余原子
    if name == 'Float':      return t.head in FLOATS
    if name == 'FloatLike':  # Fl ∨ (valsem ∧ shaped ∧ elt:Fl)
        if t.head in FLOATS: return True
        if t.head in SHAPED_VALSEM: return reduce(st, ('Float', t.args[-1]))
        return False
    if name == 'Numeric':    # linalg 体内 cast 的前提：来自 C++，登记为声明原子（D）
        return t.head in FLOATS | {'i8', 'i16', 'i32', 'i64'}
    if name == 'RankedTensor': return t.head == 'rtensor'
    raise KeyError(name)

def solve(st):
    out = []
    for p in st.preds:
        r = reduce(st, p)
        if r is False: raise TypeError(f'谓词不满足: {p[0]}({st.res(p[1])})')
        if r is not True: out += r
    st.preds = out

# ---- Schema：输入端口类型、输出端口类型、谓词；每次实例化都使用新鲜变量
def tensor(*dims, elt): return C('rtensor', *dims, elt)
def schema_addf():          # ODS：Arith_FloatBinaryOp = SameOperandsAndResultType ∧ FloatLike
    T = V('type', 'T'); return [T, T], [T], [('FloatLike', T)]
schema_mulf = schema_addf
def schema_matmul():        # ODS 只写了 AnyType/AnyShaped，形状关系由 indexing_maps 导入（Presburger，E）
    m, n, k = (V('dim', x) for x in 'mnk')
    ta, tb, tc = (V('type', x) for x in ('Ta', 'Tb', 'Tc'))
    A, B, Cc = tensor(m, k, elt=ta), tensor(k, n, elt=tb), tensor(m, n, elt=tc)
    return [A, B, Cc], [Cc], [('Numeric', ta), ('Numeric', tb), ('Numeric', tc)]

def derive(nodes, edges, inputs, outputs):
    """nodes: {名字: schema}；edges: (源节点, 输出号, 目标节点, 输入号)；
    inputs: 边界输入 → (节点, 输入号)，可一对多；outputs: 边界输出 → (节点, 输出号)。"""
    st, inst = Store(), {}
    for nm, sch in nodes.items():
        ins, outs, preds = sch(); inst[nm] = (ins, outs); st.preds += preds
    for a, i, b, j in edges: st.unify(inst[a][1][i], inst[b][0][j], f'{a}.out{i}→{b}.in{j}')
    bi = {}
    for p, uses in inputs.items():
        bi[p] = V('type', p)
        for b, j in uses: st.unify(bi[p], inst[b][0][j], f'{p}→{b}.in{j}')
    bo = {p: inst[a][1][i] for p, (a, i) in outputs.items()}
    solve(st)
    # ∃ 消去：边界项中出现的变量保留，其余残余谓词若只涉及内部变量，则检查后丢弃
    sig_in = {p: st.res(t) for p, t in bi.items()}
    sig_out = {p: st.res(t) for p, t in bo.items()}
    fv = set()
    def collect(t):
        if isinstance(t, V): fv.add(t)
        elif isinstance(t, C): [collect(a) for a in t.args]
    for t in [*sig_in.values(), *sig_out.values()]: collect(t)
    phi = []
    for q in st.preds:   # 去重 + 格上吸收：Float ⊑ Numeric，Float ⊑ FloatLike
        q = (q[0], st.find(q[1]))
        if q[1] in fv and q not in phi: phi.append(q)
    stronger = {'Numeric': 'Float', 'FloatLike': 'Float'}
    phi = [q for q in phi if (stronger.get(q[0]), q[1]) not in phi]
    return sig_in, sig_out, phi, sorted(fv, key=lambda v: v.id)

def same_operands_and_result_type(sig):   # trait = 对 Φ 的蕴含查询
    ts = [*sig[0].values(), *sig[1].values()]
    return all(repr(t) == repr(ts[0]) for t in ts)

def show(name, sig):
    i, o, phi, fv = sig
    print(f'{name}: ∀ {", ".join(map(repr, fv))}')
    print('   in :', ', '.join(f'{k}: {v}' for k, v in i.items()))
    print('   out:', ', '.join(f'{k}: {v}' for k, v in o.items()))
    print('   Φ  :', ' ∧ '.join(f'{p[0]}({p[1]})' for p in phi) or '⊤')
    print('   ⊨ SameOperandsAndResultType:', same_operands_and_result_type(sig))

if __name__ == '__main__':
    # 1 axpy: r = a*x + y
    axpy = derive({'mul': schema_mulf, 'add': schema_addf},
                  [('mul', 0, 'add', 0)],
                  {'a': [('mul', 0)], 'x': [('mul', 1)], 'y': [('add', 1)]}, {'r': ('add', 0)})
    show('axpy', axpy)
    # 2 matmul + bias: r = addf(matmul(A,B,C0), bias)
    mb = derive({'mm': schema_matmul, 'add': schema_addf},
                [('mm', 0, 'add', 0)],
                {'A': [('mm', 0)], 'B': [('mm', 1)], 'C0': [('mm', 2)], 'bias': [('add', 1)]},
                {'r': ('add', 0)})
    show('matmul_bias', mb)
    # 3 负例：把 bias 接到 A 上（m×k 与 m×n 合一，迫使 k=n；这是合法收窄，不是错误）
    neg = derive({'mm': schema_matmul, 'add': schema_addf}, [('mm', 0, 'add', 0)],
                 {'A': [('mm', 0), ('add', 1)], 'B': [('mm', 1)], 'C0': [('mm', 2)]}, {'r': ('add', 0)})
    show('matmul_plus_A', neg)
    # 4 真错误：把整数张量常量类型强加给 addf
    try:
        st = Store(); i, o, p = schema_addf(); st.preds += p
        st.unify(i[0], tensor(4, elt=C('i32')), 'bind'); solve(st)
    except TypeError as e: print('addf(tensor<4xi32>) 拒绝 →', e)
