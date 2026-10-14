"""统一核心 v3：Schema 是唯一的数据结构；原生 op（ODS 导入 + 语义覆盖层）与复合 op（推导）同构。
整数层（标量值、index、extent）用 isl；类型层见 solver.py（此处只沿用 ODS 的约束名）。
每个字段带信任等级：ods（导入）/ authored（人工语义）/ derived（推导）。
"""
import re, itertools, islpy as isl
from dataclasses import dataclass, field
from ods_import import import_op, KNOWN

BIG = 10**6
_n = itertools.count()

@dataclass
class Schema:
    name: str
    ins: list            # [(端口, 论域上界)]
    outs: list
    sem: str = 'true'    # 关于 {in}/{out} 的 Presburger 公式
    ub: str = 'false'
    poison: dict = field(default_factory=dict)   # out -> 公式
    sensitive: set = field(default_factory=set)
    eff: frozenset = frozenset()
    traits_ods: list = field(default_factory=list)
    trust: dict = field(default_factory=dict)
    prop: dict = None    # out -> 毒值传播公式，可引用 {P:port}；None 表示任一输入是毒值则输出是毒值
    exact: bool = True   # Sem 是否精确（False：被抽象成过近似，比如非线性乘法）

    def inst(s, binding):
        """binding: 端口 → 变量名；返回把占位符替换后的 (sem, ub, poison)。"""
        sub = lambda f: re.sub(r'\{(\w+)\}', lambda m: binding[m.group(1)], f)
        return sub(s.sem), sub(s.ub), {o: sub(p) for o, p in s.poison.items()}

# ---- 原生 op：ODS 事实 + 人工语义覆盖层（authored）
ARITH = open('inventory/Arith_ArithOps.td').read(); TENSOR = open('inventory/Tensor_TensorOps.td').read()
U8 = 255
def native(odsname, src, ins, outs, sem, ub='false', poison=None, sensitive=()):
    o = import_op(src, odsname, KNOWN)
    return Schema(odsname, ins, outs, sem, ub, poison or {}, set(sensitive), frozenset(), o['traits'],
                  {'traits': 'ods', 'sem/ub/poison': 'authored'})
LIB = {
  'addi_nuw': native('Arith_AddIOp', ARITH, [('lhs', U8), ('rhs', U8)], [('r', U8)],
                     '{r} = ({lhs} + {rhs}) mod 256', poison={'r': '{lhs} + {rhs} > 255'}),
  'subi':     native('Arith_SubIOp', ARITH, [('lhs', U8), ('rhs', U8)], [('r', U8)], '{r} = ({lhs} - {rhs}) mod 256'),
  'divui':    native('Arith_DivUIOp', ARITH, [('lhs', U8), ('rhs', U8)], [('r', U8)],
                     '0 <= {r} <= {lhs}', ub='{rhs} = 0', sensitive=['rhs']),
  'dim':      native('Tensor_DimOp', TENSOR, [('extent', BIG)], [('r', BIG)], '{r} = {extent}'),
  'extract_slice': native('Tensor_ExtractSliceOp', TENSOR, [('d', BIG), ('o', BIG), ('n', BIG)], [('r', BIG)],
                     '{r} = {n}', poison={'r': '{o} + {n} > {d}'}),
  # tensor.extract：语义覆盖层故意写成“越界即 UB”，用来演示与 ODS 的 Pure 冲突会被审计出来
  'extract':  native('Tensor_ExtractOp', TENSOR, [('t', BIG), ('i', BIG)], [('r', U8)],
                     '0 <= {r} <= 255', ub='{i} >= {t}', sensitive=['t', 'i']),
}

# ---- 复合 op 推导：结果仍是 Schema，可以再作为节点使用
def S(vs, cs): return isl.Set('{ [%s] : %s }' % (', '.join(vs), ' and '.join(f'({c})' for c in cs) or 'true'))
def as_formula(st, names):
    """把 isl 集合转成以 {port} 为占位符的公式文本，供再次实例化。"""
    st = st.coalesce()
    for i, n in enumerate(names): st = st.set_dim_name(isl.dim_type.set, i, f'P_{n}')
    if st.is_empty(): return 'false'
    parts = []
    for bs in st.get_basic_sets():                     # 并集按 basic set 逐个转成析取
        t = str(bs); parts.append(t[t.index(':') + 1:t.rindex('}')].strip() if ':' in t else 'true')
    txt = re.sub(r'(\d)P_', r'\1*P_', ' or '.join(f'({x})' for x in parts))    # isl 会打印成 3P_b 这样的并置形式
    return re.sub(r'(?<![A-Za-z0-9_])P_(\w+)\b', r'{\1}', txt)

def derive(name, ins, outs, nodes):
    """ins: [(端口, 上界)]；nodes: [(id, schema, {输入端口: 来源})]，来源是边界端口名或 'id.out'；outs: {端口: 'id.out'}"""
    var = {p: f'{p}_{next(_n)}' for p, _ in ins}; dom = [f'0 <= {var[p]} <= {h}' for p, h in ins]
    P = {p: 'false' for p, _ in ins}; sem, ub = [], []
    for nid, sch, wire in nodes:
        b = {port: var[src] for port, src in wire.items()}
        for o, h in sch.outs:
            key = f'{nid}.{o}'; var[key] = f'{nid}_{o}_{next(_n)}'; b[o] = var[key]; dom.append(f'0 <= {var[key]} <= {h}')
        s, u, po = sch.inst(b); sem.append(f'({s}) or ({u})'); ub.append(u)   # Sem 在 UB 处取全关系，保证 Sem 是全的
        pin = ' or '.join(f'({P[src]})' for src in wire.values()) or 'false'
        for o, _ in sch.outs:
            if sch.prop and o in sch.prop:
                pin = re.sub(r'\{P:(\w+)\}', lambda m: f'({P[wire[m.group(1)]]})', sch.prop[o])
                pin = re.sub(r'\{(\w+)\}', lambda m: b[m.group(1)], pin)
            P[f'{nid}.{o}'] = f'({pin}) or ({po.get(o, "false")})'
        ub += [P[wire[x]] for x in sch.sensitive]
    bnd = [var[p] for p, _ in ins]; ov = [var[outs[o]] for o, _ in outs_list(outs, nodes)]
    rest = [v for v in var.values() if v not in bnd and v not in ov]
    order = bnd + ov + rest; nb, no = len(bnd), len(ov)
    proj = lambda X, k: X.project_out(isl.dim_type.set, k, len(order) - k)
    semS = S(order, sem + dom)
    bad = proj(semS.intersect(S(order, [' or '.join(f'({u})' for u in ub)])), nb)
    domb = proj(S(order, dom), nb); pre = domb.subtract(bad)
    names_in = [p for p, _ in ins]; names_out = list(outs)
    rel = proj(semS, nb + no)                          # 不能限制到 Pre 上：否则外层会看不到 UB 输入（v3 初版的 bug）
    poison = {o: as_formula(proj(semS.intersect(S(order, [P[outs[o]]])), nb).intersect(pre), names_in) for o in outs}
    sch = Schema(name, ins, [(o, BIG) for o in outs], as_formula(rel, names_in + names_out),
                 as_formula(domb.subtract(pre), names_in), poison, set(), frozenset(), [],
                 {'*': 'derived'})
    sch.exact = all(n[1].exact for n in nodes)
    return sch, pre

def outs_list(outs, nodes): return [(o, None) for o in outs]

def caps(s, ins_dom):
    spec = s.ub == 'false'
    return {'Speculatable': spec, 'Pure': spec and not s.eff,
            'MayPoison': {o: p for o, p in s.poison.items() if p != 'false'}}

def audit(lib):
    """ODS 声明的 caps 与人工语义是否一致：声明 Pure/AlwaysSpeculatable 却有非空 UB ⇒ 冲突。"""
    for k, s in lib.items():
        if ('Pure' in s.traits_ods or 'AlwaysSpeculatable' in s.traits_ods) and s.ub != 'false':
            print(f'  ✗ {k}: ODS 声明 Pure，语义覆盖层却有 UB「{s.ub}」 → 二者必有一错')
        elif 'ConditionallySpeculatable' in s.traits_ods and s.ub == 'false':
            print(f'  · {k}: ODS 为 ConditionallySpeculatable，语义上恒可推测 → 可以加强')
        else: print(f'  ✓ {k}: {s.traits_ods}')

if __name__ == '__main__':
    print('■ ODS caps 与语义覆盖层的一致性审计'); audit(LIB)

    # 闭包测试：内层复合 op 当作节点使用，与“全部内联”的推导结果比较
    inner, _ = derive('sum_nuw', [('a', U8), ('b', U8)], {'s': 'add.r'},
                      [('add', LIB['addi_nuw'], {'lhs': 'a', 'rhs': 'b'})])
    print('■ 内层 sum_nuw:', caps(inner, None), ' sem:', inner.sem)
    nested, pre_n = derive('f_nested', [('x', U8), ('a', U8), ('b', U8)], {'q': 'div.r'},
                           [('s', inner, {'a': 'a', 'b': 'b'}), ('div', LIB['divui'], {'lhs': 'x', 'rhs': 's.s'})])
    flat, pre_f = derive('f_flat', [('x', U8), ('a', U8), ('b', U8)], {'q': 'div.r'},
                         [('add', LIB['addi_nuw'], {'lhs': 'a', 'rhs': 'b'}), ('div', LIB['divui'], {'lhs': 'x', 'rhs': 'add.r'})])
    print('■ x / (a +nuw b)  Pre(嵌套):', pre_n.coalesce())
    print('  嵌套推导 ≡ 全内联推导:', pre_n.is_equal(pre_f.reset_space(pre_n.get_space())), ' caps:', caps(nested, None))
    # 再套一层：复合中的复合中的复合
    g, pre_g = derive('g', [('x', U8), ('a', U8), ('b', U8)], {'q': 'f.q'}, [('f', nested, {'x': 'x', 'a': 'a', 'b': 'b'})])
    print('  三层嵌套 ≡ 全内联:', pre_g.is_equal(pre_f.reset_space(pre_g.get_space())))
