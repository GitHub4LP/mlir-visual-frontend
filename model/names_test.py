"""第 27 轮：清点“按名字共享”的语义身份（spec/names.md）。
第 26 轮的教训：随机证伪看不到“不在文本里的语义来源”，只能逐项清点“名字相同 ⇒ 语义相同”的前提。清点结果：
  N1 编辑洞：旧实现中洞 = 与环境无关的常量 hole.{hid}
     (a) 同一定义调用两次：f(a) ≡ f(b) 被“证明”；填 h:=x 后不成立（JIT 见证）。
     (b) 两个定义里 hid 相同：f(a) ≡ f2(a) 被“证明”。
     修（L20′）：洞 = 以（定义, hid）命名、作用于此处全部已定义值的未解释函数（Hazel 闭包语义）。
     (c) 洞在循环摘要体内：L20 代换看不进摘要（会“证明”fill(1) ≡ fill(0)）⇒ complete() 对这种洞改为真填后重推导。
     (d) 正对照：同一定义同一实参调用两次仍可证相等（洞内容是确定的）。
  N2 端口重名：两个端口共用一个占位项 ⇒ x == y 被“证明”。修：拒绝（值层未知 + 诊断）。
  N3 复合 op 与原生 op 同名：静默覆盖原生语义（REG 只有一个命名空间）。修：拒绝，不登记。
  N4 端口占位项名不单射：定义 'a.b' 的端口 c 与定义 'a' 的端口 b.c 同名。修：长度前缀 + 位置。
  N5 循环摘要名（第 26 轮 L18′）回归：体内有洞时名字带所属定义。
清点为“不构成问题”的（理由见 spec/names.md）：Ctx.env 按 SSA 名（每次推导独立）、fv() 全局计数、FreshConst、
  edit.Workspace 的 Merkle 键（含定义名与被调指纹）、fingerprint 的 H{i}/P.{nm}（只作字符串）。
保留风险：复合 op 参数常量 f'{name}.${nm}' 与 defs.derive_params 共享命名（契约），未做单射化；参数未在调用点给出时才会悬空。
"""
import copy, z3
import core as C, edit as E
from core import blk, node

I = 'i64'
def D(ins, outs, ops, args, mode='transparent'): return dict(ins=ins, outs=outs, mode=mode, blocks=[blk('entry', ops, args=args)])
def proves_true(S):
    v, p = S.outv[0]; return C.prove(z3.Implies(S.ax, z3.Or(p, S.ubf, v == 1)))
def eq_def(name, f1, f2, a1, a2):
    return D([('a', I), ('b', I)], [('e', 'i1')], [node('u', f1, ins=[a1], outs=[('u', I)]), node('w', f2, ins=[a2], outs=[('w', I)]),
             node('e', 'arith.cmpi', ins=['u', 'w'], outs=[('e', 'i1')], props={'predicate': 0}), C.ret('r', 'e')], [('a', I), ('b', I)])

FILLED_MLIR = '''func.func @f(%x: i64) -> i64 { return %x : i64 }
func.func @main(%a: i64, %b: i64) -> i1 {
  %u = func.call @f(%a) : (i64) -> i64
  %w = func.call @f(%b) : (i64) -> i64
  %e = arith.cmpi eq, %u, %w : i64
  return %e : i1
}'''

def N1():
    C.ORACLE_ON = False; out = {}
    for mode in ('transparent', 'opaque'):
        defs = {'f': D([('x', I)], [('y', I)], [E.hole('h', I), C.ret('r', 'h')], [('x', I)], mode),
                'f2': D([('x', I)], [('y', I)], [E.hole('h', I), C.ret('r', 'h')], [('x', I)], mode),
                'g': eq_def('g', 'f', 'f', 'a', 'b'), 'g2': eq_def('g2', 'f', 'f2', 'a', 'a'), 'g3': eq_def('g3', 'f', 'f', 'a', 'a')}
        ws = E.Workspace(defs); s = ws.all()
        out[f'{mode}: (a) f(a)≡f(b) 被证明'] = proves_true(s['g'])
        out[f'{mode}: (b) f(a)≡f2(a) 被证明'] = proves_true(s['g2'])
        out[f'{mode}: (d) f(a)≡f(a) 可证'] = proves_true(s['g3'])
    import valimport_test as T
    out['JIT（填 h:=x 后）main(1,2)'] = T.jit_run(FILLED_MLIR, 'main', ['i64', 'i64'], 'i1', [[1, 2], [3, 3]])
    return out

def N1c():
    """洞在被摘要的循环体内：complete() 的分类 ≡ 逐候选真填后重推导。"""
    C.ORACLE_ON = False
    inner = [blk('b', [E.hole('h', I), node('q', 'arith.addi', ins=['acc', 'h'], outs=[('q', I)]), node('y', 'scf.yield', ins=['q'])],
                 args=[('i', 'index'), ('acc', I)])]
    ops = [node('one', 'arith.constant', outs=[('one', I)], props={'value': 1, 'type': I}),
           node('zero', 'arith.constant', outs=[('zero', I)], props={'value': 0, 'type': I}),
           node('c0', 'arith.constant', outs=[('c0', 'index')], props={'value': 0, 'type': 'index'}),
           node('c1', 'arith.constant', outs=[('c1', 'index')], props={'value': 1, 'type': 'index'}),
           node('u', 'arith.index_cast', ins=['n'], outs=[('u', 'index')]),
           node('f', 'scf.for', ins=['c0', 'u', 'c1', 'x'], outs=[('r', I)], regions=[inner]), C.ret('ret', 'r')]
    ws = E.Workspace({'L': D([('x', I), ('n', I)], [('y', I)], ops, [('x', I), ('n', I)])}); sch = ws.schema('L')
    summarized = ('L', 'h') in sch.hole_sum
    # complete 的候选来自顶层（洞的“第一个使用处”按顶层 op 计），这里直接逐候选对照
    full = {}
    for v in ('one', 'zero'):
        w2 = E.Workspace(ws.defs); E.fill(w2, 'L', 'h', v); full[v] = w2.schema('L')
    sub = {v: E.subst_fill(sch, 'h', *sch.env[v], _unsafe=True) for v in ('one', 'zero')}
    try: E.subst_fill(sch, 'h', *sch.env['one']); refused = False
    except C.ValueUnknown: refused = True
    # 真值：填 1 ⇒ x + n，填 0 ⇒ x（n ≥ 1 时不同）。L20 代换看不进摘要 ⇒ 两种填法得到同一个项
    return {'洞被循环摘要吞掉': summarized,
            '真填：fill(1) ≡ fill(0) 被证明': C.equivalent(full['one'], full['zero']),
            'L20 代换（看不进摘要，强制）：fill(1) ≡ fill(0) 被证明': C.equivalent(sub['one'], sub['zero']),
            'subst_fill 默认拒绝': refused}

def N2():
    S = C.derive('dupp', [('a', I), ('a', I)], [('e', 'i1')],
                 [blk('entry', [node('e', 'arith.cmpi', ins=['x', 'y'], outs=[('e', 'i1')], props={'predicate': 0}), C.ret('r', 'e')], args=[('x', I), ('y', I)])])
    return {'值层': S.value_known, '诊断': getattr(S, 'name_diag', None), '已登记': C.REG.get('dupp') is S}

def N3():
    before = C.REG['arith.addi']
    T = C.derive('arith.addi', [('a', I), ('b', I)], [('r', I)],
                 [blk('entry', [node('s', 'arith.subi', ins=['x', 'y'], outs=[('s', I)]), C.ret('r', 's')], args=[('x', I), ('y', I)])])
    ok = C.REG['arith.addi'] is before
    C.REG['arith.addi'] = before
    return {'原生保持': ok, '诊断': getattr(T, 'name_diag', None)}

def N4():
    U = C.derive('a.b', [('c', I)], [('r', I)], [blk('entry', [C.ret('r', 'x')], args=[('x', I)])])
    V = C.derive('a', [('b.c', I), ('d', I)], [('r', I)], [blk('entry', [C.ret('r', 'x')], args=[('x', I), ('y', I)])])
    return {'占位项同名': U.holes[0][0].eq(V.holes[0][0])}

if __name__ == '__main__':
    r1 = N1(); print('── N1 编辑洞:', r1)
    r1c = N1c(); print('── N1c 洞在循环摘要内:', r1c)
    r2, r3, r4 = N2(), N3(), N4()
    print('── N2 端口重名:', r2); print('── N3 与原生同名:', r3); print('── N4 占位项名单射:', r4)
    ok = (not any(v for k, v in r1.items() if '被证明' in k) and all(v for k, v in r1.items() if '可证' in k)
          and r1['JIT（填 h:=x 后）main(1,2)'] == [0, 1] and r1c['洞被循环摘要吞掉'] and not r1c['真填：fill(1) ≡ fill(0) 被证明'] and r1c['subst_fill 默认拒绝'] and not r2['值层'] and not r2['已登记']
          and r3['原生保持'] and not r4['占位项同名'])
    print('── 结论:', '全部符合' if ok else '有不符')
