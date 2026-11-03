"""第 55 轮：M3.9 ADT 降级后的毒值层外部化（L11 的毒值子句）。

L11 的数据精化：¬UB_A(abs x̂) ⇒ ¬UB_L(x̂) ∧ (¬P_A ⇒ ¬P_L ∧ abs(out_L) = out_A)。A11 只在无毒值输入上检验了值子句。
这里在 MLIR 里给降级后的输入分量 x̂ = (a, b, tag, val) 真正注入毒值（shli x, 位宽），抽象层按抽象函数 abs_fn 算出抽象输入的毒值：
  a/b 毒化 ⇒ 抽象 a/b 毒值；tag 毒化 ⇒ option 毒值；val 毒化 ⇒ 仅当 tag = 1（some）时 option 毒值，none 的载荷是什么都行。
抽象层（z3 Datatype，不经降级）给出确定值 ⇒ 降级程序 JIT（O0 与 O3）经 abs 后必须等于它。

阈值在运行前写定（第 55 轮）：
  Q1 程序 ≥ 300、JIT 失败 0、反例 0；确定值中“抽象输入含毒值”的 ≥ 50（毒值被吸收）；
     确定值中“降级输入含毒值而抽象输入无毒值”（none 的载荷毒化）的 ≥ 50。
  Q2 消融 a：抽象层不看输入毒值（毒值标志代入假），O3 反例 > 0。
  Q3 消融 b（ADT 专属）：抽象函数忽略 tag 的毒值（只看载荷），O3 反例 > 0；若为 0，说明本检验对 tag 毒值不灵敏，照实报告。
用法：PYTHONPATH=model python3 model/adtpoison_test.py [程序数] [每程序输入数]
"""
import random, sys, ctypes, collections, itertools, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
import core as C, defs as D, defs_test as DT, verdict as V
from core import blk, I3
from adtjit_test import emit, PIPE, ALL

W = {'i1': 1, 'i3': 3}
COMP = [('a', 'i3'), ('b', 'i3'), ('t', 'i1'), ('u', 'i3')]

def jit(low, rtys, mask, inputs, opt):
    call_ty = '(i3, i3, i1, i3) -> (' + ', '.join(rtys) + ')'
    rs = ', '.join(f'%r{i}' for i in range(len(rtys)))
    body = []
    args = []
    for i, (nm, t) in enumerate(COMP):
        body.append(f'%{nm}0 = arith.trunci %x{i} : i64 to {t}')
        if mask[i]:
            body += [f'%{nm}s = arith.constant {W[t]} : {t}', f'%{nm}1 = arith.shli %{nm}0, %{nm}s : {t}']; args.append(f'%{nm}1')
        else: args.append(f'%{nm}0')
    body += [f'{rs} = func.call @f({", ".join(args)}) : {call_ty}', '%acc0 = arith.constant 0 : i64']
    for i, t in enumerate(rtys):
        body += [f'%e{i} = arith.extui %r{i} : {t} to i64', f'%s{i} = arith.constant {8 * i} : i64', f'%h{i} = arith.shli %e{i}, %s{i} : i64',
                 f'%acc{i + 1} = arith.ori %acc{i}, %h{i} : i64']
    body.append(f'return %acc{len(rtys)} : i64')
    src = emit(low, rtys) + '\nfunc.func @main(%x0: i64, %x1: i64, %x2: i64, %x3: i64) -> i64 attributes {llvm.emit_c_interface} {\n  ' + '\n  '.join(body) + '\n}'
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify(), src
        PassManager.parse(PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=opt); out = []
        for xs in inputs:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r)
            out.append([(r[0] >> (8 * i)) & ((1 << W[t]) - 1) for i, t in enumerate(rtys)])
        return out

def abs_eval(A, opt, xs, mask, mode='model'):
    """抽象层求值，抽象输入毒值由 abs_fn 的规则给出。mode：model / abl_a（输入毒值全假）/ abl_b（忽略 tag 毒值）。"""
    a, b, t, u = xs
    ov = opt.ctors[1](z3.BitVecVal(u, 3)) if t else opt.ctors[0]()
    pa, pb = bool(mask[0]), bool(mask[1])
    po = (bool(mask[2]) if mode != 'abl_b' else False) or (bool(t) and bool(mask[3]))
    if mode == 'abl_a': pa = pb = po = False
    pairs = []
    for (h, hp), x, p in zip(A.holes, [z3.BitVecVal(a, 3), z3.BitVecVal(b, 3), ov], [pa, pb, po]):
        pairs += [(h, x), (hp, z3.BoolVal(p))]
    sub = lambda f: z3.simplify(z3.substitute(f, pairs))
    if not all(z3.is_true(sub(g)) for _, g in A.guards): return ('skip', '守卫'), (pa or pb or po)
    if not z3.is_false(sub(A.ubf)): return ('skip', 'UB'), (pa or pb or po)
    v, p = A.outv[0]
    if not z3.is_false(sub(p)): return ('skip', '毒值'), (pa or pb or po)
    return ('val', sub(v)), (pa or pb or po)

def agree(val, jo, opt, rty):
    if rty == I3: return z3.is_true(z3.simplify(val == z3.BitVecVal(jo[0], 3)))
    t, u = jo
    return z3.is_true(z3.simplify(val == (opt.ctors[1](z3.BitVecVal(u, 3)) if t else opt.ctors[0]())))

def run(n, k, seed=55):
    opt = D.define_adt('option', [('none', []), ('some', [('val', I3)])])
    rng = random.Random(seed); st = collections.Counter()
    for t in range(n):
        ins, ops, (rv, rty) = DT.gen_opt(rng, 55000 + t)
        prog = [blk('entry', ops + [C.ret('r', rv)], args=ins)]
        A = C.derive(f'AP{t}', ins, [('q', rty)], prog)
        low = C.inline_all(D.lower_blocks(prog)); rtys = opt.lowered if rty == 'option' else [rty]
        mask = [0, 0, 0, 0]
        while not any(mask): mask = [int(rng.random() < 0.4) for _ in range(4)]
        inputs = rng.sample(ALL, k)
        ev = {m: [abs_eval(A, opt, xs, mask, m) for xs in inputs] for m in ('model', 'abl_a', 'abl_b')}
        try: js = {o: jit(low, rtys, mask, inputs, o) for o in (0, 3)}
        except Exception as e: st['JIT 失败'] += 1; print('  JIT 失败', str(e)[:300]); continue
        st['程序'] += 1
        for o in (0, 3):
            for i, xs in enumerate(inputs):
                st['输入'] += 1
                (r, absp) = ev['model'][i]
                lowp = any(mask)                                    # 每个程序至少毒化一个降级分量
                if r[0] == 'skip': st[f'模型 {r[1]}'] += 1
                else:
                    st['确定值'] += 1; st['确定值/抽象输入含毒值'] += absp; st['确定值/仅降级输入含毒值'] += lowp and not absp
                    if agree(r[1], js[o][i], opt, rty): st['一致'] += 1
                    else:
                        st['反例'] += 1
                        if st['反例'] <= 2: print(f'  ✗ O{o} mask={mask} x={xs} 模型={r[1]} JIT={js[o][i]}\n', emit(low, rtys))
                for m in ('abl_a', 'abl_b'):
                    (r2, _) = ev[m][i]
                    if r2[0] == 'val' and not agree(r2[1], js[o][i], opt, rty): st[f'{m} 反例/O{o}'] += 1
    return st

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    st = run(n, k); print(dict(sorted(st.items())))
    V.check('Q1 ADT 毒值层 vs JIT', st['程序'] >= 300 and st['JIT 失败'] == 0 and st['反例'] == 0
            and st['确定值/抽象输入含毒值'] >= 50 and st['确定值/仅降级输入含毒值'] >= 50,
            f"程序 {st['程序']}，输入 {st['输入']}（O0+O3）：确定值 {st['确定值']} 中一致 {st['一致']}、反例 {st['反例']}；"
            f"其中抽象输入含毒值 {st['确定值/抽象输入含毒值']}、仅降级输入含毒值（none 载荷）{st['确定值/仅降级输入含毒值']}；"
            f"模型毒值 {st['模型 毒值']}、UB {st['模型 UB']}、守卫 {st['模型 守卫']}；JIT 失败 {st['JIT 失败']}")
    V.check('Q2 消融 a 输入毒值', st['abl_a 反例/O3'] > 0, f"O0 {st['abl_a 反例/O0']}、O3 {st['abl_a 反例/O3']}")
    V.check('Q3 消融 b tag 毒值', st['abl_b 反例/O3'] > 0, f"O0 {st['abl_b 反例/O0']}、O3 {st['abl_b 反例/O3']}")
    V.finish()
