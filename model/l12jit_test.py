"""L12 外部化（第 31 轮）：接口定律的定向判定（defs.implement → (左→右, 右→左)）对照同一提交 MLIR 的 JIT。
L12 进入改写路径：law_ok 只放行“已证明的方向”。此前只有 D5 的 4 个实现 × 3 条定律，而且只有自证。
检验：
  - 随机生成 i3 上的 Magma 实现：combine = 由 addi/subi/muli/shli/shrui/shrsi/divui/divsi/remui/remsi/andi/ori/xori
    组成的深 ≤2 的表达式（随机 nsw/nuw/exact 标志、常量叶子），unit = 随机常量；
  - 定律：assoc、left_id、right_id、comm、idem，每条两个方向，都走真实的 D.implement（与生产路径相同的 law 构造与 C.refines）；
  - 每条定律的两边按模型内联成纯 arith，交给 MLIR 降级 + JIT，穷举 8^k 个输入。
判据（JIT 看不到毒值与 UB，只能看值）：
  T  模型判某方向有效 ⇒ 在模型认为源侧有定义（无 UB、非毒值）的每个输入上，JIT(源) = JIT(目标)，且 = 模型值。不符就是不可靠。
  F  模型判无效 ⇒ 分成“值层可见的反例”（双方都有定义而模型值不同，JIT 也必须不同）和“只在毒值/UB 层”（JIT 不可观测，计数）。
灵敏度：值层可见的反例数，就是“如果检查器把这些方向错判为有效，JIT 会抓到”的条数。
用法：PYTHONPATH=model python3 model/l12jit_test.py [实现数]
"""
import random, sys, re, ctypes, itertools, collections
import mlir.ir as ir
from mlir.passmanager import PassManager
from mlir.execution_engine import ExecutionEngine
import core as C, defs as D
from core import blk, node, I3
from defs import call
from valimport_test import model_eval

OVF = ('arith.addi', 'arith.subi', 'arith.muli', 'arith.shli')
EXA = ('arith.shrui', 'arith.shrsi', 'arith.divui', 'arith.divsi')
OPS = list(OVF + EXA) + ['arith.remui', 'arith.remsi', 'arith.andi', 'arith.ori', 'arith.xori']

D.IFACES['Magma'] = D.Interface('Magma', {'unit': (0, 1), 'combine': (2, 1)}, D.IFACES['Monoid'].laws + [
    ('comm', lambda m, t: ([call('q', m['combine'], ['x0', 'x1'], t)], 'q'),
             lambda m, t: ([call('q', m['combine'], ['x1', 'x0'], t)], 'q'), 2),
    ('idem', lambda m, t: ([call('q', m['combine'], ['x0', 'x0'], t)], 'q'),
             lambda m, t: ([], 'x0'), 1)])

def gen(rng, tag):
    ops, k = [], [0]
    def ex(d):
        r = rng.random()
        if d == 0 or r < 0.3:
            if rng.random() < 0.25:
                i = f'{tag}k{k[0]}'; k[0] += 1; ops.append(C.cst(i, rng.randrange(8))); return i
            return rng.choice(['a', 'b'])
        op = rng.choice(OPS); x, y = ex(d - 1), ex(d - 1); i = f'{tag}n{k[0]}'; k[0] += 1
        p = {}
        if op in OVF:
            if rng.random() < 0.3: p['nsw'] = True
            if rng.random() < 0.3: p['nuw'] = True
        if op in EXA and rng.random() < 0.3: p['exact'] = True
        ops.append(C.op2(i, op, x, y, **p)); return i
    r = ex(rng.choice([1, 1, 2]))
    return ops, r

def emit(blocks, k):
    nm = {}
    def v(x):
        if x not in nm: nm[x] = '%v' + str(len(nm)) + '_' + re.sub(r'[^A-Za-z0-9_]', '_', x)
        return nm[x]
    b = blocks[0]; out = []
    for n in b['ops']:
        op, pr = n['op'], n.get('props', {})
        if op == 'func.return': out.append(f'  return {v(n["ins"][0])} : i3'); continue
        if op == 'arith.constant': at = f'<{{value = {pr["value"] % 8} : i3}}>'
        elif op in OVF:
            fl = [f for f in ('nsw', 'nuw') if pr.get(f) is True]
            at = f'<{{overflowFlags = #arith.overflow<{", ".join(fl)}>}}>' if fl else ''
        elif op in EXA: at = '<{isExact}>' if pr.get('exact') is True else ''
        else: at = ''
        sig = '(' + ', '.join('i3' for _ in n['ins']) + ') -> (i3)'
        out.append(f'  {v(n["outs"][0][0])} = "{op}"(' + ', '.join(v(x) for x in n['ins']) + f') {at} : {sig}')
    args = ', '.join(f'{v(x)}: i3' for x, _ in b['args'])
    return f'func.func @f({args}) -> i3 {{\n' + '\n'.join(out) + '\n}'

PIPE = 'builtin.module(convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)'
def jit(blocks, k):
    xs = ', '.join(f'%x{i}: i64' for i in range(k))
    body = [f'%t{i} = arith.trunci %x{i} : i64 to i3' for i in range(k)]
    body += [f'%r = func.call @f(' + ', '.join(f'%t{i}' for i in range(k)) + ') : (' + ', '.join(['i3'] * k) + ') -> i3',
             '%e = arith.extui %r : i3 to i64', 'return %e : i64']
    src = emit(blocks, k) + f'\nfunc.func @main({xs}) -> i64 attributes {{llvm.emit_c_interface}} {{\n  ' + '\n  '.join(body) + '\n}'
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify(), src
        PassManager.parse(PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=0); out = []
        for x in itertools.product(range(8), repeat=k):
            a = [(ctypes.c_int64 * 1)(y) for y in x]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); out.append(r[0] & 7)
        return out, src

def side(meth, f, k, nm):
    ins = [(f'x{i}', I3) for i in range(k)]
    ops, r = f(meth, I3)
    prog = [blk('entry', ops + [C.ret(nm + '.r', r)], args=ins)]
    S = C.derive(nm, ins, [('q', I3)], prog)
    low = C.inline_all(prog)
    return S, low

def run(n, seed=31):
    rng = random.Random(seed); st = collections.Counter(); bad = []
    for t in range(n):
        cops, cr = gen(rng, f'm{t}')
        C.derive(f'mg{t}.c', [('a', I3), ('b', I3)], [('q', I3)], [blk('entry', cops + [C.ret(f'mg{t}.cr', cr)], args=[('a', I3), ('b', I3)])])
        C.derive(f'mg{t}.u', [], [('q', I3)], [blk('entry', [C.cst(f'mg{t}.z', rng.randrange(8)), C.ret(f'mg{t}.ur', f'mg{t}.z')])])
        meth = {'unit': f'mg{t}.u', 'combine': f'mg{t}.c'}
        rep = D.implement('Magma', I3, meth)           # 生产路径
        st['实现'] += 1
        for law, lf, rf, k in D.IFACES['Magma'].laws:
            SL, lowL = side(meth, lf, k, f'L12.{t}.{law}.L'); SR, lowR = side(meth, rf, k, f'L12.{t}.{law}.R')
            jl, srcL = jit(lowL, k); jr, srcR = jit(lowR, k)
            xs = list(itertools.product(range(8), repeat=k))
            ml = [model_eval(SL, [I3] * k, x) for x in xs]; mr = [model_eval(SR, [I3] * k, x) for x in xs]
            for (src, tgt, js, jt), ok in zip([(ml, mr, jl, jr), (mr, ml, jr, jl)], rep[law]):
                st['方向'] += 1
                dfd = [i for i, m in enumerate(src) if m[0] == 'val']
                st['翻译不符'] += sum(src[i][1] != js[i] for i in dfd)
                if ok is True:
                    st['判有效'] += 1; st['有效·源有定义输入'] += len(dfd)
                    d = sum(js[i] != jt[i] for i in dfd); st['有效·JIT不符'] += d
                    if d: bad.append((t, law, srcL, srcR))
                elif ok is False:
                    st['判无效'] += 1
                    vis = [i for i in dfd if tgt[i][0] == 'val' and tgt[i][1] != src[i][1]]
                    if vis:
                        st['无效·值层可见'] += 1; st['无效·JIT确认'] += all(js[i] != jt[i] for i in vis)
                    else: st['无效·仅毒值/UB层'] += 1
                else: st['未判定'] += 1
    return st, bad

if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    st, bad = run(n)
    print('── L12 接口定律方向 vs JIT:', dict(st))
    for b in bad[:3]: print('  ✗', b[0], b[1], '\n', b[2], '\n', b[3])
