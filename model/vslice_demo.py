import time
from vslice import *

def N(id, op, ins=(), **attrs):
    d = dict(id=id, op=op, **{'in': list(ins)}); regions = attrs.pop('regions', None)
    if attrs: d['attrs'] = attrs
    if regions: d['regions'] = regions
    return d
def R(nodes, yld, args=()): return dict(nodes=nodes, **{'yield': list(yld)}, args=list(args))

P = Program()
# 1. 多态复合 op：axpy = a*x + y
P.define('axpy', ['a', 'x', 'y'], R([N('m', 'arith.mulf', ['a', 'x']), N('s', 'arith.addf', ['m', 'y'])], []), ['s'])
# 2. 迭代 fib：scf.for + iter_args；常量类型由使用处推出
P.define('fib', ['n'], R([
    N('lb', 'arith.constant', value=0), N('st', 'arith.constant', value=1), N('ub', 'arith.index_cast', ['n'], type='index'),
    N('a0', 'arith.constant', value=0), N('b0', 'arith.constant', value=1),
    N('loop', 'scf.for', ['lb', 'ub', 'st', 'a0', 'b0'],
      regions=[R([N('t', 'arith.addi', ['a', 'b'])], ['b', 't'], args=['i', 'a', 'b'])])], []), ['loop'], ptypes={'n': 'i64'})
# 3. 递归 fact：symbol 引用自身 + scf.if
P.define('fact', ['n'], R([
    N('one', 'arith.constant', value=1), N('le', 'arith.cmpi', ['n', 'one'], pred='sle'),
    N('if', 'scf.if', ['le'], regions=[R([], ['one']),
        R([N('m', 'arith.subi', ['n', 'one']), N('r', 'fact', ['m']), N('p', 'arith.muli', ['n', 'r'])], ['p'])])], []),
    ['if'], ptypes={'n': 'i64'})
# 4. 局部可变状态：Σ i²，累加器放在 memref 里
P.define('sumsq', ['n'], R([
    N('buf', 'memref.alloca', size=1), N('z', 'arith.constant', value=0, type='index'), N('c0', 'arith.constant', value=0),
    N('init', 'memref.store', ['c0', 'buf', 'z'], size=1),
    N('lb', 'arith.constant', value=0), N('st', 'arith.constant', value=1), N('ub', 'arith.index_cast', ['n'], type='index'),
    N('loop', 'scf.for', ['lb', 'ub', 'st'], regions=[R([
        N('v', 'memref.load', ['buf', 'z'], size=1), N('ii', 'arith.index_cast', ['i'], type='i64'),
        N('sq', 'arith.muli', ['ii', 'ii']), N('s', 'arith.addi', ['v', 'sq']), N('w', 'memref.store', ['s', 'buf', 'z'], size=1)],
        [], args=['i'])]),
    N('res', 'memref.load', ['buf', 'z'], size=1)], []), ['res'], ptypes={'n': 'i64'})
# 5. 使用者：复合 op 作为节点（f64 与 f32 两个实例）
P.define('axpy_test', ['a', 'x', 'y'], R([N('r', 'axpy', ['a', 'x', 'y'])], []), ['r'], ptypes={'a': 'f64'})
P.define('axpy_f32', ['a', 'x', 'y'], R([N('r', 'axpy', ['a', 'x', 'y'])], []), ['r'], ptypes={'a': 'f32'})

print('■ 推导出的 Schema（复合 op 与函数一视同仁）')
for f in ['axpy', 'fib', 'fact', 'sumsq', 'axpy_test']: print(f'  {f:10s}', P.schema(f))

print('■ 编辑期诊断（定位到节点和边）')
bad = [('类型冲突', ['a', 'k'], {'a': 'f64', 'k': 'i64'}, [N('r', 'axpy', ['a', 'a', 'k'])], ['r']),
       ('类约束', ['k'], {'k': 'i64'}, [N('r', 'arith.addf', ['k', 'k'])], ['r']),
       ('悬空端口', ['a'], {'a': 'f64'}, [N('r', 'arith.addf', ['a', 'q'])], ['r'])]
for title, ps, pt, nodes, res in bad:
    P.define('bad', ps, R(nodes, []), res, ptypes=pt)
    try: P.schema('bad')
    except DiagErr as e: print(f'  {title}：', e.errs)
del P.defs['bad']

print('■ 补全：端口类型为 f64 / i64 时可接的 op（复合 op 同样出现）')
print('  f64 →', P.completions('f64'))
print('  i64 →', P.completions('i64'))

print('■ 生成 MLIR → lowering → 执行')
E_ = Emitter(P)
names = {f: E_.func(f, pts, export=True) for f, pts in
         [('fib', ['i64']), ('fact', ['i64']), ('sumsq', ['i64']), ('axpy_test', ['f64', 'f64', 'f64']), ('axpy_f32', ['f32', 'f32', 'f32'])]}
src = E_.module(); open('model/vslice_out.mlir', 'w').write(src)
print('  生成的函数：', [l.split('(')[0][10:] for l in src.splitlines() if l.startswith('func.func')], ' 缺省规则命中', len(DEFAULTED), '处')
t0 = time.time(); ee, diags = compile_and_load(src); print(f'  编译 {time.time()-t0:.2f}s', diags or '')
print('  fib(10) =', call(ee, names['fib'], [(10, 'i64')], 'i64'), ' fact(10) =', call(ee, names['fact'], [(10, 'i64')], 'i64'),
      ' sumsq(10) =', call(ee, names['sumsq'], [(10, 'i64')], 'i64'),
      ' axpy(2,3,4) =', call(ee, names['axpy_test'], [(2.0, 'f64'), (3.0, 'f64'), (4.0, 'f64')], 'f64'))

print('■ 后端诊断回映（模拟发射器出错：把 sumsq 中的 muli 写成 mulf）')
ee2, diags = compile_and_load(src.replace('arith.muli %ii', 'arith.mulf %ii'))
print('  ', diags[:1])

print('■ 增量：大定义 3000 节点')
n = 3000
P.define('big', ['x'], R([N('n0', 'arith.addf', ['x', 'x'])] + [N(f'n{k}', 'arith.addf', [f'n{k-1}', 'x']) for k in range(1, n)], []), [f'n{n-1}'])
P.define('user', ['a'], R([N('r', 'big', ['a'])], []), ['r'], ptypes={'a': 'f64'})
t0 = time.time(); P.schema('user'); t1 = time.time()
P.define('user', ['a'], R([N('r', 'big', ['a']), N('r2', 'axpy', ['r', 'a', 'a'])], []), ['r2'], ptypes={'a': 'f64'})
t2 = time.time(); P.schema('user'); t3 = time.time()
body = P.defs['big']['body']; body['nodes'][1500]['in'] = ['n1499', 'n1498']; P.define('big', ['x'], body, [f'n{n-1}'])
t4 = time.time(); P.schema('user'); t5 = time.time()
print(f'  首次推导 {1000*(t1-t0):.1f}ms；编辑使用者（big 的 Schema 命中缓存）{1000*(t3-t2):.2f}ms；编辑 big 内部 {1000*(t5-t4):.1f}ms')
