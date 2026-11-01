"""第 39 轮（M1.8，spec/frontend-model §1）：caps（Speculatable、Pure）的外部对照。

真值的界定（批判性）：
  · “可推测” = 对所有输入（含毒值）不触发 UB。UB 在 JIT 中通常不可观测（AArch64 sdiv 除零返回 0、越界读不一定崩溃），
    所以 JIT 不能直接检验 caps；本测试的外部参照是 **MLIR 自己的判定**：loop-invariant-code-motion 只外提
    isMemoryEffectFree ∧ isSpeculatable 的 op（含各 op 的 getSpeculatability 定制，如 divsi 的常量除数、tensor.dim 的常量下标）。
  · 分类：一致；core 保守（MLIR 外提而 core 不判 Pure）——安全；core 宽松（core 判 Pure 而 MLIR 不外提）——危险，逐例归因：
    或 MLIR 保守（core 的 UB 模型证明无 UB，须引用 MLIR 的 UB 定义作依据），或 core 的 UB 模型有缺口（缺陷，须修）。
  · 前端把 caps 用于外提/改写，只能在“一致”或“core 保守”的方向上直接使用；宽松方向须有逐例的语义依据。
C1：原生 op 实例（固定类型；变量/常量操作数）core caps vs MLIR LICM。
C2：复合（多 op 函数体）core caps vs MLIR：整体外提须逐 op 外提——MLIR 的结论是逐 op 的合取，core 可以更精（整体证明）。
C3：义务经复合 op 调用传播（第 39 轮修的缺口：specialize 与摘要路径此前丢掉被调体的 L22/L24 义务）。
    宿主调用含 tensor.extract 的复合 op（透明特化 / 精确摘要两条路径）：宿主的 guards() 非空、caps 不可推测；
    在宿主里把下标换成常量 0 + 静态 tensor ⇒ 守卫可证、可推测（不过度保守）。
C4：MLIR 判 Pure 而 core 判不可推测的 arith.remui/remsi：JIT A/B（LICM + LLVM O3）观测到误编译 ⇒ core 对、MLIR 错
    （本提交 remui/remsi 标 Pure，文档写除零 UB；上游 #188263（2026-03-24）已改为 ConditionallySpeculatable，晚于 9bae84b0）。
"""
import os, sys, re, collections, multiprocessing as mp

# (名字, 参数类型, 外部常量行, 循环内 op 行（结果 %r）)
I = 'i32'; F = 'f32'
def bin_(op, t=I, flags=''): return (f'{op}{flags}', [t, t], [], f'%r = {op} %a0, %a1 {flags} : {t}')
INST = [bin_(f'arith.{o}') for o in ('addi', 'subi', 'muli', 'andi', 'ori', 'xori', 'shli', 'shrui', 'shrsi', 'divui', 'divsi', 'remui', 'remsi')]
INST += [(f'arith.{o} nsw', [I, I], [], f'%r = arith.{o} %a0, %a1 overflow<nsw> : i32') for o in ('addi', 'subi', 'muli', 'shli')]
for o in ('divui', 'divsi', 'remui', 'remsi', 'shli', 'shrui', 'shrsi'):
    for c in (3, -1, 0, 40):
        INST.append((f'arith.{o} 常量{c}', [I], [f'%k = arith.constant {c} : i32'], f'%r = arith.{o} %a0, %k : i32'))
    INST.append((f'arith.{o} 常量被除数7', [I], [f'%k = arith.constant 7 : i32'], f'%r = arith.{o} %k, %a0 : i32'))
INST += [('arith.divsi exact', [I, I], [], '%r = arith.divsi %a0, %a1 exact : i32'),
         ('arith.cmpi', [I, I], [], '%r = arith.cmpi slt, %a0, %a1 : i32'),
         ('arith.select', ['i1', I, I], [], '%r = arith.select %a0, %a1, %a2 : i32'),
         ('arith.trunci', [I], [], '%r = arith.trunci %a0 : i32 to i8'),
         ('arith.extui', ['i8'], [], '%r = arith.extui %a0 : i8 to i32'),
         ('arith.extsi', ['i8'], [], '%r = arith.extsi %a0 : i8 to i32'),
         ('arith.index_cast', [I], [], '%r = arith.index_cast %a0 : i32 to index'),
         ('arith.index_castui', [I], [], '%r = arith.index_castui %a0 : i32 to index'),
         ('arith.addi index', ['index', 'index'], [], '%r = arith.addi %a0, %a1 : index'),
         ('arith.divui index', ['index', 'index'], [], '%r = arith.divui %a0, %a1 : index'),
         ('arith.constant', [], [], '%r = arith.constant 5 : i32'),
         ('ub.poison', [], [], '%r = ub.poison : i32')]
INST += [bin_(f'arith.{o}', F) for o in ('addf', 'subf', 'mulf', 'divf', 'remf', 'maximumf', 'minimumf')]
INST += [('arith.addf nnan', [F, F], [], '%r = arith.addf %a0, %a1 fastmath<nnan> : f32'),
         ('arith.negf', [F], [], '%r = arith.negf %a0 : f32'),
         ('arith.cmpf', [F, F], [], '%r = arith.cmpf olt, %a0, %a1 : f32'),
         ('arith.extf', ['f16'], [], '%r = arith.extf %a0 : f16 to f32'),
         ('arith.truncf', [F], [], '%r = arith.truncf %a0 : f32 to f16'),
         ('arith.sitofp', [I], [], '%r = arith.sitofp %a0 : i32 to f32'),
         ('arith.uitofp', [I], [], '%r = arith.uitofp %a0 : i32 to f32'),
         ('arith.fptosi', [F], [], '%r = arith.fptosi %a0 : f32 to i32'),
         ('arith.fptoui', [F], [], '%r = arith.fptoui %a0 : f32 to i32'),
         ('arith.bitcast i→f', [I], [], '%r = arith.bitcast %a0 : i32 to f32')]
INST += [(f'math.{o}', [F], [], f'%r = math.{o} %a0 : f32') for o in ('absf', 'ceil', 'floor', 'trunc', 'round', 'roundeven', 'sqrt', 'isnan', 'isfinite')]
INST += [('math.fma', [F, F, F], [], '%r = math.fma %a0, %a1, %a2 : f32')]
INST += [(f'math.{o}', [I], [], f'%r = math.{o} %a0 : i32') for o in ('absi', 'ctlz', 'cttz', 'ctpop')]
T4, TD = 'tensor<4xi32>', 'tensor<?xi32>'
INST += [('tensor.splat', [I], [], f'%r = tensor.splat %a0 : {T4}'),
         ('tensor.empty', [], [], f'%r = tensor.empty() : {T4}'),
         ('tensor.from_elements', [I, I], [], '%r = tensor.from_elements %a0, %a1 : tensor<2xi32>'),
         ('tensor.dim 常量0', [TD], ['%k = arith.constant 0 : index'], f'%r = tensor.dim %a0, %k : {TD}'),
         ('tensor.dim 常量1（越界）', [TD], ['%k = arith.constant 1 : index'], f'%r = tensor.dim %a0, %k : {TD}'),
         ('tensor.dim 变量', [TD, 'index'], [], f'%r = tensor.dim %a0, %a1 : {TD}'),
         ('tensor.extract 变量', [T4, 'index'], [], f'%r = tensor.extract %a0[%a1] : {T4}'),
         ('tensor.extract 常量2', [T4], ['%k = arith.constant 2 : index'], f'%r = tensor.extract %a0[%k] : {T4}'),
         ('tensor.extract 常量9（越界）', [T4], ['%k = arith.constant 9 : index'], f'%r = tensor.extract %a0[%k] : {T4}'),
         ('tensor.insert 变量', [I, T4, 'index'], [], f'%r = tensor.insert %a0 into %a1[%a2] : {T4}'),
         ('tensor.insert 常量1', [I, T4], ['%k = arith.constant 1 : index'], f'%r = tensor.insert %a0 into %a1[%k] : {T4}'),
         ('tensor.extract_slice 静态', [T4], [], f'%r = tensor.extract_slice %a0[1] [2] [1] : {T4} to tensor<2xi32>'),
         ('tensor.extract_slice 动态', [TD, 'index', 'index'], [], f'%r = tensor.extract_slice %a0[%a1] [%a2] [1] : {TD} to {TD}'),
         ('tensor.insert_slice 静态', ['tensor<2xi32>', T4], [], f'%r = tensor.insert_slice %a0 into %a1[1] [2] [1] : tensor<2xi32> into {T4}'),
         ('tensor.cast 静→动', [T4], [], f'%r = tensor.cast %a0 : {T4} to {TD}'),
         ('tensor.cast 动→静', [TD], [], f'%r = tensor.cast %a0 : {TD} to {T4}'),
         ('memref.load', ['memref<4xi32>', 'index'], [], '%r = memref.load %a0[%a1] : memref<4xi32>'),
         ('memref.alloc', [], [], '%r = memref.alloc() : memref<4xi32>')]
# C2：复合体（多 op）
COMP = [('守卫除法 select', [I, I], ['%z = arith.constant 0 : i32', '%one = arith.constant 1 : i32'],
         ['%c = arith.cmpi ne, %a1, %z : i32', '%d = arith.select %c, %a1, %one : i32', '%r = arith.divui %a0, %d : i32']),
        ('ori 1 后除法', [I, I], ['%one = arith.constant 1 : i32'], ['%d = arith.ori %a1, %one : i32', '%r = arith.divui %a0, %d : i32']),
        ('remui 4 作下标', [T4, 'index'], ['%c4 = arith.constant 4 : index'], ['%i = arith.remui %a1, %c4 : index', f'%r = tensor.extract %a0[%i] : {T4}']),
        ('min 夹紧下标', [T4, 'index'], ['%c3 = arith.constant 3 : index'], ['%i = arith.minui %a1, %c3 : index', f'%r = tensor.extract %a0[%i] : {T4}']),
        ('纯算术链', [I, I], [], ['%s = arith.addi %a0, %a1 : i32', '%r = arith.muli %s, %a0 : i32'])]

def texts(args, outer, inner_lines):
    """(函数体文本 for core, LICM 文本)。inner_lines 最后一行定义 %r。"""
    sig = ', '.join(f'%a{i}: {t}' for i, t in enumerate(args))
    rty = re.search(r':\s*([^:]+)$', inner_lines[-1]).group(1).strip()
    rty = rty.split(' to ')[-1].split(' into ')[-1].strip()
    if 'tensor.extract ' in inner_lines[-1]: rty = 'i32'
    if 'tensor.dim' in inner_lines[-1]: rty = 'index'
    if 'cmp' in inner_lines[-1] or 'math.is' in inner_lines[-1]: rty = 'i1'
    if 'memref.load' in inner_lines[-1]: rty = 'i32'
    body = '\n'.join('  ' + l for l in outer + inner_lines)
    f = f'func.func @f({sig}) -> {rty} {{\n{body}\n  return %r : {rty}\n}}'
    sep = ', ' if sig else ''
    loop = '\n'.join('    ' + l for l in inner_lines)
    g = (f'func.func @g({sig}{sep}%n: index) {{\n' + '\n'.join('  ' + l for l in outer) +
         f'\n  %lc0 = arith.constant 0 : index\n  %lc1 = arith.constant 1 : index\n  scf.for %iv = %lc0 to %n step %lc1 {{\n{loop}\n    "test.sink"(%r) : ({rty}) -> ()\n  }}\n  return\n}}')
    return f, g, len(inner_lines)

def _licm(g, q):
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    try:
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            m = ir.Module.parse(g); PassManager.parse('builtin.module(func.func(loop-invariant-code-motion))').run(m.operation)
            s = str(m); body = s[s.index('scf.for'):]
            body = body[body.index('\n') + 1:body.index('test.sink')]
            q.put(('ok', body.count(' = ')))                  # 循环体内剩下的定义数（不计 iv）
    except Exception as e: q.put(('err', str(e)[:100]))
def licm(g, timeout=60):
    ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=_licm, args=(g, q)); p.start(); p.join(timeout)
    if p.is_alive(): p.kill(); return ('timeout', None)
    if p.exitcode != 0: return ('crash', p.exitcode)
    return q.get()

def core_caps(f):
    import mlir.ir as ir, irimport as I_, core as C, valimport_test as VT
    from compcorpus_test import _funcs
    try:
        m = ir.Module.parse(f, ir.Context())
        if not m.operation.verify(): return None, 'MLIR 不接受实例'
        fn = list(_funcs(I_.import_module(m)))[0]
    except Exception as e: return None, f'实例不合法 {str(e)[:60]}'
    S, why = VT.derive_imported('caps.' + str(abs(hash(f))), fn)
    if S is None: return None, why
    return C.caps(S), None

def judge(name, args, outer, inner):
    f, g, k = texts(args, outer, inner)
    cc, why = core_caps(f)
    r = licm(g)
    if r[0] != 'ok': return name, '比较不了：MLIR ' + r[0], cc, None
    mlir_all = r[1] == 0                    # 全部外提
    mlir_none = r[1] == k
    if cc is None: return name, '未知（core）：' + (why or '')[:50], None, mlir_all
    cp = cc['Pure']
    if cp is None: return name, '未知（core caps None）', cc, mlir_all
    if cp == mlir_all: cls = '一致'
    elif mlir_all and not cp: cls = 'core 保守'
    else: cls = 'core 宽松' + ('' if mlir_none or k == 1 else '（MLIR 部分外提）')
    return name, cls, cc, mlir_all

# core 宽松且已归因（MLIR 保守、core 依 MLIR 的 UB 定义证明无 UB）。每条须写依据；空表 = 尚未归因任何一例。
ALLOWED_PERMISSIVE = {}

def C3():
    import core as C
    from structure import blk, node
    IX, I8, TD, T8 = 'index', 'i8', 'tensor<?xi8>', 'tensor<8xi8>'
    nd = lambda v, op, ins, ty: node(v, op, ins=ins, outs=[(v, ty)])
    D = lambda name, ins, outs, ops, mode='transparent': C.derive(name, ins, outs, [blk('entry', ops, args=ins)], mode)
    old = C.ORACLE_ON; C.ORACLE_ON = False
    try:
        inner = D('c3.get', [('x', TD), ('i', IX)], [('q', I8)], [nd('e', 'tensor.extract', ['x', 'i'], I8), C.ret('r', 'e')])
        host = D('c3.host', [('x', TD), ('i', IX)], [('q', I8)], [nd('c', 'c3.get', ['x', 'i'], I8), C.ret('r', 'c')])
        z = C.cst('z', 0, IX)
        host_ok = D('c3.host_ok', [('v', I8)], [('q', I8)], [nd('t', 'tensor.splat', ['v'], T8), nd('u', 'tensor.cast', ['t'], TD), z,
                                                              nd('c', 'c3.get', ['u', 'z'], I8), C.ret('r', 'c')])
        res = {'被调体自身守卫': len(C.guards(inner)), '宿主守卫（经调用传播）': len(C.guards(host)),
               '宿主 caps 可推测': C.caps(host)['Speculatable'], '常量下标宿主：守卫未证': len(C.guards(host_ok)),
               '常量下标宿主 caps 可推测': C.caps(host_ok)['Speculatable'], '被调体精确摘要': inner.exact}
    finally: C.ORACLE_ON = old
    ok = res['被调体自身守卫'] > 0 and res['宿主守卫（经调用传播）'] > 0 and res['宿主 caps 可推测'] is False \
         and res['常量下标宿主：守卫未证'] == 0 and res['常量下标宿主 caps 可推测'] is True
    return ok, res

REM_SRC = '''func.func @f(%a: i32, %b: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0 : i32
  %s = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) -> (i32) {
    %r = arith.OP %a, %b : i32
    %t = arith.addi %acc, %r : i32
    scf.yield %t : i32
  }
  %isz = arith.cmpi eq, %b, %z : i32
  %x = arith.constant 111 : i32
  %y = arith.constant 222 : i32
  %o = arith.select %isz, %x, %y : i32
  %res = arith.addi %o, %s : i32
  return %res : i32
}'''
def _rem_child(src, licm, opt, q):
    import mlir.ir as ir, ctypes, valimport_test as VT
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(src)
        if licm: PassManager.parse('builtin.module(func.func(loop-invariant-code-motion))').run(m.operation)
        m.body.operations[0].attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
        PassManager.parse(VT.PIPE).run(m.operation)
        ee = ExecutionEngine(m, opt_level=opt); r = ctypes.c_int32(0)
        ee.invoke('f', ctypes.pointer(ctypes.c_int32(5)), ctypes.pointer(ctypes.c_int32(0)), ctypes.pointer(ctypes.c_int64(0)), ctypes.pointer(r))
        q.put(r.value)
def C4():
    """b = 0、n = 0：循环一次不执行，正确结果 111。"""
    out = {}
    for op in ('remui', 'remsi', 'divui', 'divsi'):
        for licm in (False, True):
            for opt in (0, 3):
                ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=_rem_child, args=(REM_SRC.replace('OP', op), licm, opt, q)); p.start(); p.join(60)
                out[(op, licm, opt)] = q.get() if p.exitcode == 0 else f'exit {p.exitcode}'
    return out

if __name__ == '__main__':
    st = collections.Counter(); rows = []
    for name, args, outer, inner in INST:
        rows.append(('C1',) + judge(name, args, outer, [inner]))
    for name, args, outer, inner in COMP:
        rows.append(('C2',) + judge(name, args, outer, inner))
    for grp, name, cls, cc, ma in rows:
        st[f'{grp} {cls.split("：")[0]}'] += 1
        if cls != '一致':
            sp = None if cc is None else (cc.get('Speculatable'), cc.get('Speculatable|非毒值输入'))
            print(f'  {grp} {name:28s} {cls:30s} core(spec, spec|非毒)={sp} MLIR外提={ma}')
    print('──', dict(sorted(st.items())))
    ok3, r3 = C3(); print('── C3 义务经调用传播:', 'OK' if ok3 else 'FAIL', r3)
    r4 = C4(); print('── C4 LICM+O3（b=0,n=0，正确 111）:', {f'{o} {"LICM" if l else "原样"} O{k}': v for (o, l, k), v in r4.items()})
    wrong = {o for (o, l, k), v in r4.items() if v != 111}
    print('   误编译的 op:', sorted(wrong), '；core 对这些 op 的 caps:', {o: [r[3]['Pure'] for r in rows if r[1] == f'arith.{o}'] for o in sorted(wrong)})
    bad = [r for r in rows if r[2].startswith('core 宽松') and r[1] not in ALLOWED_PERMISSIVE]
    for r in bad: print('  未归因的宽松：', r[1])
    c4ok = all(not [r for r in rows if r[1] == f'arith.{o}'][0][3]['Pure'] for o in wrong)   # 被观测到误编译的 op，core 必须判不可推测
    sys.exit(0 if not bad and ok3 and c4ok else 1)

