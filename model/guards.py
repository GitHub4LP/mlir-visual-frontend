"""第 40 轮（L24，spec/values.md §13）：降低时真正插入守卫。

core 的 tensor 层比 MLIR 更有定义：越界下标、dim 下标越界、cast 静态维不符 ⇒ 毒值结果（MLIR：UB）。
此前这只是“义务”（core.guards 查询），emit/降低并不插守卫 ⇒ core 的语义没有落地：前端程序在越界输入上，
core 说“结果是毒值、其余输出照常”，而 MLIR 是 UB（JIT 可能读到垃圾或崩溃）。

做法：对 core 判定须守卫、且守卫条件的操作数可证非毒值的节点（core.guard_plan），在 MLIR 模块上改写为
  %r = scf.if (条件) { 原 op; yield } else { yield 替身 }
条件：extract/insert 各维 i ult dim(t, k)（无符号比较同时排除负数）；dim：i ult 秩；cast：源的动态维 = 目标静态维。
替身：毒值可被任何值精化 ⇒ extract/dim 取 0，insert 取原 tensor，cast 取 tensor.empty（动态维取源的 dim）。
不处理（返回原因，调用方判未知、不发射）：切片类（替身 tensor 的形状需另行论证）、守卫条件操作数可能是毒值（MLIR 以毒值分支是 UB，arith 无 freeze）。
"""
GUARDABLE = {'tensor.extract', 'tensor.insert', 'tensor.dim', 'tensor.cast'}

def insert_guards(module, names):
    """names：须守卫 op 的结果名（AsmState 局部作用域名，与 irimport 相同）。原地改写；返回 (已插数, 未处理原因)。"""
    with module.context: return _insert(module, names)
def _insert(module, names):
    import mlir.ir as ir
    st = ir.AsmState(module.operation, use_local_scope=True)
    todo = []
    def visit(op):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations: visit(o.operation)
        if len(op.results) == 1 and op.results[0].get_name(st) in names: todo.append(op)
    visit(module.operation)
    why, k = [], 0
    idx = ir.IndexType.get()
    i1 = ir.IntegerType.get_signless(1)
    for op in todo:
        if op.name not in GUARDABLE: why.append(f'{op.name} 的守卫未实现'); continue
        loc = op.location
        with ir.InsertionPoint(op), loc:
            def cidx(v): return ir.Operation.create('arith.constant', results=[idx], attributes={'value': ir.IntegerAttr.get(idx, v)}).result
            def cmp(pred, a, b): return ir.Operation.create('arith.cmpi', results=[i1], operands=[a, b], attributes={'predicate': ir.IntegerAttr.get(ir.IntegerType.get_signless(64), pred)}).result
            def dim(t, kk): return ir.Operation.create('tensor.dim', results=[idx], operands=[t, cidx(kk)]).result
            def andi(a, b): return ir.Operation.create('arith.andi', results=[i1], operands=[a, b]).result
            ops = list(op.operands); rt = op.results[0].type
            conds = []
            if op.name == 'tensor.extract': t, ix = ops[0], ops[1:]; conds = [cmp(6, i, dim(t, kk)) for kk, i in enumerate(ix)]
            elif op.name == 'tensor.insert': t, ix = ops[1], ops[2:]; conds = [cmp(6, i, dim(t, kk)) for kk, i in enumerate(ix)]
            elif op.name == 'tensor.dim': conds = [cmp(6, ops[1], cidx(ir.RankedTensorType(ops[0].type).rank))]
            else:
                src, dst = ir.RankedTensorType(ops[0].type), ir.RankedTensorType(rt)
                for kk in range(dst.rank):
                    if not dst.is_dynamic_dim(kk) and src.is_dynamic_dim(kk): conds.append(cmp(0, dim(ops[0], kk), cidx(dst.get_dim_size(kk))))
            if not conds: conds = [ir.Operation.create('arith.constant', results=[i1], attributes={'value': ir.IntegerAttr.get(i1, 1)}).result]
            c = conds[0]
            for x in conds[1:]: c = andi(c, x)
            ifop = ir.Operation.create('scf.if', results=[rt], operands=[c], regions=2)
        thenb, elseb = ifop.regions[0].blocks.append(), ifop.regions[1].blocks.append()
        with ir.InsertionPoint(thenb), loc:
            y = ir.Operation.create('scf.yield', operands=[op.results[0]])
        op.move_before(y)
        with ir.InsertionPoint(elseb), loc:
            if op.name == 'tensor.insert': fb = ops[1]
            elif op.name == 'tensor.cast':
                dst = ir.RankedTensorType(rt)
                dyn = [ir.Operation.create('tensor.dim', results=[idx], operands=[ops[0], ir.Operation.create('arith.constant', results=[idx], attributes={'value': ir.IntegerAttr.get(idx, kk)}).result]).result
                       for kk in range(dst.rank) if dst.is_dynamic_dim(kk)]
                fb = ir.Operation.create('tensor.empty', results=[rt], operands=dyn).result
            elif ir.FloatType.isinstance(rt): fb = ir.Operation.create('arith.constant', results=[rt], attributes={'value': ir.FloatAttr.get(rt, 0.0)}).result
            else: fb = ir.Operation.create('arith.constant', results=[rt], attributes={'value': ir.IntegerAttr.get(rt, 0)}).result
            ir.Operation.create('scf.yield', operands=[fb])
        op.results[0].replace_all_uses_except(ifop.results[0], y)
        k += 1
    return k, why

def result_names(fn, nids):
    """irimport 的函数节点 + 节点 id 集合 → 结果名集合（嵌套区域也找）。"""
    out = set()
    def walk(blocks):
        for b in blocks:
            for n in b['ops']:
                if n['id'] in nids and n['outs']: out.add(n['outs'][0][0])
                for r in n['regions']: walk(r)
    walk(fn['regions'][0]); return out

def plan(derive):
    """derive()：在当前语义开关下推导同一程序（返回 Schema 或 None）。
    返回 (可降低?, 原因, S_core, S_low)。可降低 ⇔
      (1) core 语义 ⊑ 插守卫后的语义（refines，入口非毒值）：守卫的替身只精化毒值，且没有毒值流进守卫条件（否则 UB）；
      (2) 插守卫后仍有的义务（切片、splat/empty 尺寸等未实现守卫的 op）可证（推测版：相关操作数非毒值 ∧ 良定义）。"""
    import core as C, z3
    S = derive()
    if S is None: return False, 'core 值层未知', None, None
    old = C.GUARD_LOWER; C.GUARD_LOWER = True
    try: S2 = derive()
    finally: C.GUARD_LOWER = old
    if S2 is None: return False, '插守卫语义下值层未知', S, None
    hyp = z3.And(C.no_poison(S), *[f for nid, f in S.wf if str(nid).startswith('入口')])   # 入口：参数非毒值、静态维符合类型（调用方的责任）
    r = C.refines(S, S2, hyp, cross_sem=True)
    if r is not True: return False, '插守卫后的语义不精化 core 语义（毒值流进守卫条件，或无法判定）', S, S2
    ctx = z3.And(S2.ax, C.no_poison(S2), *[f for nid, f in S2.wf if str(nid).startswith('入口')], z3.Not(S2.ubf))
    rest = [nid for nid, f in S2.gspec if not C.prove(z3.Implies(ctx, f))]
    if rest: return False, f'未实现守卫的义务不可证：{rest[:3]}', S, S2
    return True, '', S, S2
