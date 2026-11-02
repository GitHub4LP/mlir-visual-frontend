"""第 40 轮（L24，spec/values.md §13）：降低时真正插入守卫。

core 的 tensor 层比 MLIR 更有定义：越界下标、dim 下标越界、cast 静态维不符 ⇒ 毒值结果（MLIR：UB）。
此前这只是“义务”（core.guards 查询），emit/降低并不插守卫 ⇒ core 的语义没有落地：前端程序在越界输入上，
core 说“结果是毒值、其余输出照常”，而 MLIR 是 UB（JIT 可能读到垃圾或崩溃）。

做法：对 core 判定须守卫、且守卫条件的操作数可证非毒值的节点（core.guard_plan），在 MLIR 模块上改写为
  %r = scf.if (条件) { 原 op; yield } else { yield 替身 }
条件：extract/insert 各维 i ult dim(t, k)（无符号比较同时排除负数）；dim：i ult 秩；cast：源的动态维 = 目标静态维。
替身：毒值可被任何值精化 ⇒ extract/dim 取 0，insert 取原 tensor，cast 取 tensor.empty（动态维取源的 dim）。
第 43 轮加切片：条件 = 各维 s ≥ 1 ∧ o ≥ 0 ∧ n ≥ 0 ∧ (n = 0 ? o ≤ e : o < e ∧ n−1 ≤ (e−1−o) / s′)（s′ = s≥1 ? s : 1，避免除零；
  不用 o+(n−1)·s < e，因为 index 运算回绕时该式可能为真而数学上为假）；insert 另要求源各维 = 对应尺寸。
  替身：extract_slice 取 tensor.empty（动态尺寸夹到 ≥ 0，元素毒值），insert_slice 取原 tensor。
不处理（返回原因，调用方判未知、不发射）：守卫条件操作数可能是毒值（MLIR 以毒值分支是 UB，arith 无 freeze）。
"""
GUARDABLE = {'tensor.extract', 'tensor.insert', 'tensor.dim', 'tensor.cast', 'tensor.extract_slice', 'tensor.insert_slice'}   # 切片：第 43 轮

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
            def subi(a, b): return ir.Operation.create('arith.subi', results=[idx], operands=[a, b]).result
            def divui(a, b): return ir.Operation.create('arith.divui', results=[idx], operands=[a, b]).result
            def sel(c, a, b): return ir.Operation.create('arith.select', results=[a.type], operands=[c, a, b]).result
            if op.name in ('tensor.extract_slice', 'tensor.insert_slice'):
                big_i = 0 if op.name == 'tensor.extract_slice' else 1
                big = ops[big_i]; r = ir.RankedTensorType(big.type).rank
                dyn = iter(ops[big_i + 1:])
                KD = ir.ShapedType.get_dynamic_size()
                def vals(nm):
                    out = []
                    for v in ir.DenseI64ArrayAttr(op.attributes[nm]):
                        out.append(next(dyn) if v == KD else cidx(v))
                    return out
                offs, szs, sts = vals('static_offsets'), vals('static_sizes'), vals('static_strides')
                one, zero = cidx(1), cidx(0)
                for kk in range(r):
                    o_, n_, s_ = offs[kk], szs[kk], sts[kk]; e = dim(big, kk)
                    s1 = cmp(5, s_, one); sp = sel(s1, s_, one)                       # sge
                    c = andi(andi(s1, cmp(5, o_, zero)), cmp(5, n_, zero))
                    lim = divui(subi(subi(e, one), o_), sp)
                    nz = andi(cmp(2, o_, e), cmp(3, subi(n_, one), lim))              # slt, sle
                    c = andi(c, sel(cmp(0, n_, zero), cmp(3, o_, e), nz))
                    conds.append(c)
                if op.name == 'tensor.insert_slice':
                    srcT = ir.RankedTensorType(ops[0].type)
                    if srcT.rank != r: why.append('insert_slice 降秩：守卫未实现'); continue
                    keep = list(range(r))
                    for j, kk in enumerate(keep): conds.append(cmp(0, dim(ops[0], j), szs[kk]))
            if op.name == 'tensor.extract': t, ix = ops[0], ops[1:]; conds = [cmp(6, i, dim(t, kk)) for kk, i in enumerate(ix)]
            elif op.name == 'tensor.insert': t, ix = ops[1], ops[2:]; conds = [cmp(6, i, dim(t, kk)) for kk, i in enumerate(ix)]
            elif op.name == 'tensor.dim': conds = [cmp(6, ops[1], cidx(ir.RankedTensorType(ops[0].type).rank))]
            elif op.name in ('tensor.extract_slice', 'tensor.insert_slice'): pass
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
            elif op.name == 'tensor.insert_slice': fb = ops[1]
            elif op.name == 'tensor.extract_slice':
                rtt = ir.RankedTensorType(rt); stat = list(ir.DenseI64ArrayAttr(op.attributes['static_sizes']))
                KD = ir.ShapedType.get_dynamic_size()
                dynsz = [szs[kk] for kk in range(len(stat)) if stat[kk] == KD]   # 结果的动态维 = 动态尺寸（按序；被丢的维是静态 1）
                z0 = ir.Operation.create('arith.constant', results=[idx], attributes={'value': ir.IntegerAttr.get(idx, 0)}).result
                cl = [ir.Operation.create('arith.select', results=[idx], operands=[
                        ir.Operation.create('arith.cmpi', results=[i1], operands=[v, z0], attributes={'predicate': ir.IntegerAttr.get(ir.IntegerType.get_signless(64), 5)}).result, v, z0]).result for v in dynsz]
                fb = ir.Operation.create('tensor.empty', results=[rt], operands=cl).result
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
