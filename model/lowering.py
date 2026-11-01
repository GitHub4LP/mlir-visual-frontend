"""前端降级管线：自动分配作用域的忠实降低（第 38 轮，M5.3/M5.1，spec/closure.md §8.1）。

本提交（9bae84b0）MLIR 的两处降低缺陷（allocascope_test K5/K9 的 JIT 证据）：
  D1 scf.for / scf.parallel 带 AutomaticAllocationScope trait（每次迭代结束释放 alloca），canonicalize 依此把 alloca_scope
     内联/提升进循环体；但 convert-scf-to-cf 不做每次迭代的 stacksave/restore ⇒ 栈随迭代增长。上游讨论未定论（discourse 2023-10，PR #69765）。
  D2 memref.alloca_scope 是 SingleBlock：体内含 scf 控制流时，scf-to-cf 先跑 ⇒ 体变多块、校验失败；memref-to-llvm 先跑 ⇒
     把多块内联进外层 scf 的单块区域、同样失败。即“结构化区域里、体内有控制流的 alloca_scope”在任何标准次序下都降不下去。

做法：在全部优化之后、方言降低之前，把作用域语义直接落成 llvm.intr.stacksave / stackrestore（无区域 op，任意嵌套、任意降低次序都合法）：
  · 每个 memref.alloca_scope ⇒ stacksave；体内联到原处；stackrestore；结果换成 return 的操作数；
  · 每个体内（不进入内层作用域 op）有自动作用域分配的 scf.for / scf.parallel ⇒ 体首 stacksave，终结符前 stackrestore。
这正是 trait/alloca_scope 的语义（退出即释放），也是 MemRefToLLVM 对 alloca_scope 的降低本身；结果若是作用域内 alloca 的 memref，
本来就悬垂（trait 语义下同样），不改变定义性。代价：降低后的 IR 依赖 LLVM 方言（JIT/WASM 都经 LLVM）。
不处理的情形一律返回原因（判未知，不静默）：多块循环体。
"""
import core as C

LOOPS = {'scf.for', 'scf.parallel'}     # affine.for → lower-affine 后是 scf.for；scf.forall → 降成 scf.for/parallel 后同理

def _has_scoped_alloc(block):
    for op in block.operations:
        n = op.operation.name
        if n in C.SCOPE_ALLOC: return True
        if n in C.SCOPE_OPS: continue
        for r in op.operation.regions:
            for b in r.blocks:
                if _has_scoped_alloc(b): return True
    return False

def lower_alloca_scopes(module):
    """原地改写。返回 (Counter: 'alloca_scope 展开' / '循环体加 save/restore', 未处理原因列表)。"""
    import mlir.ir as ir, collections
    loops, scopes, why = [], [], []
    def visit(op):
        for r in op.regions:
            for b in r.blocks:
                for o in b.operations: visit(o.operation)
        if op.name in LOOPS: loops.append(op)
        elif op.name == 'memref.alloca_scope': scopes.append(op)
    visit(module.operation)
    ptr = ir.Type.parse('!llvm.ptr')
    st = collections.Counter()
    def save(ip, loc):
        with ip, loc: return ir.Operation.create('llvm.intr.stacksave', results=[ptr]).result
    def restore(ip, loc, p):
        with ip, loc: ir.Operation.create('llvm.intr.stackrestore', operands=[p])
    for lp in loops:                                  # 先判循环（判定看的是原始结构；alloca_scope 是作用域 op，其内的 alloca 不算）
        blocks = list(lp.regions[0].blocks)
        if len(blocks) != 1: why.append(f'{lp.name} 多块体'); continue
        body = blocks[0]
        if not _has_scoped_alloc(body): continue
        ops = list(body.operations)
        p = save(ir.InsertionPoint(ops[0].operation), lp.location)
        restore(ir.InsertionPoint(ops[-1].operation), lp.location, p)
        st['循环体加 save/restore'] += 1
    for sc in scopes:                                 # 后序：内层先展开，外层移动的是已展开的 op
        blk = sc.regions[0].blocks[0]
        ops = list(blk.operations); ret = ops[-1].operation
        p = save(ir.InsertionPoint(sc), sc.location)
        for o in ops[:-1]: o.operation.move_before(sc)
        restore(ir.InsertionPoint(sc), sc.location, p)
        for r, v in zip(sc.results, list(ret.operands)): r.replace_all_uses_with(v)
        ret.erase(); sc.erase()
        st['alloca_scope 展开'] += 1
    return st, why
