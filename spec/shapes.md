# 形状层 v0（tensor 并入 core）

实现：`model/core.py` 的“形状层”一节。检验：`model/shape_test.py`（`PYTHONPATH=model python3 model/shape_test.py 300`，约 1 分钟）。

## 1 表示
- tensor 值 = ⟨extent ∈ ℤ^r，元素 Array(ℤ^r → E)，元素毒值 Array(ℤ^r → Bool)⟩，另有整体毒值标志。
- extent 就是 index 值，所以形状推理直接用 core 的 ℤ 层，不再是独立原型。shape.py 只保留为 Presburger 显示层的参考。
- 已实现的 op（9 个）：splat、empty、from_elements、dim、extract、insert、extract_slice（秩保持、单位步长）、insert_slice、cast。

## 2 组合律
- **L22 静态维是 extent 的投影。**
  - 边上的静态维是声明：不符时该值是毒值，同时立一条证明义务（`shape_diag`）。
  - 端口上的静态维，对体是假设，对调用点是义务。
  - 不能证明的声明，降低时退为 `?`。这时 MLIR 侧行为有定义，是毒值的精化。
  - 例：出口声明 tensor<4> 而尺寸 n 是动态的 ⇒ 诊断；n 是常量 4 ⇒ 无诊断；显式 tensor.cast ⇒ 无诊断，n ≠ 4 时输出是毒值；调用点把 tensor<?> 传给 tensor<4> 端口 ⇒ 调用点诊断。
- **L23 相等只看域内。**
  - 精化与等价比较的是 extent，以及域内各元素的毒值和值；域外不可观察。
  - 对照：`from_elements(v,v) → splat(v)` 在域内相等下成立，整体相等下不成立。
- **L24 tensor 层没有 UB。**
  - 越界、负尺寸、形状操作数是毒值 ⇒ 结果是毒值。所以每个 tensor op 都可推测，这一点不弱于 MLIR 的 Pure。
  - 结果：MLIR 把 extract_slice、extract 标为 Pure，但又把越界定为 UB，自相矛盾（type-algebra.md §8 已记录）。前端的语义没有这个矛盾。
  - 代价落在降低上：在 MLIR 里这些情形是 UB，所以在上下文不能证明在界内的节点，降低时要插守卫（`guards` 查询）。
  - 例：`extract(x, i)` 需要守卫；`extract(splat:8, 5)` 不需要；`slice(x, 0, dim x)` 不需要。
  - tensor.dim 在维下标未知时也可推测，比 MLIR 更强（MLIR 在这种情形下返回 NotSpeculatable）。

## 3 修订：L17 → L17′（index 溢出由 UB 改为毒值）
- 原 L17 下，index 上的 addi 不可推测，而 MLIR 把 addi 标为 Pure，前端因此弱于 MLIR。
- 原 L17 下，`slice(slice(x,o1,n1),o2,n2) → slice(x, o1+o2, n2)` 也不成立：内层越界只产生毒值，而 o1+o2 溢出却是 UB。
- 改为毒值后两者都成立；保留 `core.IXOVF = 'ub'` 只为做对照。
- 连带影响：不变式要自己带范围事实（loops.md L17′）。

## 4 检验
- S1：`extract(slice(x,o,n), i)` 的 UB ≡ ⊥；MayPoison ≡ 切片越界 ∨ 下标越界 ∨ 元素毒值；值 = x[o+i]。三条都已证明。
- S2：L22 的 6 个情形全部符合预期。
- S3 改写规则：
  - 有效 6 条全部证明：slice∘slice 融合、`extract(slice) → extract(x, o+i)`、`extract(insert(v,x,i), i) → v`、`insert_slice(slice(x,o,n), x, o) → x`、dim 穿过 insert_slice、`from_elements → splat`；
  - 错误 2 条全部反驳：偏移相减、`extract(insert(·,i), j)`。
- S4：index addi、extract_slice、extract、dim 都可推测；按旧 L17 做对照时，addi 不可推测，slice 融合不成立。
- S5：守卫查询的 3 个用例全部符合。
- S6 与真实 MLIR 的差分：
  - 方法：随机生成 tensor 程序，经 one-shot-bufferize 降低后 JIT 执行；
  - 300 个程序、1049 组可执行输入，一致 1049，不一致 0；
  - 另有 751 组输入，模型判定需要守卫（在 MLIR 中会越界访存），按规则跳过，不执行。
- S7 变异灵敏度（把模型语义故意改错，看 S6 能否发现）：
  - extract_slice 偏移差一：发现 87/517；
  - dim 少一：发现 410/555；
  - insert_slice 窗口多一格：只发现 6/519。这个变异只有在恰好读到窗口外一格时才可见，说明随机差分对边界类错误很弱，必须靠定义期证明（S1、S3）补足。

## 5 已知边界
- 只支持秩保持、单位步长的切片；pad、collapse/expand_shape、linalg 都未接入。
- memref 的形状与别名还没做；bufferization 契约（L24 的守卫义务 + 原地更新的别名条件）是下一块。
- 守卫只在定义自身的上下文里判定；透明内联后，调用点的上下文可以消去更多守卫，这一点还没利用。
- Presburger 显示层（∃ 消去后给用户看可读的条件）还没接到 core。
- 元素毒值按元素跟踪；整体毒值标志只来自形状层面的失败。
