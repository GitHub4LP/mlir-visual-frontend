# 语义覆盖层 v0：arith 整数部分

ODS 不提供语义。覆盖层为每个 op 给出 ⟨Sem, UB, Poison, Sensitive, prop⟩。

## 来源与取舍原则

1. **依据**：依次参考 MLIR op 的 description（源码快照 inventory/Arith_ArithOps.td）、lowering 的目标语义（ArithToLLVM 映射到 LLVM IR），以及 xdsl-smt（仅作参考）。
2. **保守并集**：MLIR 文档与 lowering 目标不一致时，UB 取两者的并集，Poison 也取并集。这会让前端的前提更强，但无论后端按哪一种语义执行都可靠。每一处这样的选择都标为“前端选择”。
3. **毒值敏感**：文档没有说明时，随 LLVM 处理——除数、模数为毒值即 UB。
4. **任何来源的语义都要经过条款测试**：对每个 op 在 i4 上穷举，检查是否满足下表各条款。导入第三方语义时必须先跑这组测试（xdsl-smt 就没有通过，见下文）。

## arith 整数 op 表（W 为位宽，s 为有符号解释）

| op | UB | Poison（输出） | 敏感 | 依据 |
|---|---|---|---|---|
| addi/subi/muli | – | nuw/nsw 下的无符号/有符号溢出 | – | 文档 |
| divui / remui | y=0 | divui exact：x mod y≠0 | y | 文档；敏感性随 LLVM |
| divsi | y=0 ∨ (x=MIN ∧ y=−1) | exact：不能整除 | y | 文档 |
| **remsi** | y=0 ∨ **(x=MIN ∧ y=−1)** | – | y | **前端选择**：文档只写了除零，但 lowering 到 llvm.srem，而 LLVM 中 srem 溢出是 UB |
| ceildivui | y=0 | – | y | 文档 |
| ceildivsi / floordivsi | y=0 ∨ (x=MIN ∧ y=−1) | – | y | 文档 |
| shli | – | y ≥ W；nuw/nsw 溢出 | – | 文档 |
| shrui / shrsi | – | y ≥ W；exact 下移出的位非 0 | – | 文档 |
| trunci | – | nuw/nsw 下截掉的位不为 0 / 不等于符号位 | – | 文档 |
| extui / uitofp / index_castui | – | nneg 标志下最高位为 1 | – | 文档 |
| select | – | prop：P(c) ∨ (c ∧ P(t)) ∨ (¬c ∧ P(f)) | – | 文档 |
| and/or/xor/max/min/cmpi | – | 只传播毒值 | – | 默认 |

## 对 xdsl-smt 的审计（commit dd30235，2026-04-08）

- 覆盖面：arith 整数 op 共 28 个，memref 只有 alloc、dealloc、load、store，另有 comb、llvm；没有浮点，没有 tensor 和 scf。
- 与 MLIR 规范不符之处：
  1. shli、shrui、shrsi 只在 y > W 时产生毒值，而规范是 **y ≥ W**。已用 xdsl-tv 实证：i4 上 `x & 0 → x >>u 4` 被判为正确变换，但按 MLIR 和 LLVM 语义，这个变换引入了毒值。用例见 model/difftest/src.mlir、tgt.mlir；对照用例 tgt2.mlir（移 5 位）被正确拒绝。
  2. 没有建模 exact 标志（divui、divsi、shrui、shrsi），缺少相应的毒值。
  3. nuw/nsw 只对 addi 建模；subi、muli、shli 忽略了这两个标志。
  4. remsi 与 MLIR 文档一致，但和 lowering 一样存在上述 INT_MIN % −1 的缺口。
- 结论：xdsl-smt 的方法（值加毒值标志、UB 效应、memory 编码）可以借鉴，但逐 op 的语义不能直接导入，只有通过条款测试的 op 才能用。

## 待办
- tensor、scf、memref 的覆盖层（越界读取取毒值还是 UB，与 frontend-model §3 的管线约束一起决定）；
- 浮点：NaN、舍入模式、fastmath 标志（fastmath 产生毒值）；
- 把条款测试写成可执行代码（i4 穷举），作为导入任何第三方语义之前的门槛。
