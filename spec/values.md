# 值层审计扩面（第 23 轮）

实现：`model/core.py`。本轮新增：
- 移位、有符号除法、取余；
- exact 标志与 trunci 的 nsw/nuw；
- `nat(reads=…)` 守卫；
- `set_iw`、`REMSI_MIN`。

检验：`model/values_test.py`，命令为 `PYTHONPATH=model python3 model/values_test.py 150 300 120`，约 15 分钟。
真值只取 MLIR 自己的东西：ODS、JIT 执行（O0/O3，index 位宽 64 与 32），以及 MLIR 自己的变换（canonicalize、int-range-optimizations、sccp）。

## 1 复盘：第 21 轮的教训没有变成机制
- 第 21 轮发现 nsw 从未建模：参数存在，但语义不读，模型因此比 MLIR 更“有定义”，属于不可靠方向。
- 当时只补了 nsw 这一个属性，没有追问“还有哪些属性被静默忽略了”。
- 本轮用 ODS 对账，又发现两处同类缺陷：
  - `arith.divui` 的 `isExact`；
  - `arith.trunci` 的 `overflowFlags`。
- 所以要修的是机制，而不是逐个属性补：
  1. **账（S0）**：值层覆盖的每个 op，它的每个 ODS 属性都必须经 `core.ATTR_MAP` 译成 core 参数并被语义读取，或者登记为语义无关（并写明理由）。反向也查：core 有而 MLIR 没有的参数。
  2. **守卫**：节点带着语义不读的参数（且值不为 False）时，值层抛出 ValueUnknown，不给事实。
  3. **读回**：MLIR → core 时，未知属性一律报错，不丢弃。
- 回溯检查：
  - 用第 20 轮的 reads 表跑这本账，会报出 addi/subi/muli 的 nsw、divui 的 exact、trunci 的两个标志——第 21 轮那个缺陷本来能由账自动发现。
  - 用第 22 轮的 reads 表跑，会报出 divui 与 trunci。

## 2 新语义（LangRef 原文，ArithOps.td @ 9bae84b0）
| op | 值 | 毒值 | UB |
|---|---|---|---|
| shli | x << y | y ≥ w；nsw：(r >>ₛ y) ≠ x；nuw：(r >>ᵤ y) ≠ x | — |
| shrui / shrsi | 逻辑 / 算术右移 | y ≥ w；exact：移出的位非零 | — |
| divui | ⌊x/y⌋ 无符号 | exact：不整除 | y = 0 |
| divsi | 向零截断 | exact：不整除 | y = 0，或 MIN / −1 |
| remui / remsi | 余数（remsi 的符号随被除数） | — | y = 0；remsi(MIN, −1)：见 §4 |
| trunci | 截低位 | nuw：截掉的位非零；nsw：截掉的位不全等于结果的最高位 | — |

index 操作数按 IW 位的位向量计算。

## 3 结果
| 检验 | 结果 |
|---|---|
| S0 属性账 | arith/ub 共 22 个 op，缺口 0，core 多出的参数 0（ub.poison 的 value 登记为无关）。回溯：第 20 轮 reads ⇒ 6 个缺口（含第 21 轮的 nsw）；第 22 轮 reads ⇒ 3 个缺口（divui exact、trunci nsw/nuw） |
| V5 JIT，IW=64 | 1200 个输入（O0 与 O3）：一致 544，模型毒值 456，模型 UB 200，**反例 0** |
| V5 JIT，IW=32 | 1200 个输入：一致 544，毒值 390，UB 266，**反例 0** |
| V5 对照：FLAGS_EXT=False | IW=64 反例 37，IW=32 反例 41（例如 `divui exact` 按常量除时被降低为乘以模逆）⇒ 本检验确实会利用被测语义 |
| V6 canonicalize + int-range-optimizations | 300 个用例（104 个有变化）。IW=64 的两种 remsi 设定：精化 297，未定 3，**反例 0**。IW=32：反例 8 和 7，全部来自 MLIR 按 64 位折叠 index 常量（见 L50） |
| V7 scf + sccp | 120 个程序（84 个含循环）：模型给出确定值的 426 个输入上，模型 = sccp 后的 JIT，426/426 |
| V8 IW=32，arith 与 index 方言对照 | 模型@32 ≡ JIT@32，363/363。arith：canonicalize、int-range-optimizations、sccp 在 IW=32 下各误编译 **36/363**，IW=64 下为 0。index 方言：int-range-optimizations 与 sccp 为 0；canonicalize 在 IW=32 下 5 例、IW=64 下 6 例（见下一行） |
| V9 index 方言 `CmpOp::canonicalize` | 把 `cmp(pred, sub(x, y), 0)` 改写成 `cmp(pred, x, y)`，对 10 个谓词都这样做。只有 eq/ne 正确；有符号的 4 个谓词在溢出时出错，无符号的 4 个谓词在 x < y 时就出错（例：0 − 2 >u 0 为真，0 >u 2 为假）。9bae84b0 与当前 upstream main 都没有限制谓词。V8 里 index/canonicalize 的 11 例误编译由 V9 定向复现，但未逐例归因 |

求解器未定 13 例，按可靠方向不计为通过。

## 4 定律
- **L49 语义参数全称量化。** 以下两处，MLIR 自身不确定或自相矛盾：
  - 目标的 index 位宽 IW；
  - remsi(MIN, −1)：LangRef 只规定除数为 0 是 UB，MLIR 的折叠器把它折成 0，而降低到 `llvm.srem` 之后它是 UB。

  这类地方建模为参数，规则如下：
  - 值层事实和改写证明必须对参数的全部取值成立（V6 的四种设定）；
  - 推测执行取最坏的取值（UB）。
- **L50 MLIR 对 index 的折叠只在 IW = 64 下保持语义（A，V8）。**
  - arith 的折叠器、int-range-optimizations、sccp 都按 64 位折叠 index 常量。在 index-bitwidth=32 的降低下，它们会改变程序的执行结果。
  - 连 convert-arith-to-llvm 在转换时也按 64 位折叠常量比较。
  - index 方言的折叠器只在 32 位与 64 位结果一致时才折叠，同样的程序 0 误编译。
  - 但 index 方言的 canonicalize 另有一个与位宽无关的错误改写（V9），所以“改用 index 方言”本身不是安全的出路。
  - 对前端的规定（WASM32 一类目标）：先按目标确定 IW，在交给任何 MLIR 变换之前，把 index 降为 i{IW}。index 不带着“位宽未定”进入 canonicalize、sccp 或 int-range-optimizations。
  - 模型本身在 IW=32 下与 JIT 一致（V5、V8）。不安全的是 MLIR 的变换，不是模型。
- **L52 MLIR 自己的变换不是无条件的真值（A，V8/V9）。** 用 MLIR 变换作真值时，必须同时跑 JIT 对照（变换前后都执行）。一个变换在 JIT 上被证明误编译时，它的结果对模型不构成反例，而是记入 MLIR 缺陷账（spec/values.md §3）。
- **L51 语义属性账（A：ODS；守卫 D）。** 值层覆盖的每个 op：ODS 属性 ⊆ 语义读取的参数 ∪ 登记的无关属性；节点带未读参数 ⇒ 无事实。
- **L17″ 升级**：由 B 升为 A。依据是 V5 在 IW=32 下对照 JIT、V8 的 model@32 ≡ JIT@32。

## 5 缺口
- tensor/memref 覆盖层的 reads 还没有对账（参数用的是 core 自己的记法），在 S0 里单列为“未对账”。
- 值层仍没有 cf 多块、func.call、浮点。
- V7 只覆盖 loop_test 的 scf.for 形状。scf.while 的值层没有建模。
