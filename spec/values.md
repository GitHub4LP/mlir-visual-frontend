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
- ~~tensor/memref 覆盖层的 reads 未对账~~：第 25 轮对账（§8）。
- 值层仍没有 cf 多块、func.call、浮点。
- V7 只覆盖 loop_test 的 scf.for 形状。scf.while 的值层没有建模。

## 8 第 25 轮：值层吃导入的 MLIR（model/valimport.py、valimport_test.py）

**复盘。** 第 21–23 轮的值层审计全部经由自己的发射器（core 记法 → MLIR 文本）。反方向——MLIR 测试集里的真实 IR 进入值层——从来没有走过：
- 导入节点的属性原样存在 `_attrs`。arith 节点因此一律被 reads 守卫判成未知：没有错，但也没有事实；
- tensor/memref 的 `reads=None`（未对账），守卫不生效。`tensor.extract_slice %t[2][4][1]` 的静态偏移/尺寸不是操作数，`_extract_slice` 按位置取操作数，结果是 z3 异常；
- `tensor.from_elements` 按 1 维建模。0 维结果直接崩溃，多维结果错位；
- `arith.index_cast` 作用于 tensor 时（core 只有标量语义），同样崩溃；`i0` 在 z3 里没有对应的 sort，也崩溃；
- `scf.for` 的 `unsignedCmp`（无符号循环界）在 core 的循环摘要中没有对应，但只要能走到那里，就会被静默按有符号处理。

第 21 轮那句“前端可导入 MLIR”在值层上从来就不成立。

**机制（L53）。** 导入 → 值层必须经过 `valimport.to_core`。对每个 MLIR 属性，只有三种处理：
- **翻译**（17 项，各带适用条件）：static_* 与动态操作数合并为 core 的全动态记法，静态项物化为 index 常量；要求步长全为 1、不降秩；insert_slice 的尺寸须静态且等于源形状。
- **语义无关**（3 项，各带理由）。
- **拒绝**：未登记的属性一律 ⇒ 未知。目前有 alignment ×3、`scf.for.unsignedCmp`，以及全部可丢弃属性。

此外，以下情况也 ⇒ 未知：条件不满足、i0、0 维 tensor、逐元素作用于 tensor/vector 的 arith、浮点/dense 字面量、后继。core 的 walker 遇到仍带 `_attrs` 的节点一律 ValueUnknown，不再崩溃。所有 nat 的 `reads` 已对账（tensor/memref 为 `()`）。

**外部真值（MLIR @ 9bae84b0）。**
| 检验 | 结果 |
|---|---|
| A0 属性账 | 翻译 17、语义无关 3、拒绝 4；无静默项 |
| A1 语料覆盖 | 值层覆盖的 op 节点 4330：翻译 3243（74.9%），其余各有原因（字面量 392、逐元素 375、未登记属性 171、降秩 48、步长 26…） |
| A2 语料函数 vs JIT | 候选 188 → 有事实 169 → 执行 141 个函数、1036 个输入，**一致 1036/1036**；core 异常 0（修前 2） |
| A3 静态/动态混合切片 vs JIT | 300 个程序、1322 个输入，**一致 1322/1322** |
| A4 变异灵敏度 | 静态偏移当 0：A3 中 259/1322 被抓出；谓词编号 ^1：A2 中 77/1036 被抓出 |

丢 nsw 不可测：nsw 只会把值变成毒值，而毒值输入本来就跳过——毒值的精化自由度无法用执行来检验（这一项由 V3 的翻译确认负责）。

**局限。** 可丢弃属性（如 `__test_pipelining_op_order__`）一律拒绝。这比 MLIR 保守：MLIR 中不带方言前缀的可丢弃属性不改变 op 语义。但带方言前缀的属性可以由方言赋予语义，所以暂不放宽。A2 的候选只覆盖标量签名的函数，tensor 签名的语料函数尚未执行。

## 9 第 32 轮：tensor 签名的语料函数接入 JIT（model/tensorjit_test.py，G1.4）

**方法**：输入 tensor 用 `tensor.generate` 按公式 e(i) = trunc(seed + Σc_k·i_k) 生成，动态维随机取 0..3。模型侧用同一个公式构造具体的 z3 tensor（extent、Lambda 元素、元素毒值全假）代入洞。外包 main 只有 i64 参数：标量结果直接比较；tensor 结果先比较 `tensor.dim`，再在 extent 内抽样下标比较元素。模型判 UB、毒值或守卫不成立的输入跳过；毒值元素跳过，因为 JIT 观测不到。降级管线在 A2 的基础上改了两处：函数边界用 identity-layout-map（否则参数的 strided 布局 cast 降不到 LLVM），并且加了 convert-linalg-to-loops（tensor.generate 会缓冲化成 linalg.map）。

**结果**（同一提交 9bae84b0）：

| 项 | 数 |
|---|---|
| 单块 tensor 签名函数 | 616 |
| 含值层未覆盖的 op | 454 |
| 有值层事实（候选） | 10 |
| 已执行 | 7（另 3 个所有输入都不满足守卫） |
| 比较（dim 52、元素 117、标量 4） | 173，一致 173 |
| 比较器灵敏度：模型侧 seed+1 | 24/24 发现 |

valimport 的 pred / static_zero 变异在这批函数上 0/100：这批函数没有 cmpi，切片偏移也全是 0。这说明它们在这里没有判别力，不能当作灵敏度证据。切片偏移的灵敏度仍由 A3/A4 提供（static_zero 92/424）。

**core 修复**：无秩 tensor、浮点元素、带 encoding 的 tensor 作为 tensor op 的操作数或结果时，以及 scf.if / scf.for 携带未建模类型的值时，此前会以 z3 或 Python 异常告终（48 个语料函数）。外层把它们当作“core 异常”判未知，路径本身是安全的，但原因不可审计，而且同一个 except 也可能吞掉真实缺陷。现在这些情况一律显式抛出 ValueUnknown，扫描结果中 core 异常为 0。

**缺口（G1.4 的分母，即 tensor 值层事实率）**，按未知原因计：
- 逐元素作用于 tensor 的 arith：约 25；
- 0 维 tensor：约 14；
- 降秩切片：22；
- 步长 ≠ 1 的切片：14；
- insert_slice 尺寸不是静态或与源形状不等：9；
- 浮点或无秩：其余大部分。

前三类可以在 core 里建模，下一步按这个顺序推进。浮点值层另立（G1.5）。
