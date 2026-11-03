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
| S0 属性账 | 第 45 轮机器判定：arith/ub 共 38 个 op 全部对账，core 多出的参数 0；缺口只剩 truncf.roundingmode“无译名”（valimport 判未知，§10.2）。**更正**：第 35 轮加入浮点 op 后 S0 实际已失败（fastmath 无译名 11 条、core 多出 nnan/ninf 22 条与 constant.dense 1 条），因退出码恒 0 未被发现；原因是 `core.ATTR_MAP` 未补登 fastmath（valimport._fastmath 已实现），已补。回溯：第 20 轮 reads ⇒ 缺口（含 nsw），第 22 轮 ⇒ 缺口（divui exact 等），敏感性仍成立 |
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

### 9.1 第 33 轮：arith 逐元素作用于 tensor（core.elementwise）

**语义**（MLIR Elementwise / ElementwiseMappable）：
- 结果 extent 等于操作数 extent。
- 元素 i 的值 = 标量语义作用于各操作数的元素 i。操作数的整体毒值并入元素毒值，所以“对毒值除”等标量规则逐元素成立。
- 运行时 extent 不一致（只可能出现在动态维）⇒ UB。
- 任一域内元素有标量 UB ⇒ 整体 UB，形式为 ∃i（例如某个元素除以 0）。
- 条件为标量的 arith.select 选择整个 tensor，走原标量路径。
- dense 整数常量（splat 或嵌套列表）由 valimport._dense 解析，嵌套结构须与形状一致；十六进制、资源、浮点、i1 列表 ⇒ 未知。

**对照管线**：前面加 convert-elementwise-to-linalg（MLIR 标准降级）。JIT 在 fork 子进程里跑，MLIR 断言只记为“JIT 不可用（崩溃）”。父进程的 MLIR Context 关掉多线程，否则 fork 出的子进程会死锁。

**结果**（9bae84b0）：

| | 有事实 | 执行 | 比较 | 不一致 | 变异 |
|---|---|---|---|---|---|
| T 语料 tensor 签名 | 35（第 32 轮为 10） | 31 | 930（dim 214、元素 708、标量 8） | 0 | seed+1 75/87 |
| T2 逐元素定向，200 程序 | 200 | 167 | 3009 | 0 | pred 39/670 |

T2 的覆盖面：13 种二元 op（含可能 UB 的除法/取余/移位）、nsw/nuw/exact、cmpi + tensor 条件 select、标量条件 select、ext/trunc、dense 常量；形状有静态的，也有动态的。

**MLIR 崩溃（上游均已修复，不提 issue）**：T2 中 28 个程序在本提交的 canonicalize/folding 处断言崩溃（`type must have static shape`）。原因有两类：
- `arith.xori %x, %x` 作用于动态形状 tensor：上游 677f13e（#212072，2026-07-26）修复；
- `arith.cmpi p, %x, %x` 作用于动态形状 tensor：上游 main 的 getBoolAttribute 已对动态形状返回空。

两者都是合法 IR，触发的是本提交 MLIR 的已知缺陷，不影响模型结论；这些程序记为不可执行。

**剩余未知**：0 维 tensor（约 14）、降秩/步长切片（约 37）、insert_slice 尺寸、浮点。

### 9.2 第 34 轮：0 维 tensor；标量条件 select 的可靠性缺陷

**0 维表示**：
- z3 数组至少要一个下标，所以 0 维 tensor 的 data/pz 数组用一个恒为 0 的哑下标，extent 列表为空。
- 取元素、建数组、写元素、量化统一走 `sel`/`lam`/`store`/`forall`/`exists`，空下标时分别退化为“取下标 0 / 常数组 K / 写下标 0 / 公式本身”。
- tparse 接受 `tensor<i8>`。
- `tensor.dim` 作用于 0 维 ⇒ 毒值（MLIR：UB，降低时插守卫）。
- `tensor.from_elements` 支持 0 维。
- valimport 去掉了第 25 轮的 0 维拒绝。

**缺陷（JIT 反例，已修）**：第 33 轮把“条件为标量的 arith.select 作用于 tensor”按“选整个 tensor”建模。MLIR 的 `OpDefinition.h` 在 Elementwise 性质 3 中写明：所有 vector/tensor 操作数与结果须同形，动态形状不一致时行为未定义。arith.select 带 ElementwiseMappable，标量条件只是广播。

反例 t2_193：`select %s, %b, %a`，其中 %a、%b 的动态维不同。
- 模型给出有定义值：dim 3。
- JIT（convert-elementwise-to-linalg）按 %b 的形状生成：dim 1。

修复：标量条件 select 也走 `core.elementwise`，条件广播，tensor 操作数形状不一致 ⇒ UB。这是“模型在 MLIR 为 UB 处给出定义值”的方向，对前端不可靠，不是保守误差。第 33 轮 T2 没抓到它，是因为同类型参数 80% 共用 extent，而形状不一致的那部分又多半被别的 UB 跳过了。教训：生成器的输入分布要刻意覆盖“声明允许而语义未定义”的区域。

**结果**（9bae84b0）：

| | 有事实 | 执行 | 比较 | 不一致 |
|---|---|---|---|---|
| T 语料 tensor 签名 | 39（第 33 轮为 35） | 35 | 945 | 0 |
| T2 定向，240 程序 | 240 | 218 | 2886 | 0（修复前 4，均来自 t2_193） |

- T2 按形状分：2x3 一致 1741，?x3 一致 899，0 维一致 246（75 个程序，含 extract[]、insert[]、from_elements）。
- 变异：pred 57/673，seed+1 88/99。
- MLIR 已知断言崩溃 16 个（xori/cmpi x,x，上游已修）。
- 语料另有 6 个函数有事实，但签名含非整数 tensor，harness 无法构造输入，记为跳过，不计入执行。

### 9.3 第 34b 轮：全形式切片（步长、降秩/升秩、insert_slice 动态尺寸）

**模型**：core 新增 `tensor.extract_slice.g` / `tensor.insert_slice.g`，端口为 offs/sizes/strides 各 r 个，参数 `drop` 记被丢弃的单位维。valimport 的切片全部翻译到这两个 op；简式切片只留给手写模型。
- extract：`r[J] = src[o + embed(J)·s]`，被丢弃的维取 0。
- insert：窗口为 `d−o ≥ 0 ∧ (d−o) mod s = 0 ∧ (d−o)/s < n`，源下标为 `(d−o)/s`，再去掉被丢弃的维。
- 界内条件：o ≥ 0、n ≥ 0、s ≥ 1；n = 0 时 o ≤ e，n > 0 时 o + (n−1)·s < e；被丢弃的维 n = 1；insert 还要求源 extent 等于对应尺寸。不满足 ⇒ 毒值，`_mlir_def` 同步给出守卫。
- s ≤ 0：MLIR 文档没有给出语义，这里按毒值处理。这比 MLIR 严，代价只是多插守卫。
- 降秩掩码按 computeRankReductionMask 的贪心匹配计算。歧义只出现在单位维之间，被丢弃维的下标恒为 0，所以不影响值。
- 静态 verifier 比动态语义更严（verifyInBoundsSlice：静态 offset ≥ 静态 dim 即非法，即使 size = 0）。这属于结构层的约束，值层不复制。

**诊断**：core 的 ValueUnknown 现在把原因带出（`Schema.value_why`），不再只报“值层未知”。

**结果**（9bae84b0 JIT）：

| | 有事实 | 执行 | 比较 | 不一致 |
|---|---|---|---|---|
| 语料 tensor 签名 | 41（第 34 轮为 39） | 37 | 1030 | 0 |
| T3 定向，400 生成，353 进入值层 | 353 | 112 | 743 | 0 |

- 语料变异：static_zero 2/386，seed+1 90/101。
- T3 按类别（程序数 / 一致数）：同秩 124 / 275，同秩+insert 84 / 217，降秩 65 / 61，降秩+insert 80 / 190；含步长 ≠ 1 的程序 324 个。
- T3 变异：static_zero 136/408。
- T3 守卫跳过 1154 组输入：负值、越界、形状不一致的区域被刻意采样，模型在这些输入上均判毒值或守卫，没有一处给出定义值。
- MLIR 断言崩溃 6 个，都是 size 0 配动态步长，不计入一致。

**分母的真实缺口**：语料仍有 95 个函数报值层未知，原因按位置统计：
- extract_slice.g 35，insert_slice.g 15；
- tensor.cast 15，tensor.empty 8；
- scf.for / scf.if 携带 10；
- 其余 12。

几乎全部是“操作数或结果的元素类型未建模”（f32 等）。切片的结构缺口已经关闭；G1.4 剩下的主要缺口是浮点值层（G1.5），属于另一类工作。

### 9.x 第 47 轮：展开资源上限（KTUNROLL）

静态趟数 ≤ KUNROLL 的循环原本一律精确展开。携带 tensor 的循环每展开一趟，上一趟的 tensor 项就被嵌进新的 z3 Lambda（insert_slice/extract_slice 的数据与毒值都是 Lambda）；z3 抽象绑定变量时按嵌套深度重复遍历共享子项，开销约 ×3.5/趟（语料 tensor_dim_of_iter_arg_insertslice：6 趟 2.8 s，10 趟超过 20 分钟）。这是推导不终止，不是语义问题。规则：展开后 tensor 产生 op 总数（趟数 × 体内 tensor 产生 op 数）> KTUNROLL = 8 时，改走摘要路径。摘要路径与 trip > KUNROLL 相同，可靠性不变，只是可能更不精确或判未知。T 语料修复前后一致数相同（1030），只是那个文件不再超时。

### 9.y 第 48 轮：根因修复（项层 β/投影化简），KTUNROLL 降为安全网

第 47 轮的 KTUNROLL 只是治标：同样的指数增长在**直线代码**里也会出现（实测：链式 12 个 insert_slice 13 s，14 个超过 25 s），scf.if 嵌套同理，与循环无关。根因在项的构造方式：`dat(mk(…))` 被保留为“访问器(构造器)”，`sel(Lambda, I)` 被保留为 `Select(Lambda, I)`，于是每个 tensor op 都把前一个 tensor 的完整项嵌进一层新的 Lambda。

修复（core `_proj`/`_sel`）是纯重写，语义不变：
- 构造器上的投影直接取对应参数；
- If 上的投影或取元素推进到两支（带记忆，DAG 规模线性）；
- Lambda 上的取元素做 β 代换（`substitute_vars`）；
- 常量数组取元素直接取其值。

修复后，每个 tensor 的元素项只有一层 Lambda，推导代价随链长呈多项式增长：直线 40 个 insert_slice 3.6 s；关闭 KTUNROLL 时 64 趟循环 7.1 s。KTUNROLL 由 8 放宽到 512，只作安全网，原先第 47 轮因它走摘要的循环重新得到精确展开。

回归（derivecost_test D1/D2/D3）以“限时内推导出事实”为判据，在 fork 子进程中运行，限时 90 s 只用于防止挂起。旧实现 D1 N=40、D2 10 趟均超时，判据能区分新旧实现。我曾试过“Lambda 嵌套深度”作为确定性度量，但它对旧实现也给 1，不灵敏，已弃用。

语义不变性由外部 JIT 证据保证，数字与修复前完全相同：T 1030/0、T2 2886/0、T3 297/0、valimport A2 1274/0、A3 1322/0。

### 9.z 第 49 轮：全局确定性资源预算（DBUDGET，L53）

第 48 轮修复后，推导仍没有全局上限。KUNROLL 和 KTUNROLL 都只看单层循环，嵌套静态循环会逐层相乘：三层 64 趟嵌套约 64³ 次展开，实测超过 90 s 仍未结束。透明特化的子推导同样不受任何上限约束。

规则：
- **计量**：一次 derive 内 walk 处理的 op 节点数，透明特化的子推导共享同一个计数器。这个量确定、可复现，不依赖机器，也不计时。
- **展开前估算**：用 `Ctx.cost` 估算 趟数 × 体的展开代价。内层静态循环按趟数相乘：界在当前环境里是常量时取实际趟数，否则按 KUNROLL 上界；超过 KUNROLL 的内层循环按摘要只计一遍。若已用量加上估算值超过 DBUDGET，就改走摘要。摘要是可靠路径，只可能损失精度。
- **硬上限**：计量超过 DBUDGET 时抛 ValueUnknown（“推导超出资源预算”），原因随警告一起带出。

**标定**：DBUDGET = 20000。在第 49 轮的全部测试与语料上（8301 次 derive），单次用量最大为 925（tensorjit 语料 tj83_13_0），预算留有 20 倍以上余量。外部 JIT 数字与改动前完全相同，说明预算没有改变任何已有事实。

**回归**：derivecost D4（三层 64 趟嵌套，现在限时内得到事实，旧实现超时）；D5（DBUDGET 临时降到 50，直线 100 个 addi 判未知且原因可读，恢复后得到事实）。这两项已登记在 M3.8 的检验列。

**仍未覆盖**：
- 单个 z3 调用（simplify、Lambda 抽象）本身的代价不计入。第 48 轮已消除已知的指数来源，但没有形式化的上界。
- 估算对 iN 界一律按有符号解释，可能低估无符号循环的趟数。即使低估，内层循环在入口仍按实际趟数检查，硬上限也兜底，所以只影响精度与资源，不影响可靠性。

## 10 第 35 轮：浮点值层（G1.5；model/core.py 浮点节、valimport、floatjit_test.py）

### 10.1 语义（逐条对 LLVM LangRef，经 arith-to-llvm 降低）
- 类型：f16、bf16、f32、f64 ↦ z3 FPSort。f8 系列、tf32、f80/f128 不登记，值层判未知。
- **NaN**：z3 只有一个 NaN，所以值层的相等就是“不分载荷、不分符号的 NaN 相等”。LLVM 的 NaN 结果载荷不确定，这一点与它一致。**前提**：本片段没有能观察 NaN 位的 op。具体措施：
  - `arith.bitcast` 浮点 → 整数判未知，整数 → 浮点照常建模；
  - copysign、`math.*` 不登记；
  - 以后新增任何读取位的 op，都必须重新检查这个前提。
- 运算：addf、subf、mulf、divf 按 RNE；negf、cmpf 共 16 个谓词（有序：NaN 为假；无序：NaN 为真）；maximumf/minimumf（任一 NaN ⇒ NaN，−0 < +0）。
- **remf**：降低为 `llvm.frem`，即 C 的 fmod（结果与被除数同号），**不是** IEEE remainder（z3 fpRem）。值层由 IEEE remainder 修正得到 fmod：数学结果可精确表示，所以这一步没有误差。若直接用 fpRem，就是可靠性错误。
- maxnumf/minnumf：±0 的结果不确定，sNaN 的语义也不稳定，因此不登记，判未知。
- 转换：
  - extf 精确；truncf、sitofp、uitofp 按 RNE；
  - fptosi/fptoui 按 RTZ，NaN、±Inf 或越界 ⇒ 毒值（LangRef）。上界用 RTZ(hi) 比较：可表示的整数值都满足 `tz ≤ RTZ(hi) ⇔ tz ≤ hi`。
- **fastmath**：
  - nnan/ninf：操作数或结果为 NaN/Inf ⇒ 毒值（精确语义）；
  - reassoc、nsz、arcp、contract、afn、fast：允许近似或不定符号，valimport 判未知。contract 会让降低把乘加融合，单个 op 的语义不足以描述结果。
- 常量：十六进制按位模式；十进制按精确有理数 RNE（APFloat convertFromString）。attrsig 已能把它打印回 MLIR。

### 10.2 MLIR 文档与降低不一致：truncf 的 roundingmode
- **文档**：“rounded using the provided rounding mode”。
- **F3 探针**：输入 ±(1+2⁻²⁴)（恰在两个 f32 的中点），5 种舍入模式在 JIT 中的结果全部是 0x3f800000 / 0xbf800000，即都按 RNE 处理。
- **原因（推断）**：降低到 LLVM 受约束内建后，舍入模式只是“对当前动态模式的假设”，不是指令。
- **处理**：值层判未知（不选文档语义，也不选 JIT 行为）。不提 upstream issue，只记录在此，待用户决定。

### 10.3 证据（MLIR 9bae84b0 JIT；包装函数以整数位模式进出，NaN 只比较类别）
| | 程序/函数 | 执行 | 比较 | 不一致 | NaN 结果 | 变异 pred |
|---|---|---|---|---|---|---|
| F1 定向（f16 115、bf16 93、f32 94、f64 98） | 400 | 397 | 3500 | 0 | 388 | 503/1182 |
| F2 语料，含浮点的标量签名函数 | 有事实 75/204 | 68 | 1333 | 0 | 71 | 160/160 |

- **输入**：60% 取特殊值（±0、±Inf、qNaN/sNaN、最小次正规、最大有限值、最小正规、带舍入位的值），其余为随机位模式。
- **变异**：cmpf 谓词取逻辑否定。异或 1 只在相等或 NaN 时有差别，第一次试只发现 0/10，太弱，已改。
- **语料里的未知（显式）**：bitcast 浮点 → 整数 9、fastmath contract/fast/nsz 5、roundingmode 1、多维 from_elements 2。
- **语料里未覆盖的 op（112 个函数）**：主要是 math.*（sin 14、absf 11、powf 10、cos 8、log2 7 等）、func.call 12、complex.*。
- **测试工具的坑**：ExecutionEngine.invoke 遇到多结果 C 接口（sret）会挂起或崩溃，已改为每个结果一个包装函数。

### 10.4 边界
- **浮点 tensor**：valimport 仍判未知。core.elementwise 对元素语义是参数化的，但还没有浮点 tensor 的 JIT 证据，按原则不开放。
- **math 方言**：超越函数在 libm 里的实现不是正确舍入，只能建成不确定的 UF 加区间约束，否则判未知。可精确建模的 absf、ceil、floor、trunc、roundeven、sqrt、fma、copysign、isnan 等可以作为下一步。copysign 会观察 NaN 符号，需要单独论证。

## 11 第 35b 轮：math 方言的可精确子集（model/core.py math 节、valimport、floatjit_test.py F4 混入 F1）

### 11.1 登记与理由（降低：convert-math-to-llvm → LLVM 内建，逐条 JIT 探针）
| op | 语义 | 依据 |
|---|---|---|
| absf | fpAbs | llvm.fabs（NaN 只清符号，NaN 不比载荷/符号 ⇒ 安全） |
| ceil/floor/trunc/round/roundeven | roundToIntegral RTP/RTN/RTZ/RNA/RNE | llvm.ceil/floor/trunc/round（远离零）/roundeven；探针 1.5/2.5/−1.5 |
| sqrt | fpSqrt RNE | llvm.sqrt 正确舍入（f16/bf16/f32 探针） |
| fma | fpFMA RNE（单次舍入） | llvm.fma；探针 (1+2⁻²³)²−(1+2⁻²²) = 2⁻⁴⁶ |
| isnan/isfinite | 谓词 | 降到 fcmp/is.fpclass；汇编格式不接受 fastmath |
| absi | INT_MIN ↦ INT_MIN | llvm.abs(is_int_min_poison=false)，探针 i8 −128 |
| ctlz/cttz | 0 ↦ 位宽 | is_zero_poison=false，探针 |
| ctpop | 位计数 | llvm.ctpop |

fastmath 与 arith 同：只建 nnan/ninf，其余 ⇒ 未知。math 只开放标量；逐元素作用于 tensor/vector 无 JIT 证据 ⇒ 未知。

不登记（⇒ 未知），各有理由：sin/cos/exp/log/tanh/erf/atan/powf/rsqrt…（libm 不保证正确舍入，结果依实现）；copysign（读 NaN 符号位，NaN 符号不确定）；fpowi（乘法次序未规定）；isinf/isnormal（本提交 convert-math-to-llvm 不降低，JIT 无法建立证据）；clampf（未审计）。

### 11.2 证据（MLIR 9bae84b0）
- F1（400 程序，math 混入一元/fma/谓词/整数 math）：比较 3522，一致 3521，**不一致 0**，已归因外部缺陷 1；NaN 结果 416；毒值跳过 478；cmpf 变异 458/1147。
- F2 语料：含浮点标量签名 204 → 有事实 102（第 35 轮 75）→ 执行 94（68）→ 比较 1749/1749；UB 跳过 27，毒值 144；未覆盖函数 85（超越函数、complex、func.call 为主）。

### 11.3 外部真值缺陷：LLVM AArch64 `sitofp(fptosi f16→i32/i64)→f16` 误编译
- 现象：f16 65504 经 `fptosi → i32 → sitofp → f16` 得 32768（0x7800），−65504 得 −32768；49152 得 32768。i64 中间值无此现象（直接链路 i64 正确），fptoui/uitofp 正确（u16 可容纳 65535）。
- 定位：MLIR 发出的 `llvm.fptosi`/`llvm.sitofp` 正确；单独 fptosi、单独 sitofp 均正确；两步之间插入不可内联的恒等调用后结果正确。与 AArch64 TableGen 模式 `(f16 (sint_to_fp (i32 (fp_to_sint f16))))` → `SCVTFv1i16 (FCVTZSv1f16 …)` 一致：中间值被缩成 16 位有符号饱和整数，i32 值域内的 32768..65504 被错误饱和。
- 处理：模型按 LangRef（不跟随误编译）。floatjit 对含该模式且不一致的程序做 A/B 重跑（`barrier()` 插入 `@__id_iN`），同一输入在隔断版本上与模型一致才计入“已归因”，否则仍算不一致。属 LLVM 后端缺陷，非 MLIR；**未提 upstream issue**（需用户同意）。

## 12 第 39 轮：caps 的外部对照（M1.8；model/caps_test.py）
**真值的界定**：“可推测”= 对所有输入（含毒值）不触发 UB。UB 在 JIT 中通常不可直接观测，所以外部参照取 **MLIR 自己的判定**——loop-invariant-code-motion 只外提 isMemoryEffectFree ∧ isSpeculatable 的 op（含各 op 的 getSpeculatability 定制）；分歧再用 JIT A/B（LICM + LLVM O3 利用 UB）判谁对。
**发现的 core 缺陷（已修）**：
1. caps 只看 core 语义的 UB。core 的 tensor 层把越界记作毒值（L24：比 MLIR 更有定义，降低时须插守卫），但 emit_core 并不插守卫 ⇒ tensor.extract/insert/dim/cast/切片的变量下标被判可推测，前端据此外提，在 MLIR 中就是 UB（修复前 C1：96 例“core 宽松”，其中 tensor 类 15 例是真缺陷，其余是测试工具的计数错误）。
   现在 caps['Speculatable'] = ax ∧ 入口 L22 假设 ⇒ ¬UB ∧ 推测版守卫（良定义条件用到的操作数为毒值也算违反）∧ 体内/出口 L22 义务；旧判定保留为 'Speculatable|core 语义'。
2. 义务不随复合 op 调用传播：specialize（透明特化）与精确摘要两条路径都丢掉被调体的 L22/L24 义务 ⇒ 宿主的 guards() 与 caps 漏掉复合 op 内部的越界。修为以调用路径条件守卫后并入（摘要路径做项代换）。C3 变异：去掉传播 ⇒ 宿主 caps 判可推测（不可靠），测试失败。

**结果**（MLIR 9bae84b0）：C1 113 个原生 op 实例：一致 98，core 保守 15，core 宽松 **0**；C2 复合 5 例：一致 3，保守 1，未知 1（arith.minui 未建模）。
core 保守的 15 例归因：
- arith.remui/remsi（变量除数、常量 0、常量被除数）6 例：**MLIR 错**。本提交两者标 Pure，文档却写除零 UB。C4 JIT A/B：b=0、n=0（循环不执行）时正确结果 111；LICM 外提后经 LLVM O3 得 222（LLVM 由“已执行 urem/srem”推出 b≠0），O0 与原样均为 111；divui/divsi（ConditionallySpeculatable）不外提，全部 111。上游 #188263（2026-03-24）已改为 ConditionallySpeculatable，晚于 9bae84b0 ⇒ 不提 issue。
- tensor.extract/insert 变量或越界下标、动态切片、cast 动→静 5 例：MLIR 标 Pure，但越界/形状不符按其文档是 UB——与 M5.1 的 LICM+bufferize 反例同源，MLIR 自身不一致；core 对。
- 静态下标的 extract/insert/切片 4 例：只因 tensor 操作数可能是毒值（从毒值 tensor 读 = 从毒值指针 load）而保守；'Speculatable|非毒值输入' 为真。

**对前端的指导**：caps 可直接用于外提/改写的方向只有“一致”与“core 保守”；当前无宽松例。MLIR 的 Pure 不能作为前端可推测的依据（remui/remsi、tensor 越界两类反例）。

## 13 第 40 轮：L24 落地——降低时真正插守卫（model/guards.py，guards_test.py）
**问题**：core 的 tensor 层越界 ⇒ 毒值（L24，比 MLIR 更有定义），但此前“守卫”只是一个查询（core.guards），emit/降低从不插 ⇒ core 的语义没有落地。前端写出的无分支掩码 `select(i<dim, extract t[i], 0)` 在 core 中良定义，原样降到 MLIR 是 UB（越界读在 select 之前已执行）。

**做法**：
- 语义开关 `GUARD_LOWER`（进 sem_tag）：extract/insert/dim/cast 一律插守卫之后的 MLIR 语义——相关操作数是毒值 ⇒ UB（scf.if 以毒值为条件）；否则良定义走原 op，不良定义走替身（extract/dim 取 0，insert 返回原 tensor，cast 取 tensor.empty、元素为毒值，与 core 对 empty 的建模一致）。
- `guards.plan`：分别在 core 语义与 GUARD_LOWER 语义下推导同一程序，用 `core.refines(core, 插守卫后, 入口非毒值 ∧ 入口静态维, cross_sem=True)` 证明“插守卫后的程序精化 core 程序”；剩下的、未实现守卫的义务（切片、splat/empty 尺寸）按推测版义务必须可证。两条都成立才允许降低，否则判未知、不发射。
- `guards.insert_guards`：在 MLIR 模块上改写为 `scf.if (各维 i ult dim) {原 op} else {替身}`（cast：源的动态维 = 目标静态维）。
- 每个程序的降低都附带一个 z3 证明（精化），而不是对守卫充分性的一般性论证。

**证据**（MLIR 9bae84b0 JIT；生成器只作补充，真值是 JIT）：
- G0′：nsw 溢出得到的毒值下标流进守卫 ⇒ 规划拒绝；普通下标 ⇒ 接受。
- G1：120 个程序（静态/动态/二维 tensor；变量、常量、算术下标；extract/insert/dim/cast；70% 的读用无分支掩码），规划全部通过；比较 428/428 一致，其中**越界输入**（此前一律以“守卫”跳过）344/344。
- G2 A/B：同一批输入跑未插守卫的原模块，16 个程序的 JIT 进程崩溃（SIGSEGV、堆损坏）——守卫是必要的。

**语料（第 41 轮，guards_test G3）**：tensor 签名语料函数有事实 41 个，其中需守卫 3 个，规划全部接受；越界输入下这 3 个函数的输出全为毒值，可比 0 组；常规输入 9/9。语料中几乎没有“掩码读”（MLIR 作者不会写原生 UB 的代码），所以 L24 的证据主要来自生成器。

**边界**：切片类守卫未实现（替身 tensor 的形状须另行论证）；守卫条件的操作数可能是 MLIR 级毒值时拒绝（arith 无 freeze）。core 把 index 溢出也记作毒值（L17″），所以经 index 算术得到的下标常被拒绝——这是保守，不是错误。语料回归尚未换成插守卫后的全部输入比较。

### 13.1 第 43 轮：切片守卫与 S3 改写规则外部化（guards.py、guards_test G4/G5）
**更正记录**：此前的续写摘要与 clauses.md M3.4 写着“S3 两条组合律未修”，这是**误读**。shapes.md §4 的那两条（偏移相减、`extract(insert(·,i), j)`）是故意写错的对照规则，模型早已反驳（S3：有效 6 条全证、错误 2 条全反驳），不是待修的缺陷。真正的缺口是：这 8 条规则只在 core 语义内证明过，没有经过外部真值。

**切片守卫**：
- GUARD_LOWER 下，extract_slice.g 的替身为 tensor.empty（尺寸夹到 ≥ 0，元素为毒值），insert_slice.g 的替身为原 tensor。
- insert_guards 的条件逐维为：s ≥ 1 ∧ o ≥ 0 ∧ n ≥ 0 ∧ (n = 0 ? o ≤ e : o < e ∧ n−1 ≤ (e−1−o)/s′)，其中 s′ = (s≥1 ? s : 1)，用来避免除零。
- 刻意不用 o+(n−1)·s < e：index 运算会回绕，这个式子可能为真而数学上为假，那样就会放行越界。
- insert_slice 另要求源的各维等于对应尺寸；降秩的 insert_slice 目前判未知。

**G4 S3 规则外部化**：左边按 core 语义求值，右边插守卫后交 JIT 执行。输入用拒绝采样取得：左边有非毒值结果的 60 组，加上全毒值（越界）的 20 组。

| 规则 | 一致 | 不一致 |
|---|---|---|
| slice∘slice → slice(o1+o2) | 88 | 0 |
| ✗ slice∘slice → slice(o1−o2) | 81 | **7**（反例被找到） |
| extract(slice(x,o,n), i) → extract(x, o+i) | 60 | 0 |
| extract(insert(v,x,i), i) → v | 60 | 0 |
| ✗ extract(insert(v,x,i), j) → extract(x, j) | 31 | **29** |
| insert_slice(slice(x,o,n), x, o) → x | 176 | 0 |
| dim(insert_slice(s,x,o), 0) → dim(x, 0) | 60 | 0 |
| from_elements(v,v) → splat(v) | 179 | 0 |

**G5 全形式切片**：用 tensorjit 的 gen_slice 生成 80 个程序，含步长、降秩、动态偏移/尺寸/步长和写回。其中 55 个合法，规划接受 38 个；另有 17 个因降秩 insert_slice 被拒绝。
- 测试的是此前因守卫被跳过的 581 组越界输入。这些输入下输出整体是毒值，没有值可比，所以比较的是执行是否安全。
- 插守卫后：正常结束 52，运行崩溃 0。
- 未插守卫：正常结束 30，运行崩溃或异常 22。
- 另有 2 个程序在 MLIR 管线中**编译期断言崩溃**，原程序同样崩溃，与守卫无关，不计入。这类程序是带动态步长的降秩 extract_slice（`[1,1,1]` 尺寸、`tensor<3x1x4>` 降到 `tensor<1x1>`），只记录。

## 14 第 53 轮：L6 外部化（model/poison_test.py）

毒值不可观测，但“模型说有定义”可以检验。调用方在 MLIR 里真正制造毒值：`shli x, w`，移位量等于位宽，LangRef 与 LLVM 都规定其结果为毒值。把它作为实参传给复合 op f。f 是单独的 func.func，经 func.call 调用；O3 下 LLVM 会内联它，并把 `shl x, w` 折成 poison 常量再传播。模型推导调用方时只经 f 的摘要（摘要实例化），不看 f 的体。

- **P1**：600 个随机复合 f（arith 整数 op、nsw/nuw/exact、cmpi + select，i8–i64），每个程序至少毒化一个实参，4800 个输入（O0 与 O3）。模型给出确定值 614 个，与 JIT 一致 614/614；其中 100 个输入的毒值实参在数据流上可达结果，被 select 吸收。其余输入模型判毒值 2760、UB 1426，未定 0。
- **P2 消融**：f 的摘要换成不依赖输入毒值的版本（core_test X1 的 v3 式），同一批程序 O0 反例 689、O3 反例 328。O0 的反例只说明毒值的位不确定；O3 的反例来自 LLVM 真正利用毒值。
- **边界**：只检验“确定值”方向。毒值与 UB 方向是精化自由度，执行测不出（与 V5 相同）。浮点、tensor、ADT 的毒值层未外部化，所以 M3.9 仍为部分。毒值来源在第 53 轮只有 shli 一种，第 54 轮扩展（§14.1）。

### 14.1 第 54 轮：依赖数据的毒值来源（P3）

shl 的毒值恒定，LLVM 在 O3 下直接折成 poison 常量；依赖数据的毒值更接近真实程序：只在某些输入下为毒值，LLVM 只能借标志做推理。P3 新增五种来源：`addi nsw`（加 MAX，正数溢出）、`addi nuw`（加 −1，非零即溢出）、`divui exact`（÷2，奇数）、`shrui exact`（>>1，奇数）、`trunci nuw`（i64 截断，高位非零）。

计数口径收紧：“可达确定值”只计该输入下实参确为毒值（按 LangRef 具体求值）且在数据流上可达结果的确定值。

**阈值在运行前写定**：每种来源须满足程序 ≥ 100、反例 0、可达确定值 ≥ 10、消融 O3 反例 > 0。

- **首跑**（seed 54，600 个程序，每种来源 120）：反例全为 0，但 shrui_exact 的可达确定值只有 6 < 10，判 **FAIL**。阈值不改。
- **新样本**（seed 55，1500 个程序，每种来源 300，12000 个输入）：判 **PASS**。

| 来源 | 确定值 | 其中确为毒值且可达 | 反例 | 消融 O3 反例 |
|---|---|---|---|---|
| addi nsw | 768 | 90 | 0 | 37 |
| addi nuw | 528 | 28 | 0 | 319 |
| divui exact | 718 | 36 | 0 | 13 |
| shrui exact | 786 | 34 | 0 | 14 |
| trunci nuw | 948 | 36 | 0 | 9 |

其余输入：模型判毒值 5448、UB 2804，未定 0；JIT 失败、值层未知均为 0。

**批判**：新样本的规模是看到首跑 FAIL 之后定的（可选停止）。缓解有两点：反例在两次样本中都为 0；新样本换了 seed，与首跑独立。即便如此，这仍弱于一次就预先写定样本量。消融在 exact、trunci 来源上 O3 反例只有个位到十几，说明 LLVM 对这几种标志的利用较弱，检验对它们的灵敏度也相应偏低。

### 14.2 第 55 轮：ADT 降级后的毒值层（model/adtpoison_test.py，M3.9）

L11 的数据精化含毒值子句：¬P_A ⇒ ¬P_L ∧ abs(out_L) = out_A。A11 只在无毒值输入上检验过它的值部分。这里用 option 的随机程序（defs_test.gen_opt：构造、判别、投影、scf.if、divui），在 MLIR 里给降级分量 x̂ = (a, b, tag, val) 注入毒值（shli 位宽）。抽象输入的毒值按 abs_fn 计算：tag 毒化 ⇒ option 毒值；val 毒化只在 tag = 1 时使 option 成为毒值（none 的载荷是什么都行）。抽象层不经降级直接求值。

**阈值在运行前写定，且首跑即为最终结果**：400 个程序，每个程序抽 64 个输入，共 51200 个输入（O0+O3）。

- **Q1** PASS：抽象层给出确定值 13374 个，全部一致，反例 0。其中抽象输入含毒值的 11060 个（未做数据流可达过滤，包含未使用的情形）；只有降级输入含毒值的（none 载荷被毒化）2314 个。其余输入模型判 UB 29986（主要是 divui 除数为毒值，LLVM 的 udiv 也把它当 UB）、毒值 7840。JIT 失败 0。
- **Q2** PASS：抽象层不看输入毒值的消融，O0 反例 1460，O3 反例 8266。
- **Q3** PASS：抽象函数忽略 tag 毒值的消融（ADT 专属），O0 与 O3 各 650。O0 与 O3 相同，说明这些差异来自 i1 毒值的位，而不是优化器的推理。

**发现（精度，不影响可靠性）**：抽象层里 `option.some(毒值)` 整体是毒值，所以 `is_some(some(毒值))` 也是毒值；降级后 tag 是常量 1，结果确定。也就是说，抽象层比 MLIR 更“毒”，方向可靠，代价是精度。

**边界**：只覆盖 option 一种 ADT；L12 的接口定律方向没有在毒值输入下单独检验；“抽象输入含毒值”的计数没有排除未使用的输入。

## 15 第 56 轮：管线白名单差分（model/pipezoo_test.py，M5.1，L55）

判据：Sem(g) ⊑ Sem(pipeline(⟦g⟧))。在模型给出确定值的输入上，经管线 P、再接标准降低之后的 JIT 结果必须等于这个值。

**程序族**
- **A**：复合 op 以不透明方式落地（func + func.call，M5.2），调用方随机注入 poison_test 的六种毒值来源，也可能不注入。
- **B**：loop_test.gen 生成的程序，含 i8/i32/i64/index、类型转换、select、静态 scf.for（含嵌套）。

**执行与归因**：每个程序在 fork 子进程里跑完全部管线。归因按 L52 分三类：
- 模型反例：基线 JIT 与模型确定值不同。
- 变换偏离：基线与模型一致，管线后不同。
- 管线失败：单列，不计入分母。

管线清单与阈值都在运行前写定。

**结果**：两族各 400 个程序，基线确定值 1992，模型反例 0（Z0 PASS）。

| 管线 | 程序 | 确定值 | 偏离 | 失败 |
|---|---|---|---|---|
| inline、canonicalize、cse、sccp、loop-invariant-code-motion、control-flow-sink、symbol-dce、scf-for-loop-canonicalization、scf-for-loop-specialization、int-range-optimizations、arith-unsigned-when-equivalent、arith-expand、组合（inline,canonicalize,cse,sccp,LICM,canonicalize） | 各 800 | 各 1992 | 0 | 0 |
| scf-for-loop-peeling | 800 | 1992 | **11**（基线 O3 也与模型一致：11/11） | 0 |
| remove-dead-values | 555 | 1633 | 0 | **245** |

**MLIR 缺陷账（只记录，不提 upstream）**
- **scf-for-loop-peeling（向后剥离）**：拆分点为 ub − ((ub − lb) mod step)。当 lb > ub 时，这个点落到 lb 之下，于是本应为空的循环被剥出一次迭代并执行。最小例：lb=2、ub=1、step=2，执行后 10 → 11；peel-front 不受影响（Z2）。upstream main 的 peelForLoop 已改为 ub − ((ub − min(lb, ub)) remui step)，并在常量 ub ≤ lb 时直接放弃剥离，所以该缺陷在 9bae84b0 之后已修。
- **remove-dead-values**：调用方把实参传给公开 callee，callee 不使用该参数时，这个 pass 会删掉调用方里实参的定义，留下操作数为空的 func.call，于是 pass 失败，报 "null operand found"（Z3 最小例）。upstream 现状未查。

**白名单（9bae84b0，对上述程序族）**：上表中偏离为 0 的 13 条管线，加上 bufferize、scf→cf、作用域降低三条。peeling 与 remove-dead-values 不在白名单内。

**边界**
- 程序族不含 tensor、memref、浮点和多块 cf。
- 每个程序只取 4 个输入。
- peeling 的偏离只在 B 族出现，因为只有 B 族含 scf.for。
- 白名单只对已检验的程序族成立；扩到新程序族要重新做差分。
