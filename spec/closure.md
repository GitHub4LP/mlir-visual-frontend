# 组合封闭的外部检验（第 22 轮）

实现：`model/composite.py`（复合 op 的判定）、`model/irdl_export.py`（导出为 IRDL op）。
检验：`model/compcorpus_test.py`（`PYTHONPATH=model python3 model/compcorpus_test.py 12 6`，需要 corpus/）。

## 1 复盘：L25–L27 承载在没有拒绝权的层上
- 第 8 轮的复合 op 签名（compsig.py）用 typesig.py 作节点检查器。
  - typesig 是第 7 轮的人工签名：只覆盖 23 个 op，.td 取自 inventory/，不是被测提交。
  - 第 12 轮已经查明 typesig 在真实 IR 上误拒 203 个函数，因此按 L33 降为只发警告。
- 结论：用户要求的“组合出的 op 与原生同地位”，至今建立在一个已被剥夺拒绝权的层上。
  - 复合 op 的体也只能用这 23 个 op。
  - C1 的“Φ ≡ 体 ≡ MLIR 1734/1734”是在自己的 23 op 生成器上得到的，属于 B 级证据，而且承载层本身就不对。
- 本轮的处理分两步：
  - 先在真实 IR 上量化这个问题；
  - 再把复合 op 的判定改到有拒绝权的层上（composite.py）。compsig 只保留呈现与端口投影两项职责。

## 2 做法：从 MLIR 测试集里抽复合 op
- **抽取**：取语料中 MLIR 接受的 func.func 的入口块，把其中连续的 1–4 个 op 当作“提取为复合 op”的实现体（与 edit 的 extract 相同）。
  - 端口：体外定义、体内使用的值。
  - 结果类型：ODS 有推断规则的节点，结果写成推断，随端口类型变化；其余节点的结果是字面量。
  - 这些复合 op 的形状来自 MLIR 的测试作者，不来自本项目的生成器。
- **实例**：
  - T0 = 原类型；
  - 变异 = 把一个端口的类型换成同一段 IR 里出现过的类型，或者换元素类型。内部类型按 ODS 规则前推。
- **真值**：把复合 op 在 T 上内联成独立的 func，交同一提交的 MLIR verifier（在隔离子进程里运行）。
  - 原类型实例如果独立后被拒（依赖上下文），就跳过并计数。
- **被测的三个判定**：
  - Φ_ods：自动签名 + 属性层，不开预言机。这是有拒绝权的符号层，覆盖 19 个方言。
  - Φ_full：Φ_ods 加隔离单元预言机。这是透明模式的判定。
  - Φ_ts：第 8 轮的节点检查器，只在 23 个 op 内可表示。

## 3 不透明身份：导出为 IRDL op
- 复合 op 导出为 `irdl.dialect @vf` 里的 op。实例是对该 op 的一次调用，先交 MLIR 的 IRDL verifier，再跑 canonicalize + cse，检验它确实被当作 op 处理。
- **本提交的 IRDL 对 builtin 类型能表达什么**（实测）：
  - `irdl.any`；
  - `irdl.is <具体类型>`；
  - 同一个约束变量用在多个端口上，表示类型相等；
  - `irdl.base "!builtin.integer"` 或 `"!builtin.tensor"`，只能单独使用。
- **不能表达的**：
  - builtin 容器的参数（元素类型、形状、秩）。`irdl.parametric` 只对 IRDL 自己定义的类型可用。
  - 端口间除相等以外的任何关系：getElementType、getI1SameShape、`with(x, t)`、形状相容、Elementwise。
  - C++ verify。
- **IRDL 自身的缺陷**：
  - `any_of(base, base)` 在加载时触发 UNREACHABLE（IRDLLoading.cpp:541），进程直接退出；
  - `base "!builtin.ranked_tensor"` 加载失败。
  - 所以导出只使用 any、is 和相等。
- **因此导出是上近似**。丢掉的约束逐条记账，作为“导出缺口”。这些约束留在前端，由透明判定承担。

## 4 结果
语料：246 个文件，抽出 10710 个切片，覆盖 19 个方言。其中只有 1014 个（9.5%）能用 typesig 的 23 个 op 表示；全部节点都是 auto 账目的有 1830 个。
原类型实例 10703 个，MLIR 全部接受；变异实例 40681 个，MLIR 接受 12603 个。

| 判定 | 原类型（10703） | 变异（40681） |
|---|---|---|
| Φ_ods 误拒 | **0** | **0** |
| Φ_ods 漏判（残差，交给预言机） | 0 | 6774（修复后） |
| 全 auto 复合 op：Φ_ods ≠ MLIR | 0 / 1829 | 修复前 **38** / 9145，修复后 **0** |
| Φ_full ≠ MLIR（透明判定） | 0 | 修复前 38，修复后 **0** |
| Φ_ts 误拒（第 8 轮的承载层） | **240 / 1014（24%）** | 469 / 5649 |
| IRDL 导出：误拒 | 0 | 0 |
| IRDL 导出：MLIR 拒绝的 28078 例中抓住 | — | **896（3.2%）** |
| 能无缺口导出为 IRDL 的复合 op | 251 / 10703（2.3%） | 0 |

- **Φ_ts 的误拒例**：
  - `extsi : i0`，MLIR 接受；
  - `cmpi : vector<i64>`，0 维向量；
  - extract_slice 的 _slice_res，人工改写的 C++ 条件。
  - 这些正是第 12 轮人工签名误拒的同类问题，在复合 op 上原样复现。
- **全 auto 的 38 例分歧**：这是 L28（auto 精确）在真实类型上的反例，方向是“前端判定合法，MLIR 拒绝”，属于不可靠方向。两类原因：
  1. SameOperandsAndResultType 要求 tensor 的编码相同。例：`mulf(tensor<8x8xf64, "foo">, tensor<8x8xf64>)`。
  2. verifyCompatibleShape 要求 vector 的可伸缩维一致。例：`vector<1x[4]xindex>` 与 `vector<1x4xindex>`。
  - 自动签名比较类型时忽略了修饰（第 12 轮的“上近似”假设）。在接受方向上，这个假设对“精确”是错的。
  - 修复：`typesig.compat_shape` 比较可伸缩维；`odssig` 的 samec_all 比较编码。修复后 0 / 9145。
  - 回归：C2 3487 个函数仍然误拒 0；M3 19641 个变异误拒 0、漏判 0；odssig L1 的 auto 7640/7640。
- **IRDL 漏判的归因**：每个漏判实例至少含一类导出缺口。按出现次数，最多的是：
  - C++ verify；
  - 端口约束：Index、AnyVector、AnyShaped、FloatLike……
  - Elementwise、samec、shape、tdims 等关系；
  - TypesMatchWith 的变换。
  - 详细计数见 compcorpus_test 的输出。
- **身份**：10703 个原类型实例里，导出的 op 都被 MLIR 接受，并且经过了 canonicalize + cse 而不出错。IRDL op 作为不透明身份是可用的。

## 5 定律
- **L46（auto 精确在组合下封闭）**：复合 op 的每个节点都是 auto 账目时，Φ_ods ≡ MLIR，不需要预言机。真实切片上，修复 L28 的修饰缺陷后为 10974/10974。
  - 推论：复合 op 的 Φ 不需要新的判定机制。由原生判定的合取，加上前推推断得到的类型，就足以判定。“与原生同地位”在类型层的含义因此是可检验的。
- **L47（判定承载层）**：复合 op 的判定必须用有拒绝权的层：结构层、自动签名与属性层、预言机。
  - 人工签名（typesig）在复合 op 上照样误拒（§4），所以 compsig 的 Φ 降为呈现与端口投影，不再判定。
  - 这是 L33 在组合上的推论。
- **L48（IRDL 是不透明身份，不是判定）**：导出的 IRDL op 不误拒（上近似），但只能抓住极少数错误。
  - 前端对复合 op 的判定严格强于 MLIR IR 所能表达的。L26 的 `with` 只是其中一例；实际上相等以外的全部类型关系都在 IR 之外。
  - 所以不透明模式（IRDL op、func + call）下，复合 op 的使用点仍须由前端用 Φ 判定，不能交给 MLIR 的 IRDL verifier。
- **L25 改写**：Φ(T) ⇔ 内联后逐节点用有拒绝权的层检查 ⇔ MLIR。证据由 B 级升为 A 级（§4）。
- **L27 待重做**：端口投影目前仍建立在 typesig 上。在 Φ_ods 上做投影需要 odssig 的端口谓词的符号形式，留到下一轮。
