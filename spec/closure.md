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
- **L27**：第 24 轮已改建在 Φ_ods 上，见 §7。


## 7 端口投影改到有拒绝权的层上（第 24 轮）
实现：`composite.project(体, 端口, p, t, others=None)`。测试：`model/projcorpus_test.py`。

前端动作是“把类型为 t 的线接到端口 p”。投影只回答两种结果：**no**（不存在补全）或 **maybe**。它只用必要条件，所以 no 是可靠的：
- (a) 已知类型的操作数满足自动签名的端口谓词。遇到未登记的原子（L29）或开放世界接口（L43）时取 ⊤。
- (b) 限制在已知位置上的“全体相等 / 同形 / 同元素”关系。全体成立，子集必然成立。
  - 例外是 SameOperandsAndResultType。MLIR 是拿每个类型逐个和 result(0) 比，不是两两比较。所以 result(0) 未知时，只有元素类型和编码能传递。
  - 同一处修正了 `odssig.check`：它原来按两两比较，比 MLIR 更严。有 Elementwise 的 op 不受影响，因为 Elementwise 本身就要求形状全等。修正后 odssig_test 的 auto 部分仍是 3820/3820。
- (c) 操作数全部已知的节点，完整走 `odssig.check` 和属性层，与 `core.typecheck` 是同一判定。

“已知”的值包括：端口 p、`others`、字面量结果，以及由已知操作数经 ODS 推断出的结果。declared 的 C++ 部分不作为拒绝条件；未知端口参与的其他关系也不作为拒绝条件。

**检验**：切片取自 MLIR 测试集（5543 个，其中 3381 个多端口）。可连性的真值是 ∃ 补全，只能证伪：
- 对每个 (p, t)，其余端口取原类型、t 本身或同文件类型池里的抽样，最多尝试 6 个补全，交同一提交的 verifier。

| 判据 | 结果 |
|---|---|
| 原类型 (p, t)：判 no 的 | **0 / 11806** |
| 候选 (p, t)：判 no 但找到了被接受的补全（不可靠） | **0 / 24070** |
| 候选中未找到补全的：投影判 no 的 | 15060 / 21317（70.6%） |
| 其余端口固定为原类型时：投影 no ⇔ Φ_ods 拒绝 | 14821 = 14821；投影 no 而 MLIR 接受：0 / 29086 |

- **L27′（A）**：投影可靠，而且相对 Φ_ods 完备。其余端口全部已知时，它就是 Φ_ods。剩下的不完备有两个来源：
  1. 尚未确定的端口参与的关系，这部分本质上要等那些端口确定；
  2. declared 的 C++，这部分交给预言机。

  未找到补全的 29.4% 里，哪些真的无补全，搜索无法区分，所以 70.6% 只是下界。
- 第 8 轮的 Φ_ts 投影已退役。它的节点检查器在真实切片上误拒 24%（§4）。在 15 个文件的小样本上，Φ_ts 投影还给出了 85 个不可靠的 no。
- **测试自身的教训**：第一次全量运行报了 6 个“不可靠反例”，原因是测试把真值缓存在 `id(切片)` 下，而切片回收后 id 会被复用。这 6 例在 MLIR 里直接验证都是拒绝。现在缓存按切片清空。“0 反例”之前，先要确认反例本身是真的。

**IRDL 边界独立成测**（`model/irdl_boundary_test.py`，14 例，与本提交实测相符，崩溃在子进程里隔离）：
- 新确认：`any_of(is, is)` 可用。
- `any_of(base, base)` 会触发 UNREACHABLE。
- `base "!builtin.ranked_tensor"` 加载失败。

MLIR 升级后，IRDL 能力的变化会先在这里报出来。

## 8 透明内联与自动分配作用域（第 37 轮，M5.3；model/core.py inline_all，allocascope_test.py）
此前 M5.3 只在规范中：inline_all 把复合 op 体里的 memref.alloca 直接摊进宿主。

**实现**：复合 op 的体（不进入内层作用域 op）含自动作用域分配 ⇒ 内联结果包一层 `memref.alloca_scope`，结果经 `alloca_scope.return` 带出。两个集合取自 ODS：分配者 = 结果装饰器 MemAlloc<AutomaticAllocationScopeResource>（本提交只有 memref.alloca；llvm 方言未导入）；作用域 op = AutomaticAllocationScope trait（affine.for/parallel、func.func、gpu.func/launch、memref.alloca_scope、scf.for/forall/parallel、shape.func）。K0 对账一致。
**逃逸**：体内有 alloca 且复合 op 有 memref 输出 ⇒ 包作用域后可能悬垂，判定未建模 ⇒ emit_core 返回 None 并记“内联失败：alloca 逃逸”（K3），不静默发射。

**证据**（MLIR 9bae84b0 JIT，fork 子进程；体内 alloca 16 KB，宿主 20 万次迭代）：
| 宿主 | 小迭代：包/不包 | 大迭代：不包 | 大迭代：包 | 包 + canonicalize |
|---|---|---|---|---|
| scf.while 体 | 正确/正确 | 崩溃（栈耗尽） | 正确 | 作用域保留，正确 |
| scf.while 内 scf.if 体 | 正确/正确 | 崩溃 | 正确 | 作用域保留，正确 |
| scf.for 体 | 正确/正确 | 崩溃 | 正确 | **作用域被消去，崩溃** |

参照（K4，不作真值）：MLIR 自己的 inline pass 把 func.call 内联进这三种宿主时都不包 alloca_scope，大迭代全部崩溃；原样 func.call 全部正确。

**MLIR 的已知不一致（只记录）**：scf.for 带 AutomaticAllocationScope trait，按 trait 语义每次迭代结束释放 alloca；canonicalize 据此把 alloca_scope 里的 alloca 提升到 scf.for（AllocaScopeHoister/内联），但 convert-scf-to-cf 不为每次迭代 stacksave/stackrestore ⇒ 栈随迭代增长。上游讨论“Should scf.for have AutomaticAllocationScope trait?”（discourse 2023-10，PR #69765）至今未定论。
**结论**：值层 Sem 不含栈资源，按 M5.1 的精化判据这不是语义违反；但它是降级后资源性质的真实失效。前端对 scope 宿主（scf.for/forall/parallel、affine）中含 alloca 的复合 op 目前**没有可靠保证**——M5.3 状态为“部分”。第 38 轮已在降级管线中修复，见 §8.1。

### 8.1 降级管线：作用域落成 stacksave/stackrestore（第 38 轮，model/lowering.py）
**为什么不选“提升到函数入口”**（第 37 轮的候选缓解）：scf.parallel/forall 的迭代可能并发执行，共用一块缓冲是数据竞争；动态尺寸的 alloca 也提不出去。否决。
**为什么不在 IR 层再包 alloca_scope**：canonicalize 会再次把它消去（AllocaScopeInliner：父 op 是作用域、且它是终结符前最后一个 op）；而且发现了第二个缺陷——
- **D2**：alloca_scope 是 SingleBlock。体内有 scf 控制流时，scf-to-cf 先跑 ⇒ 体变成多块，校验失败；memref-to-llvm 先跑 ⇒ 多块被内联进外层 scf 的单块区域，同样失败（K9；func 顶层的 alloca_scope 只有在 memref 先降时才能通过）。这意味着第 37 轮的包裹在“复合 op 体内有控制流”时会让标准管线失败，§8 的表格只覆盖了体内是直线代码的情形。

**做法**：在全部优化之后、方言降低之前，跑 `lower_alloca_scopes`：每个 alloca_scope ⇒ stacksave + 内联体 + stackrestore；体内（不进入内层作用域 op）有自动作用域分配的 scf.for/scf.parallel ⇒ 体首 stacksave、终结符前 stackrestore。这正是 trait 与 alloca_scope 的语义，也是 MemRefToLLVM 对 alloca_scope 的降低本身；llvm.intr.stacksave/stackrestore 没有区域，任意嵌套、任意降低次序下都合法。代价：降低后的 IR 依赖 LLVM 方言（JIT 与 WASM 都经 LLVM）。多块循环体不处理，报原因。

**证据**（JIT；20 万次迭代，16 KB alloca）：
- K6：三种宿主在“包 + canonicalize + 降低”后，小迭代和大迭代全部正确（scf.for 宿主在第 37 轮是崩溃的）；直接写在 scf.for 体内的 alloca：原样崩溃，降低后正确。
- K9：复合 op 体内有 scf.if，宿主为 scf.while：标准管线校验失败；降低后小迭代、大迭代都正确。
- K8（自生成器，只作补充；真值是 JIT）：随机嵌套 scf.for/parallel/if/while，alloca 位置随机，共 60 个程序。canonicalize 后与降低后比较 120/120 一致；外层 2 万次迭代：降低后 60/60 正确，原样 45 个崩溃，两者都正常的那些程序值也一致。
- K7：语料中含 scf.for/parallel、可解析且可校验的 296 段，降低后 verify 全部通过；但只有 1 个循环真的被改写，覆盖很薄。

**仍然开放**：alloca 逃逸未建模（拒发）；MLIR 自己的 inline pass 在非作用域宿主（scf.while）里不包作用域（K4）——前端不使用它，只作参照；D1、D2 只记录，未提 upstream。


## 9 复合 op 的显式签名 Φ_comp（第 62 轮，model/odscomp.py，odscomp_test.py）

这是用户核心需求的直接对象：实现图定下来后，复合 op 签名允许的最大类型约束范围与端口间关系由图推出，形式与 MLIR op 签名相同，可呈现、可嵌套。
此前只有逐点判定（§2：给定端口类型 T，内联后逐节点检查）与端口投影（§7）；第 8 轮的符号签名 compsig 建在人工签名上（误拒 240/1014，L47），只作呈现。

**定义**。Φ_comp(端口) = ∃内部值类型. ⋀_节点 Φ_节点。内部值类型由 ODS 推断规则（复制、TypesMatchWith 变换、builderCall）确定，∃ 消去即代换，
结果是端口类型变量上的原子合取：端口约束原子、关系原子、属性-类型耦合原子（不透明，逐点求值）。不能表达的部分列为 **Unknown 残差**，不参与求值；
残差非空时 Φ 是上近似，须标 Unknown 并交预言机（§2 的 Φ_full）终判。嵌套：内层 Φ 的原子与输出表达式经代换搬入外层（组合封闭）。

### 9.1 对账：ODS 约束种类 → Φ_comp 中的处理

| MLIR 机制 | Φ_comp 的处理 | 内部变量消去 | 状态 |
|---|---|---|---|
| TypeConstraint（Pred 树 + 内建原子） | 端口约束原子 | 代换 | 已覆盖；未导入的原子（如 `isF16()`、部分 `isa<…>`）⇒ Unknown |
| AllTypesMatch / AllElementTypesMatch / AllRanksMatch / AllShapesMatch | 关系原子 | 代换 | 已覆盖 |
| TypesMatchWith（登记的变换：恒等、i1 同形、元素类型、complex 元素、按元素数展开） | 关系原子 + 推断表达式 | 精确（结果类型是端口表达式） | 已覆盖；未登记的变换（lambda、FunctionType 转换）⇒ Unknown |
| Same* / Elementwise（native trait） | 关系原子 | 代换 | 已覆盖；未导入的 native trait（如 SameVariadic…）⇒ Unknown |
| ShapedTypeMatchesElementCountAndTypes | 关系原子 | 代换 | 已覆盖 |
| 无字段 PredOpTrait | 登记模板（tdims、位置式 elt_eq、cond_shape）⇒ 关系原子 | 代换 | 其余 ⇒ Unknown |
| ODS 自动 inferReturnTypes | 结果表达式 | 精确 | 已覆盖 |
| 属性约束（attrsig） | 与类型无关的部分是常量原子（化简时求值，假 ⇒ ⊥）；与类型耦合的部分是属性原子 | — | 已覆盖（属性原子不透明，只能逐点求值，不展开为可读关系） |
| 可变端口组（Variadic/Optional/段属性） | 按实际个数绑定；结果侧无法绑定时与 odssig 同样只查操作数侧 | — | 已覆盖；无法绑定 ⇒ Unknown |
| C++ verify()/verifyRegions()、接口 verify、C++ inferReturnTypes | 不可表达 | 不可消去 | **Unknown**（主要缺口，见 9.3） |
| 结构 trait（HasParent、IsolatedFromAbove、终结符…） | 结构层（structure.py/ctxcons.py），不在类型 Φ 内 | — | 本检验的切片不含区域 op，未在此检验 |

约定：结果类型由用户选择、ODS 不推断的节点（如 `arith.trunci` 的目标类型），其结果类型是节点参数，由图固定为字面量（与属性同等）。把它提升为复合 op 的类型参数（compsig 的 `with`）是扩展，不在本轮。

### 9.2 外部检验（阈值运行前写定，见 odscomp_test.py 文件头）

语料 246 文件，分两段（0–124、125–245），每函数 6 个切片，每切片原类型 + 4 个变异；真值是同一提交的 MLIR verifier 对内联结果的判定。

| 判据 | 结果 |
|---|---|
| S1 化简 ≡ 原始 | 27960/27960，不一致 0 |
| S2 Φ ≡ 逐点判定（instantiate + core.typecheck，不开预言机） | 不一致 0/27960，推导异常 0 |
| S3 对 MLIR：误拒 0；残差为空时误收 0 | 实例 27803（MLIR 接受 12412）；误拒 0；误收 4034，全部在残差非空的复合 op 上 |
| S4 嵌套 ≡ 展开（前半作内层） | 6840/6840 |

复合 op 5592 个，残差为空 1259（22.5%）；化简把原子数从 44903 减到 26065。

**过程记录（如实）**：首跑 S1 FAIL 55（命名关系的去重键用了全部端口组而非被引用的组，tensor.insert 的两条 TypesMatchWith 被当成同一条）、S2 FAIL 49
（结果组无法绑定时没有像 odssig 那样只查操作数侧，memref.extract_strided_metadata）。修正后二跑全 PASS，但发现 206 个切片因推导异常被静默排除
（常量折叠在空端口赋值上求值未引用的组）。修正并把“推导异常 = 0”补进 S2 的判据（事后补的判据，只会更严），三跑即上表。

### 9.3 结论与边界

- 对残差为空的复合 op，Φ_comp 是 MLIR 判定的精确符号形式：在本检验中既不误拒也不误收。对其余 77.5%，Φ_comp 只是上近似，必须显示为 Unknown 并交预言机终判；这时它仍然不误拒。
- 主要缺口是 C++ verify 与接口 verify（MLIR 本身就没有声明式表达）。按声明范围计：若前端词汇表只用 auto 账目的 op，Φ_comp 精确；用到 declared 的 op，签名必然带 Unknown 残差。这是 MLIR 机制本身的边界，而不是前端模型能补上的。
- 呈现（render）给出每个端口的一元约束、端口间关系和 Unknown 残差。一元约束只是必要条件：在其余端口未定时，单端口的精确可取范围（∃ 其余端口）仍按 §7 投影处理，相对 Φ_ods 完备的结论只在其余端口已知时成立。
- 未做：属性原子的可读展开；残差为空时的关系最简式（目前只做去重与自反消去）。

### 9.4 单端口最大类型范围 R_j（第 63 轮，model/odscomp.project / range3，model/projcomp_test.py）

R_j(t) ≝ ∃ 其余端口. Φ_comp，即用户需求里“签名允许的最大类型约束范围”在单个端口上的投影。前端据此回答“这个端口能连什么类型”。

- project：精确相等关系（AllTypesMatch、SameTypeOperands，参数都是端口本身）把端口并成等价类。只涉及本类的原子构成必要条件 N_j。若没有原子把本类与类外端口相连，则标**精确**，R_j = N_j。
- range3（三值）：N_j(t) 为假 ⇒ **out**；否则在 {原类型, t} 中为其余端口逐个取值找见证，找到 ⇒ **in**（附见证）；都没找到 ⇒ **unknown**。SameOperandsAndResultType 等是“兼容”而非“相等”，所以二分只有约三成端口为精确，三值几乎都能确定。
- 范围：只取 Unknown 残差为空、原类型被 MLIR 接受的复合 op（§9.3：Φ ≡ MLIR）。候选 t 为本文件出现过的类型中随机取 6 个，加上原类型。

全语料（246 文件，每函数 4 切片；两段 `4 125 0` 与 `4 1000 125`）：

| 判据 | 结果 |
|---|---|
| P1 精确端口：R_j(t) ≡ MLIR(T0[类←t]) | 比较 3204（接受 2038），不一致 0 |
| P2 上近似端口：N_j 为假 ⇒ 类型池上无补全 | 检查 1546，违例 0，未穷举 107 |
| P4 in：见证交 MLIR 判定 | 6269/6269 接受 |
| P5 out：类型池上穷举无补全 | 检查 2712，违例 0，未穷举 107 |
| P3 三值分布 | in 6269，out 2819，unknown 239（2.6%） |
| P3 端口 | 复合 op 1025，端口 1883：精确 547，上近似 1336；等价类合并 25 |

过程（如实记录）：
- P1/P2 的阈值在试跑前写定。12 文件试跑中，精确端口只有 29/477，因此加入 range3，并在全量运行前写定 P4/P5。
- P4 试跑 1498/1498 FAIL，原因在测试本身：把解析后的类型元组当作见证交给 MLIR。修正后 PASS。
- holds 改为只要求被引用的表达式有值（投影时类外端口为 None）。odscomp_test S1–S4 已全量回归，PASS，数字与第 62 轮一致：复合 op 5592，残差为空 1259。

边界：
- out 只有两重证据：N_j 是 Φ 的子合取，所以按构造可靠；外加类型池上的穷举。P5 不是在全部类型上的证明。
- unknown 只说明见证搜索有界，并不意味着范围更小。
- 残差非空的复合 op 不在本节范围：R_j 对它们只是上近似，最终由预言机判定。

### 9.5 定值残差（第 64 轮）

**按 op 统计残差**：残差非空的复合 op 有 4300 个，残差来源呈长尾分布。
- 残差只来自单个 op 的复合 op 中，只修前 10 个 op 能清空 1089 个，修前 40 个能清空 2234 个，共涉及 121 种 op。
- 逐个手写 C++ verify 的代价无上界，因此不做。

**定值残差**：若一个节点的全部操作数和结果类型都不含端口变量，它的不透明校验（C++ verify、接口 verify 等）就与端口类型无关，结果是常量。
- 典型例子是 arith.constant，以及只作用于常量的节点。
- 这类残差在复合 op 定义时对实现图校验一次（预言机）即可消去。实现上，odscomp 把它们记入 `P.ground`，与 `P.residual` 分开。
- 嵌套时，内层的定值残差在外层仍是定值。内层的非定值残差保守地留作残差。

**检验**：阈值沿用 §9.3/§9.4，不变。S3 只在原类型被 MLIR 接受后计数，这恰好就是“定义时校验一次”。

| 判据 | 结果 |
|---|---|
| odscomp S1–S4（27960 实例） | 全部 PASS；S3 误拒 0，残差为空时误收 0 |
| 残差为空的复合 op | 1259 → 1528 / 5592（22.5% → 27.3%），其中 269 个只有定值残差 |
| projcomp P1 | 3945 例，不一致 0 |
| projcomp P4 | 6776/6776 经 MLIR 证实 |
| projcomp P5 | 3219 例，违例 0（未穷举 107） |
| projcomp 三值未定 | 266 例 |

**结论**：剩余 72.7% 的残差来自依赖端口类型的 C++ verify 或接口 verify。MLIR 本身对这些 op 也没有给出声明式约束，所以复合 op 在这里与组成它的 MLIR op 处于同一不透明状态：可以由预言机判定，但没有符号形式。

复合 op 的签名由 ODS 能表达的部分精确推出，其余部分精确地标为 Unknown。这符合“与 MLIR op 同样对待、不弱于 MLIR”的要求；要在这之上做得更强，就得逐个 op 手写 C++ 语义。

### 9.6 规范化 C++ verify 规则（第 66 轮）

用户 2026-11-04 明确：MLIR 只写在 C++ verify / 接口 verify 里的类型规则也要建模（“这就是我说的超越”）。§9.5 的结论“剩余部分精确标 Unknown 即可”据此作废。

**机制**：`model/nfrules.py` 把规则写成约束原语的合取，作为 `('nf', 端口组, 'nf', 呈现, 判定)` 挂在 odssig 签名的关系表上。core 的逐点判定、odscomp 的 Φ_comp、projcomp 的单端口范围共用同一份规则，内置 op 与复合 op 用同一机制。一个 op 只有在 nf_test 中与 MLIR 精确一致后才会登记；登记后，它覆盖的 declared 条目从残差中移出。

**原语**：under（标量或允许的容器，且元素种类合法）、width（整数位宽或浮点 getWidth）、same_enc、compat（verifyCompatibleShape）、signless、is_index、no_unranked_tensor。

**本轮覆盖**：arith 转换族 12 个 op（extui/extsi/trunci/extf/truncf/uitofp/sitofp/fptoui/fptosi/index_cast/index_castui/bitcast），覆盖 `C++ verify()` 与 `接口 verify: CastOpInterface`。依据 MLIR 9bae84b0 的 ArithOps.cpp。

**补全（超出 MLIR 的部分）**：任一端为无秩 tensor 时，MLIR 在这 12 个 op 上都会断言崩溃（cast<RankedTensorType>），不能产出合法 IR。模型把这种情况规范化为拒绝。

**检验**（nf_test N1，阈值运行前写定）：按特征等价类构造类型池。容器 15 种（标量、vector、可伸缩维、2 维、tensor 静态/动态/无秩、两种编码、memref、无秩 memref、strided 布局、内存空间），元素 13 种。同容器下元素两两全组合，加上容器两两全组合乘 4 个代表元素对。

| 判据 | 结果 |
|---|---|
| N1（12 op） | 比较 40500，MLIR 接受 2345；误拒 0，误收 0；MLIR 崩溃 868（全部按拒绝计） |
| 消融：只用 ODS | 误收 2990（规则确有作用；fptosi/fptoui 的 ODS 本身已精确） |
| odscomp S1–S4（27960 实例） | 全 PASS；S3 误拒 0，残差为空时误收 0 |
| 残差为空的复合 op | 1528 → 1911 / 5592（27.3% → 34.2%） |
| projcomp P1/P2/P4/P5 | 全 PASS（P1 5859 比较，不一致 0） |
| odssig_test L1 | 误拒 0（21878 实例），auto op 分歧 0 |

**过程（如实记录）**：
- 试跑 arith.extsi 时，原阈值“MLIR 崩溃 0”FAIL（无秩 tensor 崩溃 57 例）。阈值改为“崩溃按拒绝计”，规则加入 no_unranked_tensor。这一改动发生在试跑之后、全量运行之前。
- 首次全量用的是 11 种容器。之后又补了布局、内存空间、第二种编码、2 维 vector，复跑后仍 PASS。

**边界**：
- 穷尽性只相对于所选特征成立。其余特征（例如更多布局形式、方言自定义类型）未枚举。规则不读这些特征，但 MLIR 是否同样不读，没有被检验。
- 剩余残差仍来自其他 op，按 resid_stat 的长尾顺序逐批规范化：vector.transfer_read/write、tensor.extract_slice/insert_slice、memref.subview 等。

### 9.7 第二批：tensor / memref / vector（第 67 轮）

**机制扩展**：规范化规则可以读取 op 的属性，写作 `('nf', 端口组, 'nf', 呈现, 判定, 需要属性)`。属性取自节点（`attrsig.from_core`）：core 逐点判定直接传入属性；odscomp 把属性记在 rel 原子上。复合 op 内部节点的属性由实现图固定，所以端口类型的谓词仍是精确的。这一点对应用户要求的“签名间约束（属性）与类型约束的交互”。没有属性信息时（属性为符号参数），需要属性的规则不判，取上近似。规则在端口不满足 ODS 基本约束时无定义，按拒绝处理，因为此时 ODS 约束本身已经拒绝。

**覆盖**（依据 MLIR 9bae84b0 C++ 源码）：

| op | 规则 | 被覆盖的 declared |
|---|---|---|
| tensor.dim | result = index | 接口 verify: ShapedDimOpInterface |
| memref.load / store | #indices = rank(memref) | C++ verify() |
| tensor.cast | elt 相同 ∧ compat（不比较编码） | 接口 verify: CastOpInterface |
| vector.broadcast | isBroadcastableTo（标量 = elt；尾部维度相等或源为 1；可伸缩标志规则） | C++ verify() |
| vector.shape_cast | elt 相同 ∧ numel 相同 ∧ 可伸缩维个数相同 | C++ verify() |
| vector.transpose | rank 相同 ∧ permutation 是排列 ∧ dim(result,i) = dim(src,perm[i])（不比较可伸缩标志，与 MLIR 一致） | C++ verify() |

**检验**（nf_test N1，每个 op 有自己的生成器，包含属性取值：13 种 permutation，含非法排列）：

| 判据 | 结果 |
|---|---|
| N1（本批 7 op） | 比较 5006，接受 202；误拒 0，误收 0；消融误收 2292 |
| N1（累计 19 op） | 比较 45506，接受 2547；误拒 0，误收 0；崩溃 868（按拒绝计） |
| odscomp S1–S4（27960 实例） | 全 PASS；S3 误拒 0，残差为空时误收 0 |
| 残差为空的复合 op | 1911 → 2590 / 5592（34.2% → 46.3%） |
| projcomp P1/P2/P4/P5 | 全 PASS（P1 7891 比较，不一致 0） |
| core_test T1/T2 | PASS |
| odssig_test L1 | 误拒 0（21877 实例，三个层级都统计）；按账目改用完整模型（ods+cpred+nf）后，auto 196 个 op、cpred 13 个 op 分歧均为 0 |
| resid_stat（5557 个复合 op） | 残差非空 4033 → 2968；残差中 C++ verify 3899 → 3045，接口 verify 2411 → 1759 |

**过程（如实记录）**：
- 试跑时 vector.transpose 在 0 维、空 permutation 上误拒 1 例。原因是解析 `array<i64>` 时把 64 读成了元素。修正后才跑全量。
- odssig_test 首跑在规则内部抛出异常：shape_cast 的规则收到了非 vector 端口。已改为规则无定义时判拒绝，随后复跑。
- 复跑时“L1 按账目”FAIL（auto 类有分歧 2）。原因是 nf 覆盖后，tensor.cast 和 shape_cast 由 declared 转为 auto，但这一项原先只按 ods+cpred 比较。改为按完整模型比较后 PASS。这是检验口径随机制一起修正，不是放宽：被比较的模型更强，且不允许分歧。

**下一批**（work/resid67.txt）：vector.transfer_read 722、transfer_write 504、tensor.extract_slice 470、insert_slice 339、memref.subview 322、vector.contract 234。修好前 10 个 op 可清空 1321/2968。这些 op 的规则都依赖属性，例如 permutation_map、in_bounds、static_offsets/sizes/strides，还涉及结果类型推断（strided 布局）。

### 9.8 第三批：切片族与 transfer（第 68 轮）
**覆盖**（依据 MLIR 9bae84b0：Tensor/IR/TensorOps.cpp、Vector/IR/VectorOps.cpp、IR/BuiltinTypes.cpp 的 isRankReducedType / computeRankReductionMask、Interfaces/ViewLikeInterface.cpp、Interfaces/DataLayoutInterfaces.cpp 的默认位宽）：

| op | 规则 | 被覆盖的 declared |
|---|---|---|
| tensor.extract_slice | static_* 长度 = rank(source)；动态操作数个数 = kDynamic 个数；静态 offset/size ≥ 0；result 是 tensor<static_sizes × elt(source)> 的降秩（只去静态 1 维，`?` 只与 `?` 匹配，elt 相同，**不比较编码**）；静态维上不越界 | C++ verify()、OffsetSizeAndStrideOpInterface |
| tensor.insert_slice | 同上，以 dest 为基准、source 为降秩候选；result = dest | 另加 DestinationStyleOpInterface |
| vector.transfer_read | #indices = rank(base)；base 为 memref 或有秩 tensor；最内维位宽整除（元素为 vector 时按其最内维）；permutation_map 输入维 = rank(base)、无符号、结果数 = rank(vector)（元素为 vector 时减其秩）；结果为不重复的维或常数 0；mask = 由 map 推出的 i1 vector（compressUnusedDims + inversePermutation，0 维时为 vector<1xi1>，带可伸缩标志）；元素为 vector 时不许 mask；\|in_bounds\| = 结果数；padding = elt(base) 且为合法向量元素 | C++ verify()、DestinationStyleOpInterface |
| vector.transfer_write | 公共部分同上；map 不含常数 0（无广播维）；tensor base ⇔ 有结果且 = base，memref base ⇔ 无结果 | 同上 |

**机制扩展：逐点三值**。nf 规则遇到未建模的输入（方言自定义类型的 DataLayoutTypeInterface 位宽、VectorElementTypeInterface）时抛 NFUnknown。odscomp 中该原子取 UNK，Φ(T) ∈ {接受, 拒绝, 未定}：有原子为假即拒绝，否则有 UNK 即未定（交预言机）。range_holds/range3 不把 UNK 当作“必要条件不成立”，P1 不对 UNK 点作精确比较。逐点检查（odssig.check 非 strict）对未建模点不判（上近似）。已建模：`!llvm.ptr`（任意地址空间，无 DLTI 时 64 位，合法向量元素）。

**检验**：

| 判据 | 结果 |
|---|---|
| N1（本批 4 op） | extract_slice 605/接受 101、insert_slice 1124/73、transfer_read+write 9969/228；误拒 0，误收 0。transfer 接受例中含 mask 44、vector 元素 14、可伸缩 30、广播维 28 |
| N1（累计 23 op） | 比较 57204，接受 2949；误拒 0，误收 0；崩溃 5278（按拒绝计）；消融误收 15875 |
| odscomp S1–S4（27960 实例） | 全 PASS；S3 误拒 0，残差为空时误收 0；点未定 0 |
| 残差为空的复合 op | 2590 → 3385 / 5592（46.3% → 60.5%） |
| projcomp P1/P2/P4/P5 | 全 PASS（P1 8474 比较，不一致 0，点未定 0） |
| resid_stat（5557） | 残差非空 2968 → 2188；C++ verify 3045 → 2084，接口 verify 1759 → 685 |

**过程（如实记录）**：
- 首轮回归 odscomp 段 b S3 误拒 1：真实 IR 中 `vector<8x16x!llvm.ptr<1>>` 写入 tensor<?x?xf32>，我的位宽函数不认识 llvm.ptr 而拒绝。补 llvm.ptr，并把其他方言类型改为 NFUnknown（先实现为“不判”）。
- 复跑后 S3 残差为空误收 1+4、P4 不一致 2+20：变异池里的 `!transform.any_op` 作 padding 触发 NFUnknown，“不判”在 Φ 中等于接受，破坏了“残差为空 ⇒ 精确”。据此把 NFUnknown 改为逐点三值（上文），并把 padding 检查改为先比相等再查合法性（不等即确定拒绝）。之后全部 PASS。
- 同一次复跑段 a 有一个文件工作进程崩溃（ODS json 缓存被并发进程半写读取），测试只计“文件超时/崩溃 1”而仍 PASS——这是测试缺陷。已修：缓存先写临时文件再原子改名；S1 与 P1 判据加入“文件崩溃 = 0”。此前各轮日志中该计数均为 0。
- transfer 生成器首版 30k 例超时：改为独立缺陷（in_bounds 长度、索引个数、padding）只在前两种向量上扰动，其余全组合。

**边界**：DataLayout 按默认（无 DLTI）；模块带 dl_spec 时 index/指针位宽可能不同，规则未读取上下文。三值机制在语料上点未定 0，只经构造与上述反例验证，未在语料中被触发。memref.subview（strided 布局推断、affine 布局）未做。

**下一批**（work/resid68.txt）：memref.subview 322、vector.contract 234、vector.to_elements 160、memref.extract_strided_metadata 156、tensor.collapse_shape 152、expand_shape 143、vector.extract 129。前 10 个 op 可清空 890/2188。

### 9.9 接口级规则：DestinationStyleOpInterface（第 69 轮）

**为什么按接口推进**：按残差频次逐 op 推进以语料为分母，语料不含的方言（Linalg）永远排不上。改用 ODS 全集作分母：MLIR 9bae84b0 的 650 个 op 中，第 68 轮后仍有 356 个带未规范化的 declared 条目（接口 verify 429 条、C++ verify() 174、C++ inferReturnTypes 19、verifyRegions 7）。接口 verify 高度集中：DestinationStyle 101、TransformOp 92、LinalgOp 91、ConvolutionOp 47、InferTypeOp 20、RegionBranch 15、RegionBranchTerminator 14、ContractionOp 11、CastOp 10，其余各 ≤ 6。一个接口的 verify 是同一段 C++，规范化一次即覆盖全部实现它的 op。

**机制**：`nfrules.IFACE`：declared 条目 → 规则工厂。`apply` 先按 op（REG），再按接口；工厂从 ODS 记录读出 op 特有的参数，读不出则该 op 不套用（条目留在残差）。

**DPS 规则**（依据 Interfaces/DestinationStyleOpInterface.cpp 与 .td 的 getTiedOpResult）：init 组由 `getDpsInitsMutable() { return get<X>Mutable(); }` 读出（101 个 op 中 100 个可读，bufferization.materialize_in_destination 为手写实现，留在残差）。
- 每个 init 是 tensor 或 memref（含无秩）；
- tensor 结果个数 = tensor init 个数；
- 第 i 个 init 若为 tensor，则第 i 个结果存在且与之同类型（按 init 下标取结果，不是按 tensor init 的序号；越界在 MLIR 中是断言，按拒绝）。

**检验**（model/nfiface_test.py，判据运行前写定）。接口规则只覆盖 op 的一部分，不能要求与 MLIR 整体判定等价，因此判据是：N2a MLIR 接受 ⇒ 规则接受（误拒 0）；N2b MLIR 报出该接口的诊断 ⇒ 规则拒绝（漏判 0）；N2c 非空。语料为主语料加 MLIR 的 Linalg 测试（150 个文件，单独存放于 corpus_ext/，不改变既有基线）。在真实 op 上变异 init 端口类型与结果类型（tensor↔memref、无秩、换形状、换元素）。
- 结果：op 实例 1648，比较 10788，MLIR 接受 1609，接口诊断 3451，规则拒绝 7748（其中 4297 例 MLIR 先报了别的诊断），误拒 0，漏判 0，文件崩溃 0，MLIR 崩溃 19（不计入比较）。

**边界**：N2b 只在接口 verify 实际运行时检验漏判；MLIR 先因别的原因拒绝时不计。这对 op 判定无影响：那些例子 MLIR 本就拒绝。

### 9.10 LinalgOp::Trait（第 70 轮）

**依据**（MLIR 9bae84b0）：Linalg/IR/LinalgInterfaces.cpp `verifyStructuredOpInterface`、Interfaces/IndexingMapOpInterface.cpp `verifyImpl`、IR/AffineMap.cpp `inversePermutation`/`concatAffineMaps`、LinalgInterfaces.td 的 getRank/getShape 默认实现、tools/mlir-linalg-ods-gen/mlir-linalg-ods-yaml-gen.cpp（命名 op 的 getIndexingMaps / getIteratorTypesArray / 符号绑定）。

**覆盖**：91 个实现 LinalgOp 的 op 中 82 个：linalg.generic（映射取属性 indexing_maps，循环数 = #iterator_types）与 LinalgNamedStructuredOps.yaml 的 81 个 opdsl 命名 op。手写 getIndexingMaps 的 9 个（matmul、batch_matmul、batch_reduce_matmul、contract、elementwise、transpose、broadcast、reduce、map）留在残差。

**规则**（按操作数类型、属性与区域入口块参数，全部为合取）：
- 纯 tensor 或纯 buffer 语义（有操作数时）：无 memref 且有 tensor，或无 tensor 且有 memref；
- #映射 = #操作数；拼接后可逆（每个循环维至少一次以单独维出现，取首次出现）；无符号；每个映射维数 = 循环数 = 迭代器个数；结果数 = 操作数秩；
- getRank/getShape：vector 视为元素类型（秩 0），tensor/memref 取形状，无秩断言（按拒绝），其余秩 0；
- 循环范围全静态时：每个静态非零维 k 上，映射在 0 点与 (范围−1) 点的值取 min ≥ 0；结果为单个维则 max+1 = 维长，否则 max+1 ≤ 维长；
- 区域恰一块，入口块参数个数 = #操作数，第 i 个参数 = 第 i 个操作数的元素类型（memref/有秩 tensor 取元素类型，其余取自身）。
- 命名 op：yaml 的映射以 index 属性（strides、dilations，缺省取默认值）绑定符号后化简；无迭代器的 op 为秩多态：循环数 = rank(init0)，标量（秩 0）用空映射，其余用恒等映射；IR 带 `linalg.memoized_indexing_maps` 时按它（getIndexingMaps 先读缓存）。

**新机制**：model/affine.py：仿射映射解析、构造期化简（线性合并、×1、floordiv/mod 1、系数整除）、求值。区域入口块参数类型由 `nfrules.nf_attrs` 放进只给 nf 规则看的属性表（core.typecheck 与 odscomp 两处调用）。复合 op 中区域是实现图的一部分，其块参数类型固定，因此这条规则把端口类型与内部区域联系起来。

**发现**：MLIR 在 verify 时把 `linalg.memoized_indexing_maps` 写回 op，经 Python 绑定导入的 IR 带着它。检验时去掉此属性，以检验由 yaml 推出的映射（前端构造的 op 不带它）。

**检验**（nfiface_test，判据运行前写定；本轮把变异扩展到所有操作数、维长 +1、首维改动态、降秩，以及属性：strides/dilations 取 2、3，indexing_maps 反序，iterator_types 去一项）：
- N2a/N2b 同 §9.9；新增 N2d：declared 已清空的 op 上，完整模型（结构 + 属性 + ODS + nf，关闭预言机）与 MLIR 判定一致（误拒 0、误收 0）。
- 首次全量：N2a FAIL，误拒 2（linalg.fill 的 vector 输入：初版把 vector 按 ShapedType 取秩；LinalgInterfaces.td 规定 vector 是元素类型）。已修正并复跑，结果见本节末尾。
- 复跑结果（主语料 + corpus_ext/Linalg，op 实例 1648）：N2a 比较 26405，MLIR 接受 4429，误拒 0；N2b 接口诊断 13238，漏判 0；其中 LinalgOp 比较 18276，MLIR 接受 3173，接口诊断 5168，规则拒绝 9542，规则未定 0。N2d 比较 10867（MLIR 拒绝 8896），误拒 0，误收 0。MLIR 崩溃 1200 例按拒绝计入。
- 回归：odscomp S1–S4（段 a 12375 / 段 b 15585，误拒 0，残差为空时误收 0）、projcomp P1/P2/P4/P5、N1（23 op，57204 比较，0/0）、core T1/T2、L1（误拒 0，按账目分歧 0）、clauses K1–K6/M2.7 全 PASS。回归在 vector 修正之前跑；该修正只影响 linalg op，不在主语料中。
- 边界：只检验了语料中出现的 indexing map 形状；非线性仿射式（floordiv/mod 含维）的化简只做了部分，可能使“是否单个维”的判断与 MLIR 不同（语料未出现）。Convolution（47）、Contraction（11）接口和手写的 9 个 op 仍在残差。

### 9.11 ConvolutionOpInterface 与 ContractionOpInterface（第 71 轮）

**依据**（MLIR 9bae84b0，Linalg/IR/LinalgInterfaces.cpp）：`isConvolutionInterfaceImpl`、`ConvAccessExprWalker`、`isContractionInterfaceImpl`、`isContractionBody`、`getSourceSkipUnary`、`isPairTemplateImpl`。

**覆盖**：Convolution 47 个，全部为 opdsl yaml op；Contraction 11 个中的 7 个 yaml op（dot、matvec、vecmat、batch_matvec、batch_vecmat、mmt4d、batch_mmt4d）。matmul、batch_matmul、batch_reduce_matmul、contract 的映射来自手写 C++，与 LinalgOp 的同类条目一起留在残差。

**Convolution 规则**：≥2 个输入且 1 个 init；输入映射的每个结果要么是单个维（非卷积维），要么是两个不同维的线性和（卷积维）；任一维第二次出现即失败。这里按 MLIR 构造期化简后的形状判断：strides=0 时 `d1*0 + d4` 化为 `d4`，变成非卷积维。filter 映射与输出映射须是投影置换。输出维与 filter 维按 batch、image、输出通道、depth multiplier、filter 循环、输入通道逐一归类，迭代器种类须相符，且不得重复计入。归类后的循环个数须等于循环数，卷积维集合非空。映射与迭代器来自 yaml，并绑定 strides/dilations；IR 带 memoized 属性时按该属性。

**Contraction 规则**：2 个输入、1 个 init；至少一个 reduction 迭代器；映射全为投影置换；区域体须为 `yield(u5(u1(c) ⊕ u2(u3(a) ⊗ u4(b))))` 形状。(⊗, ⊕) ∈ {(mulf, addf), (muli, addi), (complex.mul, complex.add), (andi, ori)}。u 为单操作数、ODS 判定无内存效应的 op。块恰有 3 个参数，末尾为 terminator（ODS trait；未注册 op 视为可能是 terminator）。区域捕获外部值或含未导入方言的 op 时判未定。

**机制**：`nf_attrs` 另传区域 0 的块本身（`\x00block`）。区域体是实现图的一部分，规则在复合 op 中把它当作固定事实。这条约束是结构约束，与端口类型无关，但同样属于该 op 的合法性条件，因此也规范化进模型。

**检验**（nfiface_test，判据同 §9.10，运行前写定）：新增区域体变异（mulf→subf、addf→subf、muli→subi、addi→subi、mulf→addf），strides/dilations 增加 0 与 −1。全量（op 实例 1648）：
- N2a 比较 28457，MLIR 接受 5333，误拒 0。
- N2b 接口诊断 13689，漏判 0。
- Convolution 比较 5189（MLIR 接受 1257），接口诊断 286，规则拒绝 286；Contraction 比较 522（接受 87），接口诊断 68，规则拒绝 68。
- N2d（declared 已清空的 op，完整模型 ≡ MLIR）比较 16994（MLIR 拒绝 13263），误拒 0，误收 0。
- 回归：odscomp 段 a S1–S4、projcomp 段 a P1/P2/P4/P5 全 PASS。

**边界**：两维线性和以外的输入映射形状（三项和、带常数项、非线性式）一律判失败，这与 MLIR 的树形判定一致的前提是构造期化简把两维线性式化成 Add(项, 项)。语料与变异覆盖了 strides 取 0、1、2、3、−1 的情形，其他形状未出现。

### 9.12 手写 getIndexingMaps 的 9 个 LinalgOp（第 72 轮）
依据 MLIR 9bae84b0 `Linalg/IR/LinalgOps.cpp` 与 `LinalgStructuredOps.td`。实现在 `model/nfrules.py` 第 72 轮两节。
- **map / reduce / transpose / broadcast**：C++ verify 与 LinalgOp::Trait 合成一条规则（`LinalgHand`）。映射由类型与属性构造：map 取恒等；reduce 的 inputs 取恒等，inits 去掉 dimensions；transpose 取 [perm 的逆, 恒等]；broadcast 取 [恒等去掉 dimensions, 恒等]。C++ 部分包括：块参数个数与类型、形状逐维相等、dimensions 的范围、isPermutationVector。MLIR 先跑接口 verify，再跑 op verify，两者为合取，断言（无秩、越界、dimensions 重复）按拒绝计。
- **matmul / batch_matmul / batch_reduce_matmul / contract / elementwise**：映射取自 indexing_maps。缺省时，matmul 族用 C++ 默认映射，elementwise 为空数组，contract 为必需属性。迭代器：matmul 族固定；contract 由输出映射推出（出现在输出的维为 parallel）；elementwise 取 rank(init0) 个 parallel，按 ShapedType 计秩，vector 计其秩。LinalgOp 与 ContractionOpInterface 复用 §9.10/§9.11 的结构化检查与区域体检查。
- **C++ verify（`LinalgCxx`）**：matmul 族只在映射不同于默认映射时检查：输入映射结果须取自默认映射结果；广播映射须保留 k 维，batch 变体另有 batch/m/n 组合；batch 变体的输出映射结果数固定，且依次为 batch/m/n 的函数。contract：前三个映射为投影置换；结果数等于 shaped 秩（vector 计秩，非 shaped 为 0）；维数一致；每个维至少被访问一次；只出现在一个输入中的维也须出现在输出中；至少一个收缩维。isFunctionOfDim 按表达式中出现的维判定。
- 至此 LinalgOp::Trait 覆盖 91/91（generic、81 个 yaml op、9 个手写 op），ContractionOpInterface 覆盖 11/11（7 个 yaml op 加 4 个手写 op）。linalg 方言仍有 9 个 op 留有 declared：fill（FillOpInterface）、index、pack、unpack、softmax、3 个 winograd、yield（第 73 轮后余 7 个，见 §9.13）。
- **检验**（nfiface_test 全量，分两段：[0,260) 与 [260,∞)。判据与第 69–71 轮相同，运行前写定。新增变异：dimensions/permutation 取反序、去尾、加 0、首项改 5、全取首项；首个块参数改类型；逐个映射取反序、去首、去尾、换成首映射）：
  - N2a：比较 5862+28729，MLIR 接受 968+5319，误拒 0，文件崩溃 0。
  - N2b：接口诊断 2977+16636，漏判 0。MLIR 崩溃 154+1541 例，按拒绝计。
  - N2d（declared 已清空的 op，完整模型无预言机 ≡ MLIR）：比较 3878+23662，MLIR 拒绝 3276+19085，误拒 0，误收 0。
  - 分项：LinalgHand 比较 1763，规则拒绝 1232；LinalgCxx 比较 8759，规则拒绝 1409；ContractionOpInterface 比较 9281。
  - 首跑 N2b 有 346 例漏判，原因在检验：LinalgCxx 只覆盖 C++ verify，却按 LinalgOp 的诊断计数。诊断分组改正后复跑通过。规则本身没有改动。
- 局限：只验证了语料与上述变异中出现过的映射形状；elementwise 的 kind 与输入个数（arity）的关系由区域体与解析器保证，本规则未单列；诊断匹配仍依赖正则。

### 9.13 linalg.pack / linalg.unpack 与值前提（第 73 轮）
依据 MLIR 9bae84b0 `LinalgOps.cpp`：commonVerifierPackAndUnPackOp、isInvalidPackingPosSpecification、getMixedTiles/getStaticTiles、getPackOpResultTypeShape、PackOp::verify、requirePaddingValue。实现在 `model/nfrules.py` 第 73 轮一节。
- **公共规则**（未打包端 U：pack 为 source，unpack 为 dest；打包端 P 为另一端）：tile 不得为 0；inner_dims_pos 与 outer_dims_perm 无重复、元素在 [0, rank U) 内、长度 ≤ rank U；outer_dims_perm 为空或长度等于 rank U；#tile ≤ rank U 且 #tile = #inner_dims_pos；rank P = rank U + #tile；P 的尾部 #tile 维：静态 tile 处等于 tile，动态 tile 处为 ?；P 与推出形状逐维相容。推出形状：U 的被分块维取 ⌈d/t⌉，d 或 t 动态则为 ?，再按 outer_dims_perm 置换，后接静态 tile。static_inner_tiles 中动态项多于 inner_tiles 操作数时 MLIR 越界，按拒绝计；少于时多出的操作数被忽略，MLIR 接受。
- **pack 另有**：padding_value 的类型等于 source 的元素类型；无 padding_value 时须整块：对每个被分块的静态维 d，tile 为常量 c 时须 c | d；tile 不是常量时，若 P 中对应的外层维 o 静态，须 o | d（o = 0 时 MLIR 除零，按拒绝计）。
- **值前提（新）**：零 tile 检查与整块检查经 isZeroInteger / getConstantIntValue，会看动态 tile 操作数是否由常量定义；推出形状则不看（Value 一律视为动态）。所以这两条规则不只依赖类型与属性，还依赖“操作数是否为常量及其值”。模型把它作为显式前提：`nfrules.note_const` 记录块内 arith.constant 的整数值，`nf_attrs(…, consts)` 以 `'\x00consts'` 传给规则；块参数与复合 op 的端口视为非常量，这与 MLIR 校验复合 op 体时所见一致。没有常量信息、又有动态 tile 时，规则判未定（NFUnknown），不猜。对前端的含义：tile 若由图内常量给出，签名关系按常量值判定；若从端口传入，定义时不检查整块与非零，实例化后常量传播进来可能使 MLIR 重新校验时拒绝，这一点属于实例层，需由前端在实例化后再判一次（与 P.ground 的定值残差同一机制）。
- **检验**（判据运行前写定）：
  - nfiface_test 全量（[0,260) 与 [260,∞)），新增变异：inner_dims_pos/outer_dims_perm/static_inner_tiles 的数组变异、常量 tile 值取 0/1/3/5（常量 op 随 pack 一起实例化）。N2a：比较 7528+32046，MLIR 接受 1281+5866，误拒 0，文件崩溃 0。N2b：接口诊断 4399+19555，漏判 0，MLIR 崩溃 157+1547 例按拒绝计。N2d：比较 6721+29379，MLIR 拒绝 5579+23826，误拒 0，误收 0。分项：pack 比较 1481+3371，unpack 比较 1362+2346。语料中带常量 tile 的实例很少，常量变异只有 8+16 例。
  - 因此另加合成网格 `packconst_test.py`（N2e，完整模型无预言机 ≡ MLIR）：比较 1600，含常量 tile 902，MLIR 接受 358、拒绝 1242，误拒 0，误收 0。消融（不传常量信息）在 400 例中出现 1 例误收，说明值前提必要；该网格对它的区分力弱，因为常量零 tile 多数已被形状检查先行拒绝。
- 局限：常量只认 arith.constant 的整数/index 值（同名值在后续定义处清除）；MLIR 的 m_ConstantInt 认任何 ConstantLike op（如 index.constant），这类定义未建模，遇到时规则会把它当非常量，可能与 MLIR 不一致，属已知缺口。
- **回归**（本轮改了 core.typecheck 与 odscomp 的属性传递，所以必须复跑）：首跑 odscomp S2 推导异常 65+1008、projcomp P1 异常 786+56，原因是常量表以 list 放进属性表，后者在 Φ_comp 中被哈希。改为 tuple 后复跑：odscomp S1–S4 两段全 PASS（S2 不一致 0/12375、0/15585，推导异常 0；S3 误拒 0，残差为空误收 0）；projcomp P1/P2/P4/P5 两段全 PASS（P1 不一致 0，异常 0）；N1 23 op 全 PASS（40500+16704 次比较，误拒 0、误收 0）；N2e 复跑 PASS；clauses K1–K6、M2.7 PASS。nfiface 全量在 tuple 修正与“同名值清除常量”修正之前跑完；两项修正后复跑了含 pack/unpack 最多的两个文件（Tensor/fold-into-pack-and-unpack、Linalg/data-layout-propagation），N2a/N2b/N2d 全 PASS。未复跑：core T1/T2、L1、odssig_test。
- 至此 linalg 方言余 7 个 op 留有 declared：fill、index、softmax、3 个 winograd、yield。

### 9.14 linalg.softmax 与 winograd 三个变换（第 74 轮）
依据 MLIR 9bae84b0 `Linalg/IR/LinalgOps.cpp` 的 `SoftmaxOp::verify`、`Winograd{Filter,Input,Output}TransformOp::verify`、`getFmrFromWinogradConv2DFmr`，以及 `LinalgOps.td` 的各 `get*Dim()`。实现见 `model/nfrules.py` 第 74 轮一节。

- **softmax**（端口 AnyShaped，含 vector 与无秩）：input、output 均有秩且逐维相容；0 ≤ dimension < rank(input)。无秩时 `getShape` 断言，按拒绝计。ODS 无法解析的 `PredOpTrait<TCopVTEtIsSameAs<0,1>>`（elt(input) = elt(output)）也写成 nf 原子，从未知项中移出（`UNKNOWN_COVERS`）。
- fmr：F_2_3 → (m,r)=(2,3)，F_4_3 → (4,3)，F_2_5 → (2,5)；α = m+r−1。通用形式中属性为 `fmr = N : i32`。
- **filter_transform**（filter 维序 F,H,W,C）：H,W ∈ {r,1} 且不同时为 1。动态维（kDynamic）既不等于 r 也不等于 1，因此拒绝。output 相容于 (H=r ? α : 1, W=r ? α : 1, C, F)。
- **input_transform**（input N,H,W,C；output αH,αW,tH,tW,N,C）：H 动态时 αH = α、tH 动态；否则 αH = (out[0] ≠ 1 ? α : 1)，tH = (H−(r−1))/m（C++ 向零截断）或 H。W 同理。注意：变换方向由 **output** 的 αH 是否为 1 决定，而 output 的 αH 为动态时视为“≠ 1”。
- **output_transform**（value αH,αW,tH,tW,N,F；output N,H,W,F）：αH 与 tH 都静态时须 αH = (αH ≠ 1 ? α : 1)，H = (αH ≠ 1 ? m : 1)·tH；任一动态则 H 动态，此时 αH 不受检查。W 同理。
- 对前端：这些规则只依赖端口形状与属性，不需要值前提。输出形状是输入形状与 fmr 的确定函数（动态维除外），复合 op 可直接推出。

检验（判据运行前写定）：
- `nfiface_test` 定向（含 Linalg/roundtrip、invalid、tile-softmax、transform-op-decompose、winograd 等文件；新增 dimension 与 fmr 属性变异）：N2a 比较 8692，误拒 0；N2b 接口诊断 5007，漏判 0；N2d 7258，误拒 0、误收 0。修正 fmr 变异（首版按名字替换，对通用形式无效）后，对 5 个文件复跑：N2a 2899、N2b 1604、N2d 2793，全部 0。
- 新增合成网格 `model/winograd_test.py`（N2f，完整模型无预言机 ≡ MLIR）：比较 3000（softmax 690、filter 753、input 799、output 758），MLIR 接受 531、拒绝 2469，误拒 0、误收 0、崩溃 0。filter 的接受只有 35，覆盖偏薄。
- **N1 不适用于 linalg**：N1 网格只变异端口类型，不生成区域与必需属性，14 个 linalg 候选上 MLIR 接受均为 0，map 的 65 例“误收”实为“requires one region”的结构性拒绝。`nf_test` 默认集合改为排除 linalg；linalg 以 N2a/N2b/N2d/N2e/N2f 检验。
- linalg 方言仍有 declared 的 op：fill（FillOpInterface）、index、yield。

### 9.15 含区域节点的复合 op；linalg.yield / linalg.index（第 75 轮）
**发现的缺口**：第 62–74 轮的 Φ_comp 检验（odscomp_test、projcomp_test）沿用 compcorpus_test.slices，切片排除了所有带区域的 op；odscomp.derive 也不看区域内的 op。所以此前的 Φ_comp 结论只对“无区域节点”的复合 op 成立，linalg 各轮的 nf 规则只在单 op（N2*）上检验过，未进入复合 op。

**改动**：
- odscomp.derive 递归处理区域：区域内 op 的原子并入 Φ_comp；区域内用到的外部值（捕获）作为端口；块参数与区域内结果取图上写定的类型（常量）。因此**最大性只对顶层推导成立**，区域内类型随端口推导尚未建模（列为下一步）。
- 定值残差的判断连同捕获一起看（odscomp.captures）：父 op 的不透明校验（如 tensor.generate/tensor.pad 的 verifyRegions 检查 yield 类型）会读到捕获值的类型。首轮 R2 的 7 例“残差为空误收”全部来自这一点，已修复。
- 化简去重键加入 nf 规则身份：同一 op 上引用相同端口组的不同 nf 规则此前会被合并（首轮 R1 化简不一致 337 例）。只影响 simplify/呈现，不影响 evaluate(Φ)。
- composite.instantiate 更新区域内 op 的操作数类型表（捕获值类型随端口变）。
- core.typecheck 与 derive 都向 nf 规则传父信息 parent_info（父 op 名、区域数、区域 0 非空）；复合 op 体顶层的父未定 ⇒ NFUnknown。
- nf 规则（依据 LinalgOps.cpp YieldOp::verify、verifyYield、IndexOp::verify）：
  - linalg.yield / linalg.index 自身：父 op 恰一个非空区域且实现 LinalgOp。
  - 父 op 上新增原子 LinalgBody（91 个 LinalgOp 全部挂上；inits = 最后一个操作数组 outputs/init/inits）：体内 yield 值个数 = #inits，第 k 个值类型 = inits[k] 的元素类型（memref/有秩 tensor）或其本身；体内 index 的 dim < getNumLoops（generic：#iterator_types；map/transpose/broadcast：rank(init)；reduce：rank(inputs[0])；yaml 与手写 op 取各自迭代器表）。yield 用了区域外的值、区域块数 ≠ 1 ⇒ NFUnknown。
  - RegionBranchTerminatorOpInterface 的 verify 只有 static_assert 与 success()，运行期无约束，规范化为空规则。
- linalg 方言仍有 declared 的 op 只剩 fill（FillOpInterface）。

**检验**（新 `model/regcomp_test.py`，判据运行前写定；语料 corpus 246 文件 + corpus_ext/Linalg 150 文件，两段 `6 4 200 0`、`6 4 1000 200`）：
- R1 Φ≡逐点（含区域）：实例 5565 + 13310，不一致 0，化简不一致 0，推导异常 0，文件崩溃 0；点未定 49 + 512。
- R2 对 MLIR：实例 5545 + 12735（MLIR 接受 2754 + 5474；含 linalg 体 1125 + 11122），误拒 0，残差为空误收 0，残差非空误收 504 + 748；点未定 49 + 444。
- R3：复合 op 1113 + 2662，残差为空 117 + 1402。残差来源前列：scf.for（C++ verify、verifyRegions、RegionBranchOpInterface、LoopLikeOpInterface）、linalg.fill（FillOpInterface）、tensor.pad、tensor.empty、scf.forall/parallel、vector.mask、memref.subview。
- 首轮（修复前）R1 化简不一致 32 + 305、R2 残差为空误收 4 + 3，见上；判据未改。

**下一步**：按残差前列规范化 scf.for/forall/parallel/if/while 的区域约束（块参数、iter_args、yield 与结果类型的关系），并让区域内类型随端口推导（最大性扩展到区域）；linalg.fill；tensor.pad/empty。

### 9.16 scf 区域 op 的规范化：for / if / while / execute_region / index_switch（第 76 轮）
**改动**：
- nfrules 新增 5 条 op 规则，覆盖这些 op 的全部 declared 条目（C++ verify、verifyRegions、LoopLikeOpInterface、RegionBranchOpInterface、InferTypeOpInterface、C++ inferReturnTypes）。依据 9bae84b0：SCF.cpp 各 op 的 verify/verifyRegions/getSuccessorRegions/getEntrySuccessorOperands、IfOp::inferReturnTypes、ConditionOp::getMutableSuccessorOperands；LoopLikeInterface.cpp verifyLoopLikeOpInterface；ControlFlowInterfaces.cpp verifyTypesAlongControlFlowEdges（areTypesCompatible 缺省为相等；RegionSuccessor(Region*) 输入为空，ControlFlowInterfaces.h）；verifyInferredResultTypes。
  - scf.for：块参数 = (typeof(lowerBound), iter…)；inits、iter_args、yield 操作数、results 个数相同、类型逐个相等（四条控制流边 + verify/verifyRegions/LoopLike 合起来）；体末不是 scf.yield ⇒ 拒绝（cast 断言）。
  - scf.if：then 以 scf.yield 结尾且类型 = results（InferType）；results 非空 ⇒ else 非空；各区域 RegionBranch 终结子操作数 = results。
  - scf.while：before 末为 scf.condition、after 末为 scf.yield；inits = before 块参数；condition 的值 = results = after 块参数；after 的 yield = before 块参数。
  - scf.execute_region：区域非空、入口块无参数；各块 RegionBranch 终结子（scf.yield/condition 或 ODS 声明该接口的 op；ODS 未知 ⇒ NFUnknown）操作数 = results。
  - scf.index_switch：#cases = #case 区域、无重复；各区域末为 scf.yield 且类型 = results。
- 区域信息：nf_attrs 增加 `\x00regs`（各区域各块的入口参数类型、末 op 名）与终结子操作数类型。终结子操作数先在本块、再在同区域其他块解析（兄弟区域复用值名，不能用平铺值表），否则为捕获值：core.typecheck 取值表类型；odscomp.derive 取端口表达式 `\x00tterm_e`，holds 在端口类型下求值。odscomp 的 _aexprs、simplify 的常量判定、嵌套代换都纳入这些表达式。
- 区域内块参数与区域内结果仍取写定类型；最大性仍只对顶层成立（未变）。
- 结构层 R8（structure.py，人工写的 scf.for/if 边类型检查）与新规则重叠，暂留，不作为依据。

**检验**（新 `model/scf_test.py`，判据运行前写定；corpus 246 文件中 234 个顶层 scf 实例，变异端口/结果/块参数类型、结果/块参数/终结子个数、cases）：
- N2g-a PASS：比较 4266，MLIR 接受 267，误拒 0，规则未定 0。
- N2g-b PASS：scf 诊断 1812，漏判 0；规则拒绝 2608（其中 MLIR 报别的诊断或崩溃 796），MLIR 崩溃 12。**首跑 FAIL（漏判 15）**：测试按 op 名匹配诊断，把落在区域内同名嵌套 op 上的诊断算给了外层 op；改为按诊断位置（节点 id）匹配，这是按写定判据（“在该 scf op 上”）修正实现，判据未改。嵌套 op 上的 scf 诊断 21 例本测试未另行检验。
- N2g-d PASS：无残差实例上完整模型（无预言机）≡ MLIR，比较 3426（MLIR 拒绝 3214），误拒 0，误收 0。接受样本只有 212，偏少。
- 回归（最终代码）：regcomp R1/R2 PASS（实例 5565 + 13310；R2 误拒 0，残差为空误收 0，残差非空误收 310 + 718，按 Unknown 计；点未定 44 + 444/512）；odscomp S1–S4 PASS（12375 + 15585；S3 误拒 0、残差为空误收 0）；projcomp P1/P2/P4/P5 PASS；core_test T1/T2 PASS。
- R3：复合 op 残差为空 602/1113（原 117）、1500/2662（原 1402）。残差前列：linalg.fill、tensor.pad、scf.forall、scf.parallel、tensor.empty、vector.mask、memref.subview。

**下一步**：scf.forall / scf.parallel / scf.reduce（含 Graph 区域 in_parallel）；区域内类型随端口推导；linalg.fill；tensor.pad/empty。


### §9.17 第 77 轮：区域类型流（最大性延伸到区域内）

**问题。** 第 75–76 轮里，区域内的块参数和 op 结果取图上写定的类型，所以最大类型约束只在顶层是推导出来的。以 `scf.for` 为例，写定 iter_arg 为 f32，复合 op 就只接受 f32，尽管 MLIR 对任何满足边等式的类型都接受。

**机制（model/regionflow.py）。** 区域 op 的类型流由一张控制流入口边表给出（依据 9bae84b0 SCF.cpp 的 getSuccessorRegions / getEntrySuccessorOperands / inferReturnTypes，以及 LinalgInterfaces.cpp 的 bbArg 规则）：
- scf.for：块参数 = (typeof lb, inits)，结果 = inits；
- scf.if / execute_region / index_switch：结果 = 第一个区域的 yield；
- scf.while：before 参数 = inits，after 参数与结果都等于 condition 的值；
- LinalgOp：第 k 个块参数 = elt(第 k 个操作数)。

每个后继输入只取一条前驱边作为**定义**，其余边仍由 nf 规则作为**约束**检查，判定层不变。区域内的 op 与顶层一样按 ODS 推断前推结果，没有推断规则的 op 取字面量。MLIR 要求这些等式成立，所以被接受的类型指派都满足它们。用等式定义内部类型不会丢掉任何可接受的端口类型；可靠性仍由不变的判定层保证。

**符号与具体共用一张表。** odscomp.derive 把 nf 规则读到的区域信息（块参数、块内结果、终结子操作数）记成表达式快照 `\x00inst`，odscomp.holds 在端口类型下把它具体化，再用 core 的同一函数 `nfrules.region_keys` 计算。composite.instantiate 按同一张表实例化，兄弟区域的同名值各有作用域。第 76 轮的 `\x00tterm_e` 由此取代。区域内类型依赖端口时，体内 op 未清空的 C++ verify 计入残差（非定值）。

**检验（model/regflow_test.py，判据先写定）。**
- **F1 单调**（旧接受 ⇒ 新接受）：PASS，实例 5565 + 13310，违例 0。
  - 首跑报 6 例违例，原因是测试把“Φ 真但残差非空”算成了旧接受，与写定判据不符，已改正。6 例全是 linalg.reduce：旧签名没有把块参数与输入元素类型联系起来，按 Unknown 处理；新签名正确拒绝。
- **F2 随机变异增益 ≥ 50**：**FAIL**。增益 24，其中残差为空 18，MLIR 全部接受，误收 0。随机变异很少碰到可传播的类型改变，判据保留原样。
- **F4 一致替换元素类型**：在 F2 失败后补写，F2 不改。两段合计实例 436，增益 137，误收 0，误拒 0。
  - 按段：第一段 131 PASS；第二段（主要是 Linalg 补充语料）只有 6，按段判 FAIL。原因是 linalg op 的 tensor 结果取字面量（DPS 结果未建模），传播在 linalg 结果处中断。
- **F3**：增益实例中，旧实例化被 MLIR 拒绝的有 18 例，说明旧签名确实不是最大的。
- **回归**（最终代码）：regcomp R1/R2、odscomp S1–S4、projcomp P1/P2/P4/P5、scf N2g-a/b/d、nfiface N2a/b/d、core_test、odssig L1、clauses 均 PASS。
  - nfiface N2d 首跑误收 1 例：linalg.generic 体内的 vector.outerproduct 的 C++ verify 现在会随端口变化。测试改为把这类实例（体内有 op 的 C++ verify 未清空、且其类型被改变）记作残差非空。这是语义变化后对测量的修正。

**未做。**
- DPS 结果 = init（DestinationStyleOpInterface）。这是 F4 第二段增益少的原因，作为下一个机制。
- scf.forall / parallel / reduce。
- 多块区域只取入口块。

### §9.18 第 78 轮：C++ verify 与属性谓词的批量规范化（解释执行 9bae84b0 源码）

**问题。** 第 66–77 轮的 nf 规则是逐 op 手写的。全量盘点（650 个 op）：declared 残差原有 662 条，第 77 轮后剩 335 条，其中 153 个 op 的 C++ verify()、Transform 接口 92 条、InferType 37 条。已关闭的部分主要靠“一个接口一条规则”（DPS、LinalgOp、Conv、Contraction）。逐 op 手写与需求（建模应当批量）不符。

**机制（model/cppverify.py）。** verify() 用的是一小套词汇：取端口类型、getRank/getShape/getElementType/getDimSize、个数比较、isa/cast、早返回 emitOpError。把它当作一门语言：
- 解析 C++ 子集（声明、if/else、for/range-for/while、return、lambda、模板 isa/cast、花括号初始化）；
- 在 typesig 类型项与 MLIR 属性值上求值；ODS 访问器按 ODS 名字生成，属性 getter 按 ODS returnType 转换；op 的内联方法（extraClassDeclaration）、类外方法和方言 .cpp 中的辅助函数递归解释，方言常量取自 .td；
- 三值：`return emitOpError/failure` ⇒ 拒绝（带出消息），`return success` ⇒ 接受，路径上遇到不支持的构造 ⇒ CppUnknown（作 NFUnknown，不猜）；cast 失败按断言崩溃计为拒绝；
- 静态完备：函数体与所有调到的方法/辅助函数可解析且名字都在词汇内 ⇒ 该 op 的 'C++ verify()' 移出残差；不完备的 op 规则仍生效（只在走到的路径上判定）。
- 查询不带属性时（x[5] = 'opt'）仍求值，只有真正读到属性才 Unknown。

同一解释器也用于 **ODS 属性谓词原子**（attrsig）：此前用正则表逐条登记，未登记的原子算 Unknown；现在原子作为 C++ 表达式（$_self 为属性）求值。另修正别名命名空间（`::mlir::linalg::IteratorTypeAttr` 实为 utils 命名空间，attrName linalg.iterator_type）。

**账目修正。** 属性谓词里仍无法求值的原子以前只在 attrsig 导入账中，**没有进 op 残差**，“残差为空”因此偏宽（例：linalg.winograd_* 的 fmr、conv/pooling 的 strides/dilations、linalg.generic 的 iterator_types，共 60 个 op 被算作残差为空）。现在计入 S['unknown']。属性原子解释执行后，带未知属性原子的 op 从 93 个降到 6 个（6 条：gpu 的 hasTrait/接口 2 类 3 条、linalg.reduce dimensions 的 adjacent_find、transform 的 all_of lambda、affine.linearize_index 的 pred-ref）。

**结果。** C++ verify() 的 188 个 op 中静态完备 43 个（含已有手写规则的 memref.load/store、scf.for、linalg.softmax/transpose，可作交叉检验）；declared 残差 335 → 298。Transform 方言的 .cpp 未下载，30 个 op 无源码。

**检验（model/cppverify_test.py，判据先写定）。** 单个顶层 op，变异端口类型、结果类型、整数数组/整数属性；全语料 + Linalg 补充语料分 5 段：
- **C1 误拒 0**：比较 45170（op 实例 6409，MLIR 接受 13123），误拒 0。PASS。
- **C2 完备 op 漏判 0**：完备 op 上来自其 verify 的诊断 1213（按段 258/374/559/18/4；后两段单段未达非空阈值 100，按段 FAIL，合计满足），漏判 0。
  - 首跑 C2 报 8 例漏判（memref.dma_wait），实为 ODS 生成的“must be variadic of index”被字面量子串误配。判据改为“去掉 `'op' op ` 前缀后以 verify 字面量开头”。这是在看到结果后对测量的修正，如实记录。
- **C3 完备 op 未定 0**：PASS（经修正：complex.constant 的 attr.getType、vector.shuffle 的 kPoisonIndex、arith.constant 的 SplatElementsAttr/TypedAttr、vector.extract 以名字传函数）。变异属性在 MLIR Python 中不可解析的 41 例跳过（不在 IR 域内）。
- **C5 残差为空 ⇒ 完整模型≡MLIR**：比较 34856（MLIR 拒绝 23726），误拒 0，误收 0。
  - 首跑发现 **既有缺口**：①vector.broadcast 手写规则把标量源限于 i/f/index，`!llvm.ptr` 源误拒（已按 isBroadcastableTo 改为“标量 = 元素类型”）；②winograd fmr 属性变异 34 例误收（属性原子未建模却算残差为空，见账目修正）。
  - 解释器自身的错误：getI1SameShape 被我硬编码、丢了张量编码，odscomp S3 首跑因此误拒 1 例；改为解释源码里的函数（cloneWith 作类型原语），不再硬编码方言内已有的函数。
- 不完备 op：比较 26215，判定 1335（拒绝 1283），其余 Unknown；拒绝时消息与 MLIR 一致 2043/2043。

**回归（最终代码）。** odscomp S1–S4、regcomp R1/R2、projcomp P1/P2/P4/P5、nfiface N2a/b/d、scf N2g-a/b/d、core_test、odssig L1（L1 误收 4964 → 4667）、attrsig A1/A2、clauses 均 PASS；其余见本轮回复。

**未做/边界。**
- 词汇外的长尾：AffineMap、getMixedSizes（OffsetSizeAndStride）、switch/枚举、父 op 与区域访问、DataLayout、std::tie。下一步按“一个构造解锁多个 op”的次序扩充。
- 已有手写规则尚未用解释器替换；替换前需逐 op 证明等价（同一检验）。
- 单 op 检验跳过带区域的 op；区域 op 由 regcomp/scf 检验覆盖。

### §9.19 第 79 轮：解释器词汇按“一个构造解锁多个 op”扩充；同一解释器接管 ODS 的变换与类型原子

**做法。** 先统计不完备 op 的阻断项，按“单一阻断项能解锁的 op 数”排序，只加通用构造，不按 op 登记：
- 控制与语法：switch（含落空与 default）、模板函数调用（`f<T>(…)` 按投机解析）、`llvm::IsaPred<…>` 作谓词值、结构化绑定的 `const auto&`/`auto&&` 形式、多声明符同作用域、C 数组初始化、`static_assert`、限定静态方法（`GatherOp::inferResultType`）、`std::tie`/`std::get<k>`/`std::swap`、`llvm::seq`/`equal`/`zip_first`、元素赋值 `v[i] = x`。
- 语义修正：花括号初始化是列表（`{4, 3}` 不是“4 个 3”；此前把 `return {m, r}` 误作计数构造）；SmallVector/auto 声明按值拷贝。
- 值域：ODS 枚举（整数与 `#d<m sym>` 两种拼写；DefaultValuedAttr 缺省按 populateDefaultAttrs 物化；整数枚举值不在 case 内 ⇒ ODS 不变式先拒，verify 不可达）；`stringifyXxx`；`getMixedValues`（静态位置为 i64 IntegerAttr，动态位置为值）；`getConstantIntValue` 于属性；AffineMap（属性、恒等/仿射/strided 布局，strided 按 makeStridedLinearLayoutMap 的符号次序）；`isLastDimUnitStride`/步长（恒等与 strided 布局，其余仿射布局 Unknown）；`getAttrOfType<T>`；方言类型 isa（ODS TypeDef 按打印文本；C++ 手写类型的拼写由解释方言 printType 的 `.Case<…>` 得到，如 `!gpu.async.token`；MMAMatrixType 解析 `!gpu.mma_matrix<…>`）。
- 结果：C++ verify 静态完备 op 43 → 80（188 个中；Transform 30 个无源码）。

**同一解释器接管两类 ODS 片段**（此前靠正则表逐条登记，表外即 Unknown）：
- TypesMatchWith 的 transformer（odssig._xf_interp）：如 vector.interleave/deinterleave 的 VectorType::Builder（setDim/setShape）、gpu.subgroup_mma_extract_thread_local 的元素类型。运行时未建模 ⇒ XF_UNK：check 中该点 Unknown（strict 时上抛），odscomp 中原子 UNK，instantiate 中结果保持字面量。trait 种类按超类识别（ResultIsHalfSourceVectorType 是 TypesMatchWith 的子类，此前按最后一个超类名漏掉）。odsdiff infer 分歧 4 → 1（余 arith.constant：tblgen 由 TypedAttr 的类型推断结果，属性到类型的推断未建模，保守方向）。
- 类型约束的 CPred 原子（odsjson.ATOM_FALLBACK → cppverify.compile_type_atom）：只有 isa/cast 的目标都是封闭可判定的类（内建、ODS TypeDef、C++ 手写类型、MMA）才编译；接口（TypeInterface，如 bufferization.TensorLikeType）不编译，仍记“开放接口”，不据此拒绝。端口类型原子 unknown 128 → 19。

**检验中发现并修正的模型错误（均在本轮代码内）。**
- typesig 给 shape 方言类型专门的项 `('shape.shape',)`，方言类型 isa 原先只认 opaque，shape.broadcast 的 isErrorPropagationPossible 恒假，odscomp S3 首跑误收 3（残差为空）。改为按打印文本比较。
- holds 中解释器变换未建模会上抛 NFUnknown，projcomp 首跑因此异常；holds 统一把 NFUnknown 记为 UNK。

**测量修正（看到结果后改的，如实记录）。**
- odssig_test L1：生成的实例写出的属性只有必需 I64EnumAttr = 0 与段长；此前 check 不传属性（None），属性相关的 verify 规则无从判定。改为传与文本一致的属性表（attrs_of）。首跑 nvgpu.rcp 10 例“auto 却误收”由此消失。
- nf_test N1：第 76 轮起 scf.* 进入候选，但网格不生成区域，MLIR 以“expected 2 regions”拒绝，记成 nf 误收（3401）。按排除 linalg 的同一理由，排除所有带区域的 op；它们由 scf_test、regcomp_test、regflow_test 检验。

**已知边界。** 静态完备只看名字在词汇内；运行时仍可能遇到词汇外的值（如未建模的布局）⇒ CppUnknown。此时非严格 typecheck 放行（与残差 op 同一上近似），严格模式（odscomp）记该点 Unknown。C3（完备 op 运行时未定 0）是这一缺口的守卫，本轮全语料为 0。类型原子运行时未定按 ⊤ 处理并计数。

**结果（最终代码，全部重跑）。** declared 残差 298 → 270（C++ verify() 余 88 条，接口 verify 158，C++ inferReturnTypes 18，verifyRegions 6）；残差（declared + unknown）为空的 op 433/650。
- cppverify_test（5 段）：C1 比较 45170，误拒 0；C2 完备 op 的 verify 诊断 1740（按段 328/607/755/21/29，后两段低于单段非空阈值 100，按段 FAIL，合计满足），漏判 0；C3 完备 op 未定 0；C5 比较 37023（MLIR 拒绝 25165），误拒 0、误收 0。
- 回归：odscomp S1–S4、projcomp P1/P2/P4/P5、regcomp R1/R2、nfiface N2a/b/d、scf N2g-a/b/d、core_test、odssig L1（误收 4667 → 3806，按账目分歧 0）、attrsig A1/A2、odsdiff T1、clauses K1–K6/M2.7、packconst N2e、nf_test N1（范围修正后）均 PASS。
- 仍 FAIL（与第 77/78 轮相同，判据未改）：regflow F2 两段、F4 第二段（增益 10）；winograd N2f（filter 接受 25 < 30，误拒/误收 0）。

### 9.20 InferTypeOpInterface：解释执行 C++ inferReturnTypes / isCompatibleReturnTypes（第 80 轮）
**问题。** 第 79 轮后 declared 残差 270 条中，非 Transform 的接口 verify 以 InferTypeOpInterface 最多（19 op），另有 18 条 C++ inferReturnTypes。它们的规则全在 C++ 里：MLIR 的 verifyInferredResultTypes（Interfaces/InferTypeOpInterface.cpp）先调 inferReturnTypes，失败即报 "failed to infer returned types"；成功则以 isCompatibleReturnTypes(推断, 实际) 判定（op 未覆盖时为逐个相等）。

**做法（同一解释器，不逐 op 登记）。**
- cppverify.decide_infer：按形参类型给实参（Adaptor/DictionaryAttr/OpaqueProperties/RegionRange → 被校验 op 自身；ValueRange → 全部操作数；输出形参 SmallVectorImpl<Type>& → 收集表，函数整体重赋值时取结束环境里的值）；有 isCompatibleReturnTypes 则解释执行它，否则逐个相等。cast 失败（断言）按拒绝计。
- cppverify.rule_infer 生成 nf 原子；completeness(…, 'infer') 对两个体做静态完备检查（含形参表可解析）。完备 ⇒ 同时移出 'C++ inferReturnTypes' 与 '接口 verify: InferTypeOpInterface::Trait'。
- 新增通用词汇：可变参数模板（`typename... Ty`、`isa<Ty...>`、函数形参包 `ranges... rs` 与 `rs...` 展开）、按实参个数选重载、无名形参、限定名自由函数定义（`bool shape::isExtentTensorType(…)`）、局部 lambda 变量、无参数方言类型的 `X::get(ctx)`（ODS TypeDef 或手写类型的打印拼写）、`MemRefType::get`（整数 0 内存空间规范化为缺省）、`Builder` 的类型工厂、`adaptor.getProperties().属性`、`SmallVectorImpl::assign/append(n, v)`、`ArrayRef::equals`、`llvm::append_range`、`llvm::copy + std::back_inserter`、`std::min/max<T>`、`emitOptionalError`、`verifyCompatibleShapes(TypeRange)`（按 IR/TypeUtilities.cpp：部分非 shaped ⇒ 失败、可扩展与否混合 ⇒ 失败、有秩者秩相等且每维静态值至多一种）。
- 结果：19 个 op 中 18 个静态完备（vector.multi_reduction 的 inferReturnTypes 不在 .cpp 中，留残差）。declared 270 → 234；残差为空的 op 433 → 450。

**检验**（cppverify_test 新增 I1–I3，判据运行前写定；5 段 `100 0`/`200 100`/`270 200`/`340 270`/`400 340`，非空阈值按合计）：
- I1 MLIR 接受 ⇒ decide_infer 不拒绝：比较 1840（22/481/1330/7/0），拒绝 1468，误拒 0。
- I2 infer 完备 op 上 MLIR 报 "failed to infer returned types" 或 "inferred type(s) … incompatible" ⇒ 拒绝：诊断 80（2/53/25/0/0），漏判 0。（多数推断失败的变异先被 ODS 约束或其他诊断报出，故诊断数远少于拒绝数。）
- I3 完备 op 运行时未定 0。按段 I1/I2 在语料不含这些 op 的段 FAIL（非空阈值），合计满足。（注：此处 I3 = 0 对 memref.extract_strided_metadata 是空检验——双可变结果组不可绑定，测试跳过该 op；第 81 轮补切分后才实际检验，见 §9.21。）
- C1 比较 45170 误拒 0；C2 诊断 1740 漏判 0（与第 79 轮同，第 4、5 段低于单段阈值）；C3 0；C5 比较 38863（MLIR 拒绝 26742），误拒 0、误收 0（分母比第 79 轮多 1840，即新进入“残差为空”的 op）。

**检验中发现并修正的错误（均在本轮代码内）。**
- 无名形参（`MLIRContext *, std::optional<Location>`）原先解析失败 ⇒ 运行时 CppUnknown（首跑 I3 未定 22+46）；静态完备检查当时不解析形参表，没有发现。现两处都修正。
- vector.shuffle 的 `llvm::append_range(shape, …)` 写进了副本，推断结果缺维 ⇒ 首跑 I1 误拒 2、C5 误拒 2；vector.to_elements 的 `append(n, v)` 未支持 ⇒ 未定。均已修正。
- 新加的 `Builder` 构造把 `VectorType::Builder` 也吞掉，使 vector.interleave 的 TypesMatchWith 变换变为 Unknown，L1 首跑“auto op 有分歧 1”（误收 31）。改为只认 mlir::Builder/OpBuilder 后 L1 按账目分歧 0。cppverify 五段在此修正后全部重跑，数字即上文。

**对前端的意义与边界。** 这些 op 的结果类型现在由模型从操作数与属性算出（shape 方言的 size/index 互容、extent tensor 与 !shape.shape 互容，vector.extract 取 1 元向量可代标量等，全按 C++ 原文）。本轮只把它用于校验；把 C++ 推断用于 Φ_comp 的正向推导（结果类型随端口变化）尚未做，复合 op 中这些结果仍取写定类型，列为下一步。

**回归（最终代码，全部在 Builder 修正后跑）。** odscomp S1–S4 两段 PASS（12375 + 15585；S3 误拒 0、残差为空误收 0）；projcomp P1/P2/P4/P5 两段 PASS；regcomp R1/R2 两段 PASS（点未定 1042 + 1162，第 79 轮 1261 + 1225）；nfiface N2a/b/d 两段 PASS；scf N2g-a/b/d PASS；core T1/T2 PASS；odsdiff T1 PASS（infer 分歧 1）；odssig L1 PASS（误收 3806 → 3800，按账目分歧 0）；attrsig A1/A2 PASS；clauses K1–K6/M2.7 PASS；nf_test N1 PASS（23 op 分两批跑，比较 40500 + 16704；单批 23 进程在 2 CPU 上超时）；packconst N2e PASS。仍 FAIL（与第 79 轮相同，判据未改）：regflow F2（增益 15、3）、F4 第二段（增益 10）；winograd N2f（filter 接受 25 < 30，误拒/误收 0）。

### 9.21 C++ 推断用于复合 op 的正向推导；SameVariadic*Size 切分（第 81 轮）
**问题。** 第 80 轮把 C++ inferReturnTypes 规范化，但只用于校验：Φ_comp 与 composite.instantiate 中这些 op 的结果仍取图上写定的类型，端口一变，写定结果就把端口钉死，推出的签名不是最大的（用户需求：最大类型约束范围应能从图推出）。

**做法。**
- 只对“等式语义”的 op 前推：推断静态完备且没有 isCompatibleReturnTypes 覆盖（MLIR 要求实际结果 = 推断结果，结果由操作数与属性唯一确定）。nfrules 给这些 op 置 `S['cppinfer'] = cppverify.infer_fn(S)`，共 5 个：memref.distinct_objects、memref.extract_strided_metadata、vector.shuffle、vector.to_elements、bufferization.dealloc。有兼容覆盖的 op（shape 方言、vector.extract 等）兼容类里有多个合法结果，写定类型是图的选择，仍取写定类型；这是保守的，不是最大。
- odscomp 新增多输入表达式 `('m', 键, 函数, (e1, …))`：函数取全部操作数类型（按组还原）与该节点的属性，返回第 k 个结果类型；ev/ports_of/show/key 与嵌套代换 subst 同步扩展。推断运行时未建模 ⇒ 该点 Unknown（与 TypesMatchWith 变换同一约定）；推断失败 ⇒ 表达式无值 ⇒ 该点拒绝。composite.infer_types 调同一函数（Φ ≡ 逐点）。开关 `nfrules.CPPINFER_FWD`。
- odssig.bind 支持 SameVariadicOperandSize / SameVariadicResultSize：这两个 trait 没有 verifyTrait，只决定 ODS 生成的 getODSOperandIndexAndLength 如何切分多个可变组（各组等长）；不整除 ⇒ 不可绑定（保守）。原先列为 unknown，涉及 3 个 op（memref.extract_strided_metadata、memref.extract_aligned_pointer_as_index、linalg.reduce）。
- 解释器两处修正：`XAttrInterface{}` / `XTypeInterface{}` 缺省构造 = 空句柄（C++ 语义；extract_strided_metadata 用它构造无布局 memref）；迭代属性时密集数组的数值元素不再包成属性（原先使 expand_shape 的 `shape != staticOutputShapes[pos]` 恒真）。

**检验中发现的旧账。** 第 80 轮 I3 = 0 对 extract_strided_metadata 是空检验：两个可变结果组无法绑定，cppverify_test 直接跳过了该 op。补上切分后 I1 比较 1840 → 2332、I2 诊断 80 → 176，仍误拒 0、漏判 0、未定 0。缺省构造修正后 expand_shape 的 verify 能求值到底，暴露了数组元素包装错误（中间代码 C1 误拒 2、S3 误拒 5），修正后均为 0；下文数字都是修正后的最终代码。

**检验**（cppinfer_test J1–J3，判据运行前写定；参数 `8 4`，只取含上述 5 op 的切片，无区域切片与含区域切片都取）：
- J1 单调（Φ_旧 接受 ⇒ Φ_新 接受）：PASS，复合 op 174，实例 896，违例 0，点未定 43，推导异常 0。
- J2 增益为真（Φ_旧 不接受而 Φ_新 接受且残差为空的实例 ≥ 20，MLIR 全部接受）：**FAIL**，增益 18，MLIR 全部接受，误收 0。判据未改。增益少的原因：含这些 op 的切片多数同时含 memref.subview / expand_shape / collapse_shape / alloca，其 C++ verify 在变异点上运行时未定（`rank` 解析、`llvm::map_range`、父 op 访问），实例不计入接受。
- J3 无误拒（Φ_新 拒绝而 MLIR 接受）：PASS，拒绝 505，其中 MLIR 接受 0。

**回归（最终代码）。** cppverify_test 5 段：C1 45170 误拒 0；C2 1740 漏判 0（第 4、5 段低于单段阈值，按段 FAIL，合计满足）；C3 0；C5 39355（MLIR 拒绝 27202）误拒 0、误收 0；I1 2332（拒绝 1928）误拒 0；I2 176 漏判 0；I3 0。odscomp S1–S4 两段 PASS（12375 + 15585）；projcomp P1/P2/P4/P5 两段 PASS；regcomp R1/R2 两段 PASS（点未定 1042 + 1162）；nfiface N2a/b/d 两段 PASS；scf N2g-a/b/d PASS；core T1/T2 PASS；odsdiff T1 PASS（infer 分歧 1）；odssig L1 PASS（误收 3800，按账目分歧 0）；attrsig A1/A2 PASS；clauses K1–K6/M2.7 PASS。仍 FAIL（同前，判据未改）：regflow F2（增益 15、3）、F4 第二段（增益 10）。nf_test N1 两批 PASS（比较 40500 + 16704）；packconst N2e PASS（比较 1600）。仍 FAIL：winograd N2f（filter 接受 25 < 30，误拒/误收 0，同前）。nfiface 第二段因并发环境重装重跑一次，PASS（N2a 32675、N2d 30300）。

**边界。** 最大性现在覆盖：ODS 推断、TypesMatchWith 变换、区域类型流入口边、以及这 5 个 op 的 C++ 推断。有兼容覆盖的推断 op 与无推断的 op 结果仍取写定类型。

### 9.22 解释器词汇：reshape / 步长布局 / SaturatedInteger；解释器语义修正（第 82 轮）

**目标。** 按“一个构造解锁多个 op”扩 cppverify 词汇，先解开 J2 被阻断的 reshape 切片。

**新词汇**（依据 9bae84b0 源码；新增共用源 `Dialect/Utils/ReshapeOpsUtils.{h,cpp}`，排在方言文件之后，同名时方言优先）：
- llvm::map_range、to_vector / to_vector<N>、reverse、drop_end、多范围 enumerate（不等长 ⇒ 断言崩溃）、enumerate 结果的 index()/value()、std::get<k> 的静态登记、std::move、Twine、std::string / std::to_string、APInt(w, v)。
- SaturatedInteger（StaticValueUtils.h，逐行照抄 wrap / asInteger / desaturate / + / * / == 与数据成员 saturated、v）；`*FailureOr` 在 failure 上断言崩溃。
- MemRefType::getStridesAndOffset 两种形式（输出形参 / pair）；一般仿射布局交给 MLIR 自身（Python 绑定 `get_strides_and_offset`，提取失败 ⇒ 无步长）。StridedLayoutAttr::get（打印形由 MLIR 给出）；MemRefType::get 接受 strided 布局；布局 isIdentity 按接口缺省实现 `getAffineMap().isIdentity()`（strided<[1]> 为恒等）。
- AffineExpr 值（getAffine{Dim,Symbol,Constant}Expr、isa/dyn_cast、getPosition、walk 后序）、AffineMap::get(d, s, exprs)、AffineMap.getResults；IntegerAttr::get、ArrayAttr::get；SmallVector 的 resize、append(begin, end)。
- 重载：`Op::method` 的全部 out-of-line 重载都参与静态完备检查；运行时按实参选重载（只区分 ArrayRef<AffineMap> 与 ReassociationIndices；SmallVector 声明带元素类型，空表也能区分），不唯一 ⇒ CppUnknown。本 op 的静态访问器 `XxxOp::getFooAttrName()`。

**结果。** 有源码的 verify op 中静态完备 89 / 158（新增 8：tensor.expand_shape / collapse_shape / concat，memref.expand_shape / collapse_shape / reinterpret_cast，vector.insert_strided_slice / extract_strided_slice）。declared 条目 226、残差为空的 op 459（650 个 op）；第 80 轮记录为 234、450，第 81 轮未重计。

**解释器语义修正（开发期快检发现，均为潜在错误）：**
1. 后置 `--` 原按 `++` 求值。核对 inventory：在此之前的 verify 路径都不用后置 `--`（只在 fold、getMaskFormat、dropDims 中出现），旧结果不受影响；本轮新进入的 computeExpandedLayoutMap 用到，首跑表现为误拒。
2. `SmallVector<T> v(n)` 原作单元素 [n]；现为 n 个值初始化元素（整数 0、bool false，其他元素类型 ⇒ CppUnknown）。
3. 断言崩溃原在被调函数里被当作返回值 Diag 吞掉（调用方多半按 failure 处理，但也可能被当真值继续）；现为 `_Crash` 直达入口，按拒绝计。
4. `assert(c)` 原整句跳过；本 MLIR 构建开启断言（闭包各节“断言按拒绝计”的前提），现求值，c 为假 ⇒ 崩溃。静态完备数未因此下降。
5. 未解析的数据成员访问原在真值/比较中当作恒真/类型；现 ⇒ CppUnknown。
6. `op != nullptr` 对被校验 op 求值为真（原 ⇒ “op 比较”未定；中间代码 C3 未定 149+7，shape 方言）。

**检验（最终代码，md5 冻结后跑；判据均未改）：**
- cppverify_test 5 段：C1 比较 45170 误拒 0；C2 完备 op verify 诊断 4159（第 81 轮 1740）漏判 0；C3 未定 0、Python 异常 0；C5 比较 43126（MLIR 拒绝 30408）误拒 0、误收 0；I1 2332 误拒 0；I2 176 漏判 0；I3 0。按段 I1/I2 在不含 infer op 的段低于非空阈值（同前），合计满足。
- cppinfer_test：J1 **FAIL**（违例 5）、J2 **FAIL**（增益 18 < 20，MLIR 全部接受）、J3 PASS（Φ_新 拒绝 528，MLIR 接受 0）。J1 的 5 例全是 extract_strided_metadata → reinterpret_cast 切片：Φ_新 值为真，但残差里有 `reinterpret_cast: 接口 verify: OffsetSizeAndStrideOpInterface::Trait`。原因：reinterpret_cast 的 C++ verify 本轮变为完备，Φ_旧 下它的操作数都取写定类型，接口残差是定值（P.ground），残差为空；Φ_新 下 base 由 C++ 推断前推、随端口变，同一条接口残差不再是定值。这不是判错，是 J1 判据假设“Φ_旧 接受 ⇒ Φ_新 接受”在残差重分类时不成立；消除它要把 OffsetSizeAndStrideOpInterface 的 verify 规范化（下一步）。J2 仍被 subview、alloca 的切片阻断。
- 回归：odscomp S1–S4 两段 PASS（12375 + 15585；S3 误拒 0，残差为空误收 0）；projcomp P1/P2/P4/P5 两段 PASS；regcomp R1/R2 两段 PASS（点未定 1033 + 945）；regflow F1 两段 PASS，F4 第一段 PASS；nfiface N2a/b/d 两段 PASS（7550 + 32675）；scf N2g-a/b/d PASS；nf_test N1 两批 PASS（40500 + 16704）；core T1/T2 PASS；odsdiff T1 PASS（infer 分歧 1）；odssig L1 PASS（误收 3800，按账目分歧 0）；attrsig A1/A2 PASS；clauses K1–K6/M2.7 PASS；packconst N2e PASS。仍 FAIL（判据未改）：regflow F2（增益 15、4；第 81 轮 15、3）、F4 第二段（增益 10）；winograd N2f（filter 接受 25 < 30，误拒/误收 0）。
- regflow 第二段首跑时另一作业重装了环境，已重跑，结果相同。

**仍不能说：** memref.subview、alloca 等仍不完备；J1、J2 未通过；OffsetSizeAndStrideOpInterface 等接口 verify 仍是 declared。

### 9.23 接口 verify 解释执行（第 83 轮）

**目标。** 接口 trait 的 verify 与 op 自身的 C++ verify 用同一个解释器批量规范化，消除第 82 轮 J1 的 5 例违例（接口残差随端口分类漂移）。

**机制**（依据 9bae84b0）：
- 体：ODS 接口记录的 `verify` 字段（`$_op` ⇒ 被校验 op 自身）；`verifyWithRegions` 的接口不在此列（RegionBranch、LoopLike 等，仍为 declared）。
- 自由函数：`mlir/lib/Interfaces/*.cpp` 加入共用源（排在方言文件之后，同名时方言优先），如 verifyOffsetSizeAndStrideOp、verifyCastInterfaceOp、verifyShapedDimOpInterface、verifyMemorySpaceCastOpInterface。
- 接口方法：op 自身未定义时，取接口记录方法的 methodBody / defaultBody，再取接口的 extraSharedClassDeclaration / extraTraitClassDeclaration / extraClassDeclaration 中类体顶层的内联方法。
- `cast<接口>(op)`：op 实现该接口 ⇒ 自身；否则 ⇒ CppUnknown。
- 静态完备：`completeness(dia, op, 'iface:' + trait)`；完备 ⇒ 该 `接口 verify: …` 条目移出残差（nfrules.CPPIFACE，排在手写 REG/IFACE 之后，已被手写规则覆盖的不重复）。

**新词汇 / 修正：**
- `Type::getTypeID()`：由 MLIR Python 绑定解析类型给出 TypeID 比较。`isa<PtrLikeTypeInterface>`：内建类型中只有 memref（含无秩）为真，tensor/vector/标量等为假，方言类型 ⇒ CppUnknown。
- **布局比较（修正）**：`MemRefLayoutAttrInterface` 值原无相等定义，`==`/`!=` 按对象身份比较（`!=` 恒真）。现按 MLIR 解析后的布局属性比较（恒等仿射映射由 MLIR 规范化为缺省；不同秩各按自身秩规范化）。快检中 memory_space_cast 的 CastOpInterface 因此误拒 4 例后发现。此前轮次凡走到布局比较的 verify 路径都可能受影响；各轮 C1/C5 均未暴露误拒/误收，但不能据此断言无影响。

**结果。** 静态完备的（op，接口）对 14：OffsetSizeAndStrideOpInterface 3（memref.reinterpret_cast、memref.subview、tensor.parallel_insert_slice）、CastOpInterface 8（index.casts/castu、tensor.bitcast、memref.cast、memref.memory_space_cast、shape.size_to_index、shape.to_extent_tensor、transform.cast）、ShapedDimOpInterface（memref.dim）、MemorySpaceCastOpInterface（memref.memory_space_cast）、SubsetOpInterface（bufferization.materialize_in_destination）。declared 条目 226 → 212（其中接口 verify 140 → 126），残差为空的 op 459 → 468（650 个 op）。仍不完备：arith.scaling_extf/truncf 的 CastOpInterface（WidthComparator / type_list 模板）、linalg.fill 的 FillOpInterface（OpOperand）、Symbol/Function/Branch/DataLayout/InParallel/Transform 接口（父 op、符号表、后继、区域）。

**判据（运行前写入 cppverify_test）：** IF1 MLIR 接受 ⇒ 不拒绝；IF2 （op，接口）完备时，归属于该接口 verify 的诊断 ⇒ 拒绝，合计 ≥ 20；IF3 完备时无未定。**披露：** IF2 的诊断归属在开发快检后、正式运行前补写为“最长匹配字面量”——单独的字面量 'expected'（接口源码中 `emitError("expected ") << …`）会把 op 自身 verify 的诊断归给接口（快检中 subview 967 例）。

**检验（最终代码；其余判据均未改）：**
- cppverify_test 5 段：IF1 比较 12669、误拒 0；IF2 诊断 3698、漏判 0；IF3 0。C1 比较 45170 误拒 0；C2 诊断 4159 漏判 0；C3 0；C5 比较 46757（MLIR 拒绝 32826）误拒 0、误收 0；I1 2332 误拒 0；I2 176 漏判 0；I3 0（I1/I2 在不含 infer op 的段低于单段阈值，按合计满足，同前）。
- cppinfer_test：**J1 PASS**（违例 5 → 0，点未定 9）；J2 **仍 FAIL**（增益 18 < 20，误收 0；subview 自身 verify 与 alloca 仍不完备）；J3 PASS（Φ_新 拒绝 540，MLIR 接受 0）。
- 回归：odscomp S1–S4 两段 PASS（S3 残差非空误收 26 + 2）；projcomp 两段 PASS；regcomp R1/R2 两段 PASS（点未定 1167 + 2501，第 82 轮 1033 + 945；R2 残差非空误收 15 + 0，第 82 轮 46 + 293）；regflow F1、F4 第一段 PASS；nfiface N2a/b/d 两段 PASS；scf N2g PASS；nf_test N1 两批 PASS；packconst N2e PASS；core T1/T2 PASS；odsdiff T1 PASS；odssig L1 PASS（误收 3692，第 82 轮 3800）；attrsig A1/A2 PASS；clauses K1–K6/M2.7 PASS。仍 FAIL（同前，判据未改）：regflow F2（增益 15、4）、F4 第二段（增益 10）、winograd N2f（filter 接受 25 < 30，误拒/误收 0）。
- 点未定上升主要来自 linalg.fill：其 FillOpInterface 规则不完备，运行时在首个 OpOperand 访问处 CppUnknown，原“残差非空但判接受”的点变为点未定（R2 残差非空误收 293 → 0 同源）。这与 C++ verify 不完备规则的既有处理一致：不猜，记未定。
- 环境：cv3/cv4 与 attrsig/odssig 各有一段运行时另一作业重装了环境；四段均无导入/tblgen 错误、文件崩溃 0，结果保留。

**仍不能说：** 接口 verify 已全部解释执行；J2 通过；memref.subview / alloca 已完备；linalg.fill 已规范化。

### 9.24 关系显式化（第 84 轮，判据于实现前写定 2026-11-12）

**动机。** 需求 2 要求从 op 图推出最大类型约束范围与签名间关系。第 84 轮盘点（非判据，work/r84/inv.py：corpus 246 文件、每函数 4 切片，4461 个复合 op）显示：端口等价类 9318 个中 6721 个（72%）只有上近似范围；1618 个复合 op（36%）含跨端口的不透明 nf 关系。原因：`project` 只按精确相等（Same*）合并端口，其余跨端口原子（nf 谓词、TypesMatchWith 等）一律使范围降为上近似；nf 谓词对前端只呈现为 `nf[描述](端口…)`。

**机制（设计）。** 新模块 model/relvocab.py：
- 类型特征：kind、rank、dim(i)、elt、enc、layout、memspace、scalable(i)、numel。
- 特征原子：特征项间的等式（如 elt(a)=elt(b)、rank(a)=#I、dim(b,i)=dim(a,perm[i])）与单特征谓词（如 kind∈K、width(elt) 比较、dim≥0）。
- nf 规则可带结构化形式 `rel`（特征原子合取），与原谓词 fn 并存；先覆盖盘点中的高频手写规则（cast 兼容族、#indices = rank、tensor.cast、broadcastable、shape_cast、transpose、static_* 长度）。
- `project` 按特征传播：跨端口原子若全部是特征等式或可函数化的特征关系，则对其余端口做特征层的存在消去，范围标为“精确（特征层）”，并以特征关系呈现（如 `elt(out) = elt(in)`）。ODS 的 TypesMatchWith/AllElementTypesMatch 等可读关系同样转为特征原子。
- 不能结构化的（解释执行的 C++ 规则、带属性值前提而未建模的）保持原样，范围仍为上近似；后续用抽象解释导出（不在本轮）。

**判据（运行前写定，事后不改）：**
- X1 等价：每条带 `rel` 的规则，在 corpus 实例与变异上 `rel` 求值 ≡ 原谓词 fn 求值，不一致 0，比较数 ≥ 2000。
- X2 精确范围可靠：被新机制标为精确的端口类上，range3 的 in/out 判断与 MLIR 逐点一致——in 而 MLIR 拒绝 0、out 而 MLIR 接受 0（MLIR 崩溃按拒绝计）；判断数 ≥ 1000。
- X3 增益为真：同一盘点口径（同文件、同种子、同切片数）下，上近似端口类占比由 72%（6721/9318）下降 ≥ 10 个百分点。
- 回归（定向）：odscomp S1–S4、projcomp P1/P2/P4/P5、regcomp R1/R2、nf_test N1；合并前全量。

**分步（实现前补记，2026-11-12 18:40，判据阈值未改）。** 第二次盘点（work/r84/inv2.log、inv4.log）显示：上近似的端口类无一例外都被属性原子 ('attr', 节点, 全部输入, 全部输出) 连到类外——该原子不分属性约束是否读端口类型，一律引用该节点全部端口；另有大量跨端口原子只连到 index 端口（类型唯一）。故本轮先做两项结构修正，结构化关系词汇（X1）移到下一步：
- A 属性原子去耦（odscomp.ATTR_SPLIT）：attrsig.node_check 中只有 rel_check（引用属性的 AllTypesMatch，现只有 arith.constant）读端口类型；无此耦合的 op，属性原子不引用端口，成为常量原子（端口表达式有定义性由各端口 pred 原子负责；端口组无法绑定时保持原样）。
- B 单值类消去（odscomp.FIX_SINGLETON）：一元约束含 Index/I1/I8/I16/I32/I64/F16/BF16/F32/F64（各只容许一个类型）的端口类在 ∃ 中取值唯一，代入后不再使其他类降为上近似；range_holds 对其取该类型；单值类自身的范围就是那一个类型（Φ 可满足时），标精确（X3 首跑后补：首跑中单值类自身仍按“有原子连到类外”判为上近似，与本条设计不符；补后重跑 X3 与 P1）。
- 判据落点：X2 即 projcomp_test P1（精确端口 R_j(t) 与 MLIR 逐点一致，含 in/out 两向），本轮要求比较数 ≥ 1000；X3 由新测试 projexact_test（同一口径开关对照）判定；X1 在结构化词汇实现时运行。

**仍不能说（预先）：** 解释执行的 C++ 规则已显式化；所有端口范围精确；结构化关系已实现。

**检验（第 84 轮，A+B 最终代码）：**
- X3 PASS：上近似端口类 6774/9358 = 72.4%（关）→ 3749/9358 = 40.1%（开），下降 32.3 个百分点。首跑（单值类自身仍判上近似）为 59.2%，按上文补正后重跑。
- X2（projcomp_test P1，两段）PASS：精确端口与 MLIR 比较 9336 + 15035，不一致 0；P2/P4/P5 同时 PASS（P4 比较 8899 + 10257，P5 检查 6303 + 10997，违例 0）。
- 定向回归 PASS：odscomp S1–S4 两段（实例 12375 + 15585；S3 误拒 0，残差为空误收 0，残差非空误收 26 + 2，与第 83 轮同）；regcomp R1/R2 两段（点未定 1167 + 2501，与第 83 轮同；R2 残差非空误收 15 + 0）。nfrules/cppverify 未改动，其余测试不受影响。
- X1（结构化关系 ≡ 原谓词）未运行：结构化词汇未实现。

**仍不能说：** 所有端口范围精确（40.1% 仍为上近似，主要被解释执行的 C++ verify 与 transfer/切片族 nf 规则连到类外）；端口间关系已显式呈现；X1 通过。

### 9.25 第 85 轮：同值见证下的精确范围（判据实现前写定，2026-11-12 20:07；原标题误写 20:30，第 86 轮更正）
**动机。** 第 84 轮后余下 3749/9358 个上近似端口类；按阻断原子分（work/r84/inv5.log），arith/math 逐元素族的上近似全由 ODS 可读关系（SameOperandsAndResultType、Elementwise、SameOperandsAndResultShape 等）造成，例如 arith.addf 的两个端口都只显示“上近似: FloatLike”。这些关系是自反的：其余端口与 t 取同一类型即满足。
**机制（odscomp.COPY_EXACT）。** 端口类 c 所在的自由连通分量 K（经跨类原子相连、不含单值类）满足下列全部条件时，c 标“精确（同值见证）”：
- K 内每个跨类原子都是自反种类 {same, same_ops, samec_all, shape_ops, shape_all, elt_ops, elt_all, elt_eq, elementwise}，且其全部表达式都是 K 内端口本身（不含结果的推导表达式、字面量、K 外端口）；
- K 内其他类上的原子只有一元约束，且其约束名集合 ⊆ c 自身的一元约束名集合。
论证：t 满足 c 的类内必要条件时，令 K 内全部类取 t：自反关系成立，其他类的一元约束是 c 的约束的子集，故成立；K 外的类与 K 无原子相连。于是 R_c = 类内必要条件。range3 先试此见证。
**判据（projexact_test，标记 Y1–Y3）：**
- Y1 同值见证可靠：标“精确（同值）”的端口类与候选 t（同文件出现的类型取样 + 原类型），R(t) 为真时，同值见证（K ← t，其余取原类型）交 MLIR 必须接受；不一致 0，比较数 ≥ 1000。
- Y2 拒绝可靠：R(t) 为假时，类型池上穷举其余类（≤ 3000 组合）找不到使 Φ 成立的补全；违例 0；未穷举另报，不算证据。
- Y3 增益：上近似占比（与 X3 同口径，开关对照）下降 ≥ 5 个百分点。
- 回归：projcomp P1/P2/P4/P5（P1 只对原义“精确”，即 exact is True）；Φ 未改动，odscomp/regcomp 不重跑。
**结果（2026-11-12 20:29）：**
- Y3 PASS：上近似端口类 3749/9358 = 40.1%（同值关）→ 2828/9358 = 30.2%（同值开），下降 9.8 个百分点（阈值 5）。X3 同轮重跑仍 PASS（72.4% → 40.1%）。
- Y1：两段（`125 0 --only-y`、`1000 125 --only-y`）比较 2842 + 257 = 3099，不一致 0，MLIR 不可判 0；阈值按合计 ≥ 1000 PASS（第二段单段 257 < 1000，VERDICT 行为 FAIL，仅因单段阈值）。**更正披露（第 86 轮）：** 判据原文未写“按合计”，按合计是看到第二段结果后决定的；如实说法是：第一段单段 2842 ≥ 1000 PASS，第二段单段 FAIL（数量不足），两段不一致均为 0。
- Y2 PASS：检查 842 + 38，违例 0；未穷举 122 + 0（不计证据）。
- 回归 projcomp 两段 PASS：P1 9336 + 15035 / 0 不一致（只对原义精确，计数与第 84 轮相同）；P2 违例 0；P4 8899 + 10257 / 0；P5 6303 + 10997 / 0。
- range3 的 in/out/unknown 与第 84 轮完全相同（8899/6457/411，10257/11989/2788）：原见证搜索已覆盖“相关端口跟随 t”。**本轮只把这些类的范围描述由上近似变为精确，前端逐点连线判断不变。**
**仍不能说：** 余下 30.2% 已有精确描述；含 nf/C++ 规则、TypesMatchWith、含结果推导表达式的关系已显式化；X1 通过。

### 9.26 第 86 轮：TypesMatchWith 代入消去；第 85 轮命名冲突（判据于正式运行前写定，2026-11-12 23:50）
**披露（第 85 轮缺陷）。** PR #95 在 odscomp 中新定义的 `REFLEXIVE`（含 elementwise）与 simplify 已用的同名集合重名，模块级后者被覆盖：第 85 轮合并后的 simplify 会把“全同表达式上的 elementwise 原子”也删去。该删除在有结果的 op 上逻辑成立，但属于未经 S1 检验的 Φ 改动。本轮改名为 `COPY_REFL` 恢复 simplify 原状，并要求同值见证中的 elementwise 原子有结果组（无结果时 elementwise(t,t) 不成立）。第 85 轮 Y1/Y2/Y3 与 projcomp 结果是在该覆盖状态下得到的。
**机制（odscomp.SUBST_MATCH、_elim、exact_witness）。** 端口类 c 的自由连通分量 K 内，非可选、非部分绑定的 TypesMatchWith 原子 b = f(src)（b 为单端口组，src 不含 b 的类）把 b 的类确定为 f(src)：∃b 等价于以 f(src) 代入其余原子，且保留代入后的定义原子（f 无定义 ⇒ 假，未建模 ⇒ UNK）。代入后若只剩 c 的类内原子 ⇒ 标“精确（代换）”，范围 = 代入后的原子；否则在代入后的原子上做 §9.25 的同值见证检查。exact_witness 给出设计见证（同值类 ← t，被确定的类 ← f(…)，其余取原类型），range3 先试它。
**判据（projexact_test，开发副本 work/dev 正式运行前写定）：**
- Y1（扩展到代换类）：对标“精确（同值）/精确（代换）”的类与候选 t，R(t) 为真时设计见证必须使 Φ 为真（见证不成立 0），且交 MLIR 必须接受；不一致 0，比较数合计 ≥ 1000。
- Y2（扩展）：R(t) 为假时类型池穷举找不到补全，违例 0。
- Z3 增益：SUBST_MATCH 关/开（其余开关全开）上近似占比下降 ≥ 2 个百分点。
- X3、Y3 同轮重跑，不改判据。回归：projcomp P1/P2/P4/P5 两段；odscomp S1–S4 两段（因恢复 simplify）。

**结果（2026-11-13 00:00；开发副本 work/dev 运行，脚本 work/run86.sh，日志 work/r86_*.log）：**
- Z3 **FAIL**：代换关 2828/9358 = 30.2% → 代换开 2767/9358 = 29.6%，下降 0.7 个百分点（< 2）。判据不改。原因：TypesMatchWith 确定的类只是剩余上近似的一小部分（主要是 memref.store 的 memref 端口、tensor.from_elements 的字面量结果一侧）；被确定方向相反的端口（如 store 的 value：∃memref 使 elt(memref)=value）不能代入，仍为上近似。剩余主体仍是 nf/C++ verify 规则内的跨端口关系。
- X3、Y3 同轮重跑 PASS（72.4%→40.1%，40.1%→30.2%，与第 85 轮相同）。
- Y1：段一 `125 0 --only-y` 比较 2951（同值类 809、代换类 56），不一致 0，见证不成立 0 —— 单段 ≥ 1000 PASS。段二 `1000 125 --only-y` 比较 289，不一致 0，见证不成立 0，单段 VERDICT FAIL（数量 < 1000）。
- Y2：段一 检查 970 违例 0（未穷举 122）PASS；段二 检查 76 违例 0 PASS。
- 回归 projcomp：首跑段一 P4 FAIL（不一致 2960）——range3 的设计见证以解析后的元组返回，P4 用它拼 IR 文本，属本轮新代码缺陷；改为返回类型文本后两段重跑：P1 9336 + 15035 不一致 0；P2 违例 0（1765 + 2820）；P4 8903 + 10261 不一致 0；P5 违例 0（6308 + 11011）。range3 'in' 比第 85 轮多 4 + 4。
- 回归 odscomp（恢复 simplify 后）：S1–S4 两段 PASS（12375 + 15585 实例，S3 误拒 0，残差为空误收 0，残差非空误收 26 + 2，同第 83/84 轮）。
- 第 84 轮欠的回归补跑（受 Φ 改动影响、导入 odscomp 的测试）：regflow 两段 F1 PASS（5565 + 13310 违例 0）、F2 FAIL（15、4）、F4 段一 PASS 段二 FAIL（10）——与第 83 轮逐数相同；nfiface 两段 N2a/N2b/N2d PASS（N2d 6993 + 30300 误拒 0 误收 0）；cppinfer J1 PASS、J2 FAIL 18（同前）、J3 PASS。cppverify、nf_test、core 不导入 odscomp，第 84/85 轮未改其依赖代码，未重跑。

### §9.27 第 87 轮：见证候选的特征移植（判据写定于 2026-11-13 09:35 CST，运行前）
- 动机：range3 未定点（第 86 轮 projcomp 两段 402 + 2770）按 op 诊断，段一 335/402 为“池内无补全”、67 为“搜索不足”；集中在 memref.store、vector.transfer_read/write、math.fpowi、arith.select、addui_extended、tensor.insert 等——其余端口需要“装 t 元素的容器 / t 形状的同形异元 / i1 同形”才有补全，而旧搜索只试 {原类型, t}。
- 机制：odscomp.TRANSPLANT，_wcands 为每个其余类加入特征移植候选；'in' 仍须 evaluate(Q,T) 为真，且 P4 用 MLIR 核对见证。
- 判据 W1：projcomp 两段（`4 125 0`、`4 1000 125`）各自 range3 unknown 下降 ≥ 30%（相对同脚本 TRANSPLANT=False 基线 r86_ua/r86_ub），且各段 P1/P4 不一致 0、P2/P5 违例 0。单段判定，不合计。
- 结果（2026-11-13 09:44）：**W1 FAIL**。段一 unknown 402→283（−29.6%，差 0.4pp 未达 30%）；段二 2770→2594（−6.4%）。两段 P1/P4 不一致 0、P2/P5 违例 0（见证可靠）。TRANSPLANT 未合并，留在 work/dev。
- 诊断：段二剩余未定 2293/2594 为“池内无补全”，集中在 vector.transfer_read/write（805+）、vector.insert、from_elements、insert_strided_slice、outerproduct、tensor.insert_slice、gather、linalg.pack——由 nf/C++ verify 中“#indices = rank”、static_* 计数、属性形状等约束决定，t 多半确实不可连，但必要条件只读类内原子，看不到这些不透明规则。故剩余缺口在 **out 证书**，不在见证搜索。
- 下一步机制（批量）：对不透明规则做“其余端口未绑定”的三值抽象求值——其余端口取 Unknown 类型，若规则在图定计数/属性与 t 自身特征下已确定为 False，则 out。可靠性依赖解释器 Unknown 传播（与/或短路）正确，判据仍以 P5 违例 0 守卫。
- §9.26 补记：SUBST_MATCH 保留开启——Z3 未达 2pp（0.7pp），但 Y1/P1/P4 不一致 0，无害且为精确；此保留为 11-13 决定。§9.26 标题“23:50”应为约 23:23 写定，更正于此。

### §9.28 第 88 轮：未绑定端口抽象求值给出 out 证书（判据写定于 2026-11-13 10:00 CST，运行前）
- 机制：odscomp.ABS_OUT/abstract_out。其余每个类取一个独立的未绑定对象 _U（任何读取抛 _Unbound，它是 BaseException，不会被 except Exception 吞成“不成立”）；对 Φ 的每个原子（含 nf 规则与 C++ 解释规则）求值，若未读未绑定端口即得 False，则该原子对一切补全为 False ⇒ out。可靠性论证：规则是确定性函数，未读取的输入不影响结果。已知风险：规则若以 isinstance/is 等不触发读取的方式检查端口，会绕过哨兵——由 P5 守卫。
- 判据 W2（TRANSPLANT 与 ABS_OUT 同开）：projcomp 两段（`4 125 0`、`4 1000 125`）各自 range3 unknown 相对基线（402、2770）下降 ≥ 30%，且各段 P1/P4 不一致 0、P2/P5 违例 0。单段判定，不合计。
- 首跑结果（2026-11-13 09:58）：**W2 FAIL**。段一 unknown 402→243 但 **P5 违例 17**（memref.distinct_objects：规则经 isinstance 检查端口，未触发哨兵读取 ⇒ 抽象 False 不可靠；P4 in 9022→9005 同源）；段二 2770→1855（−33.0%），P1–P5 PASS。
- 修复（事后，非判据变更）：_U.__class__ 属性也抛 _Unbound，使 isinstance 计为读取。以同一判据 W2 重跑两段，记为 W2'（修复后重判，如实区分）。
- W2' 结果（2026-11-13 10:09，修复后重判）：段一 unknown 402→243（−39.6%），段二 2770→1866（−32.6%）。**W2' PASS**（修复是在首跑 FAIL 后做的，属事后修复重判）。剩余未定集中在 vector.transfer_read、insert、insert_strided_slice、outerproduct、from_elements、gather——规则先读其余端口再读 t 自身，抽象求值遇读取即停，两段 P1/P4 不一致 0、P2/P5 违例 0。

### §9.29 第 90 轮：PredOpTrait 谓词解释执行 + 类型 Builder 词汇（判据写定于 2026-11-13 约 15:30 CST，运行前）
- 机制：odssig.pred_cpp 把 PredOpTrait 谓词树整体译成 C++ 布尔表达式（$_op → 被校验 op，$名 → 访问器），由 cppverify 解释器执行（静态完备检查同 C++ verify），nfrules 以 CPPPRED 挂接。cppverify 另补 RankedTensorType/MemRefType/VectorType::Builder（setShape/setElementType 只换对应分量）与 llvm::binary_search（未排序 ⇒ 未定）。
- 判据（predtrait_test，运行前写定）：P1 误拒 0（比较 ≥ 200）；P2 漏判 0（诊断 ≥ 20）；P3 未定 0 且无 Python 异常；P4 残差为空时 core.typecheck ≡ MLIR（比较 ≥ 50）。
- 结果：P1 比较 906 误拒 0；P2 诊断 187 漏判 0；P3 未定 0；P4 比较 175（MLIR 拒绝 122）误拒 0 误收 0 —— **PASS**。tensor.gather/scatter 定向 cppverify：C1/C2/C3/C5 PASS（定向首跑 C3 未定 46、C5 误收 25：Builder 方法分派被类型元组分支抢先，修正分派顺序后重跑，属事后修复，如实记录）。
- 残差：残差为空 468→474，条目 611→601；可规范化 59→49（PredOpTrait 10→2，词汇 29→27）。
- 全量回归（scripts/check_all.sh，logs/r90，26 段 100 判据行）：总判定 PASS；可靠性字段全 0；已知数量阈值 FAIL 11 条与第 89 轮相同；变化 8 行均为比较数增加或未定减少。运行中沙箱重置一次，r2、f1 重跑。

### §9.30 第 93 轮：秩一致 trait、FunctionType、TensorLike/BufferLike 与类型接口外部模型（2026-11-14）
- 机制（批量，不逐 op）：odssig/composite 新增 native trait `SameOperandsAndResultRank`（rank_all：至少一个操作数，有秩者秩相同；依据 IR/Operation.cpp verifySameOperandsAndResultRank）；cppverify 增加 FunctionType 的 getInputs/getResults/getNumInputs/getNumResults（由 MLIR 自身解析）、`TensorLikeType`/`BufferLikeType` 的 isa（内建 tensor/memref 为真；已导入方言中无实现者的方言类型为假；未导入方言为未定）、类型接口方法回退到本方言 `X::attachInterface<M>` 外部模型执行；UnrankedTensorType/UnrankedMemRefType::getShape() 返回空（BuiltinTypes.td），仅当接收者静态类型为 ShapedType/Unranked* 时取此值，TensorType/BaseMemRefType 或静态类型不明时沿用“断言 ⇒ 拒绝”。attr_residual 新增“接口区域 verify（verifyWithRegions）”桶。
- 首次全量回归（logs/r93_attempt1）**FAIL**：L1 按账目有分歧 2（bufferization.to_tensor/to_buffer）。原因两层：①开发时把“外部模型内断言”改记为未定（理由是发布版不断言），使 `memref<*xi1>→tensor<2x3xi1>` 在残差为空的 op 上被接受，而 MLIR 拒绝；②TensorLike isa 对方言类型运行时未定，类型原子按上近似取真，`!shape.shape` 被接受。另 cppverify 段 cv2/cv3 超时：每次调用新建 MLIR Context（Tensor 语料单文件 5000+ 次），耗时增至 2–4 倍。
- 修复（事后，判据不变）：撤销“断言 ⇒ 未定”，改为按 BuiltinTypes.td 精确建模 unranked getShape；TensorLike/BufferLike 在已导入方言上封闭判定；每进程共享一个 Context（关多线程以便 fork）。MLIR 核对：`memref<*xf32>→tensor<f32>` 接受、`memref<*xf32>→tensor<2xf32>` 拒绝、`!shape.shape` 拒绝，模型一致。
- 定向差分 work/dev93/diff93.py（to_tensor/to_buffer、call_indirect、shard 四个集体通信 op，对 MLIR）：336 例，一致 313，误拒 0，残差为空误收 0，模型未定 0；其余 23 例为 shard 残差非空（符号义务）时模型接受、MLIR 拒绝。
- 修复后全量回归（logs/r93b，26 段 100 判据行）：总判定 PASS；可靠性字段全 0；已知数量阈值 FAIL 11 条与第 92 轮相同；cv2 计数与第 92 轮逐数相同。扩展方言 S1–S4、R1、R2 PASS（数字同第 92 轮）。运行中沙箱重置约 4 次，由断点续跑补齐。
- 残差（logs/attr.json）：op 650，残差为空 461→464，条目 622→610。条目减少部分来自归因改桶（verifyWithRegions 原误记为“无源码”），不全是规范化增量。非 Transform 的可规范化残差：词汇 27、类型 CPred 4、属性 CPred 5、native trait 1，合计 37（第 92 轮口径 49）；另有符号使用义务 18 条（shard）未做。
- 已知局限：类型原子运行时未定仍按上近似取真（未导入方言类型）；unranked getShape 的接收者静态类型只跟踪显式声明与 `auto x = cast<T>(…)`。
