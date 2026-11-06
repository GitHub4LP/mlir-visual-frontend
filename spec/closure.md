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
