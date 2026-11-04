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
