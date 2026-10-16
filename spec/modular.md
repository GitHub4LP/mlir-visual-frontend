# 预言机按模块上下文发射（第 16 轮）

## 1. 复盘：预言机层自己是空检验

第 12–15 轮一直报告“C2 真实 IR 3487 函数误拒 0”，但 `corpus_test.py` 在 `run_file` 开头写着 `C.ORACLE_ON = False`：
**有拒绝权的四层里，预言机层从未在真实 IR 上被检验过不误拒**；`obligations_test` 的 `full()` 也关着它。
打开后再看可达性：真实语料 2706 个 func.func 体里 738 个（27%）根本到不了预言机：

| 不可发射原因（修前） | 体数 | 性质 |
|---|---|---|
| 缺分段 | 537 | 导入节点的分段在 `_attrs.operandSegmentSizes`，发射器只看手工字段 `seg` |
| insert_slice 的源秩与偏移数不符 | 42 | 把导入的 MLIR 形式当成 core 简写去“降级” |
| func.call 需要模块级符号 | 26 | 发射单位是孤立函数 |
| MLIR 不认识 linalg/affine/bufferization/shard… | ~130 | 守卫用的是**我们**导入的 ODS，而预言机（MLIR 自己）认识它们 |

即：第 15 轮 R1/R2 漏判的大头（“未导入方言”）本来就该由预言机补上，却被发射器自己挡在门外。

## 2. 改动（model/oracle.py、core.py、symtab.py）

- **导入节点逐字发射**：`props` 带 `_attrs` 的节点不经 ODS 守卫，属性原样写入属性字典，操作数/结果类型取当前值类型（前端编辑后的类型，而非导入时的 `_types`），区域递归，入口块也写块标签（否则空块的区域变成 0 块——`scf.forall.in_parallel` 的 16 例误拒即此）。core 原生节点仍走原守卫（参数归属、必需属性、分段、切片降级）。
- **符号上下文**：`symtab.func_scopes(g)` 给每个有体的 func.func 求最近符号表（名字 → 定义者）与最近 builtin.module 的属性。`emit_core(…, symbols, mod_attrs)` 收集体内全部符号引用（core 原生 `func.call` 的 callee、导入节点属性文本里的 `@ref`，字符串字面量除外），补成私有声明：
  - func.func ⇒ 无体 `private` 声明（S5：声明不能 public）；
  - 无区域的定义者（memref.global、shard.grid…）⇒ 原样；
  - 自递归不补；嵌套引用 `@a::@b`、有区域的非函数定义者、`dense_resource` ⇒ **不发射**（缺符号/缺资源会被 MLIR 以缺失误拒；预言机有拒绝权，发射必须忠实）。
  未导入 ODS 的方言里带 `sym_name` 的 op 也登记为定义者——只用于补声明，不赋予结构层拒绝权。
- **未注册方言**：`Oracle.verify(text, unreg)`；只在模块本就允许时打开（语料：源文件带 `--allow-unregistered-dialect`）。未注册 op 对 MLIR 不透明，其自身义务不查（上近似），其余照查。
- `core.typecheck(blocks, ftype, symbols, name, mod_attrs, unreg)`；我们未导入其 ODS 的导入节点计入“需预言机”。

## 3. 检验（model/modular_test.py，需 corpus/）

- **M1 单元**：core 原生 func.call —— 无上下文 ⇒ 不发射；有上下文 ⇒ 发射且通过；实参类型错 ⇒ 拒并归位到调用节点；自递归无需声明。5/5。
- **C2（corpus_test，C2 段打开预言机，C3 仍关闭以免 MLIR 判自己）**：3535 函数 / 3487 有体全部接受，**误拒 0**；预言机调用 2982 次，需预言机而不可发射的体只剩 4 个（dense_resource 3、`llvm.mlir.global` 定义者 1）。修复过程中真实出现过 16 例误拒（入口空块），说明这项检验不是空的；未注册方言体 175 个先表现为“发射缺陷”，改为 `unreg` 送检后消失。
- **M2 体级预言机 ≡ 整模块 verifier**：2108 个合法模块，第 15 轮逐义务变异 + T1（体内 op 结果类型改变），19271 例，真值为 MLIR 对整个变异模块的判定：

| | 例数 |
|---|---|
| 两者都拒 | 18658 |
| 两者都收 | 315 |
| **误拒（我们拒、MLIR 收）** | **0** |
| 无预言机时漏判 | 881 |
| 其中体级预言机抓住 | 583 |
| 漏判 | 298，**全部**在 func.func 体外（transform.named_sequence 体内的 transform.* 结果类型、shard 等） |
| 发射伪影（变异后 SSA 名字层不可解析，跳过） | 267 |

另查：func.func 体置空的 1 例漏判是方言属性校验（`bufferization.writable` 不能出现在外部函数参数上，`verifyRegionArgAttribute`），不属于任何层。

## 4. 定律

- **L40（体级发射忠实）**：对 func.func 体，结构层 ∧ 符号层 ∧ 类型层 ∧ 预言机（体 + 被引用符号的私有声明 + 模块属性）的拒绝 ⊆ MLIR 对整模块的拒绝（M2：19271 例 0 反例）；在 func.func 体内二者相等（体内漏判 0）。
- **L41（可达性是义务的一部分）**：一层的“0 误拒”只对送达它的实例成立，须同时报告送达率与不可送达原因（本轮：C2 需预言机的体只有 4 个不可送达，原因 2 类全部列账）。这是 L38 对预言机层的推论。

## 5. 未解决

1. ~~非 func.func 的函数体~~：第 17 轮已解决（§6）。
2. 函数自身的 arg_attrs 在单元发射中已随单元带上（§6）；方言属性 verifier 仍只由预言机覆盖，符号层无对应。
3. 方言资源表（dense_resource）未导入，3 个体不可发射。
4. 嵌套符号引用 `@a::@b` 不发射。

## 6. 第 17 轮：发射单位 = 隔离单元（oracle.units / emit_unit / check_unit，core.check_module）

第 16 轮的发射单位是 func.func 体（重写 `func.func` 头）：丢了函数自身的属性（arg_attrs/res_attrs），也覆盖不了 transform.named_sequence、gpu.func、llvm.func 等，M2 的 298 例漏判全在此。

- **单元** = 带区域、有 `sym_name`、自身不是符号表的 op，取最外层。识别是启发式的，但**选错不会误判**：若体引用单元外的值，发射文本解析不了 ⇒ 无结论（计入不可送达），而不是拒绝。
- **发射** = 单元原样（含自身全部属性、`loc("节点")`）+ 祖先链外壳（属性原样、单区域单块、只含本单元与所需声明；带操作数/结果/多区域的祖先 ⇒ 不发射）+ 被引用符号的声明，按 MLIR 最近符号表规则在父表解析：
  - func.func ⇒ 无体私有声明（不带 arg_attrs：外部函数上的方言参数属性另有约束）；其它定义者 ⇒ 原样整段（要求其中不再引用符号、不含 dense_resource）；
  - `@a::@b` ⇒ a 的外壳里放 b 的声明；三层以上 ⇒ 不发射；
  - 目标在单元内部 ⇒ 随单元发射；**在最近符号表里也不存在 ⇒ 原样保留**（整模块里 MLIR 同样找不到它，两边判定一致；典型是 transform 指向载荷 IR 的名字，如 `transform.func.replace_func_signature @func_with_…`）。
- **判定**：只认落在单元节点上的诊断；全落在单元外（被引用的符号本身不合法）⇒ 无结论；崩溃 ⇒ 拒（与真值同口径）。
- `core.check_module(g, unreg)`：结构+符号层整模块一次；每个单元先跑类型层（自动签名、属性层），通过后交预言机。

检验：
- 2112 个合法模块（含未注册方言文件）：3707 单元，预言机判定 3704，**误拒 0**；无结论 3（dense_resource）。
- **M3（modular_test 默认；`--m2` 复现第 16 轮）**：19271 例变异对照整模块 verifier——两者都拒 18956、两者都收 315、**误拒 0、漏判 0**；不开预言机时漏判 881，全部由单元级预言机抓住。变异后单元 6053，判定 6050。

**批判**：M3 的“漏判 0”是 MLIR 判 MLIR，度量的是**发射的忠实与局部性**（单元 + 外壳 + 声明 ≡ 整模块），不是我们对义务的理解；我们自己理解的部分仍是“无预言机漏判 881/19271”，其归因见 obligations.md（未导入方言、C++ declared）。局部性的用处在前端：编辑一个单元只需重验该单元（缓存键 = 单元文本 + 声明），其余单元的判定不变——这是 L42 要求的前提。

- **L42（单元局部性）**：在 M3 的分布上，整模块 verifier 的判定 = 结构+符号层整模块判定 ∧ 各单元（单元 + 祖先外壳 + 被引用符号声明）的判定。因此单元外的编辑只经由被引用符号的**声明**（func 的类型、无区域定义者的属性）影响本单元：声明不变 ⇒ 本单元判定不变。

## 7. 第 18 轮：按归因导入方言（odsjson.DIALECTS 12 → 19）

先归因再动手：M3 中“不开预言机的漏判”881 例，按预言机诊断落在的 op 及其导入账分：未导入方言 725、declared 154、unknown 2。据此导入 affine、gpu、bufferization、linalg（LinalgOps + RelayoutOps）、transform（核心 + Linalg/SCF/Func 的 TransformOps）、nvgpu、shard 的 ODS（同一提交的 .td）。

导入暴露的缺陷（**每一项都是 L33 的再现：新获得拒绝权的层先在真实 IR 上误拒**）：
1. 端口类型含方言自有类型原子（`!gpu.async.token`、`!transform.any_op`…）时 odssig 直接抛出，整个方言导入失败 ⇒ 改为记入 unknown、谓词取 ⊤（上近似，L29 方向）。
2. 一个方言跨多个 .td：tblgen 的匿名记录编号每次运行各自从头数 ⇒ 合并时撞名，取到别的文件的约束 ⇒ 后续文件的匿名名加前缀。
3. 属性层 isa 假设方言属性打印成 `#dia<mn …>` 或 `#dia<mn>`；shard 是 `#shard<axisarray[[0]]>` ⇒ 合法模块误拒 6 ⇒ 助记符按最长标识符取。
4. **S5（声明不可 public）原来对所有 FunctionOpInterface 成立**：MLIR 的 SymbolOpInterface::verify 只在 `isDeclaration() && isPublic()` 时报错，而 isDeclaration 缺省返回 false；只有 ODS extraClassDeclaration 写明 `return isExternal();` 的 op（func.func、shape.func）才有此约束 ⇒ transform.named_sequence 外部 public 合法，M3 误拒 126、Z2 误拒 87 ⇒ S5 改由 `symtab.decl_is_external` 从 .td 判定。这是一条从 func.func 外推出的手写规则——第 12 轮以来的同一类错误。
5. 属性谓词树里出现非 def 引用的子项 ⇒ attrsig 崩溃 ⇒ 进导入账（pred-ref）。
6. M3 的发射伪影正则把 `expected 'transform.yield' as terminator`（隐式终结子的解析期校验）当成伪影跳过；它是真拒绝 ⇒ 改正。

结果（MLIR 9bae84b0 语料）：
- 新方言导入账：affine 14（declared 14）、gpu 66（auto 20 / declared 10 / unknown 36）、bufferization 7、linalg 8、transform 111（auto 15 / declared 2 / unknown 94）、nvgpu 24（unknown 24）、shard 22（auto 12）。linalg 的命名结构化 op（fill、matmul、transpose、generic…）定义在构建期生成的 `LinalgNamedStructuredOps.yamlgen.td`，wheel 不带 ⇒ 仍是“未知 op”。
- 合法模块：2112 模块 3707 单元误拒 0；C2 3487/误拒 0；Y1 误拒 0、Y2 6040；A1 14905 实例误拒 0、A2 24701 变异误拒 0（ODS 属性层漏判 375，新方言的属性原子未登记）；O1/O5 一致；Z2 9049 变异误拒 0、漏判 228（原 249）。
- **M3**：19641 变异（变异集随导入的 ODS 变化），误拒 0、漏判 0；**不开预言机的漏判 698（3.55%），原 881（4.57%）**。剩余归因：declared 268、unknown 256、未知 op 161、auto 10、未导入 3。
- auto 10 例不反驳 L28：诊断都是“block with no terminator”，MLIR 把它落在块里最后一个 op（arith.addf、math.exp）上，实际义务属于父 op（linalg 命名 op 的区域，未导入）。归因按诊断位置是近似的。

批判：导入把 725 例中的一部分从“未导入”挪到了 unknown/declared，**真正被符号层自己抓住的只有 183 例**。transform 方言 94/111 是 unknown（端口类型全是方言类型），即导入给了结构与属性信息，但类型义务几乎全部仍在预言机。下一步缩小漏判的杠杆不在“再导入方言”，而在：(a) 方言类型原子（!transform.*、!gpu.*）的登记——它们是类型构造子的 isa，属于类型代数而非方言语义；(b) 区域分支类型（RegionBranchOpInterface，scf.while/tensor.pad/execute_region/named_sequence 共约 140 例）作为一条通用义务导入，而不是逐 op 手写。

## 8. 第 19 轮：类型原子的封闭与开放；构建期生成的 ODS

**(a) 方言类型原子。** 端口约束 `isa<C>($_self)` 分两类，性质不同：
- C 是具体类型类（TypeDef）：类型类是**封闭**的——一个类型恰属一个 C++ 类。odsjson.typedefs() 从全部已导入方言的 TypeDef 记录取 (命名空间, 类名) → 类型名（`transform.op`），项是 MLIR 打印的 ('opaque', 文本)，按 `!dia.mn` / `!dia.mn<…>` / `!dia<mn…>` 判定，精确。共 49 个 TypeDef。C++ 手写、无 TypeDef 的类型（gpu.async.token、gpu.mma_matrix、gpu 稀疏句柄）在 ODS 里**没有拼写**，留作 unknown。
- C 是类型接口（TransformHandleTypeInterface、TensorLikeType…）：**开放世界**——实现者可由外部模型在运行时附加（builtin tensor 经扩展实现 bufferization::TensorLikeType；pdl 类型经 transform 扩展实现句柄接口）。ODS 只看得到一部分实现者。

**L43（接口约束开放）**：接口 isa 只能正向接受，不能作为拒绝依据；拒绝权只属于预言机（MLIR 运行时知道全部外部模型）。对前端：接口端口的“可连”候选只能给出已知实现者（补全可靠不完备），“不可连”必须问 MLIR。登记为 unknown（“开放接口 isa”），不是缺陷。

实测：TypeDef 原子使 nvgpu 12、shard 3 个 op 升为 auto，但漏判只减 6（698 → 692）。原因：transform 端口 185 处约束是接口而不是具体类型——上一轮“登记方言类型原子即可缩小漏判”的预期是错的，transform 的 136 例（structured.match）原则上就在符号层之外。

**(b) 构建期生成的 .td。** linalg 命名结构化 op（fill/matmul/generic/transpose…）定义在 LinalgStructuredOps.td，它 include 构建期由 `LinalgNamedStructuredOps.yaml` 经 `mlir-linalg-ods-yaml-gen` 生成的 yamlgen.td。wheel 不带生成物，但工具轮子带同一提交的 `mlir-linalg-ods-yaml-gen`。scripts/fetch_tools.sh 取同一提交（9bae84b0）的 yaml 生成到 tools/gen/，odsjson 把它加进 -I。linalg 由 25 → 99 op（全部 declared：DestinationStyleOpInterface/LinalgOp 接口 verify），结构信息（区域、终结子）进入结构层。

同时修复导入缺陷：attrsig._has_upper 在未知属性原子（linalg::IteratorTypeAttr）上重新抛出，typecheck 异常 ⇒ 8 个合法单元的类型层被跳过（C2 一度 3479）。**异常不是误拒，但它让单元静默失去符号层——须单列计数**（MOD_STATS['类型层异常']，现为 0）。

**(c) 区域分支义务的纠偏。** 上一轮计划把 RegionBranchOpInterface 的类型义务“作为通用义务导入”。查实：`verifyTypesAlongControlFlowEdges` 是通用的，但边关系 `getSuccessorRegions`、`getEntrySuccessorOperands`、`areTypesCompatible` 都是逐 op 的 C++，ODS 不携带。能从 ODS 导入的只有“该 op 有此接口”这一事实；边关系要么手写（违背 review-r7 原则），要么问 MLIR。故不做：scf.while/tensor.pad/execute_region/named_sequence 的义务留在预言机。

验证（MLIR 9bae84b0）：合法模块 2112/3707 单元误拒 0、类型层异常 0；C2 3487 误拒 0（预言机调用 2982）；Y1/Y2、A1 14984 实例、A2 25395 变异误拒 0；O1/O5 一致；odssig 非 declared 分歧 0；Z2 9049 误拒 0、漏判 **120**（原 228）；**M3 19641 误拒 0 漏判 0，不开预言机漏判 580（2.95%，原 698 / 3.55%）**。剩余：declared 314、unknown 250、未知 op 13、未导入 3、auto 0（上轮 auto 10 是父 op 未导入导致的归因错位，导入 linalg 后消失）。

**结论（L44，残差的来源）**：剩余 580 例全部落在三类 ODS 不携带的机制上——C++ verify（declared）、开放接口（L43）、逐 op 的 C++ 关系（区域分支边）。它们唯一忠实的来源是 MLIR 自身。因此以“符号层漏判率”为目标继续推进已到边际：符号层的职责是快路径与引导（补全候选、端口取值域、编辑代价），拒绝的最终权威是隔离单元预言机（L42）。后续轮次转向：L42 增量重验（单元缓存键接 edit.md）、方言资源表、IRDL（用户自定义 op 的载体，前端组合封闭的关键）。
