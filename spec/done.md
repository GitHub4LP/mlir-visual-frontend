# 完成标准（第 28 轮草案）

目标是把“三条正确标准”（frontend-model.md）变成可测量的门槛。每一项都给出：度量方法、门槛、当前值。

门槛只能由外部真值来满足：MLIR verifier、ODS、JIT、上游测试集。模型自生成的检验只作补充，不计入门槛。

## G1 覆盖：MLIR 机制都能导入或降级，且不越权

| # | 度量 | 门槛 | 当前（第 28 轮） |
|---|---|---|---|
| G1.1 | coverage.md 中的每个机制：✓（A/B 证据），或明确走“未知 + 警告”路径 | 无空白格、无 ◐ 未注明去向 | 有 ◐（例如 #22 只覆盖 arith 整数子集） |
| G1.2 | 语料（corpus/）函数的结构层导入率 | ≥ 99%，未导入的须逐条归因 | 见 corpus.md |
| G1.3 | 语料函数的值层事实率：有事实的函数 / 候选函数 | ≥ 90%；缺口须归因到具体未建模 op 或属性 | 169/188 ≈ 90%（只计标量签名候选） |
| G1.4 | 值层事实经 JIT 执行对照的函数占比 | ≥ 80%，tensor/memref 签名也要计入 | 标量签名 141 个函数；第 34b 轮 tensor 签名：有事实 41 个，执行 37 个，比较 1030 处，0 不一致（全形式切片已建模，剩余缺口主要是浮点元素，values §9.3；第 34 轮 0 维；修标量条件 select 缺陷，values §9.2。第 33 轮：有事实 35、执行 31、930 处；第 32 轮为有事实 10、执行 7（另 3 个所有输入都不满足守卫，例如 4x8→8x4 的 cast），比较 173 处）。**达标的是比例（37/41，另 4 个的输入全被守卫或 UB 排除），缺口在分母**：616 个单块 tensor 签名函数里只有 41 个有值层事实，见 values.md §9–9.3 |
| G1.5 | 值层未覆盖的类别（func.call、多块 cf、scf.while、浮点）：要么建模，要么判未知 | 每一类都有“判未知”的回归 | 判未知已实现；浮点标量已建模并有 JIT 证据（第 35 轮，values §10：F1 3500/3500、语料 1333/1333）；math 可精确子集已建模（第 35b 轮，values §11：F1 3521/3521+1 归因 LLVM 缺陷、语料有事实 102、1749/1749）；超越函数/copysign/fpowi、func.call/多块 cf/scf.while/浮点 tensor 仍判未知 |

## G2 组合律可靠

| # | 度量 | 门槛 | 当前 |
|---|---|---|---|
| G2.1 | 所有 JIT/verifier 差分中，模型比 MLIR 更“有定义”的反例数 | 0，且报告触发数 | 0（W2、V1、V5@64、S6、A2、A3） |
| G2.2 | 进入拒绝路径或对前端作结论的定律中 C 级的条数 | 0 | 第 29 轮：L13/L14/L19 升 B，剩余 C 9 条；须逐条核对是否对前端作结论 |
| G2.3 | 按名字共享或按键缓存的项：都在 names.md 清单中，并写明名字编码了哪些语义来源 | 清单无“未审” | 第 27/28 轮清完，保留 1 项风险（参数常量命名契约） |
| G2.4 | 已知的不可靠缺陷 | 0 未修 | 0 |
| G2.5 | 求解器 unknown/超时 | 都按“未证明”处理，并报告计数 | 已做；values_test 13 次 |

## G3 对前端有指导性

| # | 度量 | 门槛 | 当前 |
|---|---|---|---|
| G3.1 | frontend-model.md 的每条规范性条款都有对应检验，或注明为元理论 | 100% | 第 36 轮：47 条编号条款（clauses.md，clauses_test 机械核对，自检 5/5）：检验 32、部分 11、元理论 3、缺口 1（M1.8 caps 无外部真值）。按门槛“有检验或元理论”计 46/47；“部分”行的缺口逐条写明（alloca_scope 包裹无回归、L31 真实 IR 归位率、L10 样本小、管线差分只覆盖两条）。注意：核对的是检验存在，不是检验通过 |
| G3.2 | 前端可见的每类诊断（拒绝、警告、未知）都能定位到节点或端口 | 有回归 | modular M1 已覆盖一部分 |
| G3.3 | 编辑期查询（补全、caps、Pre）的增量结果 ≡ 全量结果 | 0 不一致 | E1 720/720，E2 168/168 |

## 用法
- 每轮选“离门槛最远、且对可靠性影响最大”的一项推进，优先 G2。
- 门槛本身也可以修改，但只能放宽到“有外部真值支撑”的程度，并在这里记录理由。
- 下一步（第 31 轮后）：G2.2 的 C 级只剩 L6（保持）与元理论/内部一致性类 → G1.4 第 32 轮已接入 tensor 签名 JIT，第 33/34 轮逐元素 arith、0 维已建模，切片已全形式；剩余主要是浮点（G1.5，第 35 轮标量完成）→ math 可精确子集/浮点 tensor → G3.1（条款-检验对照表）。

## G2.2 剩余 C 级逐条分类（第 30 轮）
判据：这条定律是否让前端对 MLIR 程序的语义作出结论。若是，就必须有外部真值。

| 律 | 是否对前端作结论 | 处理 |
|---|---|---|
| L1、L2 | 否：元理论（全关系、摘要不限制到 Pre），保证的是推导的可靠性 | 需要证明，不靠测试；列入 spec 证明义务，不阻塞 G2 |
| L6 | 是：毒值/UB 依赖 | MLIR 没有可执行的毒值语义。外部证据只有 V3 的间接证据（int-range-optimizations 的翻译确认）。**保持 C，前端不得只凭毒值层结论做变换**。第 53 轮：确定值方向经 poison_test 外部化，升 B；毒值/UB 方向仍不可测，上述限制对该方向保留 |
| L7 | 是：纯性 | 第 30 轮 X7 外部化，升 B |
| L11 | 是：用户 ADT 的降级正确性 | 第 30 轮 A11 外部化（adtjit_test），升 B |
| L12 | 是：接口定律的方向 | 第 31 轮 l12jit_test 外部化：1500 个方向判定，判有效的 76177 个输入 JIT 不符 0，值层反例 884/884 被 JIT 确认；升 B（值层）。只差在毒值/UB 层的 171 个判定（含 option.first 的单向性）JIT 不可观测，这部分依赖 L6（毒值语义），JIT 无法佐证：law_ok 放行的方向在值层有外部证据，在毒值层的正确性仍以 L6 为前提 |
| L20′、L21 | 否：编辑与增量的内部一致性，真值按定义就是全量重推导（E1、E2 已对照） | 不需要外部真值；JIT 只作反例见证 |

- 第 37 轮（M5.3）：inline_all 实现 alloca_scope 包裹 + 逃逸不发射；allocascope_test K0–K5（closure §8）。scf.for 宿主经 canonicalize 后栈资源无保证（MLIR trait/降低不一致，已知上游讨论），M5.3 仍为“部分”。
- 第 38 轮（M5.3 → 检验）：lowering.lower_alloca_scopes（作用域 ⇒ stacksave/restore，在优化之后、方言降低之前）；修复 D1（scf.for trait 未被 scf-to-cf 尊重）与 D2（结构化区域内含控制流的 alloca_scope 在任何标准次序下都降不下去）。allocascope_test K6–K9。
- 第 39 轮（M1.8 缺口 → 检验）：caps 对照 MLIR LICM（113 实例，宽松 0）；修 caps 忽略守卫、义务不随复合 op 调用传播两处缺陷；JIT 证实本提交 MLIR remui/remsi 标 Pure 导致 LICM+O3 误编译（上游 #188263 已修）。G3.1：缺口 0。
- 第 40 轮（L24 落地）：guards.plan（refines(core, GUARD_LOWER) 逐程序证明）+ insert_guards；guards_test G0′/G1/G2：越界输入 344/344，未插守卫 16 程序崩溃。切片守卫与语料全输入回归待做。
- 第 41 轮（M3.8 → 检验）：iN 界的 scf.for 补越界 UB（此前不可靠）、符号界 iN 摘要、unsignedCmp；loopjit_test JIT 211/211，UB 归因 5/6 与 LangRef 不同。guards_test G3 语料：41 候选中仅 3 个需守卫，越界输入下输出全为毒值，0 组可比（证据薄，如实记录）。
- 第 42 轮（M2.8/M5.4 → 检验）：L31 真实 IR 统计（diagloc_test，600 函数、253 个被 MLIR 拒绝的变异体，回映 253/253）；修 Oracle：解析时属性约束诊断按行号回映（只限 op 级诊断）。
- 第 43 轮（M3.4 → 检验）：更正“S3 两条组合律未修”的误读（那是对照规则，早已反驳）；切片守卫；S3 8 条规则经 JIT 外部化（有效 6 条 0 不一致，错误 2 条找到反例）；G5 越界执行安全 52/52（未插守卫 22 崩溃）。
- 第 44 轮（回归可判定 + M1.2 的 L3 部分）：发现 28 个测试文件退出码恒为 0，“通过”靠人读输出；新增 verdict.check（失败 ⇒ 退出码 1）与 clauses_test K7 统计：检验引用 118 条中有逐标记机器判定 19、仅文件级退出码 21、无机器判定 78（检验条款 62、部分条款 16）。capture_test：真实 IR 提取端口 4435 段端口错误 0（旧算法 8，修三处：词法作用域、区域其它块、先用后定义）；structure_fuzz 入回归。M1.2 仍为部分（L5 只有定向；上下文约束 0/14 未预先指出）。
- 第 45 轮（检验引用转机器判定）：11 个测试文件（oracle、modular（含 --m2）、compsig、obligations、symtab、odssig、odsdiff、typesig、effects、transform、values）新增 37 个逐标记 verdict，全部 PASS（依据 VERDICT 行）。K7：检验引用 118 中逐标记判定 19 → 65；无机器判定 78 → 32（检验条款 19、部分条款 13）；仅文件级 21。转换中暴露两处：values S0 自第 35 轮起实际失败（ATTR_MAP 未补登 fastmath，已修）；modular M2 的判据须限定在体级预言机覆盖范围内（漏判 305 全在 func.func 体外）。
- 第 46 轮（检验条款的引用全部有逐标记判定）：attrsig、defs、irdl_boundary、names（N1/N1c/N5/N7 拆成独立判据，另加 N2–N4）、loopsum、loop、floatjit、valimport、adtjit、l12jit、edit 共 11 个文件新增逐标记 verdict，全部 PASS；modular/defs 的标记改为字面量（K7 按字面量匹配，第 45 轮 M2 实为 PASS 但未计入）。K7：逐标记 65 → 97/118；检验条款无判定 19 → 0；部分条款无判定 13 → 0；仅文件级 21（未变）。转换中暴露：valimport A0 把 scf.for 等 4 个 op 的属性硬编码为“拒绝”，与第 41 轮开放 unsignedCmp 矛盾（报表错，实现对）；改为同一分类，并加探针——账上 4 个拒绝属性（truncf.roundingmode、memref alloc/load/store.alignment）全部经 to_core 真抛 Unknown。
- 第 47 轮（文件级判定全部细化 + 一处推导挂起）：tensorjit（T/T2/T3）、sem（V1/V2/V3）、caps（C1/C2/C3/C4）、loopjit（L1/L3）、allocascope（K0/K1/K2/K3/K6/K8/K9）、clauses（M2.7）改为逐标记 verdict，全部 PASS；floatjit 的 MUN 单独计含 math 的已执行函数（81/197）。K7：118/118 逐标记，此后任何无逐标记判定的引用都算 clauses_test 错误。细化暴露一处真缺陷：T 语料每次运行都有 1 个文件（SCF/for-loop-canonicalization.mlir）超时被计为“文件超时/崩溃”，此前因判据只看“不一致 = 0”而被忽略。根因是静态循环精确展开时 tensor 项逐趟嵌进新 Lambda，z3 抽象绑定变量的开销随趟数指数增长（2 个 insert_slice/趟：6 趟 2.8 s，10 趟 > 20 分钟）。修复：KTUNROLL = 8，携带 tensor 的循环展开后 tensor 产生 op 总数超限时改走摘要路径（与 trip > KUNROLL 同一条可靠路径）。修复后 T 语料 4 分钟跑完，超时 0，一致 1030、不一致 0、候选 41（与修复前相同）。
- 第 48 轮（推导指数挂起的根因修复）：第 47 轮的 KTUNROLL 只是治标，直线链式 insert_slice 也会指数增长（12 个 13 s，14 个 > 25 s）。core `_proj`/`_sel` 改为项层 β/投影化简（纯重写），每个 tensor 的元素项只剩一层 Lambda：直线 40 个 3.6 s；关闭 KTUNROLL 时 64 趟循环 7.1 s。KTUNROLL 8 → 512（只作安全网）。新增 derivecost_test D1/D2/D3（fork 限时，旧实现 D1/D2 FAIL，可区分新旧）。外部证据数字不变：T 1030/0（超时 0）、T2 2886/0、T3 297/0、valimport A0 PASS、A2 1274/0、A3 1322/0（KTUNROLL=8 版与新版各跑一次，均 PASS，补上第 47 轮未重跑的 A1–A3）。回归 PASS：guards、values（6）、loop、loopsum、caps、obligations、transform、effects、compsig、typesig、capture、clauses（K7 118/118）。未重跑 loopjit/floatjit/sem/allocascope：改动只涉及 tensor 类型投影与取元素，这几项不含 tensor。条款状态不变：检验 38、部分 6。
- 第 49 轮（全局确定性资源预算，L53）：KUNROLL/KTUNROLL 只看单层循环，嵌套静态循环逐层相乘（三层 64 趟，旧实现 > 90 s）。新增 core.DBUDGET=20000：计量为一次 derive 内 walk 的 op 节点数（与特化子推导共享）；展开前用 Ctx.cost 估算（内层静态趟数相乘），超出 ⇒ 摘要；硬上限 ⇒ ValueUnknown（原因可读）。标定：全部测试与语料 8301 次 derive，最大用量 925。derivecost D4/D5 新增并登记到 M3.8（K7 120/120）；旧实现 D4 FAIL。验证（只依据 VERDICT）：derivecost 5/5、valimport 3/3（A2 1274/0、A3 1322/0）、tensorjit T 1030/0、T2 2886/0、T3 297/0、loop、loopsum、loopjit、guards、caps、compsig、obligations、transform、effects、typesig、capture、sem、allocascope、floatjit、clauses 全 PASS。条款状态不变：检验 38、部分 6。
- 第 50 轮（M1.2 上下文约束）：capture_test 的 14 段上下文失败中，没有一段来自 ODS HasParent。其中 7 段是子 op 的 C++ verify 读父 op，7 段是祖先的 C++ 或接口校验读体。新增 model/ctxcons.py，分三级判定：HasParent ⇒ reject（MLIR 原生 trait；650 个 op 中 15 个带此约束，全部可解析；定向 8/8 与 MLIR 一致）；已声明校验（沿调用闭包）在同提交 C++ 源中文本上读父 op 或读体、或找不到源 ⇒ unknown；其余 ⇒ ok。真实 IR（1500 个函数，4435 段）：14/14 预先指出（未知），判 ok 的 3908 段中 MLIR 上下文失败 0，误拒 0；精度代价是 MLIR 接受段中 513/4421 判未知。更正检验缺陷：capture_test 默认 300 个函数里只有 1 个含嵌套，上下文段为 0，默认改为 1500。新增 scripts/fetch_cpp.sh。K7 125/125。M1.2 仍为部分（L5 只有定向 X5）。
- 第 51 轮（M1.2 精度）：按原因拆开第 50 轮 513 个未知段，412 个来自“祖先校验读体”，多数只取入口块、块参数或终结 op，而这些在段换成调用后不变。读体判据收窄为读体内 op，调用闭包不跟别的 op 类的成员函数。调参集未知 513 → 102/4421，14/14 仍预先指出，误拒 0，ok 段失败 0。新增留出集检验 C6（1116 个函数，3345 段，未参与收窄）：P3 172/172 未知，漏 0，误拒 0，P1 未知 0.9%。M1.2 仍为部分（L5）。
- 第 52 轮（L5 外部证据；M1.2 → 检验）：needs_ssacfg 判据由“任何处有后继”更正为“顶层区域多块”（MLIR Verifier：Graph 区域 0 或 1 块）。真实 IR 2599 个函数内联进 builtin.module，一致 2599/2599；旧判据语料误报 3。同时修复结构层误拒：ODS op 的 Graph 区域与未注册 op 的区域原按 SSACFG 查 R5 支配，先用后定义被误判；现按 ODS 取区域种类。新增 l5_test（L5a/L5b/L5c）。回归 PASS：core、oracle、symtab、structure_fuzz、obligations、modular、incr。条款：检验 39、部分 5。
- 第 53 轮（L6 外部化；M1.6 → 检验）：新增 poison_test。调用方在 MLIR 里制造毒值（shli x, w），经 func.call 传入随机复合 op；模型只经摘要推导。600 个程序、4800 个输入（O0+O3）：确定值 614/614 与 JIT 一致（毒值实参可达结果 100）；摘要去掉输入毒值依赖后 O3 反例 328。L6 由 C 升 B。clauses_test K6 自检改用 L21 作 C 级注入。条款：检验 40、部分 4。
- 第 54 轮（L6 毒值来源扩面）：poison_test P3，五种依赖数据的毒值来源，阈值预先写定。首跑 seed 54（每种来源 120 个程序）中 shrui_exact 可达样本 6 < 10，判 FAIL；新样本 seed 55（每种来源 300 个程序）PASS：确定值 3748/3748，确为毒值且可达 224，消融 O3 反例每种来源都 > 0。样本量是在看到首跑结果后才定的，已在 values §14.1 写明。
- 第 55 轮（M3.9 → 检验）：新增 adtpoison_test，在 ADT 降级分量里注入毒值，阈值预先写定，首跑即通过。确定值 13374/13374 与 JIT（O0+O3）一致，其中 none 载荷毒化的 2314 个；两种消融 O3 反例分别为 8266、650。条款：检验 41，部分 3（M0.2、M1.4、M5.1）。
