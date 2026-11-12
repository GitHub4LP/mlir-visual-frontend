# 条款—实现—检验—证据对照（G3.1，第 36 轮）

每条规范性条款（frontend-model.md 的〔Mx.y〕）一行。model/clauses_test.py 机械核对：
1. 编号与 frontend-model.md 双向一致；
2. “检验”列的每个 `文件:标记` 都存在（文件在 model/，标记是该文件里的字面子串）；
3. “律”列的每条定律都在 laws.md 中，且条款声称 A/B 时，所引定律没有 C/✗（D 级是定义而非经验命题，允许，但计数报告）；
4. laws.md 中每条 A/B/C 级定律至少被一条条款引用（反向完备），否则列入文末“未引用”并说明理由；
6. K7（第 44 轮）：统计“检验”列引用中有机器判定（`verdict.check('标记', …)`，失败 ⇒ 退出码 1）的比例；此前多数测试只打印统计、退出码恒为 0，“通过”靠人读输出。 第 45 轮：逐标记判定 65/118，无机器判定 32（检验条款 19、部分条款 13），仅文件级 21。
5. 状态只能是：检验（有外部真值的回归）、部分（有回归但覆盖不全，缺口写明）、元理论（定义/证明义务，不靠测试）、缺口（无回归）。

“外部数据”列的数字是最近一次运行的结果，会过时；以测试输出为准。

| ID | 条款 | 实现 | 检验 | 律 | 级 | 状态 | 外部数据 / 缺口 |
|---|---|---|---|---|---|---|---|
| M0.1 | 前端合法 ⇒ verifier 接受；有拒绝权的层不误拒 | structure、odssig、attrsig、symtab、oracle | odssig_test:L1 差分; obligations_test:Z2 变异差分; symtab_test:Y1 误拒; modular_test:M3 单元级预言机; attrsig_test:A1 真实属性 | L29, L30, L34a/b, L36a/b, L39, L42 | A | 检验 | M3 19641 误拒 0；Z2 8858；Y1 2108 |
| M0.2 | 导入语义不变，否则未知 | irimport、valimport | valimport_test:A2 一致; tensorjit_test:T 语料; floatjit_test:F2 语料; ftensorjit_test:FT 语料; ftensorjit_test:FT 灵敏度; ftensorjit_test:FT 求解输入; ftensorjit_test:FT2 浮点逐元素; ftensorjit_test:FT2b 转换逐元素 | L53 | A | 部分 | 1274/1274（第 46 轮复跑）、1030/1030、F2 1781/1781（第 46 轮）；分母缺口：tensor 签名 41/616、浮点 102/204 有事实；FT（第 60–61 轮）浮点 tensor 签名有事实 120（开放浮点逐元素后 +26），执行 100，比较 2235/2235；守卫恒假 7、界外才有解 5、全跳过 12、JIT 不可用 8；FT2 定向 181 程序 2846/2846 |
| M0.3 | 前端证明的改写 ⇒ MLIR 精化 | transform、core | transform_test:R5 L13/L14 JIT; sem_test:V2 MLIR canonicalize; values_test:V6 canonicalize | L13, L14, L15 | B | 检验 | R5 21971/0 不符 |
| M0.4 | 强于 MLIR 只在无对应物处，标 D | — | — | L23, L24, L26 | D | 元理论 | 设计选择；L24 的守卫义务另有 B（shape S6） |
| M0.5 | 外部真值缺陷：按规范建模，A/B 隔断才可归因 | floatjit.barrier、values_test V8/V9 | floatjit_test:barrier(; values_test:V9 index 方言 cmp; values_test:V8 IW=32 | L50, L52 | A | 检验 | 归因 4 类；每类有隔断对照 |
| M1.1 | Op=⟨Schema,Impl⟩；嵌套 ≡ 内联 | core | core_test:T1 随机复合 op; compsig_test:C2 嵌套 ≡ 展开 | L25 | A | 检验 | |
| M1.2 | 结构字段：原生取 ODS，复合推导 captures/free_syms/needs_ssacfg | structure、symtab、core | oracle_test:O5 区域/后继/多块; symtab_test:Y0 导入账; capture_test:P 端口 vs MLIR; capture_test:C1 上下文拒绝 0 误拒; capture_test:C2 上下文约束预先指出; capture_test:C3 判 ok 的段 MLIR 接受; capture_test:C6 留出集上下文判定; ctxcons_test:C4 HasParent 拒绝权; ctxcons_test:C5 HasParent 导入账; structure_fuzz:structure_fuzz 不可靠; l5_test:L5a Graph 区域 op 的审计; l5_test:L5b 内联进 Graph 区域 vs MLIR; l5_test:L5b 消融灵敏度; l5_test:L5c 定向; graphorder_test:G1 Graph 区域副作用定向; graphorder_test:G2 消融灵敏度; graphorder_test:G3 副作用分类 vs MLIR 死代码删除 | L3, L4, L5, L56 | B | 检验 | L3：真实 IR 提取端口 4435 段 MLIR 接受 4421、端口错误 0（旧算法 8，消融）；structure_fuzz 1000 例不可靠 0。L5（第 52 轮）：needs_ssacfg 改为“顶层区域多块”（MLIR Verifier：已注册 op 的 Graph 区域 0 或 1 块）；真实 IR 2599 函数内联进 builtin.module 判据一致 2599/2599（顶层多块仅 5，语料所限，定向补充），旧判据语料误报 3。上下文约束：14/14 预先指出（未知级；HasParent 有拒绝权，定向 8/8），留出集 172/172；精度代价 MLIR 接受段判未知 2.3%（留出集 0.9%）。Graph 区域副作用次序（第 57 轮，L56）：可能有副作用的 op 落在 Graph 区域 ⇒ 未知并警告，不拒绝；副作用分类与 MLIR 死代码删除对照：207 个 op 名、2415 个实例，判 none 而 MLIR 保留 0 |
| M1.3 | 类型签名：自动签名 + 规范化 C++ verify 规则；复合显式推导 Φ_comp 与单端口三值范围 | odssig、nfrules、cppverify、attrsig、composite、odscomp | odssig_test:L1 差分; odsdiff_test:T1 odssig vs; compsig_test:C1 Φ ≡ 体 ≡ MLIR; compsig_test:C3 端口投影; odscomp_test:S3 对MLIR; odscomp_test:S4 嵌套≡展开; projcomp_test:P4 in可靠; projcomp_test:P5 out可靠; nf_test:N1 nf≡MLIR; nfiface_test:N2a 接口规则无误拒; nfiface_test:N2b 接口诊断必拒; nfiface_test:N2d 清空 op 完整模型≡MLIR; packconst_test:N2e pack/unpack 常量 tile 完整模型≡MLIR; winograd_test:N2f softmax/winograd 完整模型≡MLIR ; regcomp_test:R1 Φ≡逐点（含区域）; regcomp_test:R2 对MLIR（含区域）; scf_test:N2g-a scf 规则无误拒; scf_test:N2g-b scf 诊断必拒; scf_test:N2g-d 无残差实例 完整模型≡MLIR; regflow_test:F1 区域类型流单调; regflow_test:F2 区域增益为真; regflow_test:F4 一致替换元素类型的增益; cppverify_test:C1 解释器无误拒; cppverify_test:C2 完备 op 无漏判; cppverify_test:C3 完备 op 无未定; cppverify_test:C5 残差为空 完整模型≡MLIR; cppverify_test:I1 infer 无误拒; cppverify_test:I2 infer 诊断必拒; cppverify_test:I3 infer 无未定; cppinfer_test:J1 C++ 推断前推单调; cppinfer_test:J2 C++ 推断增益为真; cppinfer_test:J3 C++ 推断前推无误拒| L25, L27, L28, L29, L46, L47 | A | 检验 | |
| M1.4 | 属性：ODS 值域；复合参数约束 = 使用点合取 | attrsig、defs | attrsig_test:A2 变异差分; defs_test:D1 attribute; defs_test:D3; l10_test:E1 L10 使用点合取（MLIR 自证）; l10_test:E2 我们的合取不误拒 | L8, L9, L10, L34a/b, L35 | B | 检验 | 第 59 轮：L10 真实 IR 非符号 895 组 6972 例违例 0；符号属性跨使用点（373 例违例）⇒ 不得作参数，判 Unknown；E2 误拒 0/3958，精度：MLIR 拒绝中先抓 75%（二跑加 rel_check；首跑 46% 为低估） |
| M1.5 | declared 义务只由预言机裁决 | oracle、modular | modular_test:M2 体级预言机; modular_test:M3 单元级预言机 | L40, L43, L44 | A | 检验 | 残差 580 例全部归因 |
| M1.6 | 值层 Sem/UB/Poison；复合摘要推导 | core、valimport | sem_test:V1 无标志程序; values_test:V5 JIT; loop_test:W2 与 MLIR 执行; adtjit_test:A11 L11; poison_test:P1 毒值实参的确定值 vs JIT; poison_test:P2 消融灵敏度; poison_test:P3 依赖数据的毒值来源 | L6, L17″, L49 | B | 检验 | 原生子集 B/A；复合摘要的毒值依赖（L6）第 53 轮外部化：毒值实参下确定值 614/614 与 JIT 一致（可达结果 100），消融 O3 反例 328；第 54 轮扩到五种依赖数据的毒值来源，确定值 3748/3748（确为毒值且可达 224），首跑 shrui_exact 样本不足 FAIL，新样本 PASS。毒值/UB 方向不可测（精化自由度）；浮点、tensor 的毒值未外部化 |
| M1.7 | 效应 compose_eff | core | effects_test:X7 L7 效应组合 | L7 | B | 检验 | 48 例，不可靠 0 |
| M1.8 | caps 由字段派生（对降低后的 MLIR，含守卫与 L22 义务） | core.caps、gspec、义务传播 | caps_test:C1：原生 op; caps_test:C3：义务经复合 op 调用传播; caps_test:C4：MLIR 判 Pure; defs_test:caps= | — | B | 检验 | 第 39 轮：113 原生实例 一致 98/保守 15/宽松 0；保守中 remui/remsi 经 JIT A/B 证实 MLIR 错（上游已修）；C3 变异可抓 |
| M1.9 | 复合与原生同构；声明只收窄且检查 | core、composite | compsig_test:C6 定义期可实例化; core_test:T2 跨层组合律 | L25, L46 | A | 检验 | |
| M1.10 | 名字单射编码全部语义来源 | core、names | names_test:N1 编辑洞; names_test:N5 跨语义开关; names_test:N7 ADT 重定义; loopsum_test:U3 语义开关进名字 | L18′, L54 | B | 检验 | |
| M2.1 | 结构层有拒绝权 | structure | obligations_test:Z2 变异差分; obligations_test:Z1 触发数 | L39 | A | 检验 | |
| M2.2 | 自动签名有拒绝权 | odssig | odssig_test:L1 按账目 | L29 | A | 检验 | |
| M2.3 | 属性层有拒绝权 | attrsig | attrsig_test:A1 真实属性 | L34a/b | A | 检验 | A1 14984 实例误拒 0；A2 变异 25395 误拒 0、ODS 属性层漏判 465（第 46 轮） |
| M2.4 | 符号层有拒绝权 | symtab | symtab_test:Y1 误拒 | L36a/b | A | 检验 | |
| M2.5 | 预言机是最终权威 | oracle | modular_test:M3 单元级预言机; oracle_test:O1/O2 判定与分工 | L42 | A | 检验 | |
| M2.6 | typesig 无拒绝权，只警告 | typesig | typesig_test:T4 接入 core | L33 | D | 检验 | 元规则，由 A 级数据支撑 |
| M2.7 | 值层/效应/caps 无拒绝权；必然 UB 的程序合法 | core、structure | clauses_test:M2.7 | — | A | 检验 | 本轮新增回归：MLIR 接受 + 结构层无诊断 + 推导不抛异常 + UB 恒真 |
| M2.8 | 判定顺序；诊断归位 | oracle | oracle_test:O1/O2 判定与分工; oracle_test:O3 诊断归位; diagloc_test:R1; diagloc_test:R2; diagloc_test:R3 | L30, L31 | B | 检验 | 第 42 轮真实 IR：回映 253/253，落在编辑处/直接使用者 253/253；我们的层先抓 201，节点同 MLIR 200 |
| M2.9 | 无结论是一等结果 | oracle、modular | oracle_test:O4 崩溃隔离; modular_test:M1 | L41 | D | 检验 | |
| M3.1 | iN 语义 | core | sem_test:V1 无标志程序; sem_test:V3 定向 | L15 | B | 检验 | |
| M3.2 | index IW 回绕；IW≠64 先降 iIW | core.set_iw | values_test:V5 JIT; values_test:V8 IW=32 | L17″, L49, L50 | A | 检验 | |
| M3.3 | 未读参数/未翻译属性 ⇒ 未知 | core、valimport | values_test:S0 语义属性账; valimport_test:A0 属性账 | L51, L53 | A | 检验 | |
| M3.4 | tensor：越界毒值、守卫；逐元素 | core | shape_test:S6 与 MLIR 执行; tensorjit_test:T2 逐元素定向; tensorjit_test:T3 全形式切片; valimport_test:A3 切片定向; guards_test:G1 生成器; guards_test:G0′ 定向; guards_test:G4 S3 改写规则外部化; guards_test:G5 全形式切片 | L22, L24 | D | 检验 | 越界=毒值是 D；守卫落地（第 40 轮越界 344/344；第 43 轮切片守卫，越界执行 52/52 安全、未插守卫 22 崩溃）；S3 规则外部化：有效 6 条 0 不一致，错误 2 条找到反例；降秩 insert_slice 守卫未实现（判未知） |
| M3.5 | 浮点语义 | core 浮点节 | floatjit_test:F1 浮点定向; floatjit_test:F3 truncf roundingmode | — | B | 检验 | F1 默认 n=200：1737/1738 一致 + 1 例 A/B 归因（第 46 轮；旧记 3521 来自更大 n）；F3 五种舍入模式 JIT 结果相同 |
| M3.6 | math 精确子集 | core math 节 | floatjit_test:MUN = | — | B | 检验 | 混入 F1（第 47 轮：verdict 要求含 math 的已执行函数 > 0，默认 n=200 为 81/197，且 F1 不一致 0；math 例仍与 F1 共用逐输出比较，不单独分 op 计）；isinf/isnormal 无降低 |
| M3.7 | 精化判据；MLIR 变换作真值须有 JIT | core.refines | sem_test:V2 MLIR canonicalize; values_test:V7 scf + sccp; values_test:V9 index 方言 cmp | L52 | A | 检验 | |
| M3.8 | 循环摘要、不变式、越界步进 UB | core 循环节 | loopsum_test:U0 复合 op 参数; loop_test:W6 L19 JIT; loop_test:W6 归纳变量越界; loopjit_test:L1 生成器; loopjit_test:L3 trip_count; derivecost_test:D4 全局预算回退摘要; derivecost_test:D5 预算耗尽判未知 | L13, L18′, L19, L53 | B | 检验 | 第 41 轮：iN 界补越界 UB、unsignedCmp 开放；JIT 211/211（8 种类型×比较方式），trip_count 45/45 有事实 |
| M3.9 | ADT 降级与接口定律 | defs、core | adtjit_test:A11 L11; l12jit_test:L12 接口定律方向; adtpoison_test:Q1 ADT 毒值层 vs JIT; adtpoison_test:Q2 消融 a 输入毒值; adtpoison_test:Q3 消融 b tag 毒值 | L11, L12 | B | 检验 | 第 55 轮 ADT 毒值层外部化：降级分量注入毒值，确定值 13374/13374 与 JIT（O0+O3）一致，含 none 载荷毒化 2314；两种消融 O3 反例 8266、650。L12 方向未在毒值输入下单独检验；只有 option 一种 ADT |
| M4.1 | 放置节点：connectable 补全 | odssig | odssig_test:L1 差分 | L29, L37 | A | 检验 | 精确性由 L36b/L29 推出 |
| M4.2 | 连边/填洞 = 项代换（摘要体内重推导） | edit、core | edit_test:E2 L20 连接洞; names_test:N1c 洞在循环摘要内 | L20′ | C | 元理论 | 内部一致性：真值按定义是全量重推导 |
| M4.3 | 改参数：属性层 | attrsig | attrsig_test:A2 变异差分 | L34a/b | A | 检验 | |
| M4.4 | 改块/控制边：结构层 | structure | obligations_test:Z2 变异差分; oracle_test:O5 区域/后继/多块 | L39 | A | 检验 | |
| M4.5 | 符号编辑：失效集与补全 | symtab、incr | symtab_test:Y2 变异差分; incr_test:I 增量重验 | L36a/b, L37, L45 | A | 检验 | |
| M4.6 | 单元内编辑：dirty ∪ dependent | modular、incr | incr_test:I 增量重验 | L45 | A | 检验 | 12958 次编辑 |
| M4.7 | 提取为复合：捕获提升、签名推导 | core、composite | compsig_test:C1 Φ ≡ 体 ≡ MLIR; compsig_test:C4 with 与字面量 | L3, L25 | B | 检验 | |
| M4.8 | 应用改写：定义期证明 + 前同余 | transform | transform_test:R5 L13/L14 JIT; transform_test:R2 局部 → 整体 | L13, L14 | B | 检验 | |
| M4.9 | 预览常量：fold 由 Sem 派生 | transform | transform_test:R3 派生的 fold; values_test:V6 canonicalize | L15 | B | 检验 | |
| M4.10 | 增量 ≡ 全量 | edit | edit_test:E1 增量 ≡ 全量 | L20′, L21 | C | 元理论 | E1 720/720、E2 168/168 |
| M5.1 | 降级含管线，Sem(g) ⊑ Sem(pipeline(⟦g⟧)) | pipezoo_test | shape_test:S6 与 MLIR 执行; loop_test:W2 与 MLIR 执行; allocascope_test:K6 第 38 轮; pipezoo_test:Z0 基线; pipezoo_test:Z1 管线差分; pipezoo_test:Z2 peeling lb>ub 定向; pipezoo_test:Z3 remove-dead-values 定向; floatzoo_test:FZ0 浮点基线; floatzoo_test:FZ1 浮点管线差分; floatzoo_test:FZ2 偏离定向（事后） | L22, L52, L55 | B | 检验 | 第 56 轮按白名单判定（L55）：13 条管线 0 偏离，加上 bufferize、scf→cf、作用域降低三条；peeling 与 remove-dead-values 两条为 MLIR 缺陷，排除。第 58 轮扩到浮点族（values §15.1）：11 条 0 偏离；arith-expand（浮点 max/min 不保号）和 math-expand-ops（ceil、fma）对浮点排除；math-uplift-to-fma 为空检验。tensor 管线仍只有 bufferize 一条 |
| M5.2 | 三种落地；IRDL 是不透明身份 | irdl_export、composite | irdl_boundary_test:any_of | L48 | A | 检验 | |
| M5.3 | 内联保持作用域：alloca_scope、逃逸、自由符号；降级落成 stacksave/restore | core.inline_all、needs_alloca_scope、lowering.lower_alloca_scopes | allocascope_test:K2 A/B; allocascope_test:K6 第 38 轮; allocascope_test:K8 生成器; allocascope_test:K9 D2; allocascope_test:K3 逃逸; allocascope_test:K0 ODS 对账; core_test:自由符号 | L4 | B | 检验 | JIT：3 宿主包/不包 A/B；包+canonicalize+降低 3/3 大迭代正确；K8 60 程序 120/120 值一致、大迭代 60/60 正确（原样崩溃 45）；语料 296 段降低后 verify 全过（仅 1 个循环被改，覆盖薄）；逃逸拒发 |
| M5.4 | 诊断经 Location 回映 | oracle | oracle_test:O3 诊断归位; diagloc_test:R1; diagloc_test:R2; diagloc_test:R3 | L31 | B | 检验 | 同 M2.8（第 42 轮） |

## 未引用的 A/B/C 定律
| 律 | 理由 |
|---|---|
| L1、L2 | 元理论（Sem 全关系、摘要不限制到 Pre），是推导可靠性的证明义务（obligations 见 laws.md），不对应单条前端条款 |
| L16 | 终止度量；合流未证明，MLIR 也不保证；前端不据此作结论 |
