# 可视化编程前端模型 v2

## 0 定位与判据
- 前端 IR（VIR）是一门独立的一等语言，不是 MLIR 的外壳。前后端的表示和交互可以不一一对应。
- “不弱于 MLIR”的定义：(a) MLIR 的每个机制在 VIR 里都有对应；(b) 这些对应在编辑期可判定、可增量计算。
- “比 MLIR 强”的定义：MLIR 要等构造完才验证的东西，前端在编辑时就能预判和引导。
- 编译路径：VIR → vis dialect → 标准 dialect → LLVM → native 或 wasm32。另留一条 VIR → WASM 的直出路径，用于预览。

## 1 机制对应
| 前端 | MLIR 对应机制 | 前端需要强化的地方 |
|---|---|---|
| 节点库/包 | Dialect | 版本、迁移、依赖 |
| 节点 schema | OpDef/ODS | 作为唯一事实源，生成 UI、校验、推断、lowering |
| 端口 | operand/result | 多重性 1/?/*/分组、默认值、内联常量 |
| 端口类型 | Type + TypeConstraint | 约束限定在可判定片段，禁止 opaque 谓词 |
| 参数 | Properties/Attr | 表单 schema、值域 |
| 能力 | Trait/Interface | 能力代数：蕴含、前提、互斥；支持外部模型 |
| 容器/子图 | Region、IsolatedFromAbove | 闭包捕获规则可视化 |
| 命名定义/引用 | Symbol/SymbolTable | 重命名重构、跨图引用 |
| 执行顺序 | MemoryEffects/Pure | 区分 exec 边和数据边 |
| 控制 | Branch/RegionBranch | 优先用结构化控制 |
| 校验 | 分阶段 Verifier | 增量校验、允许带洞的部分图、错误定位到节点/边 |
| 推断 | InferType、TypesMatchWith | 双向：连线时传播类型，并反向筛选候选节点 |
| 预览 | fold/canonicalize | 实时显示常量结果 |
| 重构 | Pass/Pattern/Transform | 每个编辑操作都是带契约的变换，支持 undo |
| 复用 | func/call/inliner | 复合节点、泛型 |
| lowering | DialectConversion | 语义保持义务；用 Location 存节点 ID，后端诊断回映到图上 |

## 2 核心演算（相当于 MLIR 规范化后的超集）
- 节点 schema：`N : ∀X̄. ports × params × slots → ports where Φ, caps C`
- Φ 只能用可判定的约束片段：类型族成员、投影相等、函数依赖、Presburger 关系、参数值域。于是“两个端口能不能连”就是约束可满足性问题，用增量合一求解。
- 两级良构：W_partial 表示允许有洞但没有冲突；W_complete 表示可以编译。MLIR 只有后一级。
- 每个编辑操作都是变换，都要保持 W_partial。
- 节点库加载时检查能力代数的闭包。

## 3 前后端映射义务
- 语义映射 ⟦·⟧: VIR → MLIR，要求 W_complete(g) ⇒ W(⟦g⟧)。也就是前端判定合法的图，后端一定不报错。这是“不弱于”的可检验形式。
- **每个 op 有三个谓词，不能混在一起**（v1 原型中发现）：
  - Wf：后端的静态接受条件，也就是 MLIR verifier 的条件，**包括它的怪癖**。它定义在**静态投影**上：哪些检查会被触发，取决于维度是不是静态。
  - Def：运行期定义性，即不触发 UB。它定义在 **extent** 上。
  - Sem：结果关系。
- 复合 op 推导时三者分别投影（量词规则见 §5 第 3 条）：
  - Wf 必须在 lowering 之前对静态投影判定为真；
  - Def 推出来的是边界前提。能静态证明的直接消去，不能的 lowering 为运行期断言，可接 MLIR runtime-verification。
- 对 MLIR 的偏差统一由 lowering 消解，例如把 size=0 的 slice 规范为 tensor.empty，这样语义层就不用继承后端的怪癖。
- **⟦·⟧ 必须包含 pass 管线，而不只是 IR 翻译**（v3 用真实 MLIR 验证过，见 model/difftest/）。W_complete(g) ⇒ W(⟦g⟧) 只能保证后端不报错，不能保证语义保持。
  - 实例：tensor.extract 标为 Pure，于是 LICM 把它提到可能 0 次迭代的循环外面；bufferize 之后变成无条件执行的 memref.load。原程序从不访问内存，变换后的程序可能越界。
  - 所以前端需要固定两样东西：一份“MLIR 语义剖面”（每个 op 是 UB 还是毒值，由覆盖层规定），以及一份 pass 白名单和顺序约束。例如：bufferize 之前不跑会做推测的 pass，或者对未证明在界内的 extract 先做索引钳制。
  - 判据升级为：Sem(g) ⊑ Sem(pipeline(⟦g⟧))。
- 后端诊断通过 Location 回映到节点/边。
- 数据流图转 SSACFG：按数据依赖加效应做拓扑调度；图中的环必须经过显式的循环或状态容器。

## 4 迭代
- R1 端口类型、约束与推断（做深）
- R2 容器、作用域、效应、控制
- R3 节点库、能力代数、版本
- R4 ⟦·⟧ 与 WASM 路径
- R5 原型：TS 校验器 + MLIR lowering

## 5 复合 op：由实现图推导签名
**统一抽象**：Op = ⟨Schema, Impl⟩，其中 Schema = ∀X̄. ports × params × slots → ports where Φ, caps C；Impl 是 Primitive(mlir 名)、Graph(G)、External 三者之一。
所有消费者（编辑器推断、校验、trait 查询、lowering）只读 Schema，不关心 Impl 是哪一种。内置 op 的 Schema 由 ODS 导入后规范化得到；复合 op 的 Schema 从实现图推导。两者是同一种对象。

**推导（主签名）**：
1. 对图中每个实例 nᵢ，以新鲜变量实例化它的 Schema，得到约束 Φᵢ。
2. 每条边 p→q 生成约束 type(p)=type(q)；如果允许隐式转换，改为 ⊑。
3. 量词按变量的种类区分（v2 修正：原先统一用 ∃，对运行期量是**不可靠**的）：
   - 类型和静态变量（编译器或用户选定）用 ∃：Φ_comp = ∃Y_静态. ∧Φᵢ ∧ ∧边约束。
   - 运行期内部量（extent、index、条件）求最弱前提：Pre(b) = ¬∃r. Sem(b,r) ∧ ¬Def(b,r)；Post = ∃r. Sem ∧ Def。
   - 反例：if(数据相关条件) 两支对 n 的要求不同时，∃ 得到的是两支要求的析取，但正确的前提是合取。
4. Φ_comp 就是使实现体良构的最弱条件：它可靠，也最大。
**前提**：约束语言必须对 ∧ 和 ∃ 封闭，并且蕴含可判定。可选的组合是：项合一 + Presburger（维度、秩、计数）+ 有限类型族 + 边界上的接口原子。opaque 谓词无法被消去，这时推导结果就退化为“可靠但不是最大”。因此禁止 opaque 是硬性要求。

**签名间关系是查询，不是声明**：trait T 对复合 op 成立 ⇔ Φ_comp ⊨ φ_T。例如 SameOperandsAndResultType、AllShapesMatch、TypesMatchWith 都可以这样判定。
**Schema v3（统一记录，原生 op 与复合 op 同构）**：
`Schema = ⟨ports, params, slots; Φ_type（∃，合一）; Sem; UB; Poison; Sensitive; Eff; Wf 引理; caps⟩`
- Sem：运行期量之间的关系（Presburger）。UB：立即 UB 的条件。Poison：每个输出产生毒值的条件。Sensitive：哪些输入遇到毒值即 UB。
  - 依据：MLIR 自己就区分这两种。arith 的 nuw/nsw 溢出产生毒值，divui 除以 0 是 UB（ArithOps.cpp getDivUISpeculatability）。
- 复合 op：Pre = ¬∃r. Sem ∧ UB*，其中 UB* 包含“毒值流入敏感输入”；MayPoison_o = ∃r. Sem ∧ Pre ∧ P_o；Eff 按下面的规则组合。
- **caps 由上述字段派生，不单独声明**：
  - Speculatable ⇔ Pre ≡ 合法输入域。前端还可以给出“在 Pre 下可推测”这个精确谓词，MLIR 只有一个枚举值。
  - Pure ⇔ Speculatable ∧ Eff=∅。
  - nuw 一类的标志 ⇔ 对应的 MayPoison 恰好等于溢出条件。
- 修正 v2 之前的过强断言：
  - “体内全部 Pure ⇒ 复合 op Pure”只是充分条件，不是充要条件；
  - IsolatedFromAbove 只在没有闭包捕获时成立；
  - Commutative 一般不可判定，只能声明，再用规范化同构去检查（D）。

**组合律（core.py 的闭包测试暴露出来的两条）**：
- L1：Sem 必须是全关系。在 UB 处取任意输出，即 Sem' = Sem ∨ UB。否则 UB 输入找不到对应的输出，会从“bad”集合中漏掉，结果不可靠。
- L2：复合 op 的 Sem 摘要不能限制到 Pre 上，理由同 L1。
- 满足这两条之后，“摘要组合 ≡ 全部内联”在 2 层和 3 层嵌套下都由 isl 判定为相等。这正是“推导出的实体享有与原生实体同等地位”的可检验形式。

**约束语言改为双类（v3.1，取代“值也用 Presburger”的做法）**：
- 值层用 QF_BV（位向量）。op 是确定性的，Sem 是函数，所以 wp 只需要项代换：Pre = ¬UB(f(x)) 是无量词公式，精确且不需要投影。只有在提问时才求解：是否恒真（可推测）、是否可满足、取一个反例。
- 下标、extent、循环计数、类型和静态变量用 Presburger（ℤ）。这一层负责 ∃ 消去、循环闭包，以及编辑器里可读的前提显示（例如“需要 0<n≤x”）。
- 注意：MLIR 的 index 是目标位宽的环绕整数。把它当作 ℤ 处理，只有在不溢出时才精确，所以要附带一条义务：extent 和下标运算不超过 2^(w−1)。
- 非线性运算（mul、div）在 Presburger 层抽象成过近似，结果仍然可靠（S）；需要精确时交给 QF_BV。
- 语义覆盖层不必从零手写：xdsl-smt（Fehr 等，PLDI 2025）已经给出 arith、comb、func 等方言的毒值、UB、内存语义，表示方式与本规范同构（值加毒值标志，UB 效应）。上游的 arith-to-smt 不建模毒值，不能直接使用。**修正（v3.2）**：审计后发现 xdsl-smt 有 3 类与规范不符的语义，所以只能借鉴方法，逐 op 导入前必须先过条款测试，见 spec/semantic-overlay.md。覆盖层的原则是：规范与 lowering 不一致时，UB 和毒值都取保守并集（例如 remsi 的 INT_MIN % −1）。

**信任等级与一致性审计**：
- 每个字段标注来源：ods（导入）、authored（人工语义覆盖层）、derived（推导）。
- ODS 只提供类型约束和 caps；Sem、UB、Poison 不在 ODS 里，必须由覆盖层提供。这是建设前端的主要成本。
- 没有覆盖层的 op，其 UB 视为未知：派生的 caps 只能取 ODS 的声明（S），并在 UI 上标出来。
- 审计规则：ODS 声明了 Pure，覆盖层却写了 UB ⇒ 两者必有一错。tensor.extract 就被这条规则抓了出来；tensor.dim 则相反，语义上恒可推测，ODS 的声明可以加强。

**效应组合（model/effects.py）**：
- 并集，目标重定位到边界端口；
- 体内新分配的缓冲，在返回之前外部不可能持有它的别名，所以对它的 Read/Write 外部不可见；
- 局部 alloc…free 抵消；逃逸时 Allocate 重定位到输出端口；未释放又不逃逸的报告为泄漏；
- 体内有任何未建模效应的 op ⇒ ⊤；
- 这条“新缓冲无别名”规则依赖逃逸分析：对 builtin memref 成立（memref 不能存放 memref），对 llvm.ptr 需要真正的逃逸分析，只能做到 S；
- alloca：透明内联会改变语义（栈分配跑到外层作用域，放在循环里会让栈不断增长）。规定：含 alloca 的复合 op 在透明模式下必须包一层 memref.alloca_scope，这样两种模式语义一致；alloca 的结果逃逸则违反 Def（悬垂引用），两种模式下都拒绝。

**边界情形**：
- slot（高阶 region 参数）的签名，由实现体对它的使用方式推出
- 可变端口需要实现体中有对该组的 map/fold 结构，对应约束 ∀i<n
- 递归复合 op：必须显式声明签名，再用实现体去检查（多态递归的推断不可判定）
- 用户可以收窄签名：要求 Φ_user ⊨ Φ_comp；放宽则禁止
- 实现体修改导致 Φ_comp 变化时，检查已有使用点是否仍满足，以此做版本兼容
- 内置 op 带有 C++ verify() 的，需要人工整理成规范化 Schema，否则推出的签名会偏宽

**落到 MLIR**：
(a) 实例化时内联，单态化 —— 最简单
(b) 每个单态实例生成一个 func.func，使用处 call
(c) 把 Φ_comp 导出为 IRDL 动态 dialect 中的 op，从而获得真正的 MLIR op 身份（有 verifier、能参与 pattern），lowering 时再内联。IRDL 表达不了的约束（比如 Presburger 关系）放进动态 verifier 回调。
