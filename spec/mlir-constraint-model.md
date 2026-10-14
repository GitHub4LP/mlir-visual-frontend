# MLIR 设计与机制交互模型 v1

## 0 元模型
MLIR = ⟨Σ, G, T, W⟩
- Σ 声明库：dialect、op、type、attr、trait、interface、hook。都是**事实**，大多由 ODS 生成。
- G IR 状态：op、region、block、value 构成的图，加上 use-def 链。
- T 变换：G → G 的偏函数，包括 pass、pattern、conversion、transform 脚本。
- W 良构谓词，即 verifier。
核心组织原则：**一次声明、多处消费**。子系统之间不直接相互依赖，而是通过 Σ 中的事实耦合（事实总线）。每个子系统有两部分：它读取哪些事实，以及它对 G 的前提与保证。

## 1 静态层 Σ
- **Context**：负责 Type/Attr/Location 的唯一化（可在多线程下进行），以及 dialect 注册和加载（不是线程安全的）、诊断。推论：类型相等就是指针相等；IR 不可变，所以改类型意味着产生新的 value。
- **Dialect**：一个命名空间，加上 op/type/attr 的集合，再加上若干钩子：materializeConstant、verifyOperationAttribute、规范化 pattern、dialect interface（Inliner、OpAsm、ConvertToLLVM 等）。DialectExtension 在 dialect 加载时注入额外功能。
- **Op 定义**：签名、traits、interfaces、fold、canonicalize、verify、汇编格式，以及 Properties（固有状态）和 discardable attrs（可丢弃属性，开放世界）。
- **Trait vs Interface**：trait 是静态事实，带 verifier 和 foldTrait；interface 是动态契约，可以有外部模型、promised 声明、接口继承。
- **ODS/TableGen 是唯一事实源**：同一份声明生成 C++ 类、builder、verifier、parser/printer、类型推断、文档和 Python 绑定。
- **未注册 op**：没有任何事实，所有消费者都必须对它做最保守的假设（效应未知，不能 fold，不能移动）。

## 2 动态层 T
| 子系统 | 机制 | 读取的事实 |
|---|---|---|
| Parser/Printer/Bytecode | 自定义 / 通用形式，别名，版本化 | 汇编格式、OpAsm 接口、Properties 编码、类型推断 |
| Verifier | 见 §4 | 全部约束性事实 |
| PassManager | 以 op 名作锚点的嵌套管线；在 IsolatedFromAbove 边界处并行；pass 之间验证 | IsolatedFromAbove、pass 的 dependentDialects |
| AnalysisManager | 缓存分析结果，依据 preserved 集合失效 | pass 声明保留了哪些分析 |
| Folding | fold → OpFoldResult；用 materializeConstant 生成常量；常量去重，并提升到隔离区域入口 | ConstantLike、Commutative（常量排到右边）、Involution、Idempotent、dialect 钩子 |
| Greedy driver / Canonicalizer | 工作表 + fold + 死代码删除 + pattern 按 benefit 优先；有迭代上限 | 各 dialect/op 的规范化 pattern、效应（判断能否删除） |
| Dialect Conversion | ConversionTarget 合法性、TypeConverter、源/目标物化、回滚、unrealized_conversion_cast | 目标 op 的约束、类型映射 |
| CSE | 判断操作等价（名字、操作数、Properties/attrs、结果类型），受支配性约束 | Pure/效应、区域、IsolatedFromAbove |
| LICM / 控制流下沉 | 外提 / 下沉 | LoopLike、可推测性、效应 |
| Inliner | 调用图 + 合法性回调 | Call/Callable、SymbolTable、DialectInlinerInterface |
| SymbolDCE/符号 | 嵌套作用域、可见性、SymbolRef 解析 | Symbol、SymbolTable、SymbolUser |
| Dataflow/SCCP | 格上的不动点迭代 | Branch、RegionBranch、Call、fold |
| Bufferization | 把张量改写为 memref，分析原地更新 | BufferizableOpInterface、DPS、效应 |
| Transform dialect | 用 IR 写的变换脚本；handle 指向 payload IR | transform 接口、handle 失效规则 |

## 3 事实 × 消费者（交互矩阵，体现机制间的有机耦合）
- **Pure / 效应** → CSE、DCE、LICM、greedy 删除、推测执行外提、bufferization、别名分析。区域里的效应要靠 RecursiveMemoryEffects 才会被算进来。
- **IsolatedFromAbove** → verifier 的作用域、pass 锚点与并行、常量提升的位置、CSE 和值捕获的边界。
- **类型约束 / TypesMatchWith** → verifier、汇编格式省略类型、生成推断式 builder、InferTypeOpInterface、Python 绑定、conversion 合法性。
- **RegionBranch / Branch** → verifier 的控制边类型检查、dataflow、liveness、bufferization、LICM。
- **SymbolTable** → verifier 检查符号使用、inliner 调用图、SymbolDCE、链接。
- **ConstantLike + materializeConstant** → fold、m_Constant 匹配、常量去重、SCCP 结果回写。
- **Properties** → verifier、bytecode 及版本、CSE 的哈希与等价判断、通用形式打印。
- **Location** → 诊断；重写时 Loc 会被融合（FusedLoc）。
- **TypeConverter ↔ 目标 op 约束** → 物化出的 cast 必须由 reconcile pass 消掉，否则就是非法残留。

## 4 约束子模型（v0 的内容，压缩后放在这里）
- 签名范式：`op N : ∀X̄. (Oᵢ: mult × TC) × P × R̄ × S → Res where Φ`。AllTypesMatch 表示共享类型变量；TypesMatchWith 表示函数依赖；变长操作数的切分要唯一（单个可变组 / SameVariadicOperandSize / AttrSizedOperandSegments）。结果类型可推导 ⇔ 依赖闭包能覆盖所有结果，这同时决定 InferType 能否自动生成以及汇编格式能否省略类型。
- 约束语言：用投影项（elt、shape、rank、dim、encoding、memspace、attrval、|seg|）、类型族成员、相等、Presburger 整数关系、Impl 构造可分析片段；CPred 作为 opaque 原子并计数。
- 验证阶段：固有属性 ▷ ODS 不变量 ▷ traits（按声明序）▷ verify() ▷ 子区域 ▷ region traits ▷ verifyRegions()，另加支配性、dialect 属性、符号使用。规范化做法：给原子加定义域守卫，消除对验证顺序的依赖。

## 5 系统不变量
I1 每个 pass 结束后 W(G) 成立；verifier 是 pass 之间的契约。
I2 修改 IR 只能经过 Rewriter/Listener，这样 driver、handle、分析缓存才能保持同步。
I3 并发：同一 Context 下只允许做唯一化；执行期间禁止加载 dialect，所以必须预先声明 dependentDialects。
I4 支配性与隔离：value 的可见性只由 region 的种类和 IsolatedFromAbove 决定。
I5 开放世界：缺少某个事实 ⇒ 消费者必须保守处理（效应未知、不可 fold）。
I6 分析和 transform handle 在 IR 改变后失效。

## 6 MLIR 缺失的部分 → 规范化补完
G1 **事实代数**：trait/interface 之间的蕴含（SameOperandsAndResultType ⇒ Same*ElementType；Pure ⇔ NoMemoryEffect ∧ AlwaysSpeculatable）、前提（Involution 要求单操作数、单结果、同型）、互斥（NoTerminator ⊥ SingleBlockImplicitTerminator）。对 trait 集求闭包来检查冲突。
G2 **接口间依赖**：例如 LoopLike 隐含要实现 RegionBranch，Callable 的语义依赖 Symbol。目前只是约定，没有声明。
G3 **跨子系统义务**：推断结果满足约束；声明的效应 ⊇ 实际效应；fold 和 pattern 保持 W；类型只沿精化格 ⊑ 改变；TypeConverter 的映射与目标 op 约束相容。
G4 **pass 契约**：每个 pass 写明输入 dialect 集、输出 dialect 集、前提和保证，管线就可以静态类型检查（现在这些问题只能在运行时报错）。
G5 **重写系统**：规范化 pattern 要有终止度量，并用临界对检查局部合流（现在 greedy driver 只靠迭代上限兜底）。
G6 **类型精化格**：用 ⊑ 统一 compatible、cast、verifyCompatibleShape 的判定。
G7 **守卫化约束 + opaque 计数**：衡量形式化覆盖率。

## 7 迭代
v2 写可执行模型，覆盖 Σ、事实代数、交互矩阵和 pass 契约，用 arith/tensor/scf/func 抽样。
v3 用 TableGen --dump-json 导入真实 ODS，统计覆盖率和冲突。
v4 检验义务 G3 和 G5。
