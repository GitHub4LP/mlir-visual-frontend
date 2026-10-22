# MLIR 机制覆盖矩阵

这是正确性标准 (1) 的可检验形式：MLIR 的每一项机制都要落到元模型上，可以导入或降级，但不能缺。
状态：**✓** 已建模，并在统一核心上检验过；**◐** 规范里有，或者有原型但还没并入核心，或者只覆盖了一部分；**✗** 缺失；**—** 有意不映射（写明理由）。
强度记号同 composition-laws.md：E 精确推导 / S 可靠不完备 / D 只能声明再检查。

## A IR 核心
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 1 | Operation、operand、result | Node、端口 | E | ✓ | core |
| 2 | SSA use-def | 数据边 | E | ✓ | core、R5/R6 |
| 3 | Block、块参数 | Blk | E | ✓ | R1–R4；第 15 轮触发数与变异差分见 obligations.md（R2 通用化：非 NoTerminator 须以终结子结尾、SingleBlockImplicitTerminator<X> 须以 X 结尾，取自 ODS） |
| 4 | Region（SSACFG / Graph） | RegionSpec.kind | E | ✓ | R1、R7、L5 |
| 5 | Successor、非结构化 CFG | 控制边 | 结构 E / 值层 ✗ | ◐ | 多块 CFG 的值层是 Unknown |
| 6 | Dominance | R5（只看可达块） | E | ✓ | 差分 0 不可靠 |
| 7 | IsolatedFromAbove、捕获 | R6、结构摘要 captures | E | ✓ | L3、extract |
| 8 | Symbol、SymbolTable、可见性、嵌套引用、SymbolUserOpInterface 使用义务 | S1–S4（symtab.py：定义者/表/使用者取自 ODS 接口闭包） | E | ✓ | L4（X6）；第 14 轮修正：此前的 ✓ 只有自测，真实 IR 上从未触发（symbols.md §1）。现 Y1 整模块 2108 误拒 0、Y2 变异 6089/6089（L36a/b）；嵌套引用的接受方向无外部真值 |
| 9 | Location、诊断回映 | 节点 id → loc | E | ◐ | 规范 §3；没有和 MLIR 诊断联测 |
| 10 | 未注册 op 与方言 | 结构保守（R2 拒绝）、值层 Unknown | S | ◐ | 第 16 轮：预言机在模块允许时以 allow_unregistered 送检（未注册 op 自身义务不查，上近似）；规范见 modular.md §2 |

## B 类型与属性
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 11 | builtin 类型 | 类型项；值层多 sort（iN 位向量、index 为 ℤ、tensor 为 ⟨extent, 元素, 元素毒值⟩） | E | ◐ | core：任意 iN、index、tensor（shapes.md）；float/vector/memref 形状不在 core |
| 12 | TypeConstraint（211 个具名定义） | 端口谓词：Pred 树 + 35 个内建原子（import.md） | E | ✓ | 11 方言 305 op 的 744 个端口约束全部可解释；auto 账目 147 op 与 verifier 5880/5880；core.typecheck = 自动签名 + 预言机，与 verifier 987/987（oracle.md） |
| 13 | 用户定义类型（TypeDef、IRDL、ADT） | define_adt：sort + 生成的 op + 生成的降级 | E | ◐ | definitions.md §5；标量字段、≤2 个变体、无递归 |
| 14 | Attribute（Integer、Dense、Array、Dict、SymbolRef、Type、AffineMap…） | 静态项，按指称存储（L8） | E | ◐ | 第 13 轮：属性值改由 MLIR 属性解析器承载（attributes.md §1），全部种类可表示；缺方言资源表（dense_resource） |
| 15 | Properties 与可丢弃 attribute 的区分 | 参数进语义，元数据不进；命名空间分离（L9） | E | ✓ | D2 与 MLIR：不可靠 0 |
| 16 | AttrConstraint、ConfinedAttr、DefaultValued | 参数值域；复合 op 参数约束推导（L10） | E | ✓ | 第 13 轮：ODS 导入 225 参数（精确 224、上近似 1）；A1 真实 14065 实例误拒 0；A2 变异 18145 例误拒 0、ODS 层漏判仅上近似 9（attributes.md） |
| 17 | Type/Attr interface（ShapedType 等） | 接口 + 实现表（类型上） | E | ◐ | Monoid 示例；ShapedType 未导入 |
| 18 | DataLayout | 目标参数（IW） | E | ◐ | core.IW 是 index 的回绕位宽（L17″，第 21 轮）；IW ∈ {32,64} 双宽度判定未做；其余布局 ✗ |

## C 签名
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 19 | Variadic、Optional、segment | 端口组 | E | ◐ | typesig.bind：单变长组与段属性；复合 op 的变长端口未做 |
| 20 | InferTypeOpInterface | 类型推导 | E | ✓ | typesig.infer 239/239；复合 op 的结果写成端口表达式，自动获得 InferType（compsig.md） |
| 21 | 签名 trait（SameOperandsAndResultType…） | 端口间关系 | E | ✓ | typesig.py：7 种 trait 自动导入，按 MLIR 的实现语义（不是名字的字面意思）检验 |

## D 语义与效应
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 22 | UB 与毒值 | 值层（QF_BV） | E | ◐ | 第 21 轮降级：T1 900/900 是自证（C）；外部审计 sem_test 只覆盖 arith 整数子集（V1 JIT、V3 int-range-optimizations 1500/1500），并抓出 nsw 漏建模（旧语义反例 22）；L6 复合摘要仍 C；第 23 轮（values.md）：移位/divsi/remsi/remui/exact/trunci 标志补上；属性账 S0 缺口 0；IW∈{32,64} 对照 JIT 反例 0 |
| 23 | Speculatable | caps 派生（含毒值输入） | E | ◐ | 第 21 轮降级：只有 X3 自证（C）；未与 MLIR 的 getSpeculatability 差分 |
| 24 | MemoryEffects | compose_eff | S | ◐ | 第 21 轮降级：L7 只有自证（C）；llvm.ptr 的别名是 S |
| 25 | Resource、Target 细分 | effect 三元组 | S | ◐ | 只有 Default |
| 26 | AutomaticAllocationScope、alloca | 透明模式自动包 alloca_scope | E | ◐ | 只在规范 |
| 27 | 下标、extent（slice、pad…） | extent 是 index 值（ℤ），静态维是投影（L22） | E | ◐ | shapes.md：9 个 tensor op 并入 core，bufferize 后 JIT 差分 1049/1049；pad、多维 stride、memref 形状未做；Presburger 显示层未接 |
| 28 | 循环（scf.for、LoopLike、scf.while） | 静态展开 / 按结构命名的 UF 摘要（L18）/ 归纳检查过的不变式（L19） | E/S/D | ◐ | core 已有 scf.for，loops.md；while ✗；isl power 未并入 |
| 29 | RegionBranchOpInterface | scf.if 守卫；R8 控制流边类型 | E | ◐ | 第 15 轮更正：R8 只覆盖 func/scf.if/scf.for/scf.execute_region（3949 实例），scf.while、scf.parallel、tensor.pad 等 914 实例的边类型是 C++（declared），交预言机；Z2 漏判 57 全在此（obligations.md） |
| 30 | Call、Callable | S4；复合 op 调用 = 代换 | E | ◐ | func.call 的值层未建模 |
| 31 | 递归 | 必须声明签名 | D | ✗ | |
| 52 | 泛型复合 op（接口约束的多态） | Φ = 有拒绝权的层逐节点判定 + ODS 前推推断（composite.py，L25/L46/L47） | E | ◐ | 第 22 轮：语料抽取的复合 op（19 方言）10703 + 40681 例 Φ_full ≡ MLIR；端口投影（L27）第 24 轮改到 Φ_ods（composite.project），真实切片 24070 候选不可靠 0；值层仍单态化；接口约束的多态未做 |

## E 扩展机制
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 32 | Dialect（命名空间、注册、版本） | 节点库 | — | ✗ | |
| 33 | Op/Type/Attr interface、external model、promised interface | 声明与实现表分开；derived/checked/declared；定律为定向精化（L12） | E/D | ◐ | D5；op interface 只有派生的 MemoryEffects、Speculatable |
| 34 | Native trait | 已知实现的 trait 导入为关系；其余记 unknown | D→E | ◐ | odssig.NATIVE_OK 7 个；SameVariadicResultSize 等未覆盖 |
| 35 | ODS/TableGen 导入 | tblgen -dump-json（与被测 MLIR 同一提交）→ 自动签名 + 导入账 | E/D | ◐ | import.md：auto 161 / declared 135 / unknown 9；region/后继 op 未差分 |
| 36 | IRDL、动态方言 | 复合 op 的不透明身份 | S | ◐ | 第 22 轮（closure.md）：复合 op 导出为 IRDL op，在语料切片上误拒 0；IRDL 对 builtin 类型只能表达 any/is/相等，只抓住 3.2% 的错误（L48）；用户自定义类型的 IRDL 导入未做 |
| 37 | 方言扩展、延迟注册 | 节点库组合 | — | ✗ | |

## F 变换
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 38 | fold | 由 Sem 派生（L15），复合 op 也有 | E | ✓ | transform.md；毒值折为 ub.poison |
| 39 | Canonicalization | pattern 集 + 终止度量（L16） | E | ◐ | 合流未证明；MLIR canonicalize 的翻译确认 3000/3000 |
| 40 | RewritePattern、PDL/PDLL | pattern = 模板 + 前提 + 构造，定义期符号证明（L13、L14） | E/D | ◐ | 单根 DAG；有效应降为 D；PDL 未导入 |
| 41 | Transform dialect | 变换脚本 | — | ✗ | |
| 42 | Pass、PassManager | 管线契约 Sem(g) ⊑ Sem(pipeline(⟦g⟧)) | — | ◐ | difftest 的 LICM 实例 |
| 43 | DialectConversion、TypeConverter | 1:N 类型转换 + 抽象函数精化（L11） | E | ◐ | definitions.md §5；合法性目标未建模 |
| 44 | Inliner interface | 透明模式 + L3/L4/L5 | E | ◐ | inline_all；规则未接入合法性判断 |
| 45 | DataFlow 分析框架 | 事实分层求解 | — | ✗ | |

## G 外部
| # | 机制 | 元模型落点 | 状态 | 说明 |
|---|---|---|---|---|
| 46 | Bytecode、版本、升级 | 节点库版本与迁移 | ✗ | |
| 47 | 文本格式、OpAsmInterface、自定义汇编 | — | — | 前端有自己的表示，只需要通用形式发射 |
| 48 | ExecutionEngine、到 LLVM 的降级 | ⟦·⟧ | ◐ | vslice 端到端 |
| 49 | WASM 目标 | ⟦·⟧ | ✗ | 第 23 轮先行发现（L50）：MLIR 的 index 折叠在 32 位 index 下误编译 ⇒ 面向 wasm32 须先把 index 降为 i32 |

## H 编辑期（标准 3，MLIR 没有对应）
| # | 机制 | 状态 | 证据或缺口 |
|---|---|---|---|
| 50 | 编辑操作代数（变换 + 保持的事实 + 判定） | ◐ | edit.md：7 种全函数编辑 + 洞；L20 连接 = 代换；只覆盖顶层块 |
| 51 | 增量查询（增量 ≡ 全量） | ◐ | edit.md：Merkle 键 + 指纹早截断（L21），720 次编辑 0 不一致；定义内部不细分 |

## 小结
共 52 项：✓ 14，◐ 30，✗ 7，— 1（2026-10-17，第 22 轮 IRDL ✗→◐ 之后）。
- **第 21 轮评级规则**：✓ 须有 A 或 B 级证据（laws.md）；只有自证（C）的降为 ◐。据此 #22、#23、#24 由 ✓ 降为 ◐。值层的 L17′ 被否定、改为 L17″，nsw 补建模（laws.md §2–3）。
- 剩下的 ✗：递归、方言组织（命名空间、版本、IRDL、方言扩展、bytecode）、transform dialect、数据流分析框架、WASM。
- 第 16 轮（modular.md）：预言机按模块上下文发射——导入节点逐字发射、被引用符号补私有声明、模块属性与未注册方言随上下文；C2 首次打开预言机仍误拒 0（2982 次调用，不可送达 4）；M2 19271 例变异与整模块 verifier 误拒 0，func.func 体内漏判 0。
- 第 17 轮（modular.md §6）：发射单位 = 隔离单元（单元 + 祖先外壳 + 被引用符号声明）；core.check_module；M3 19271 例对照整模块 verifier 误拒 0、漏判 0（不开预言机漏判 881）；L42 单元局部性。
- 第 18 轮（modular.md §7）：按归因导入 7 个方言（共 19）；修 S5 外推（isDeclaration 从 .td 判定）等 6 处导入缺陷；M3 19641 误拒 0 漏判 0，符号层自身漏判 4.57% → 3.55%。
- 第 19 轮（modular.md §8）：类型原子封闭/开放之分（L43）；linalg 命名 op 由构建期生成 .td 导入；残差来源定性（L44：C++ verify、开放接口、逐 op 区域分支边，只能由预言机裁决）。
- 第 20 轮（modular.md §9）：增量重验（L45），edit.md L21 指纹在预言机层的对应。
- 当前最大的缺口：(0) 方言类型原子与 RegionBranchOpInterface 区域分支类型义务（见 modular.md §7 批判）；(1) 方言属性 verifier 只在预言机里；符号层自身的漏判 881 仍须靠导入（linalg/affine/transform ODS）缩小；(2) core 节点内部改用 ODS 属性名作参数（attributes.md §8）；(3) 方言资源表；(4) trait/interface 作为查询的导入与审计。


## 第 12 轮
真实 IR 导入（irimport.py）：MLIR 测试集 2111 段中 2108 段往返逐字相同；3487 个函数结构层与自动签名误拒 0。类型文法、区域块数、终结子性质改由 MLIR/ODS 导入。人工签名失去拒绝权（L33）。详见 corpus.md。
