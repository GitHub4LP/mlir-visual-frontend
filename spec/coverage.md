# MLIR 机制覆盖矩阵

这是正确性标准 (1) 的可检验形式：MLIR 的每一项机制都要落到元模型上，可以导入或降级，但不能缺。
状态：**✓** 已建模，并在统一核心上检验过；**◐** 规范里有，或者有原型但还没并入核心，或者只覆盖了一部分；**✗** 缺失；**—** 有意不映射（写明理由）。
强度记号同 composition-laws.md：E 精确推导 / S 可靠不完备 / D 只能声明再检查。

## A IR 核心
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 1 | Operation、operand、result | Node、端口 | E | ✓ | core |
| 2 | SSA use-def | 数据边 | E | ✓ | core、R5/R6 |
| 3 | Block、块参数 | Blk | E | ✓ | R1–R4 |
| 4 | Region（SSACFG / Graph） | RegionSpec.kind | E | ✓ | R1、R7、L5 |
| 5 | Successor、非结构化 CFG | 控制边 | 结构 E / 值层 ✗ | ◐ | 多块 CFG 的值层是 Unknown |
| 6 | Dominance | R5（只看可达块） | E | ✓ | 差分 0 不可靠 |
| 7 | IsolatedFromAbove、捕获 | R6、结构摘要 captures | E | ✓ | L3、extract |
| 8 | Symbol、SymbolTable、可见性、嵌套引用 | S1–S3 | E | ✓ | L4（X6） |
| 9 | Location、诊断回映 | 节点 id → loc | E | ◐ | 规范 §3；没有和 MLIR 诊断联测 |
| 10 | 未注册 op 与方言 | 结构保守（R2 拒绝）、值层 Unknown | S | ◐ | 策略已定，规范需写明 |

## B 类型与属性
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 11 | builtin 类型 | 类型项；值层多 sort（iN 位向量、index 为 ℤ、tensor 为 ⟨extent, 元素, 元素毒值⟩） | E | ◐ | core：任意 iN、index、tensor（shapes.md）；float/vector/memref 形状不在 core |
| 12 | TypeConstraint（211 个具名定义） | 规范化原子 | 190 E / 21 S | ◐ | normalize_types.py，未接入 core |
| 13 | 用户定义类型（TypeDef、IRDL、ADT） | define_adt：sort + 生成的 op + 生成的降级 | E | ◐ | definitions.md §5；标量字段、≤2 个变体、无递归 |
| 14 | Attribute（Integer、Dense、Array、Dict、SymbolRef、Type、AffineMap…） | 静态项，按指称存储（L8） | E | ◐ | D1 与 MLIR 600/600；缺 Dense、Float、AffineMap、Dict |
| 15 | Properties 与可丢弃 attribute 的区分 | 参数进语义，元数据不进；命名空间分离（L9） | E | ✓ | D2 与 MLIR：不可靠 0 |
| 16 | AttrConstraint、ConfinedAttr、DefaultValued | 参数值域；复合 op 参数约束推导（L10） | E | ◐ | D3 与 MLIR 16/16；复合 op 默认值推导未做 |
| 17 | Type/Attr interface（ShapedType 等） | 接口 + 实现表（类型上） | E | ◐ | Monoid 示例；ShapedType 未导入 |
| 18 | DataLayout | 目标参数（IW） | E | ◐ | core.IW 决定 index 溢出成毒值的界（L17′）；其余布局 ✗ |

## C 签名
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 19 | Variadic、Optional、segment | 端口组 | E | ◐ | 只在规范 |
| 20 | InferTypeOpInterface | 类型推导 | E | ◐ | solver.py |
| 21 | 签名 trait（SameOperandsAndResultType…） | Φ 上的蕴含查询 | E | ◐ | solver.py |

## D 语义与效应
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 22 | UB 与毒值 | 值层（QF_BV） | E | ✓ | T1 精确 900/900；L6 |
| 23 | Speculatable | caps 派生（含毒值输入） | E | ✓ | X3 |
| 24 | MemoryEffects | compose_eff | S | ✓ | L7；llvm.ptr 的别名是 S |
| 25 | Resource、Target 细分 | effect 三元组 | S | ◐ | 只有 Default |
| 26 | AutomaticAllocationScope、alloca | 透明模式自动包 alloca_scope | E | ◐ | 只在规范 |
| 27 | 下标、extent（slice、pad…） | extent 是 index 值（ℤ），静态维是投影（L22） | E | ◐ | shapes.md：9 个 tensor op 并入 core，bufferize 后 JIT 差分 1049/1049；pad、多维 stride、memref 形状未做；Presburger 显示层未接 |
| 28 | 循环（scf.for、LoopLike、scf.while） | 静态展开 / 按结构命名的 UF 摘要（L18）/ 归纳检查过的不变式（L19） | E/S/D | ◐ | core 已有 scf.for，loops.md；while ✗；isl power 未并入 |
| 29 | RegionBranchOpInterface | scf.if 守卫 | E | ✓ | core |
| 30 | Call、Callable | S4；复合 op 调用 = 代换 | E | ◐ | func.call 的值层未建模 |
| 31 | 递归 | 必须声明签名 | D | ✗ | |
| 52 | 泛型复合 op（接口约束的多态） | 单态化 + 实现表 | E | ◐ | 泛型本身没有摘要 |

## E 扩展机制
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 32 | Dialect（命名空间、注册、版本） | 节点库 | — | ✗ | |
| 33 | Op/Type/Attr interface、external model、promised interface | 声明与实现表分开；derived/checked/declared；定律为定向精化（L12） | E/D | ◐ | D5；op interface 只有派生的 MemoryEffects、Speculatable |
| 34 | Native trait | 导入为声明，并与派生 caps 审计 | D→E | ◐ | ods_import、审计规则 |
| 35 | ODS/TableGen 导入 | Schema 导入器 | — | ◐ | 迷你解析器；生产用 tblgen JSON |
| 36 | IRDL、动态方言 | 复合 op 的不透明身份 | — | ✗ | 只在规范（落地方案 c） |
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
| 49 | WASM 目标 | ⟦·⟧ | ✗ | |

## H 编辑期（标准 3，MLIR 没有对应）
| # | 机制 | 状态 | 证据或缺口 |
|---|---|---|---|
| 50 | 编辑操作代数（变换 + 保持的事实 + 判定） | ◐ | edit.md：7 种全函数编辑 + 洞；L20 连接 = 代换；只覆盖顶层块 |
| 51 | 增量查询（增量 ≡ 全量） | ◐ | edit.md：Merkle 键 + 指纹早截断（L21），720 次编辑 0 不一致；定义内部不细分 |

## 小结
共 52 项：✓ 13，◐ 30，✗ 8，— 1（2026-10-14，形状层 v0 之后；#27 仍为 ◐，但覆盖面扩大）。
- 剩下的 ✗：递归、方言组织（命名空间、版本、IRDL、方言扩展、bytecode）、transform dialect、数据流分析框架、WASM。
- 当前最大的缺口：(1) scf.while 和多块 CFG 的值层；(2) memref 的形状与别名（bufferization 契约）；(3) 区域内部的编辑、签名变更的传播。
