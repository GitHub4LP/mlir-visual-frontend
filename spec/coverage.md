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

## B 类型与属性（**最大缺口，下一轮**）
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 11 | builtin 类型 | 类型项 | E | ◐ | 只在 solver.py；core 只有 i1/i3/memref 名义类型 |
| 12 | TypeConstraint（211 个具名定义） | 规范化原子 | 190 E / 21 S | ◐ | normalize_types.py，未接入 core |
| 13 | 用户定义类型（TypeDef、IRDL、ADT） | 种类定义 | — | ✗ | |
| 14 | Attribute（Integer、Dense、Array、Dict、SymbolRef、Type、AffineMap…） | 带类型的静态值 | — | ✗ | core 的 props 是无类型字典 |
| 15 | Properties 与可丢弃 attribute 的区分 | params / 元数据 | — | ✗ | 需要的律：丢弃元数据后 Sem 不变 |
| 16 | AttrConstraint、ConfinedAttr、DefaultValued | 参数值域 | E/S | ◐ | 只有表中规则 |
| 17 | Type/Attr interface（ShapedType 等） | 种类上的能力谓词 | — | ✗ | |
| 18 | DataLayout | 目标参数 | — | ✗ | index 位宽义务依赖它 |

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
| 27 | 下标、extent（slice、pad…） | Presburger 层 | E | ◐ | shape.py、wp.py，未并入 |
| 28 | 循环（scf.for、LoopLike、scf.while） | isl power 与声明不变式 | E/S | ◐ | for 在 wp.py；while ✗ |
| 29 | RegionBranchOpInterface | scf.if 守卫 | E | ✓ | core |
| 30 | Call、Callable | S4；复合 op 调用 = 代换 | E | ◐ | func.call 的值层未建模 |
| 31 | 递归 | 必须声明签名 | D | ✗ | |

## E 扩展机制（**与 B 同属定义层，下一轮**）
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 32 | Dialect（命名空间、注册、版本） | 节点库 | — | ✗ | |
| 33 | Op/Type/Attr interface、external model、promised interface | 能力谓词 + 实现表 | — | ✗ | |
| 34 | Native trait | 导入为声明，并与派生 caps 审计 | D→E | ◐ | ods_import、审计规则 |
| 35 | ODS/TableGen 导入 | Schema 导入器 | — | ◐ | 迷你解析器；生产用 tblgen JSON |
| 36 | IRDL、动态方言 | 复合 op 的不透明身份 | — | ✗ | 只在规范（落地方案 c） |
| 37 | 方言扩展、延迟注册 | 节点库组合 | — | ✗ | |

## F 变换
| # | 机制 | 元模型落点 | 强度 | 状态 | 证据或缺口 |
|---|---|---|---|---|---|
| 38 | fold | 部分求值 | S | ✗ | |
| 39 | Canonicalization | 带契约的图变换 | D | ✗ | |
| 40 | RewritePattern、PDL/PDLL | 同上 + 匹配语义 | D | ✗ | |
| 41 | Transform dialect | 变换脚本 | — | ✗ | |
| 42 | Pass、PassManager | 管线契约 Sem(g) ⊑ Sem(pipeline(⟦g⟧)) | — | ◐ | difftest 的 LICM 实例 |
| 43 | DialectConversion、TypeConverter | ⟦·⟧ | — | ◐ | vslice 的手写发射 |
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
| 50 | 编辑操作代数（变换 + 保持的事实 + 判定） | ◐ | 结构层的增量依赖表；extract 契约 |
| 51 | 增量查询（增量 ≡ 全量） | ✗ | |

## 小结
在 51 项里，✓ 11 项，◐ 21 项，✗ 18 项，— 1 项。✗ 主要集中在 **B 与 E**：类型、属性、接口、方言都还不能由用户定义。这正是“元层未闭合”。所以下一轮做定义层。
