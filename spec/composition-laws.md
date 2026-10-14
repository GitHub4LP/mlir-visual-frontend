# 机制 × 组合律 全表 v1（对照 llvm-project main 分支，2026-10-13）
清单来源：inventory/ 目录，包括 IR/*.td、OpDefinition.h，以及 Interfaces/ 下的 27 个 .td 文件。

强度记号：**E** 精确，在可判定片段内是主推导；**S** 可靠但不完备；**D** 不可推导，必须声明并检查。
统一形式：复合 op = ⟨Schema 推导自 Graph, Impl=Graph⟩；记 ⟦n⟧ 为体内节点 n 的事实。

## 0 先决决策：透明 vs 不透明
- **透明**：在领域 pass 之前就内联。体内各 op 的全部机制自动生效，G 类接口不需要推导。
- **不透明**：保留复合 op 的身份（func 或 IRDL op）。此时 G 类接口只能声明（D）。
- 每个复合 op 自带一个模式位。默认透明，只有需要保留身份时（例如复用、分块粒度）才切换。

## A 类型 / 属性 / 参数
| 机制 | 组合律 | 强度 |
|---|---|---|
| TypeConstraint（211 个具名定义：Any*/TensorOf/MemRefOf/VectorOfRank…/HasStaticShape/ShapedTypeWithNthDimOfSize） | Φ_comp = ∃内部变量.∧Φᵢ∧边相等 | E；体内含 CPred 时为 S |
| Attr/Prop 约束（ConfinedAttr/ConfinedProp、IntMin/Max、ArrayCount、DenseArraySorted…） | 实现体中某个属性 = f(params)，把值域经 f 反向投影到 params | f 为线性时 E，否则 S |
| DefaultValued / Optional（Attr 和 Prop 都有） | 参数只流向一个落点时继承其默认值；流向多处且默认值冲突时 | E / D |
| TypedAttr、DerivedTypeAttr、TypeAttrOf（属性决定类型） | 作为函数依赖并入 Φ | E |
| 复合类型（别名、记录） | 别名：投影透明；记录：Type interface 由各字段合取 | E / S |

## B 签名
| 机制 | 组合律 | 强度 |
|---|---|---|
| Zero/One/N/AtLeastN 的 Operands/Results/Regions/Successors | 直接读边界 | E |
| Variadic/Optional/VariadicOfVariadic、SameVariadic*Size、AttrSized*Segments | 前端用显式端口组表示；可变组需要体内有 map/fold 结构，产生约束 ∀i<n；lowering 时生成 segment 属性 | E |
| SameTypeOperands、SameOperandsShape/ElementType、SameOperandsAndResult{Type,ElementType,Shape,Rank}、Operands/ResultsAre{FloatLike,BoolLike,SignlessIntegerLike}、All{Types,ElementTypes,Shapes,Ranks,ElementCounts}Match、(Optional/Ranged)TypesMatchWith、TypeIsPred | 是**查询**：Φ_comp ⊨ φ_T | E |
| InferType/InferShapedType/InferTypeOpAdaptor | 依赖闭包覆盖全部输出 ⇒ 推断函数 = 体内推断的复合 | E |
| ReifyRankedShapedType、DerivedAttributeOpInterface | 组合体内 reify / 派生函数 | S / E |

## C 结构 / 控制
| 机制 | 组合律 | 强度 |
|---|---|---|
| IsolatedFromAbove | 恒成立，因为只能经端口捕获 | E |
| HasParent / ParentOneOf | 体内有未被满足的父约束 ⇒ 外提为复合 op 自身的父约束 | E |
| SingleBlock、NoTerminator、SingleBlockImplicitTerminator、NoRegionArguments、RegionKindInterface/HasOnlyGraphRegion | 每个 slot 的约束 = 体内接收该 slot 的各 op 的区域约束之合取；没有 slot 则 ZeroRegions | E |
| RegionBranch(+Terminator)、LoopLike、HasParallelRegion | slot 之间的控制边由体内 scf 等结构推出。if：按条件守卫后取析取；loop：Post=R^t、Pre=∀k<t 可达状态满足 body 前提（isl power） | RegionBranch 为 E；LoopLike 在循环不变或平移型 carried 时为 E，其余为 S（Presburger 对迭代不封闭，例如减半），可用声明的不变式收窄（可判定的归纳检查） |
| Terminator、ReturnLike、Branch、WeightedBranch、SelectLike（多出口） | 不鼓励，建议只用结构化控制 | D |
| AffineScope、AutomaticAllocationScope | 含 alloca 的复合 op 在透明模式下自动包 memref.alloca_scope，两种模式语义一致；alloca 的结果逃逸 ⇒ 违反 Def | E（规则化） |

## D 符号
| 机制 | 组合律 | 强度 |
|---|---|---|
| Symbol、SymbolName、FunctionOp、Callable、ArgAndResultAttrs | 复合 op 的定义本身就是一个 symbol | E |
| SymbolUser（op/attr）、CallOp | 使用点；符号依赖集 = 体内引用之并 | E |
| SymbolVisibility、SymbolTable 容器 | 由用户选择 | D |

## E 效应 / 推测执行
| 机制 | 组合律 | 强度 |
|---|---|---|
| MemRead/Write/Alloc/Free（含 Resource 和 Target） | 取体内并集；Target 从体内值重定位到边界端口；局部的 alloc+free 互相抵消，**只限同一块内**（v4 修正 L7） | S |
| NoMemoryEffect、Pure（= AlwaysSpeculatable ∧ NoMemoryEffect）、RecursiveMemoryEffects | Eff 组合（新缓冲无别名、alloc/free 抵消、⊤ 吸收），Pure ⇔ Speculatable ∧ Eff=∅ | E（相对 Eff 的精度）；llvm.ptr 的逃逸为 S |
| Conditionally/Always/RecursivelySpeculatable | Speculatable ⇔ Pre ≡ 输入域，并能给出精确的“在 Pre 下可推测” | E（相对 Sem/UB 的精度） |
| TokenProducer/Consumer、DistinctObjectsTrait | 沿体内数据流传播 | S |

## F 代数 / 折叠 / 分析
| 机制 | 组合律 | 强度 |
|---|---|---|
| ConstantLike | 零输入，且实现体能完全 fold | E |
| fold | 对实现体做部分求值：逐 op fold 并传播 | S |
| 规范化 pattern、DialectFoldInterface | 体内 pattern 只作内部优化；复合 op 自己的 pattern 不可推（唯一可生成的是“全部输入为常量 ⇒ 折叠为常量”） | D |
| Commutative / Idempotent / Involution | 声明后验证：交换输入后（或自复合后）实现体在规范化意义下同构 | D，验证为 S |
| Elementwise/Scalarizable/Vectorizable/Tensorizable/ElementwiseMappable | 体内全部 Elementwise，且没有改变形状的边 | S |
| InferIntRange、InferIntDivisibility、ValueBounds、InferStridedMetadata | 组合体内的传递函数（抽象解释） | S |
| RuntimeVerifiable | 取体内检查之并 | E |

## G 领域变换接口（只有不透明模式才需要）
DestinationStyle 在 DPS 链能把输出连到 init 时为 S；Tiling/PartialReduction 在实现体为 linalg 融合链时为 S。
其余 IndexingMap、Subset、ParallelCombining、VectorUnroll/Transfer、ViewLike、OffsetSizeAndStride、ShapedDim、MemorySpaceCast(Consumer)、Promotable*（mem2reg/SROA）、Cast、Alignment、DataLayout* 均为 **D**。

## H 语法 / 元信息
| 机制 | 组合律 | 强度 |
|---|---|---|
| OpAsm 接口、汇编格式 | 用端口名作结果名，自动生成格式 | E |
| Location | 实例 loc 与体内 loc 组成 CallSiteLoc / FusedLoc，可回映到节点 | E |
| materializeConstant、discardable 属性验证 | 委托给体内所属 dialect | E / S |
| 验证 | 复合 op 的 verifier = 检查守卫化后的 Φ_comp，与验证顺序无关 | E |
| 合法性 / lowering | 透明模式内联；不透明模式自动生成 lowering pattern | E |
| bytecode 版本、Schema 演化 | 用旧使用点是否仍满足新 Φ_comp 来判兼容 | S |

## 事实间依赖（不动点求解顺序）
A 类型 → B 签名 → C 结构 → E 效应 → F 代数/分析 → G 领域。
后一层只读取前一层的结论。存在环的地方（例如效应依赖类型是 tensor 还是 memref）在格上迭代求不动点。

## 对照源码发现的 MLIR 缺口（已核实）
1. `IsIdempotent::verifyTrait` 的注释写明“要求无副作用，但该检查在 FIXME 下并未执行”。→ 前端的事实代数要把它补成真正的前提。
2. trait 前提只在 C++ 中以 static_assert 表达（Involution、Idempotent），没有声明式表示。→ 前端改为能力代数中的前提。
3. 具名约束库是有限的（211 个类型约束 + 属性约束），可以一次性人工整理成规范化原子。真正的不可分析风险集中在第三方 dialect 的 CPred 和 C++ verify()。
4. Pure、ElementwiseMappable 等是 TraitList（组合 trait），说明 MLIR 已有“合取”，但没有“由实现推导”。
