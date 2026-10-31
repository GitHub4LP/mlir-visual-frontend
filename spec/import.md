# 导入面（第 9 轮）：tblgen JSON 导入，11 个方言 305 个 op 的账目与差分

实现：`model/odsjson.py`（读取 tblgen JSON，解释 Pred 树，内建原子表）、`model/odssig.py`（自动签名、导入账）。
检验：`model/odssig_test.py`。运行前先执行 `scripts/fetch_tools.sh`，再运行 `PYTHONPATH=model python3 model/odssig_test.py 40`，约 4 分钟。

## 1 两处纠偏
- **版本固定。** 第 1–8 轮的 .td 取自 llvm main，而被测 verifier 来自 wheel 的提交 9bae84b0，两者已经有差异：main 里 `arith.addi` 用 `Arith_SignlessIntegerOrIndexLike`，wheel 里是 `SignlessIntegerOrIndexLike`。
  - 现在改用 wheel 自带的 .td（`mlir/_mlir_libs/include`），tblgen 取自同一提交的 `mlir_native_tools`。
  - 原则：**一份导入只对应一个 MLIR 提交**，导入结果是按提交生成的产物。
- **不再解析文本。** 第 1 轮的正则迷你解析器换成 `llvm-tblgen -dump-json`。导入器读取的是 tblgen 求值后的记录：继承已展开，`Variadic`、`Arg`、`TraitList` 都是结构化数据。

## 2 关键事实：端口类型约束只需 35 个内建原子
- 11 个方言 305 个 op 的 744 个端口约束，全部化归为 Pred 树（CPred / And / Or / Neg / Concat / SubstLeaves）。
- 树的叶子一共只有 **35 个不同的 C++ 原子**，外加 **2 个变换**（取元素类型、取 complex 的元素类型）。
- 这些原子全是内建类型系统的查询，例如 `isa<RankedTensorType>`、`isSignlessIntOrIndex()`、`getRank()==N`，没有一个属于方言语义。它们在 `odsjson.ATOM_RE` 里登记一次，所有方言共用。
- 推论：给前端加一个新方言，不需要为它写任何端口约束代码。这条规律对 IRDL/用户定义方言同样成立，只要它们的约束也由这些原子组合而成。

## 3 导入账（`odssig.ledger`）
| 方言 | op | auto | declared | unknown |
|---|---|---|---|---|
| arith | 51 | 35 | 16 | 0 |
| math | 46 | 46 | 0 | 0 |
| index | 26 | 24 | 2 | 0 |
| complex | 29 | 27 | 2 | 0 |
| shape | 40 | 16 | 24 | 0 |
| func | 5 | 2 | 2 | 1 |
| cf | 4 | 1 | 3 | 0 |
| vector | 39 | 4 | 29 | 6 |
| memref | 32 | 4 | 26 | 2 |
| tensor | 21 | 2 | 19 | 0 |
| scf | 12 | 0 | 12 | 0 |
| **合计** | **305** | **161（53%）** | **135（44%）** | **9（3%）** |

- **auto**：只用 ODS 就能导入，没有任何 C++ 校验。包括端口约束、结构化 trait（AllTypesMatch、TypesMatchWith、AllElementTypesMatch、AllRanksMatch、ShapedTypeMatchesElementCountAndTypes）、已知实现的 native trait、tblgen 自动生成的类型推断。
- **declared**：带有 C++ 校验。来源有四种：`verify()`（98 个）、带 verify 的接口（CastOpInterface、OffsetSizeAndStride、DestinationStyle 等）、C++ `inferReturnTypes`、`verifyRegions()`。
- **unknown**：有既不能自动导入、也没有登记的条目：`SameVariadicResultSize`、FunctionType 变换、vector 中的若干 lambda 谓词。
- **结论：值语义的计算方言（math/index/complex/arith）基本都是 auto；结构性方言（tensor/memref/vector/scf）的约束大多写在 C++ 里。**
- 修正第 7 轮的一个错误：`DeclareOpInterfaceMethods<X>` 生成的匿名记录不带 X 的 `verify` 字段，必须回到接口本身的记录去取。第一版导入器因此把 7 个 arith 转换 op 和 tensor.cast、index.castu 误判为 auto（它们有 CastOpInterface 的 C++ 校验）。是 L1 差分把这个问题暴露出来的。

## 4 差分（L1，每个 op 40 个实例，共 8917 个，按 ods+cpred 级）
- 参与差分的有 223 个 op。排除了带 region/后继/结构 trait 的 op，以及有非枚举必需属性的 op（属性是下一层的工作）。
- 实例按边界采样：一半取前端接受的，其中一部分再改动一个端口。
- 只解析合法的类型；MLIR 无法解析的组合（如 `vector<4xcomplex<f32>>`）事先剔除。

| 账目 | op | 实例 | 误收 | 误拒 |
|---|---|---|---|---|
| auto | 147 | 5880 | **0** | **0** |
| declared | 70 | 2797 | 467（16.7%） | 0 |
| unknown | 6 | 240 | 36 | 0 |

- MLIR 自身崩溃 3 例（arith 转换 op 遇无秩 tensor，与第 7 轮相同），前端全部拒绝。
- 误收最多的 op：memref.dma_start、complex.bitcast、tensor.reshape、arith 的转换类、vector.outerproduct、memref.cast。

## 5 组合律
- **L28 自动导入精确。** 账目为 auto 的 op，自动签名 ≡ MLIR verifier：5880/5880。
- **L29 自动签名是上近似。** 对任何 op，自动签名都不会误拒：8917 个实例误拒为 0，即自动签名 ⊇ MLIR 实际接受的集合。所以补全和“能不能连”可以直接用自动签名，不会漏掉合法选项；只有 declared 和 unknown 账目的 op 会误收。

## 6 对架构的影响：declared 层的定位要改
- 第 7 轮把 declared 层设想为“人工把 C++ verify 改写成声明”。账目说明这条路扩展不了：仅 11 个方言就有 135 个 op 带 C++ 校验，而且每个 MLIR 提交都可能变化。
- 改为三层：
  1. **符号层**：自动签名，加上可选的人工声明，用于补全、连接、显示签名、复合 op 的 Φ。由 L29 可知，它对所有 op 都是可靠的上近似。
  2. **预言机层**：在具体实例上调用 MLIR 自己的 verifier，作为 declared/unknown 账目的最终权威。同一提交的原生 verifier 校验一个两 op 模块约 2.8 µs，解析加校验约 15 µs，可以放进编辑期，前提是前端带有 MLIR（WASM）。代价有二：预言机只能对具体实例回答是或否，不提供候选、投影和诊断理由；而且 verifier 会断言崩溃，必须隔离运行，并以符号层的拒绝在前（崩溃例前端已全部拒绝）。
  3. **declared 降级为精度改进**：只给高频 op 写，目的是让补全更准，而不是保证正确。
- 复合 op 同样适用：Φ 是上近似，实例化时由预言机内联校验（compsig C1 已经是这种做法）。
- 第 10 轮已接入 core，见 oracle.md。第 11 轮新导入 ub 方言、修正 UnitAttr 的必需性：12 个方言，共 307 个 op，auto 163 / declared 135 / unknown 9，auto 6080/6080 一致（oracle.md §6）。

## 7 缺口
- 属性：参与差分的 op 中只处理了 I64Enum 必需属性，DenseI64Array、AffineMap、SymbolRef 等属于下一层。
- 带 region/后继的 op（scf、cf、func 的一部分）没有参与差分。
- 可伸缩向量、memref 布局与地址空间不在类型池中，相应原子的实现（`isStrided` 恒真、`ScalableVectorType` 恒假）只对类型池成立。
- linalg 未导入：它的命名 op 由 yaml 生成，需要 `mlir-linalg-ods-yaml-gen`。
- 第 7 轮的 typesig.py（23 个 op、正则导入）仍然保留，作为 core 的类型检查。下一步把它切换到 odssig，两者的差分结果均为 0 分歧。

## 8 第 31 轮：对照 mlir-tblgen 自己的解释（`model/odsdiff_test.py`，T1）
odssig 从 `llvm-tblgen -dump-json` 的原始记录**重新实现**了 mlir-tblgen 的一部分语义。此前这部分只有语料间接验证。
现在用 eudsl-tblgen（mlir::tblgen 的 Python 绑定）读同一份 .td（wheel 9bae84b0 自带），逐 op 对照 `mlir::tblgen::Operator`：
端口名与多重性（one/optional/variadic/varvar）、属性与 property 是否必需、区域与后继个数、是否自动生成 inferReturnTypes（`allResultTypesKnown`）。

| 项 | 一致/总数 | 处理 |
|---|---|---|
| ins / outs / regions / successors | 650/650 | 匿名端口 None≡'' |
| attrs（必需性） | 修复前 640/650，修复后 650/650 | **缺陷**：gpu 稀疏 op（spmv、spmm、sddmm 等 10 个）的 `modeA/modeB/pruneFlag` 是 EnumAttr，`defaultValue` 写在记录里，没有包在 DefaultValuedAttr 中，odssig 误判为必需。MLIR verifier 接受缺 modeA 的 `gpu.spmv`（会补默认值）。后果只在保守一侧：预言机发射时的忠实性守卫抛 LookupError，预言机不可用，没有误拒。修复：按记录的 `defaultValue` 字段判定，与 `Attribute::hasDefaultValue` 一致 |
| infer | 646/650 | 4 个都是 odssig 判“无规则”、tblgen 判“可推断”：arith.constant（结果取自属性 value 的类型）、vector.interleave/deinterleave、gpu.subgroup_mma_extract_thread_local（TypesMatchWith 变换不在 XF 表中，已登记为 unknown）。方向保守：结果类型保持字面量，不前推，类型关系仍由 rel/预言机检查。没有反方向的分歧 |

- 版本：eudsl-tblgen 0.0.1（2026-10-29 构建，LLVM 比 9bae84b0 新）。这里只用它的 C++ 解释逻辑，.td 仍是被测提交自带的那份。分歧须逐条归因，本轮 11 处都归因到 odssig，没有版本漂移。
- 定位：交叉检验，不是新真值。真值仍是同一提交的 verifier。eudsl-tblgen 不可用时这项检验跳过（setup_env.sh 安装失败不阻塞）。
- 回归：odssig_test L1 auto 账目 191 op / 11460 实例 0 分歧；oracle_test O3–O5 不变（O5 300/300）。
- 不替换 odsjson：Pred 树的语义解释（ATOM/XFORM ⇒ Unknown）是模型自己的义务，mlir::tblgen 只给出 C++ 条件字符串，没有给出语义。
