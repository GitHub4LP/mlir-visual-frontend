# 第 12 轮：真实 IR 导入与 MLIR 测试集独立差分

## 0 复盘：为什么要做这一轮
第 10、11 轮报告的“与 MLIR 一致”（987/987、1000/1000）有自证的成分：输入来自我们自己的随机生成器，输出经过我们自己的发射器。生成器没有覆盖到的东西（类型修饰、可伸缩向量、空区域、非 index 的循环变量），这两轮测试都看不见。

另外，覆盖性标准 (1) 写的是“可导入/降级”，但到第 11 轮为止只有 core → MLIR 一个方向，**从真实 MLIR IR 导入到前端模型的方向从来没有做过**。

本轮改用 MLIR 自己的方言测试集作为外部真值：取 mlir/test/Dialect 下 12 个方言共 246 个文件、2992 段，提交与 Python 绑定相同（9bae84b0）。这批 IR 不来自我们的生成器，也不经过 core 的参数解释。

## 1 导入（model/irimport.py）
- 导入读取 MLIR 的对象模型，产出 core 的图：块与块参数、区域、后继、多结果，属性以 MLIR 文本原样存在 `props['_attrs']`。
- 导入方向**不经过 core 的参数**：第 11 轮已经发现 core 的参数空间比 MLIR 大。
- 后继实参的划分：cf.br、cf.cond_br、cf.switch 按分段属性划分；其它带后继的 op（包括未注册 op）标为 `_succ_opaque`。
- `emit_generic` 把导入得到的图回发成通用形式。

## 2 结果（`PYTHONPATH=model python3 model/corpus_test.py`，需先运行 `scripts/fetch_corpus.sh`）
| 检验 | 结果 |
|---|---|
| C1 导入忠实性：MLIR 接受的 2111 段做 import → emit → 再解析，与原 IR 的通用形式逐字比较 | **2108 段逐字相同**。另有 3 段失败：Python 绑定读不出复数 dense 属性。另有 1 段带方言资源块（模块级数据，不在 op 树上，未导入） |
| C2 合法 IR 上不误拒：3487 个有函数体的函数，过结构层与自动签名 | 修正后**全部接受（误拒 0）**；修正前误拒 21 个 |
| C2 人工签名（typesig） | 在**合法**的 IR 上误拒 203 个函数，主要是带静态偏移/步长的切片、i0 |
| C3 MLIR 负例：472 个能从诊断重建出单个 op 的负例 | 符号层在 op 局部抓住 71 个；声明层（C++ verify）373 个归预言机；auto 账目漏掉的 8 个全部是属性层（缺 predicate）或符号上下文（func.call、get_global）问题，不属于类型签名漏判 |
| C3 其余负例 | 326 个是语法层错误（自定义汇编、非法类型），80 个带区域，无法在单 op 上重建 |

## 3 真实 IR 揭出的模型缺陷（均已修正）
| 缺陷 | 位置 | 原因 | 修正 |
|---|---|---|---|
| 类型文法太窄：不认识编码 `tensor<…, "foo">`、可伸缩维 `vector<[4]x…>`、memref 布局与内存空间、`vector<42 x i32>` 这种写法、f8 系列、none、tuple、方言类型 | typesig.parse | 手写的正则文法 | 类型改由 **MLIR 自己的类型解析器**给出；修饰与元素类型分开存放。比较整个类型时修饰参与比较，与 MLIR 的类型相等一致；只看 t[0..2] 的谓词仍是上近似 |
| `getI1SameShape` 丢掉了编码和可伸缩维，导致 cmpi/cmpf 误拒 | typesig.i1same | 与 MLIR 的实现不一致 | 保留修饰 |
| ScalableVectorType 原子恒为假 | odsjson 原子表 | 旧注释写着“类型池不含可伸缩向量”，即按生成器分布取了捷径 | 按修饰判定 |
| scf.for 的归纳变量恒为 index | structure R8 | 手写假设；MLIR 允许整数边界 | 归纳变量取边界的类型 |
| 函数类型按逗号和 `->` 直接切分 | structure._ftype | 没有考虑括号深度 | 按括号深度切分 |
| “空区域非法” | structure R1 | 手写假设；scf.if 的 else 可以为空 | 区域块数取自 **ODS 的区域约束**（SizedRegion、MaxSizedRegion、AnyRegion、VariadicRegion） |
| 未登记的 op 被当作非终结子 | structure R2/R3 | 手写的 STRUCT 表不全 | 终结子性质取自 ODS 的 IsTerminator；ODS 也没有的 op 视为未知，不判 R2 |
| 未注册 op 的后继实参 | structure R4 | 划分未知却仍做检查 | `_succ_opaque` 时跳过 |
| UnitAttr 被当作必需属性 | odssig | 第 11 轮已修正 | — |

这些缺陷的共同点：**凡是手写的假设，都在真实 IR 面前出了错；凡是从 MLIR 导入的（ODS 约束、MLIR 类型解析器），都没有出错。** 这把第 7 轮的原则从“值语义不手写”推广到类型文法、区域约束、终结子性质。

## 4 决定：人工签名不再有拒绝权
人工签名在合法的真实 IR 上误拒 203 个函数，而结构层、自动签名的误拒都是 0。因此 core.typecheck 的第三层改为只给提示（`core.TYPE_WARN`）、不拒绝。

规则（**L33**）：**只有在真实 IR 上证明过不误拒的层，才有拒绝权**，目前是结构层、自动签名、预言机三层。前端可以把人工签名的提示画成警告，不能画成错误。回归测试的输出不变：所有回归里人工签名本来就从未成为决定性的一层。

## 5 剩下的缺口
- **方言资源块**（`{-# dialect_resources #-}`）是模块级数据，不在 op 树上。core 的模块表示需要一个资源表，否则 `dense_resource` 常量在往返时会丢失内容。
- **类型与属性的良构性**：非法布局等由 MLIR 的类型 verify 拒绝（C3 中 3 例）。前端构造类型时需要同一套检查，应当直接调用 MLIR 的类型构造器，而不是自己维护文法。
- **op 局部可抓的负例只占 15%**（71/472）。其余要么是 C++ verify（归预言机），要么依赖属性（下一轮属性层）或符号上下文（结构层 S1–S2 需要接入 func.call、memref.get_global 的类型义务）。
- **C2 只检查了不误拒**。在合法 IR 上能检查的只有这一条；判定的完备性仍由预言机保证。

## 6 回归
- core、shape（S6 421/421）、transform、loop、edit、defs 的输出与修正前逐行相同；
- structure 的 R 系列全部一致；
- typesig_test、compsig_test 无 traceback；
- odssig_test：auto 6080/6080，非 declared 的分歧为 0；
- oracle_test 400：O1 391/391，O5 400/400。
