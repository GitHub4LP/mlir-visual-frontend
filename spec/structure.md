# 结构元模型 v0（region / block / 后继 / symbol）

原型：model/structure.py（判定、摘要、发射）、model/structure_fuzz.py（随机差分）。
运行：`PYTHONPATH=model python3 model/structure.py`、`PYTHONPATH=model python3 model/structure_fuzz.py 1500 0`

## 1 为什么要单独建模
Schema v3 只描述端口、类型、语义和效应，没有覆盖 MLIR 的结构机制，而它们决定了一张图是不是程序：哪些值在哪里可见、控制往哪里走、名字解析到谁。
所以 Schema 扩为 v4：`Schema = ⟨Schema v3; Struct⟩`。原生 op 的 Struct 来自 ODS（RegionKindInterface、IsolatedFromAbove、Terminator/HasParent、Symbol/SymbolTable、RegionBranchOpInterface、CallOpInterface），复合 op 的 Struct 由推导得到（§4）。

## 2 Struct 字段
| 字段 | 含义 | MLIR 来源 |
|---|---|---|
| regions[i].kind | SSACFG / Graph | RegionKindInterface |
| regions[i].isolated | 区域内不可引用外层值 | IsolatedFromAbove |
| regions[i].term | 要求的终结子（None 表示 NoTerminator） | SingleBlockImplicitTerminator 等 |
| regions[i].single_block | 只允许一个块 | SingleBlock |
| regions[i].table | 区域是一张符号表 | SymbolTable |
| terminator / succs | 自身是终结子，后继数的区间 | Terminator、successors |
| sym / ref | 定义符号 / 引用符号 | Symbol、SymbolUserOpInterface |
| branch | 父 op ↔ 区域之间的控制流边与类型规则 | RegionBranchOpInterface |

## 3 结构良构 = 12 条局部义务
| 义务 | 内容 | 依赖的编辑操作（增量重算的范围） |
|---|---|---|
| R1 | 区域个数、单块 | 增删区域或块：该 op |
| R2 | 每块以终结子结尾，且是规定的那个 | 增删块尾 op：该块 |
| R3 | 后继个数合法，目标在同一区域 | 连/断控制边：该终结子 |
| R4 | 后继参数与目标块参数类型逐一相等 | 控制边、块参数类型：该边 |
| R5 | 定义支配使用（只在可达块上） | 增删控制边会改支配树：**整个区域**；数据边：该边 |
| R6 | 可见性：不跨 IsolatedFromAbove 引用外层值 | 数据边、移动节点进出容器：该边 |
| R7 | Graph 区域里不允许块间跳转 | 同 R3 |
| R8 | 控制流边类型：yield/return ↔ 父结果，入口块参数 ↔ 父输入或迭代变量 | 父 op 签名、终结子输入：该 op 及其终结子 |
| S1 | 同一符号表内不重名 | 改名、增删定义：该表 |
| S2 | 符号能解析到 | 改名、移动定义：所有引用该名字的使用点（需反向索引） |
| S3 | 可见性：private/nested 不可跨表引用 | 改可见性：引用者 |
| S4 | 调用签名与被调函数一致 | 改函数签名：所有调用者 |

结论：只有 R5（改控制边重建支配树）和 S2/S4（名字和签名变化要扇出到使用点）不是 O(1) 局部。前端需要维护两份索引：每个区域的支配树（增量更新），以及符号 → 使用点的反向索引。这正是 MLIR 用 SymbolTableCollection、DominanceInfo 缓存的东西；前端把它们变成常驻、增量的结构。

## 4 真实 MLIR 确认的事实（模型照此实现）
- 平坦符号引用（func.call @f）**只在最近的符号表里解析**：不向外层上溯，也看不见嵌套表的内部。跨表要用嵌套引用 @m::@f，且目标必须是 public。
- private 在本表内（包括本表里非符号表的子区域）都可见。
- 支配只在**可达块**上检查。不可达块里先用后定义，MLIR 接受。
- 未注册的 op 可能是终结子，MLIR 因此放行；模型不放行（见 §6）。
- 函数体为空区域表示声明而不是定义；空入口块必须显式写出。

## 5 复合 op 的结构摘要（推导，声明只能收窄）
`derive_struct(体) = ⟨captures, free_syms, needs_ssacfg, multi_block, isolated_ok⟩`
- 组合律 L3（捕获）：透明内联进 IsolatedFromAbove 的宿主时，captures 必须提升为显式端口；captures 为空 ⇔ 可声明 IsolatedFromAbove（这是推导，不是声明）。
- 组合律 L4（符号）：透明模式下 free_syms 变成宿主所在符号表的解析义务（S2/S3）；不透明模式（func + call）下，它们是复合定义所在表的义务。两种模式解析到不同的表时，**语义不同**，所以必须判定一致或者拒绝透明模式。这一点与 alloca_scope 的问题（derivation-v3.md §5）同构：透明内联都要保持作用域。
- L5（区域种类）：needs_ssacfg 的复合 op 不能内联进 Graph 区域；它只能以不透明模式出现，或者先结构化（cf → scf）。

## 6 差分结果
- 15 个手写用例，覆盖全部 12 条义务，每个负例都触发了预期的那条义务：与 MLIR 判定 15/15 一致。
- 随机程序（多块 CFG、块参数、cond_br、调用、嵌套模块、重名、跨表引用），3 个种子各 1500 个：
  - 不可靠方向（模型接受、MLIR 拒绝）为 **0**；
  - 保守方向 1 到 5 个，全部来自“未注册 op 收尾”，是有意的更严。
- 过程中发现并修正了 3 处模型错误，都是随机差分抓出来的：后继操作数漏算使用（R5/R6）；不可达前驱污染支配计算；调用签名原来靠声明，改为从被调函数推出（S4）。

### 6.1 提取端口的真实 IR 差分（第 44 轮，capture_test）
语料 1500 个 MLIR 接受的函数（196 个含嵌套区域或多块），随机取 4435 段连续非终结 op（嵌套深度 1–3+ 共 526 段，多块区域 12 段），
用 `core.extract_ports` 求端口；被提取函数单独发射（func 隔离上方 ⇒ 漏捕获会被 MLIR 判“未声明值/支配”），宿主把段换成 `func.call`（漏输出 ⇒ 使用者引用不存在的值）。
发射前预置全部可见名字的类型，端口漏算不会被“无法发射”掩盖。

| | 段 | MLIR 两者接受 | 端口错误 | 上下文约束 |
|---|---|---|---|---|
| 第 44 轮算法 | 4435 | 4421 | **0** | 14 |
| 第 43 轮算法（消融） | 4437 | 4412 | **8** | 14 |

第 43 轮算法的三处缺陷（均由 MLIR 判出）：
1. 名字按全段扁平解析。导入的 IR 沿用 MLIR 文本名，兄弟区域会重用同一个名字；嵌套区域里定义的名字被当成全段已定义 ⇒ 段顶层对同名外部值的使用漏成捕获（“use of undeclared SSA value”）。改为词法作用域。
2. 只扫本块后续 op 找输出；同一区域其它块（被本块支配）的使用被漏掉。改为连同区域其它块一起扫。后继实参也算使用。
3. 段内“先用后定义”（只在不可达块或 Graph 区域合法）被当成内部定义；提取进 SSACFG 函数后变成支配错误。现在 ⇒ `NotExtractable`（2 段）。

**上下文约束 14 段**（memref.generic_atomic_rmw 体要求无副作用、tensor.parallel_insert_slice 要求 InParallelOpInterface 父 op 等）：提取改变了 op 的父上下文。
这不是端口问题，但对前端是真约束；结构层目前一例也没有预先指出（0/14）——缺口，见 §8。

**第 50 轮更正与补全（model/ctxcons.py）**：14 段里没有一段来自 ODS 的 HasParent。7 段是子 op 的 C++ verify 读父 op（parallel_insert_slice），7 段是祖先 op 的 C++/接口校验读体（generic_atomic_rmw 纯度 2 段，matmul 的 add/mul 体 5 段）。所以只导入 HasParent 只能抓到 0/14。现在分三级判定：
- **reject**：ODS HasParent/ParentOneOf。这是 MLIR 原生 trait 校验（`OpTrait::HasParent::Impl::verifyTrait`）。650 个 op 中 15 个带此约束，类名全部可解析。定向检验 8/8 与 MLIR 一致（ctxcons_test C4）。
- **unknown**：已声明校验（C++ verify/verifyRegions、接口 verify，沿调用闭包展开，深度 3）在同一提交的源码（inventory/cpp/）里文本上读父/祖先 op 或读体，或者源码里找不到。这一级只给警告，没有拒绝权。
- **ok**：两者都不命中。

真实 IR 结果（capture_test，1500 个函数，4435 段）：
- 14 段上下文失败全部预先指出（未知 14，漏 0）。
- 判 ok 的 3908 段中，MLIR 上下文失败 0 段。
- 判拒 0 段，误拒 0。
- 精度代价：MLIR 接受的段里有 513/4421（11.6%）被判为未知。原因是文本扫描与调用闭包过近似，例如 scf.for 的校验既读体，又经闭包触及 getParent。

另外更正一处检验缺陷：capture_test 原先默认只跑 300 个函数，其中只有 1 个含嵌套区域，上下文段为 0，这项检验实际形同虚设。默认已改为 1500。

## 7 对前端的指导
1. 编辑器里的“连线”分两种：数据边（R5、R6，局部检查）和控制边（R3、R4、R5、R7，会触发支配重算）。前端优先提供结构化控制（scf），把非结构化 cf 限制在显式的“CFG 容器”里，让大多数编辑只走局部检查。
2. 补全和连线候选直接来自义务：可连的源 = 当前点可见（R6）且支配（R5）的值；可调用的符号 = 最近符号表里的符号，加上嵌套表里 public 的符号（S2、S3）。
3. 把节点移进或移出容器，等价于重算 R6 和捕获集；容器若声明了隔离，前端应提示“提升为端口”（L3），而不只是报错。
4. 改名和改签名是带扇出的变换，必须通过反向索引原子地完成（MLIR 的 SymbolTable::replaceAllSymbolUses 的前端版）。

## 8 未覆盖（下一步）
- 提取的上下文约束：第 50 轮已覆盖（14/14 预先指出，只到未知级；HasParent 有拒绝权）。精度缺口：MLIR 接受的段中 11.6% 被判未知，需要更细的校验语义（例如只读区域参数/yield 类型的父校验）。
- L5（needs_ssacfg）的外部证据仍只有定向 X5。
attribute/property 的类型和约束；interface（作为 Schema 上的能力谓词）；用户类型（ADT）；方言扩展；rewrite/pass 作为带契约的图变换。
