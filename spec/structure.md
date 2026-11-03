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
- L5（区域种类）：needs_ssacfg 的复合 op 不能内联进 Graph 区域；它只能以不透明模式出现，或者先结构化（cf → scf）。needs_ssacfg ≝ 体的顶层区域多于 1 块（第 52 轮更正，见 §6.2）。

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
- 提取的上下文约束：第 50 轮已覆盖（14/14 预先指出，只到未知级；HasParent 有拒绝权）。

**第 51 轮（精度）**：第 50 轮的 513 个未知段按原因拆开，412 个来自“祖先校验读体”，而其中绝大多数只是取入口块（`getRegion().front()`）、块参数或终结 op。段不含终结 op，原处换成 func.call 后，祖先看到的块结构、块参数、终结 op 及其操作数都不变，变的只有体内的非终结 op。所以“读体”收窄为读体内 op 的写法：遍历 op（walk、getOps、getOperations、范围 for）、取值的定义 op 或使用者、判体内 op 的副作用、块的 front/begin。调用闭包也不再跟“别的 op 类”的成员函数（作用于另一个 op，不属于本 op 的校验），接口类、detail 命名空间与自由函数照跟。
- 调参集（前 1500 个函数，4435 段）：未知 513 → 102（2.3%），14/14 仍预先指出，误拒 0，判 ok 4319 段中 MLIR 失败 0。
- 留出集（之后 1116 个函数，3345 段，未参与收窄）：P3 172 段全部判未知（多为 vector.mask 内的 transfer_read/write：子 op 读父，以及 vector.mask 读体），判 ok 漏 0，误拒 0，P1 未知 30/3173（0.9%）。检验 C6。
- 余下的未知大多是真依赖：affine.* 要求 AffineScope 祖先（维度/符号合法性），vector.transfer_* 在 vector.mask 内。属于噪声的主要是 scf.for 的 24 段（getForInductionVarOwner 读的是块参数属主，即 for 自己，文本扫描分不出）。
- 仍是文本过近似，不是校验语义的证明；只产生未知，不拒绝。
- L5（needs_ssacfg）：第 52 轮补上外部证据，见 §6.2。

### 6.2 第 52 轮：Graph 区域（L5 外部证据与一处误拒）

MLIR 的规则（9bae84b0）：Verifier.cpp `verifyOnEntrance` 要求已注册 op 的 Graph 区域“0 或 1 块”，与 SingleBlock trait 无关；支配只在 SSACFG 区域里检查，`mayBeGraphRegion` 把未注册 op 的区域也当作可能是 Graph（RegionKindInterface.cpp），因此不查支配。

更正：
- **判据**：needs_ssacfg 原为“体内任何处有带后继的 op”。它既误报（嵌套 op 自己区域里的 cf，内联后那个区域的种类不变；真实语料中 3 个函数），又漏报（无跳转的不可达块）。现改为“顶层区域多于 1 块”。单块区域不可能有顶层跳转，因为入口块不能有前驱。
- **误拒（有拒绝权的结构层）**：ODS 导入的 op 原先一律按 SSACFG 查 R5 支配。所以 gpu.module、builtin.module 一类 Graph 区域里先用后定义，以及未注册 op 的区域，MLIR 接受，我们却判 R5。现按 ODS 取区域种类：HasOnlyGraphRegion/GraphRegionNoTerminator ⇒ Graph；C++ 自定 RegionKindInterface 或未导入/未注册 ⇒ 未知（不查 R5、不判 R7）。R7 增加“Graph 区域多于 1 块”。

检验（l5_test）：
- L5a：ODS 中 Graph 区域 op 共 6 个，全带 SingleBlock，结构层都判为 Graph。
- L5b：语料中 MLIR 接受的 2599 个函数，体内联进 builtin.module（参数换成 ub.poison），needs_ssacfg ⇔ MLIR “0 or 1 blocks” 一致 2599/2599。其中顶层多块的只有 5 个（语料所限）。单块函数内联后 MLIR 接受 2570，其余 24 个被拒，原因是 memref.alloca 需要 AutomaticAllocationScope 祖先，属于上下文约束，与 L5 无关（对应 M5.3）。
- L5c：定向 7/7（三种 Graph/未注册区域先用后定义：MLIR 接受、结构层不报；SSACFG 区域先用后定义：两边都拒；多块 Graph 区域：R7 且 MLIR 拒；嵌套 cf 与不可达块：新判据对、旧判据错）。

Graph 区域没有顺序语义：体内有副作用的 op 内联进去以后，执行次序由谁定义，MLIR 不规定。第 57 轮写成 L56，见 §6.3。
attribute/property 的类型和约束；interface（作为 Schema 上的能力谓词）；用户类型（ADT）；方言扩展；rewrite/pass 作为带契约的图变换。

### 6.3 第 57 轮：Graph 区域里的副作用次序（L56，model/graphorder_test.py）

**为什么是“未知”而不是“拒绝”**
- 第 52 轮 L5b 中，单块函数体内联进 builtin.module 后，MLIR 接受了 2570 个，其中不少含有 store、alloc、call。这说明 MLIR 的 verifier 接受这类 IR，但不规定其中的执行次序。
- 9bae84b0 中带 Graph 区域的 op 分三类，都没有次序语义：
  - 符号容器：builtin.module、gpu.module。
  - transform 的模式描述：apply_patterns 等 4 个。
  - scf.forall.in_parallel：唯一可执行的一类。它的 C++ verify 只允许 ParallelCombiningOpInterface op，目的地必须是 forall 的输出块参；重叠写入的结果未定义。
- 我们没有 A/B 证据证明哪种次序被 MLIR 拒绝，所以按 L33 不拒绝，只判未知并警告。值层本来就不对 Graph 区域的体推导事实。

**判据（structure.graph_order）**
- Graph 区域，或可能为 Graph 的 '?' 区域（未注册 op、C++ 自定 RegionKindInterface）里，可能有副作用的 op 都判未知。
- 副作用分类 effect_class 自下而上取 ODS：
  - NoMemoryEffect / MemoryEffects<[]>（含 Pure）⇒ none
  - RecursiveMemoryEffects ⇒ 递归看体
  - DeclareOpInterfaceMethods<MemoryEffectOpInterface>（C++ getEffects，如 linalg.generic）、操作数上的 MemRead/MemWrite、未导入 ⇒ may
- Symbol/IsolatedFromAbove 的 op 在所在处不执行，豁免：模块里的 func.func 体、gpu.func 的副作用不计入。

**检验**
- G1 定向 8/8。其中 in_parallel 里放纯 op 时 MLIR 拒绝，我们不拒（这类情况交预言机）。
- G2 消融：不递归错 1 例，无 Symbol 豁免错 4 例，不看 C++ 接口错 1 例。
- G3 外部对照：语料 1317 个块（另有 183 个块绑定无法解析，跳过）。每个直接在函数体里、无区域、有结果的 op（每块每个 op 名取 1 个），在原处克隆一份不用的副本，再跑 MLIR canonicalize（贪心驱动只删 wouldOpBeTriviallyDead 的 op）。
  - 共 207 个 op 名、2415 个实例。
  - none 删除 2070、none 保留 **0**：分类器没有低估副作用。
  - may 保留 34、may 删除 311：过近似 17 个 op 名，主要是只读的 load/transfer_read，以及 alloc、tensor 上的 transfer_write。对“次序”而言，读与写之间的次序本来就要计入，所以把只读算作 may 是对的；MLIR 删除只读 op 用的是更弱的“死”概念。

**边界**
- 无结果的 op 和 recursive 类没有外部对照，只有定向例。
- in_parallel 内不重叠的 parallel_insert_slice 本来是确定的，我们一律判未知，这是精度代价。
- “区域种类 '?'”按可能为 Graph 处理，会在 C++ 自定 RegionKindInterface 的 SSACFG 区域里产生多余的未知。
