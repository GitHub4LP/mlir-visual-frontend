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

## 7 对前端的指导
1. 编辑器里的“连线”分两种：数据边（R5、R6，局部检查）和控制边（R3、R4、R5、R7，会触发支配重算）。前端优先提供结构化控制（scf），把非结构化 cf 限制在显式的“CFG 容器”里，让大多数编辑只走局部检查。
2. 补全和连线候选直接来自义务：可连的源 = 当前点可见（R6）且支配（R5）的值；可调用的符号 = 最近符号表里的符号，加上嵌套表里 public 的符号（S2、S3）。
3. 把节点移进或移出容器，等价于重算 R6 和捕获集；容器若声明了隔离，前端应提示“提升为端口”（L3），而不只是报错。
4. 改名和改签名是带扇出的变换，必须通过反向索引原子地完成（MLIR 的 SymbolTable::replaceAllSymbolUses 的前端版）。

## 8 未覆盖（下一步）
attribute/property 的类型和约束；interface（作为 Schema 上的能力谓词）；用户类型（ADT）；方言扩展；rewrite/pass 作为带契约的图变换。
