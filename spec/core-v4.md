# 统一核心 v4

原型：`model/core.py`（Schema v4 与推导）、`model/core_test.py`（预言机与跨层用例）、`model/structure.py`（结构层义务）。
运行：`PYTHONPATH=model python3 model/core.py`、`PYTHONPATH=model python3 model/core_test.py 300`

## 1 为什么要统一
v3 之前的模型分散在 7 个原型里，各有各的 Schema 和图表示，跨层组合从来没有一起检验过。统一后立即发现一个 v3 的不可靠缺陷（§4 L6）。这说明“对组合封闭”只能在统一核心上检验。

## 2 一种 IR，一种 Schema
- **IR**：module / region / block / node（structure.py 的表示）是唯一的图表示，v3 的扁平 DAG 废弃。
- **Schema v4** = ⟨签名；值层；Eff；Struct；信任等级；Impl⟩
  - 值层：每个输出的 (值, 毒值标志) 和 UB，都是 **(输入值, 输入毒值标志)** 的函数。用 z3 QF_BV 表示（v3.1 的双类决定）。
  - Eff：{(kind, resource, 端口)} 或 ⊤。
  - Struct：区域、终结子、后继、符号、隔离性（structure.md §2）；复合 op 另有结构摘要（捕获、自由符号、needs_ssacfg）。
  - Impl：原生（覆盖层手写语义）或实现体（复合 op），复合 op 有透明/不透明两种模式。
- **derive(实现体)** 一次遍历给出值层、效应层、结构层三层事实。结果仍是 Schema，可以再作为节点使用。
- **实例化 = 项代换**。所以“嵌套推导 ≡ 全内联”可以化为一条 z3 等价判定（`equivalent`）。
- **精化** `refines(src, tgt)`：src 不 UB ⇒ tgt 不 UB，src 输出不是毒值 ⇒ tgt 输出相同且不是毒值。它是改写、降级、接口定律共用的判据（definitions.md）。
- 参数：复合 op 可带参数（摘要对参数是符号的），原生 op 的参数在 core 里是指称（definitions.md §3）。

## 3 caps 派生（修正）
| cap | 定义 |
|---|---|
| Speculatable | ∀ 输入（**包括毒值输入**）不触发 UB |
| Speculatable｜非毒值输入 | 输入都不是毒值时不触发 UB |
| Pure | Speculatable ∧ Eff = ∅ |

推测执行时，被提前的 op 的输入可能是毒值，原程序不会执行它，提前后却会执行。所以 MLIR 意义上的 Speculatable 必须对毒值输入成立。例子：`g = b≠0 ? a/b : 0` 在非毒值输入上可推测，但 b 是毒值时分支条件也是毒值，会触发 UB，所以 g 不是 Speculatable（X3）。前端可以把两者都展示出来；MLIR 只有一个枚举值。

## 4 组合律（L1、L2 见 derivation-v3.md §5，L3–L5 见 structure.md §5；全部定律的证据等级见 laws.md）
- **L6 毒值敏感性必须进入摘要**。复合 op 的 UB 和输出毒值要依赖输入的毒值标志。v3 的摘要丢掉了这一点（复合 op 的 Sensitive 为空），在复合 op 接收毒值输入时**不可靠**。
  - 实例：h(x,b,c) = f(x, b +nuw c)，其中 f(x,y) = x/y。v3 的嵌套推导允许 b+c 溢出，内联推导不允许，两者不相等（用 v3 代码复现过）。
  - 原因：v3 的“嵌套 ≡ 内联”测试只用了非毒值的边界输入。
- **L7 效应的 alloc/free 只在同一块内抵消**。只在某个分支里释放 ⇒ 其余路径泄漏。v3 的规则不看控制流，同样不可靠（X4）。

## 5 编辑操作的契约：提取为复合 op
`extract(块, 区间)`：捕获值提升为输入端口（L3），后面还用到的值成为输出端口。
契约是提取前后宿主的值层等价（`equivalent`）。这是第一个带可检验契约的编辑操作，也是后续编辑代数的模板：每个编辑操作 = 图变换 + 保持的事实 + 判定方法。

## 6 检验结果
- T1：随机复合 op，3 层嵌套调用，含 scf.if（深 2）、nuw/回绕加减、divui、select。3 个种子各 300 个：
  - 嵌套摘要 ≡ 全内联：900/900；
  - 与穷举解释器（独立实现的预言机）比较：Pre 与 MayPoison **精确相等** 900/900，不可靠 0。
- 变异测试：去掉“除数为毒值即 UB”，预言机抓到；实例化时丢掉输入毒值（v3 的做法），闭包检验 18/80 失败、预言机 1/80 失败。所以闭包检验比单点预言机敏感得多，两者都要保留。
- T2：X1–X6 全部通过（L6、L3 与提取、caps 逐层派生、L7、L5、L4）。
- 结构层沿用 structure.py 的真实 MLIR 差分：不可靠方向 0。

## 7 已知边界（诚实记录）
- 值层只覆盖单块结构化区域。多块 CFG 的值层是 Unknown，caps 降为未知（X5）。办法是先结构化（cf → scf），或按路径展开。
- 下标、extent、循环的 Presburger 层（wp.py、shape.py）与类型合一（solver.py）**尚未并入**。它们与值层的接口是：index 值作为 ℤ 项，附带不溢出义务（v3.1）。
- 原生 op 只登记了 arith 的子集与 memref 的 4 个 op。
- 位宽固定为 i3，是为了能穷举。z3 的推导与位宽无关，预言机不是。

被取代的原型：`effects.py`（并入 compose_eff，并修正了 L7）、`test_oracle.py` 与 `bv.py`（并入 core_test）。
