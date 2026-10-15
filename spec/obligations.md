# 结构义务审计（第 15 轮）

## 1 动机
第 14 轮发现：S1–S4 只读手工字段，在真实 IR 上**从未被求值**，此前的“误拒 0”是空检验。
因此本轮对全部结构义务（R1–R8、S1–S5）复查两件事：
1. **触发数**：在 MLIR 接受的真实 IR 上，每条义务实际被求值了多少次（`structure.CHECKED`）；
2. **抓得住**：只违反这一条义务的变异，我们是否和 MLIR 判定一致。

测试：`model/obligations_test.py`（需要 corpus/）。
- Z1：2108 个合法模块，整模块判定。
- Z2：以真实模块为底，每模块最多 12 个变异，交给同一提交的 MLIR verifier 判定。

## 2 Z1：触发数（2108 个模块，误拒 0）
| 义务 | 求值次数 |
|---|---|
| R1 区域数 | 7149 |
| R2 终结子 | 4509 |
| R3 后继计数 | 106 |
| R4 后继实参 | 148 |
| R5 支配 | 19695 |
| R6 作用域/隔离 | 21748 |
| R8 控制流边类型 | 3989 |
| S1 重名 | 3545 |
| S2 解析 | 110 |
| S4 使用义务 | 110 |
| S5 声明可见性（新） | 3535 |

- 每条义务都确实被触发，没有空检验。
- R3/R4 的触发数偏少（真实 IR 里多块 CFG 少），结论的统计强度相应弱一些。
- R8 只覆盖 4 种 op：func.func、scf.if、scf.for、scf.execute_region，共 3949 个实例。
- 带区域 op 中，另有 914 个实例的 R8 未覆盖，例如 scf.while、scf.parallel、tensor.pad、vector.mask。它们的区域分支类型由 C++ 实现：RegionBranchOpInterface 或各 op 自己的 verifyRegions。

## 3 Z2：变异差分（8858 个变异）
| 变异 | 拒/拒 | 收/收 | 我们收、MLIR 拒（漏判） | 我们拒、MLIR 收（误拒） |
|---|---|---|---|---|
| R1 区域置空 | 2029 | 145 | 119 | 0 |
| R2 删终结子 | 2007 | 1 | 73 | 0 |
| R2 终结子不在块尾 | 1869 | 0 | 0 | 0 |
| R4 后继实参换类型 | 4 | 0 | 0 | 0 |
| R5 定义挪到使用之后 | 1247 | 0 | 0 | 0 |
| R6 区域内的值越出作用域 | 140 | 0 | 0 | 0 |
| R8 终结子操作数换类型 | 1140 | 27 | 57 | 0 |
| **合计** | 8436 | 173 | 249 | **0** |

“我们”= 结构层 + 符号层 + 类型层（自动签名、属性层），关掉预言机。

### 漏判全部可归因
- R1 漏判 119：
  - 106 例在未导入的方言上：linalg 76、affine 28、gpu 1、async 1；
  - 13 例在已导入方言的 C++ verify 里：scf.execute_region 11（region needs at least one block）、memref.generic_atomic_rmw 1、func.func 1。
- R2 漏判 73：全是 func.func，删掉终结子后，块尾是未导入方言的 op（linalg、shard、bufferization、nvgpu）。我们不知道它是不是终结子，正确地不判。
- R8 漏判 57：tensor.pad 43、scf.while 6、tensor.generate 5、scf.parallel 2、scf.reduce 1。这些都是 C++ 区域分支类型义务（declared），交给预言机。

## 4 本轮修正（审计暴露的问题）
1. **兄弟区域复用值名**
   - 现象：MLIR 打印时，兄弟区域的值名可以相同，例如 scf.while 的 before/after 区域里都有 `%1`。`_types_of` 深收集会让它们互相覆盖。
   - 修正：判定一律逐区域下行（`deep=False`）。
   - 这是潜在误判源：此前没出事，只是因为覆盖顺序碰巧对了。
2. **R2 通用化（MLIR 核心规则）**
   - 已注册 op 只要没有 NoTerminator、也没有 RegionKindInterface，它区域里的每个块都必须以终结子结尾；
   - 有 `SingleBlockImplicitTerminator<X>` 时，块尾必须正是 X。这是按名字比较，块尾 op 未导入也成立。
   - X 由 C++ 类名经 ODS 映射到 op 名，共 12 个 op，例如 scf.parallel→scf.reduce、shape.reduce→shape.yield。
   - R2 漏判 265 → 73。
3. **TraitList 展开**
   - 问题：SingleBlockImplicitTerminator 是匿名 TraitList，以前接口闭包没有展开，SingleBlock 和终结子信息都丢了。
   - 修正：`symtab.interfaces` 展开 TraitList。
4. **S5（新）**
   - 规则：符号声明不能是 public，即 FunctionOpInterface 函数体为空且可见性为 public 时拒绝（SymbolOpInterface::verify）。
   - R1 在 func.func 上的漏判 1596 → 1。
5. **后继实参进入端口绑定**
   - 问题：ODS 的操作数表包含后继实参（cf.cond_br 的 trueDestOperands…），以前只用 ins 绑定，导致 cf.cond_br/cf.br “无法绑定”，自动签名形同未用。
   - 修正：绑定时用 ins 加上后继实参展平，段长取 operandSegmentSizes。
   - 结果：R4 那 4 例从漏判变为拒/拒（条件不是 i1）。
6. **测试伪影**
   - 问题：通用形式在 op 上写出操作数类型，变异后不刷新的话，MLIR 报的是解析错，不是 verifier 错。
   - 修正：变异后按作用域刷新 `_types`。
   - 刷新前漏判 R8 的报错是 “expects different type than prior uses”，属于假漏判。

## 5 组合律
- **L38（义务非空）**：每条有拒绝权的义务，都必须同时报告两样东西：真实 IR 上的触发数，以及至少一类只违反它的变异的差分。只有“0 误拒”不构成证据。
- **L39（结构层上近似，限于被测分布）**：在 Z2 分布上，结构层 + 符号层 + 类型层（无预言机）误拒 0；它的漏判都落在未导入方言或 C++ declared 义务上，由预言机负责。

## 6 前端指导
- 删除或替换块尾 op：有 SingleBlockImplicitTerminator 的父 op 可以直接给出唯一合法的终结子（补全精确）。其余父 op 只能给出“须是终结子”。
- 改变终结子的操作数类型：只有 R8 覆盖的 4 种 op 能即时报错；其余（while/pad/generate/parallel/reduce）必须走预言机，编辑器应把它当作“待判定”，而不是“合法”。
- 名字作用域按区域，不按函数：兄弟区域可以复用名字，前端内部应使用值 id，不使用打印名。
