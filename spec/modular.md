# 预言机按模块上下文发射（第 16 轮）

## 1. 复盘：预言机层自己是空检验

第 12–15 轮一直报告“C2 真实 IR 3487 函数误拒 0”，但 `corpus_test.py` 在 `run_file` 开头写着 `C.ORACLE_ON = False`：
**有拒绝权的四层里，预言机层从未在真实 IR 上被检验过不误拒**；`obligations_test` 的 `full()` 也关着它。
打开后再看可达性：真实语料 2706 个 func.func 体里 738 个（27%）根本到不了预言机：

| 不可发射原因（修前） | 体数 | 性质 |
|---|---|---|
| 缺分段 | 537 | 导入节点的分段在 `_attrs.operandSegmentSizes`，发射器只看手工字段 `seg` |
| insert_slice 的源秩与偏移数不符 | 42 | 把导入的 MLIR 形式当成 core 简写去“降级” |
| func.call 需要模块级符号 | 26 | 发射单位是孤立函数 |
| MLIR 不认识 linalg/affine/bufferization/shard… | ~130 | 守卫用的是**我们**导入的 ODS，而预言机（MLIR 自己）认识它们 |

即：第 15 轮 R1/R2 漏判的大头（“未导入方言”）本来就该由预言机补上，却被发射器自己挡在门外。

## 2. 改动（model/oracle.py、core.py、symtab.py）

- **导入节点逐字发射**：`props` 带 `_attrs` 的节点不经 ODS 守卫，属性原样写入属性字典，操作数/结果类型取当前值类型（前端编辑后的类型，而非导入时的 `_types`），区域递归，入口块也写块标签（否则空块的区域变成 0 块——`scf.forall.in_parallel` 的 16 例误拒即此）。core 原生节点仍走原守卫（参数归属、必需属性、分段、切片降级）。
- **符号上下文**：`symtab.func_scopes(g)` 给每个有体的 func.func 求最近符号表（名字 → 定义者）与最近 builtin.module 的属性。`emit_core(…, symbols, mod_attrs)` 收集体内全部符号引用（core 原生 `func.call` 的 callee、导入节点属性文本里的 `@ref`，字符串字面量除外），补成私有声明：
  - func.func ⇒ 无体 `private` 声明（S5：声明不能 public）；
  - 无区域的定义者（memref.global、shard.grid…）⇒ 原样；
  - 自递归不补；嵌套引用 `@a::@b`、有区域的非函数定义者、`dense_resource` ⇒ **不发射**（缺符号/缺资源会被 MLIR 以缺失误拒；预言机有拒绝权，发射必须忠实）。
  未导入 ODS 的方言里带 `sym_name` 的 op 也登记为定义者——只用于补声明，不赋予结构层拒绝权。
- **未注册方言**：`Oracle.verify(text, unreg)`；只在模块本就允许时打开（语料：源文件带 `--allow-unregistered-dialect`）。未注册 op 对 MLIR 不透明，其自身义务不查（上近似），其余照查。
- `core.typecheck(blocks, ftype, symbols, name, mod_attrs, unreg)`；我们未导入其 ODS 的导入节点计入“需预言机”。

## 3. 检验（model/modular_test.py，需 corpus/）

- **M1 单元**：core 原生 func.call —— 无上下文 ⇒ 不发射；有上下文 ⇒ 发射且通过；实参类型错 ⇒ 拒并归位到调用节点；自递归无需声明。5/5。
- **C2（corpus_test，C2 段打开预言机，C3 仍关闭以免 MLIR 判自己）**：3535 函数 / 3487 有体全部接受，**误拒 0**；预言机调用 2982 次，需预言机而不可发射的体只剩 4 个（dense_resource 3、`llvm.mlir.global` 定义者 1）。修复过程中真实出现过 16 例误拒（入口空块），说明这项检验不是空的；未注册方言体 175 个先表现为“发射缺陷”，改为 `unreg` 送检后消失。
- **M2 体级预言机 ≡ 整模块 verifier**：2108 个合法模块，第 15 轮逐义务变异 + T1（体内 op 结果类型改变），19271 例，真值为 MLIR 对整个变异模块的判定：

| | 例数 |
|---|---|
| 两者都拒 | 18658 |
| 两者都收 | 315 |
| **误拒（我们拒、MLIR 收）** | **0** |
| 无预言机时漏判 | 881 |
| 其中体级预言机抓住 | 583 |
| 漏判 | 298，**全部**在 func.func 体外（transform.named_sequence 体内的 transform.* 结果类型、shard 等） |
| 发射伪影（变异后 SSA 名字层不可解析，跳过） | 267 |

另查：func.func 体置空的 1 例漏判是方言属性校验（`bufferization.writable` 不能出现在外部函数参数上，`verifyRegionArgAttribute`），不属于任何层。

## 4. 定律

- **L40（体级发射忠实）**：对 func.func 体，结构层 ∧ 符号层 ∧ 类型层 ∧ 预言机（体 + 被引用符号的私有声明 + 模块属性）的拒绝 ⊆ MLIR 对整模块的拒绝（M2：19271 例 0 反例）；在 func.func 体内二者相等（体内漏判 0）。
- **L41（可达性是义务的一部分）**：一层的“0 误拒”只对送达它的实例成立，须同时报告送达率与不可送达原因（本轮：C2 需预言机的体只有 4 个不可送达，原因 2 类全部列账）。这是 L38 对预言机层的推论。

## 5. 未解决

1. 非 func.func 的函数体（transform.named_sequence、gpu.func、llvm.func）：发射单位须是该 op 本身（其父约束、终结子不同），M2 的 298 例漏判全在此。下一步：发射单位 = 最近的 IsolatedFromAbove 祖先 + 其符号上下文。
2. 函数自身的 `arg_attrs`/`res_attrs` 未随体发射 ⇒ 方言属性校验（`verifyRegionArgAttribute`，如 bufferization.writable）缺失；方言属性 verifier 目前不在任何层。
3. 方言资源表（dense_resource）未导入，3 个体不可发射。
4. 嵌套符号引用 `@a::@b` 不发射。
