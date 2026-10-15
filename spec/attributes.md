# 属性层（第 13 轮）

## 1 原则：不手写，导入
第 12 轮的结论是：凡是手写的文法和假设，在真实 IR 面前都出了错；凡是从 MLIR 导入的，都没有出错。属性层照此办理。

| 部分 | 来源 |
|---|---|
| 属性值的文法与查询 | MLIR 自己的属性解析器（`ir.Attribute.parse`）。属性值就是 MLIR Attribute 对象，core 不另建属性项 |
| op 的属性参数集合 | ODS `arguments` 中的属性项：名字、必需 / 可选 / 缺省值 / Unit、约束 Pred 树 |
| 方言属性类的 isa | ODS AttrDef 记录（`cppNamespace::cppClassName → attrName`），兼容 MLIR 的两种打印形式（`#arith.fastmath<…>`、`#index<cmp_predicate …>`） |
| 特化枚举属性（`genSpecializedAttr` 的 I32/I64EnumAttr，如 `CmpFPredicateAttr`） | 对应的 C++ 类就是“带取值检查的 IntegerAttr”。位宽、取值集合、BitEnum 的 validBits 都从 EnumAttrInfo 记录读出 |
| 接口原子 | TypedAttr、PoisonAttrInterface：按 MLIR 的实现做 cpred 级登记，由 A2 审计；DeviceMappingAttrInterface 的实现在未导入的方言里，因此取上近似 |
| 取值域（给前端） | 枚举的取值名、缺省值、约束摘要（`attr_args(op)[i]['domain']`） |

实现：`model/attrsig.py`。导入阶段解析不了的原子抛 `Unknown`，记入导入账，不会拖到求值时才暴露。求值时原子无定义，也单独记账（`STATS`）。

导入账（A0）：12 方言共 225 个属性参数，其中精确 224 个，上近似 1 个（`scf.forall.mapping`），未知 0 个。

## 2 core 参数的翻译只有一处
`attrsig.from_core(n)`：把 core 节点的参数翻成 `{ODS 属性名: MLIR 属性文本}`，`oracle.emit_core` 和 `core.typecheck` 都用它。
- 已导入的节点（`props['_attrs']`）原样返回。
- 符号参数（`$k`、z3 模式变量）抛 `Symbolic`：它们不是具体实例，不下结论。
- core 参数在 MLIR 上没有对应属性（如 `xori` 带 `nuw`）抛 `NotMLIR`：这样的实例无法降级到 MLIR，属于真错误。这一条落实了第 11 轮的发现：core 的参数空间比 MLIR 大。

## 3 判定位置
`core.typecheck` 中，属性层与自动签名同级（`attr:` 诊断），拒绝即终判。依据是 L33：它在真实 IR 上已证明不误拒（A1、C2）。
- 已导入节点要检查：必需性、约束、属性与类型的关系。
- core 形式的节点不查必需性，因为 core 的简写由降级补齐（例如切片的 `static_*`）。
- 关系：12 方言中引用属性的 ODS op 级关系只有一处，即 `arith.constant` 的 `AllTypesMatch<[value, result]>`。由 `rel_check` 通用地处理：属性的类型取 MLIR `attr.type`。

## 4 外部真值检验（`model/attrsig_test.py`、`model/corpus_test.py`）
- **A1**：MLIR 测试集中 MLIR 接受的段里，共 14065 个 op 实例，**属性层误拒 0**。修正前误拒 34 个，全是 `#index<cmp_predicate …>` 的打印形式不匹配。
- **A2 变异差分**：以真实 op 实例为底，每个属性参数换成取值池中的值或删除，共 18145 个变异，交给 MLIR 判定。
  - 我们拒、MLIR 收：**0**（L34a）。
  - MLIR 以 ODS 属性层诊断拒绝（`requires attribute` / `failed to satisfy constraint`）而我们接受：9 个，全部是上近似的 `scf.forall.mapping`（L34b：除登记的上近似外，对 ODS 属性层精确）。
  - 修正过程中 A2 抓出两处问题：
    - 特化枚举的 isa 原本在求值时才报未知，被静默当成接受，导致 `cmpf predicate = -1` 漏判。现在改为导入时解析。
    - 接口原子漏判 147 个，登记后为 0。
  - MLIR 以 C++ verify 拒绝的 409 个、以 PredOpTrait 拒绝的 17 个，归入 declared，不要求属性层抓。
  - 关系谓词 AllTypesMatch 剩 3 个漏判：值属性的类型是 none（如字符串），取上近似。
- **C2**（corpus_test）：属性层进入 typecheck 后，3487 个合法函数仍然误拒 0。
- **C3**：之前 C3 重建 op 时根本没有重建属性（`{'_raw': …}`），属性层在 C3 上无从检验。第一次运行时，属性层“抓住”的 219 个负例里有 150 个理由与 MLIR 不同：都是重建丢掉了属性，再被我们报“缺必需属性”。这是“答案对、理由错”，不能计为成功。现在 C3 用 MLIR 解析 `<{…}>` 和 `{…}` 重建属性，并逐例核对理由（见 §5 的数字）。

## 5 C3 结果（属性重建修正后）
| | 数 |
|---|---|
| MLIR 拒绝的可重建单 op 负例 | 436（属性文本无法重建 36 例，不参与） |
| 符号层抓住 | 78（第 12 轮报告 71） |
| 其中只由属性层抓住 | 7，理由与 MLIR 一致 7，理由不同 0 |
| 未抓住 / declared | 345：C++ verify，归预言机 |
| 未抓住 / auto | 6：全是符号上下文（func.call 结果类型、func.constant 引用、memref.get_global 引用与类型），属于下一轮 |

中间那次“290 抓住”是虚假的成功。数字涨上去时，第一件事是核对理由，而不是报告数字。

## 6 组合律
- **L34a**（不误拒）：属性层的拒绝蕴含 MLIR 拒绝。依据是 A1、A2 与 C2。
- **L34b**（ODS 层精确）：对 ODS 属性约束（必需性 + 约束 Pred）中不含上近似原子的部分，属性层与 MLIR 一致。
- **L35**（翻译唯一）：core 参数到 MLIR 属性只有一个翻译函数。发射器与判定若各写一份，就是第 10、11 轮自证偏差的来源。

## 7 对前端的指导
- 属性编辑器的控件由取值域推导：枚举给下拉，BitEnum 给多选，DefaultValued 给缺省值，Unit 给开关，DenseI64Array 给数字列表。
- 诊断可以直接对应 MLIR 原话里的属性名。
- 前端不能提供 MLIR 没有的参数（`NotMLIR`）。core 的值层扩展（如 `xori nuw`）要么删掉，要么降级前消去。

## 8 剩余
- 属性与类型、属性与属性之间、出现在 C++ verify 中的关系（切片的秩、`in_bounds` 与 `permutation_map`、`complex.constant` 的元素类型）属于 declared，由预言机负责；需要精度时再做 cpred 登记。
- 方言资源表；`DeviceMappingAttrInterface`（需导入 gpu 等方言）。
- core 的参数仍是自定义名（`nsw`/`nuw`/`predicate` 整数）。翻译已经统一，但模型内部还没改用 ODS 名字。下一步应让 core 节点直接以 ODS 属性为参数，现在的参数变成值层视图。
