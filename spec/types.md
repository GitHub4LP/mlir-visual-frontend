# 类型层 v0

实现：`model/typesig.py`。检验：`model/typesig_test.py`（`PYTHONPATH=model python3 model/typesig_test.py 300`，约 1 分钟）。
接入 core：`core.typecheck` 在值层之前运行；有违例就写入 `Schema.type_diag`，值层不运行。

## 1 签名 = 端口约束 + 端口间关系，按来源分级
- **ods**（从 .td 自动导入）：
  - 端口约束与个数（one/opt/var），模板参数沿继承链替换，例如 Arith_IToICastOp → Arith_CastOp 的 From/To；
  - 已知的原生 trait：SameOperandsAndResultType、SameTypeOperands、AllTypesMatch、SameOperandsAndResultShape、ElementwiseMappable，以及变换子已知的 TypesMatchWith；
  - ODS 自动生成的 InferTypeOpInterface 对应的条件：结果必须**等于**推断出的类型。
- **cpred**（.td 里的 C++ 片段，人工改写成原子组合）：SameInputOutputTensorDims、BooleanConditionOrMatchingShape。
- **declared**（只存在于 C++ verify() 里的条件）：
  - 整数转换的宽度方向、index_cast 的两侧；
  - extract/insert 的下标个数 = 秩；
  - empty/splat 的动态尺寸个数；
  - tensor.cast 的相容性；
  - extract_slice/insert_slice 的秩与 segment。

## 2 MLIR 的实现与 trait 名字不一致的地方（差分中发现）
- SameOperandsAndResultType 的实现不是“类型相等”，而是“元素类型相同、形状相容”。但 ODS 同时自动生成了 InferType，它要求结果等于第一个操作数。实际效果：`subi(tensor<?>, tensor<4>) -> tensor<?>` 合法，`-> tensor<4>` 不合法。
- ElementwiseMappable 要求所有非标量端口是同一种基类型：有秩 tensor 和无秩 tensor 算不同的基类型。
- AllShapesMatch 是 getShape 逐项相等，不是“相容”。

## 3 MLIR verifier 自身的缺陷
测试所用的 wheel 中，有 204 个实例让 verifier 断言崩溃，分两类：
- arith 的转换类 op 遇到无秩 tensor（例：`arith.trunci : tensor<*xi64> to tensor<*xi16>`）。SameInputOutputTensorDims 对无秩 tensor 调用了 getShape；main 分支的 .td 写法相同。
- `arith.select` 的条件是 shaped，而结果是标量。
前端对这两类一律拒绝（204/204），因为降低出去的程序会让 MLIR 崩溃。

## 4 检验结果
- T1 与 MLIR verifier 的差分：23 个 op，每个 op 300 个实例（合法模板加一处变异），共 6696 个；其中 MLIR 接受 1488 个。

  | 使用的签名部分 | 一致 | 误收（前端放行、MLIR 拒绝） | 误拒 |
  |---|---|---|---|
  | 只用 ods | 6114 | 582 | 0 |
  | ods + cpred | 6132 | 564 | 0 |
  | ods + cpred + declared | 6696 | 0 | 0 |

- T2 推断：推出结果类型的 239 例全部被 MLIR 接受；182 例推不出，因为结果类型由用户选择（例如 trunci 的目标宽度、cast 的目标类型）。前端在这些地方应当给出补全，而不是推断。
- T3 代价：connectable 0.4 µs；candidates（23 个 op）33 µs；单节点检查 6 µs。比值层查询快三个数量级。
- T4：病类型图得到诊断，不再导致 z3 异常。
- 回归：core、transform、defs、loop、edit、shape 全部通过。接入时发现 transform.py 的一处病类型（右侧结果的位宽），已修复。

## 5 缺口
- 复合 op 只有单态签名：端口类型由用户给定，还没有类型变量和 ∃ 消去（solver.py 的原型尚未并入）。
- 只导入了 23 个 op；静态 offset/size 属性的验证属于 attribute 层，没有做。
- 类型池里没有 vector 的 0 秩和 scalable 维、memref 的布局与内存空间，也没有 encoding。
