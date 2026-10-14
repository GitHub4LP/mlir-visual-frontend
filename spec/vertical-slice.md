# 纵向切片 v0（model/vslice.py、vslice_demo.py；输出见 model/vslice_out.mlir）

运行：在项目根目录执行 PYTHONPATH=model python3 model/vslice_demo.py（依赖 mlir-python-bindings）

## 已打通
图 JSON → HM 合一（带类约束，增量单位是定义）→ Schema（∀ 类型变量:类. 签名、效应、能力）→ 诊断与补全 → 单态化生成 MLIR（每个节点带 loc）→ scf/memref/arith/cf/func 降到 LLVM → ExecutionEngine 执行。

## 结果
- 执行结果：fib(10)=55、fact(10)=3628800、sumsq(10)=285、axpy(2,3,4)=10
- 推导：axpy 为 ∀T:Float.(T,T,T)→T，Pure；fact 因为递归、终止性未知，判为 NotSpeculatable；sumsq 的局部 alloca 效应相互抵消，结果为 ∅。
- 诊断：类型冲突、类约束、悬空端口都定位到节点的哪条入边。后端 verifier 的错误经 loc 回映到“函数:节点”。
- 补全：原生 op 与复合 op 一起列出。
- 增量：3000 节点的定义首次推导 10ms；编辑它的使用者时命中缓存，0.04ms；编辑它的内部 9ms。路径压缩之前是 120ms。

## 切片暴露的、做“通用”必须补的缺口（按优先级）
1. 数据抽象：MLIR 核心方言没有 struct/ADT、字符串、闭包、多态。目前多态由前端单态化处理；其余几项需要自定义方言（候选：IRDL 运行时定义）加上 lowering。
2. 控制流：scf.while、break、提前返回、非结构化 cf；递归目前只支持单态递归。
3. 推测性与越界：load/store 目前一律判为不可推测，需要接入 core 的 Pre（双类：BV 和 ℤ），放在后台档运行。
4. WASM 后端：沙箱里没有 clang/llc，需要 MLIR → LLVM IR → wasm32 的工具链。
5. 编辑器协议：把图 JSON 的 schema 和查询 API（诊断、补全、悬停时显示 Schema）定下来，做成类似 LSP 的接口。
