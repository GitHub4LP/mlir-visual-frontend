# mlir-visual-frontend

通用可视化编程前端的规范与原型：前端模型（VIR）对组合封闭，经 MLIR 编译执行（WASM 是后续目标）。

## 目录
- `spec/`：规范。`frontend-model.md` 是主规范（Schema v3、约束语言、管线契约）；`composition-laws.md` 是机制 × 组合律表；`type-algebra.md` 是类型代数与各原型结果；`semantic-overlay.md` 是 arith 语义覆盖层；`vertical-slice.md` 是端到端切片的结果和缺口。
- `model/`：原型。`core.py` 是统一核心，`ods_import.py` 是 ODS 导入器，`vslice.py` / `vslice_demo.py` 是端到端链路（图 JSON → 推导 → 诊断/补全 → MLIR → 执行），`test_oracle.py` 和 `bv.py` 做语义验证，`difftest/` 是针对真实 MLIR 的差分测试。
- `inventory/`：llvm-project main 的源码快照（ODS/TableGen）。

## 运行
```
pip install -r model/requirements.txt
PYTHONPATH=model python3 model/core.py
PYTHONPATH=model python3 model/vslice_demo.py
```
