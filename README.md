# mlir-visual-frontend

通用可视化编程前端的规范与原型：前端模型（VIR）对组合封闭，经 MLIR 编译执行（WASM 是后续目标）。

## 目录
- `spec/`：规范。`core-v4.md` 是统一核心（Schema v4、组合律 L6/L7、检验结果）；`definitions.md` 是定义层（attribute/type/property/ADT/interface 由同一机制定义，组合律 L8–L12）；`review-r7.md` 是第 7 轮复盘与新路线；`types.md` 是类型层，`compsig.md` 是复合 op 的多态签名；`shapes.md` 是形状层（tensor，L22–L24，L17′）；`edit.md` 是编辑代数与增量查询（洞、L20、L21，增量 ≡ 全量）；`loops.md` 是位宽、index 与循环（L17–L19，与 MLIR JIT 执行的差分）；`transform.md` 是变换层（pattern、fold、终止性、对 MLIR canonicalize 的翻译确认，组合律 L13–L16）；`coverage.md` 是 MLIR 机制覆盖矩阵（52 项）；`structure.md` 是结构元模型（region/block/后继/symbol 的 12 条义务、差分结果）；`frontend-model.md` 是主规范（推导规则、约束语言、管线契约）；`composition-laws.md` 是机制 × 组合律表；`type-algebra.md` 是类型代数与各原型结果；`semantic-overlay.md` 是 arith 语义覆盖层；`vertical-slice.md` 是端到端切片的结果和缺口。
- `model/`：`core.py` 是统一核心 v4（一种 IR、一种 Schema、一次推导三层事实），`core_test.py` 是预言机与跨层检验，`defs.py` / `defs_test.py` 是定义层及其与 MLIR 的差分，`transform.py` / `transform_test.py` 是变换层，`loop_test.py` 检验位宽、index 与循环，`edit.py` / `edit_test.py` 是编辑代数与增量查询，`shape_test.py` 检验形状层，`typesig.py` / `typesig_test.py` 是类型层（与 MLIR verifier 差分），`compsig.py` / `compsig_test.py` 是复合 op 签名，`structure.py` / `structure_fuzz.py` 是结构层义务与 MLIR 差分；尚未并入核心的层原型：`solver.py`（类型合一）、`shape.py` / `wp.py`（下标、extent、循环）；`ods_import.py` 是 ODS 导入器，`vslice.py` / `vslice_demo.py` 是端到端链路，`difftest/` 是针对真实 MLIR 的差分测试。

## 运行
```
pip install -r model/requirements.txt
sh scripts/fetch_inventory.sh   # 拉取原型解析用的 MLIR .td（不入库）
PYTHONPATH=model python3 model/core.py
PYTHONPATH=model python3 model/core_test.py 300
PYTHONPATH=model python3 model/structure.py
PYTHONPATH=model python3 model/defs_test.py 600
PYTHONPATH=model python3 model/transform_test.py 200
PYTHONPATH=model python3 model/loop_test.py 300
PYTHONPATH=model python3 model/edit_test.py 40 12
PYTHONPATH=model python3 model/shape_test.py 300
PYTHONPATH=model python3 model/typesig_test.py 300
PYTHONPATH=model python3 model/compsig_test.py 90
PYTHONPATH=model python3 model/vslice_demo.py
```
