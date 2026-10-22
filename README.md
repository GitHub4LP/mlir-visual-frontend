# mlir-visual-frontend

通用可视化编程前端的规范与原型：前端模型（VIR）对组合封闭，经 MLIR 编译执行（WASM 是后续目标）。

## 目录
- `spec/`：规范。`core-v4.md` 是统一核心（Schema v4、组合律 L6/L7、检验结果）；`definitions.md` 是定义层（attribute/type/property/ADT/interface 由同一机制定义，组合律 L8–L12）；`review-r7.md` 是第 7 轮复盘与新路线；`types.md` 是类型层，`compsig.md` 是复合 op 的多态签名，`import.md` 是 tblgen JSON 导入与 11 个方言的导入账，`oracle.md` 是预言机层与类型判定的三层分工；`shapes.md` 是形状层（tensor，L22–L24）；`edit.md` 是编辑代数与增量查询（洞、L20、L21，增量 ≡ 全量）；`loops.md` 是位宽、index 与循环（L17–L19，与 MLIR JIT 执行的差分）；`transform.md` 是变换层（pattern、fold、终止性、对 MLIR canonicalize 的翻译确认，组合律 L13–L16）；`coverage.md` 是 MLIR 机制覆盖矩阵（52 项）；`structure.md` 是结构元模型（region/block/后继/symbol 的 12 条义务、差分结果）；`frontend-model.md` 是主规范（第 21 轮重写：对象、分层与拒绝权、值层语义、前端动作表）；`laws.md` 是 L1–L52 的证据账本；`closure.md` 是组合封闭的外部检验（语料复合 op、IRDL 导出）；`values.md` 是值层审计扩面（属性账、IW=32 目标）；`derivation-v3.md` 是第 1–4 轮的推导规则原文；`composition-laws.md` 是机制 × 组合律表；`type-algebra.md` 是类型代数与各原型结果；`semantic-overlay.md` 是 arith 语义覆盖层；`vertical-slice.md` 是端到端切片的结果和缺口。
- `model/`：`core.py` 是统一核心 v4（一种 IR、一种 Schema、一次推导三层事实），`core_test.py` 是预言机与跨层检验，`defs.py` / `defs_test.py` 是定义层及其与 MLIR 的差分，`transform.py` / `transform_test.py` 是变换层，`loop_test.py` 检验位宽、index 与循环，`edit.py` / `edit_test.py` 是编辑代数与增量查询，`shape_test.py` 检验形状层，`typesig.py` / `typesig_test.py` 是类型层（与 MLIR verifier 差分），`compsig.py` / `compsig_test.py` 是复合 op 签名，`odsjson.py` / `odssig.py` / `odssig_test.py` 是 tblgen JSON 导入、自动签名与差分，`oracle.py` / `oracle_test.py` 是预言机层，`structure.py` / `structure_fuzz.py` 是结构层义务与 MLIR 差分；尚未并入核心的层原型：`solver.py`（类型合一）、`shape.py` / `wp.py`（下标、extent、循环）；`ods_import.py` 是 ODS 导入器，`vslice.py` / `vslice_demo.py` 是端到端链路，`difftest/` 是针对真实 MLIR 的差分测试。

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
PYTHONPATH=model python3 model/sem_test.py 300 400        # 值层外部审计（第 21 轮，laws.md §2–3）
PYTHONPATH=model python3 model/edit_test.py 40 12
PYTHONPATH=model python3 model/shape_test.py 300
PYTHONPATH=model python3 model/typesig_test.py 300
PYTHONPATH=model python3 model/compsig_test.py 90
sh scripts/fetch_tools.sh && PYTHONPATH=model python3 model/odssig_test.py 40
PYTHONPATH=model python3 model/oracle_test.py 1000
PYTHONPATH=model python3 model/vslice_demo.py
```


## 独立差分：MLIR 自己的测试集（第 12 轮）
```
sh scripts/fetch_corpus.sh            # mlir/test/Dialect/*，与绑定同一提交，不入库
PYTHONPATH=model python3 model/corpus_test.py
```
见 spec/corpus.md。

## 符号层（第 14 轮）

符号定义者 / 符号表 / 使用者取自 ODS 的 trait/interface 闭包，名字、可见性、引用取自真实属性（model/symtab.py）；结构层 S1–S4 以整模块判定。
复盘：此前 S1–S4 只读手工字段，真实 IR 上从未触发。检验：`PYTHONPATH=model python3 model/symtab_test.py 60`（需 corpus/）——Y1 2108 个模块误拒 0（修前 2031），Y2 符号变异 6089/6089 与 MLIR 一致。见 spec/symbols.md。

## 属性层（第 13 轮）
```
PYTHONPATH=model python3 model/attrsig_test.py 3     # A0 导入账 / A1 真实属性不误拒 / A2 变异差分（需 corpus/）
```
属性值由 MLIR 属性解析器承载，属性参数与约束从 ODS 导入（model/attrsig.py）；core 参数 → MLIR 属性只有一个翻译函数 attrsig.from_core。见 spec/attributes.md。

- 第 15 轮：结构义务审计（spec/obligations.md，model/obligations_test.py）：R1–R8、S1–S5 在真实 IR 上的触发数与逐义务变异差分（8858 例，误拒 0，漏判 249 全部归因于未导入方言或 C++ declared 义务）。
- 第 16 轮：预言机按模块上下文发射（spec/modular.md，model/modular_test.py）。复盘：C2 此前一直关着预言机、27% 的真实函数体到不了预言机。修后导入节点逐字发射、被引用符号补私有声明；C2 打开预言机误拒 0；M2 19271 例变异对照整模块 verifier 误拒 0，func.func 体内漏判 0（剩余 298 例全在 transform.named_sequence 等非 func 体）。
- 第 17 轮：发射单位 = 隔离单元（函数类符号 op 本身 + 祖先外壳 + 被引用符号声明，`core.check_module`）。M3 19271 例变异对照整模块 verifier 误拒 0、漏判 0；这是发射忠实与局部性的证据（L42），不是符号层的精度——符号层自身漏判仍 881。
- 第 18 轮：先归因后导入——导入 affine/gpu/bufferization/linalg/transform/nvgpu/shard 的 ODS；新层在真实 IR 上先误拒（S5 外推 126 例、shard 属性 6 例），全部修正后 M3 19641 误拒 0，符号层自身漏判 4.57% → 3.55%。环境：`scripts/setup_env.sh`。
- 第 19 轮：TypeDef 类型原子（封闭、精确）与接口原子（开放世界，L43 只能正向接受）；用同一提交的 mlir-linalg-ods-yaml-gen 生成 linalg 命名 op 的 .td（linalg 25 → 99 op）；修 attrsig 静默异常。M3 误拒 0 漏判 0，符号层自身漏判 3.55% → 2.95%，Z2 漏判 228 → 120；残差全在 ODS 之外（L44）。
- 第 20 轮：增量重验 L45——单元判定的缓存键 = 子树 + 外壳 + 被引用声明；失效集在编辑点局部可算。12958 次真实 IR 编辑：失效预测违例 0、增量 ≡ 全量；拆掉符号层后 104/144 次拒绝只有依赖单元能抓住。测试 `model/incr_test.py`。

## 第 21 轮：证据审计（spec/laws.md）
L1–L45 逐条标证据等级（A 外部语料 / B MLIR 差分 / C 自证 / D 定义 / ✗ 已否定）。审计否定两条：L17′（index 溢出 = 毒值，与 LangRef 回绕矛盾，更正为 L17″）、L32（已被 Z2 否定）；并发现 nsw 从未建模（不可靠方向）。
sem_test.py 以 MLIR 自身为真值：V1 无标志程序 vs JIT；V3 定向 + V2 随机：canonicalize + int-range-optimizations 的结果须为模型下的精化（旧语义 V3 反例 22/1500，新语义 0/1500）。

## 第 22 轮：组合封闭的外部检验（spec/closure.md）
复合 op 从 MLIR 测试集抽取（10710 个切片，19 方言），在原类型与变异类型上对照 MLIR verifier。判定改由有拒绝权的层承载（model/composite.py）；第 8 轮承载层 typesig 在真实切片上误拒 24%。抓出 L28 的修饰缺陷（tensor 编码、vector 可伸缩维），修复后全 auto 复合 op 10974/10974 ≡ MLIR。不透明身份：导出为 IRDL op（model/irdl_export.py），误拒 0，只抓住 3.2%。
```
PYTHONPATH=model python3 model/compcorpus_test.py 12 6    # 需 corpus/，约 25 分钟
```

## 第 23 轮：值层审计扩面（spec/values.md）
语义属性账与 ODS 对账，抓出 divui exact、trunci nsw/nuw 被静默忽略（第 21 轮 nsw 的同类缺陷）；补移位、divsi/remsi/remui；未读参数 ⇒ 值层不给事实。JIT（O0/O3、index 位宽 64 与 32）反例 0。发现 MLIR 的 index 折叠在 32 位 index 下误编译（canonicalize/int-range-optimizations/sccp 各 36/363），index 方言 CmpOp::canonicalize 对 8 个谓词错误。
```
PYTHONPATH=model python3 model/values_test.py 150 300 120    # 约 15 分钟
```

## 第 24 轮：端口投影改到有拒绝权的层（spec/closure.md §7）
- `composite.project`：判断“端口 p 接类型 t”有无可能补全，只用必要条件，所以判 no 是可靠的。依据有三类：
  - 自动签名的端口谓词；
  - 已知位置上的相等类关系；
  - 操作数全部已知的节点的完整判定。
- `model/projcorpus_test.py`：在 MLIR 真实切片上，用补全搜索去证伪投影的 no。结果：
  - 不可靠反例 0 / 24070；
  - 其余端口已知时，投影 ≡ Φ_ods（14821/14821）。
- `odssig.check` 的 SameOperandsAndResultType 改为与 MLIR 一样，逐个对照 result(0)。
- `model/irdl_boundary_test.py`：IRDL 边界 14 例，崩溃在子进程里隔离。

```
PYTHONPATH=model python3 model/projcorpus_test.py 6 3 6
PYTHONPATH=model python3 model/irdl_boundary_test.py
```
