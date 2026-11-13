# 保证与边界（收口文档，2026-11-13，模型 = PR #97 d5df4ff，MLIR 9bae84b0）

本文件回答三件事：模型对前端**保证什么**、**不保证什么、为什么**、**怎么一键复核**。细节与各轮历史见 `closure.md` §9.x 与 `clauses.md`。

## 1. 需求 → 模型
| 需求 | 模型中的落点 |
|---|---|
| 充分建模 MLIR 类型约束、op 签名、Traits、属性及交互（10-13） | `odsjson`/`odssig` 从 ODS 自动生成 650 个 op 的签名（类型约束、trait、属性约束、可变端口切分）；`nfrules` + `cppverify` 解释执行 C++ verify / inferReturnTypes / 接口 verify |
| 自定义 op 由 op 图实现，与原生 op 同一套建模，图定下后能推出签名间关系与最大类型范围（10-13） | `odscomp.derive` 由图得 Φ_comp（原子合取 + 残差）；`core.typecheck` 对原生与复合 op 是同一判定；`range3/project` 给单端口三值范围 |
| 只做模型（10-14） | 无前端 / UI / WASM 代码 |
| C++ verify 与接口 verify 规范化进模型（11-04） | `cppverify` 解释执行 9bae84b0 源码（三值，静态完备 ⇒ 移出残差） |
| 批量，不逐 op 手写（11-05） | 解释器 + ODS 自动导入；手写 nf 规则仅早期少数 op |

## 2. 保证（C1–C3）
三值判定：**接受 / 拒绝 / Unknown**。Unknown 只在残差非空或单点无法判定时出现，前端应交 MLIR 校验。

- **C1 确定结论与 MLIR 一致**（2026-11-13 全量回归 25 段，`work/r89_regress.md`）：
  - 误拒 0：S3 27391 例、R2 18281 例、C1 45170 次比较、IF1 12669、I1 2332、N2a 40225、N2g-a 4266、N1 57204。
  - 残差为空时误收 0：S3、R2、C5（46757 次比较）、N2d（37293）、N1、N2e、N2f、N2g-d。
  - 诊断漏判 0：C2（4159 条 verify 诊断）、IF2、I2、N2b、N2g-b。
  - 这是**语料 + 变异上的经验保证**，不是证明：只覆盖语料出现的映射、tile、形状；MLIR 崩溃按拒绝计。
- **C2 同一套建模**：复合 op 与原生 op 走同一 `typecheck`；S4 嵌套≡展开 0 不一致（6840 例）。
- **C3 Φ_comp 与逐点判定等价**：S2 不一致 0/27960，R1（含区域）0/18875；S1 化简≡原始 0。
- 范围：Φ_comp 的“最大性”对顶层推导与入口边定义的区域类型流成立（以 ODS 推断与字面量为界）。

## 3. 边界（C4）：残差逐条归因
650 op 中残差为空 468；残差条目 611，**全部落入下列有名边界，未分类 0**（`scripts/attr_residual.py`，按源码不支持构造的正则分桶，分桶本身是启发式、不是判据）。

| 边界 | 条目 | 性质 |
|---|---|---|
| Transform 方言（96 个 op） | 495 | 元编程 op，端口是 payload IR 句柄；TransformOpInterface verify 与 Transform native trait 未解释 |
| 父 op / 区域 / 块结构 | 38 | 约束依赖所在父 op 或区域内容，不是端口类型（affine 作用域、gpu 启动体等） |
| 无 C++ 源码（生成代码接口） | 13 | RegionBranch / LoopLike 等接口 verify 在生成代码中 |
| 区域 verify（verifyRegions） | 6 | 区域结构 |
| 解释器词汇未覆盖 | 29 | 可批量补（如 WidthComparator/type_list 模板、RankedTensorType::Builder、operand_begin） |
| PredOpTrait 的 $_op 谓词 | 10 | 跨端口 C++ 谓词未编译 |
| ODS 类型 CPred 未编译 | 8 | |
| 属性 CPred 未编译 | 5 | |
| native trait 未建模 | 5 | SameOperandsAndResultRank 4、HasDefaultDLTIDataLayout 1 |
| TypesMatchWith 变换未解释 | 2 | func.call_indirect |

- 非 Transform 残差共 116 条（86 个 op）：57 条属结构层，59 条属可规范化而未做。
- 单点 Unknown（残差为空但单端口范围无法判定）：段一 243/15767≈1.5%、段二 1866/25034≈7.5%（第 88 轮）。成因：规则先读其余端口再读本端口，抽象求值遇读取即停（§9.28）。

## 4. 一键检验（C5）
```
bash scripts/setup_env.sh          # 安装 mlir、z3；语料与 C++ 源码见 scripts/fetch_corpus.sh、fetch_cpp.sh
bash scripts/check_all.sh          # 25 段，2 CPU 约 3–4 小时，日志在 logs/check/
```
- 退出码 0 = 通过。`check_summary.py` 把可靠性字段（误拒、残差为空误收、不一致、违例、漏判、见证不成立、推导异常、文件崩溃）非零视为硬失败；故障注入（把一个“误拒 0”改成 3）已验证会被抓出。
- `scripts/check_known.txt` 列出 11 条**只因数量阈值不足**的已知 FAIL，可靠性字段都是 0：regflow F2（增益 15、4）、F4 段二（10）、cppinfer J2（18<20）、winograd N2f（filter 接受 25<30）、projexact Y1 段二（比较 289）、cppverify I1/I2 后三段（按合计判：I1 2332 误拒 0，I2 176 漏判 0）。
- L1 是 ODS 单层差分，误收按设计允许；N1 的“消融误收”是对照组。

## 5. 不保证 / 已披露
- 端口范围闭式精确表示（上近似 29.6%）、覆盖所有长尾 op、Unknown=0，均不在保证内。
- 静态完备 op 运行时若遇 CppUnknown，非严格 typecheck 放行（C3 为守卫）。
- `type(x) is …` 类检查可绕过未绑定哨兵，仅由 P5 池穷举经验守卫。
- 值层（常量、毒值、UB）冻结，不测；DataLayout 按默认；pack/unpack 值前提只认 arith.constant；Graph 区域副作用次序、无结果 op 未覆盖。
- SUBST_MATCH 增益 0.7pp 未达 2pp 判据，仍开启（§9.26–9.27）。第 88 轮 W2 首跑 FAIL 后修复重判为 W2'（事后修复，§9.28）。
