# 类型代数 v1：MLIR 具名类型约束的规范化
映射数据：model/type_constraints_nf.json，由 model/normalize_types.py 生成。CommonTypeConstraints.td 中的 211 个具名定义已全部映射，其中原子 33 个、组合子 16 个、模式 162 个。

## 1 类型项（内置部分）
```
τ ::= int(w, sign) | index | float(fmt) | complex(τ) | none | token
    | vector(d̄: (n, scalable)⁺, τ) | rtensor(d̄: (n|?)*, τ, enc) | utensor(τ)
    | rmemref(d̄, τ, layout, ms) | umemref(τ, ms) | tuple(τ̄) | function(τ̄→τ̄) | graph
    | opaque(dialect, name) | ext(dialect, K, p̄)        # 第三方类型
```

## 2 原子词汇（库中 71 处 CPred 全部可以归约为以下 12 类）
| 原子 | 论域 | 理论 |
|---|---|---|
| kind∈K | 有限集合 | 布尔代数 |
| sign=s, fmt=f, scal(none/any/trailing-only/all) | 有限集合 | 布尔代数 |
| width∈S / ≥n | ℕ | Presburger |
| ranked, rank∈S / ≥n | ℕ | Presburger |
| dim(i)∈S（i 是常量）, static ≡ ∀i<rank. dim(i)≠? | ℕ∪{?} | Presburger（rank 有界时展开） |
| shape=[..] | ℕ* | 等式 |
| elt:C, elts∀:C, μX（NestedTupleOf） | 项 | 合一 + 正则树约束 |
| strided | layout 上的判定过程 | 布尔原子 |
| opaque(d,n) | 字符串 | 等式 |
| valsem | **开放**：内置的 {rtensor, utensor, vector} ∪ 第三方声明了该 trait 的类型 | 开放世界原子 |
| numel∈S / ≥ / ≤ | ∏dims | **非线性** |
| dim(norm(n))，n<0 | 下标依赖于 rank | 按 rank 分情形 |

## 3 封闭性结论（已核实）
- 211 个定义中，有 **190 个**落在“有限域 + Presburger + 合一 + 正则树”这一组合理论里：对 ∧、∨、¬、∃ 封闭，蕴含可判定。因此由它们推导出的复合签名是**主签名（E）**。
- 例外一：**numel**，涉及 17 个定义（VectorOf*Length*）。它是维度的乘积，非线性。处理办法：在具体类型上直接求值；维度是符号时，把 numel 当作单调的未解释函数，推导降为 S。另外 vector 的维度在构造时必然是静态的，所以只有在端口未定时才会出现符号情形。
- 例外二：**负下标的 dim**，涉及 4 个定义（ShapedTypeWithNthDimOfSize 等）。在 rank 已知或有界时按情形展开，可回到 E。
- 开放世界原子：valsem，以及 VectorElementTypeInterface、MemRefElementTypeInterface 这类接口成员关系。它们只能保守处理：当边界变量是未知的第三方类型时，保留为残余原子，推导结果为 S。
- 推论：**内置约束库可以做到零 opaque**。opaque 风险全部来自第三方 dialect 的 CPred 和 C++ verify()。

## 4 前端约束语言 = 上述理论（这是规范化的结果）
- 前端节点库只能使用第 2 节的原子，加上类型变量和函数依赖。
- 导入第三方 dialect 时，每个 CPred 都必须登记成以下三者之一：(a) 改写为原子组合；(b) 声明为新的有限域或开放原子，并附带判定函数；(c) 标记为 opaque，推导降为 S，并在 UI 上提示。

## 5 推导原型 v0（model/solver.py，运行通过）
做法：合一 + 谓词归约（结果为真、假或残余）+ ∃ 消去（只保留边界变量）+ 格上吸收。trait 当作蕴含查询。
| 复合图 | 推出的主签名 | ⊨SameOperandsAndResultType |
|---|---|---|
| axpy = addf(mulf(a,x),y) | ∀T. (T,T,T)→T, FloatLike(T) | ✓ |
| matmul+bias | ∀m,n,k,Ta,Tb,Tc. A:m×k×Ta, B:k×n×Tb, C0,bias:m×n×Tc → m×n×Tc; Numeric(Ta,Tb) ∧ Float(Tc) | ✗ |
| matmul + A 当 bias | 迫使 k=n，B 收窄为 n×n | ✗ |
| addf(tensor<4xi32>) | 拒绝：FloatLike 不满足 | — |

已核实的发现：
1. **linalg.matmul 的 ODS 只声明了 Variadic<AnyType>、Variadic<AnyShaped>、Variadic<AnyRankedTensor>。** 形状关系全部来自 indexing_maps 属性和 C++ verify()。因此导入器必须把 indexing_maps（它是数据）翻译成 Presburger 约束，这部分是 E；“元素类型可以 cast”来自 C++，登记为声明原子（D）。
2. 主签名比手写的更宽：matmul+bias 的三个元素类型彼此独立，例如 i8×i8→f32 合法。手写成“∀T”反而是过窄的。
3. 维度应该建模为运行期 extent，静态类型中的 ? 只是它的投影。否则“相容”关系不可传递，∃ 消去会丢掉信息（∃y. x~y ∧ y~z ≡ ⊤）。

v0 的局限：维度只支持等式（靠合一）；仅涉及内部变量的残余谓词，没有检查可满足性就丢弃了；extent 与静态投影尚未分开。

## 6 原型 v1（model/shape.py：合一 + isl Presburger 投影，运行通过）
| 复合图 | 推出的 Φ（边界上） | 意义 |
|---|---|---|
| slice∘slice | o1≥0 ∧ n1≤x−o1 ∧ 0≤o2≤n1−n2 ∧ o2<n1 | Φ_comp ⊆ Φ_fused 判定为真，所以“融合为单个 slice”这个改写可以**机器验证** |
| crop(pad(x,l,h)) | 推出结果 extent = 输入 extent | 推出 ⊨SameOperandsAndResultShape；MLIR 在动态情形下只能给出 tensor<?> |
| tensor<8>[6:+4] | UNSAT | 静态错误，与 MLIR 一致 |
| tensor<?>[6:+4] | 推出前提 x≥10 | MLIR verifier 直接放行，运行期是静默 UB；前端把它变成边界前提或运行期断言 |
| slice(matmul, r×r) | r≤m ∧ r≤n，k 不受约束 | 形状约束从 indexing_maps 导入，并穿过 ∃ 保留下来 |

已核实的发现：
1. **MLIR 的怪癖会污染推导结果。** verifyInBoundsSlice 即使在 size=0 时也要求 offset<dim。于是 crop(pad) 推出了 h+x>0，slice(matmul) 推出了 m,n>0，这些都不是语义真正要求的。根因是 Wf 和 Def 混在了一起，解决办法见 frontend-model §3 的三谓词。
2. **MLIR 的静态检查只覆盖“全静态”的情形。** 负 offset、stride、动态 offset 都不检查；pad 的 verifier 在推断结果为动态时任何结果类型都接受。前端的 Def 层严格强于它。
3. stride 为符号时，(size−1)·stride 是非线性的，只能保留一个可靠的弱化，推导为 S；stride 为常量时（最常见）是 E。

## 7 原型 v2（model/wp.py：三谓词 + 最弱前提 + 控制层，运行通过）
| 用例 | 结果 |
|---|---|
| 逐 op 引理：extract_slice 的 Def ⇒ Wf | ✗，反例 d=o=n=0：MLIR 拒绝 tensor<0xf32> 的空切片。加上规范化（n=0 → tensor.empty）后全静态情形成立；“d、o 静态而 n 动态”的情形无法靠规范化修复，lowering 时要把 offset 发射为动态值（MLIR 自己的 fold pattern 也用 verifyInBoundsSlice 回避了同样的问题） |
| crop(pad) | 前提为 x,l,h≥0，v1 中混进来的怪癖条件已消失；Post 为 y=x |
| addf(x[1:], y) | Pre：y=x−1 ∧ x≥1（elementwise 的动态形状不匹配属于 UB，进入 Def） |
| if(n<4) x[0:+n] else x[0:+4] | Pre：n≥0 ∧ ((x≤3 ∧ n≤x) ∨ x≥4)，精确 |
| if(数据相关) 同上 | Pre：x≥4 ∧ n≤x（两支的合取）；用 ∃ 则得到析取，是不可靠的 |
| for t 次: x←x[1:] | Pre：x≥t；Post：y=x−t；exact |
| for t 次: x←pad(x,0,1) / x←x+x | y=x+t / y=x；exact |
| 减半、依赖 iv 的收缩 | isl power 的 exact=False，只能得到过近似（S）。过近似的可达集合给出的 Pre 偏强，但仍然可靠 |

结论：
1. **∃ 消去只适用于静态变量。** 运行期量必须求最弱前提。Presburger 对补集封闭，所以最弱前提依然可以精确计算。
2. **控制层的精度边界**：if 是 E；loop 只有在循环不变或平移型 carried 时是 E；一般情况下 Presburger 对迭代不封闭。补救办法是声明不变式 I，检查 I(x) ∧ R ⇒ I(f)，这个检查可判定。这与“声明 = 对推导的收窄，需要检查”的原则一致。
3. Wf 不需要在每个复合 op 上重新推导，而是**逐 op 在导入期一次性证明** Def ∧ Canon ⇒ Wf。证不出来的地方就是 lowering 必须处理的特例。

## 8 原型 v3：毒值与效应（model/effects.py，运行通过）
| 复合 | Pre | 可推测 | 输出可能为毒值 |
|---|---|---|---|
| a / (a +nuw b) | a+b≤255 ∧ a+b≠0 | ✗（仅在 Pre 下） | 永不 |
| a / (a + b)（回绕） | (a+b) mod 256 ≠ 0 | ✗ | 永不 |
| a +nuw b | ⊤ | ✓ | a+b≥256（相当于推出了 nuw） |
| slice(x,0,n) | ⊤ | ✓（与 MLIR 的 Pure 一致） | n>x |
| extract(slice(x,0,n),0) | 0<n≤x | ✗ | 永不 |

效应：临时缓冲组合后只剩 {Read in0, Write in1}；逃逸时为 {Allocate out0, …}；泄漏会被报告；含无接口 op 时为 ⊤。

已核实的 MLIR 不一致：tensor.extract_slice 标为 Pure（TensorOps.td），而 verifier 不检查动态越界；tensor.dim 却在索引未知时返回 NotSpeculatable（TensorOps.cpp DimOp::getSpeculatability）。文档没有说明越界切片是 UB 还是毒值。前端的选择是**毒值**，这与 Pure 一致；读取毒值才是 UB。这是语义决定，需要写进 lowering 义务：不能把越界切片降为会立即访问内存的形式。

## 9 统一核心 v3（model/core.py + ods_import.py）与真实 MLIR 差分测试
- ods_import.py 解析继承链。从真实的 ArithOps.td、TensorOps.td 中正确导入了 addi、subi、divui、dim、extract、extract_slice 的参数和 traits。生产实现应改用 llvm-tblgen -dump-json。
- core.py：Schema 是唯一的数据结构；复合 op 的推导结果仍然是 Schema。x/(a+nuw b) 的嵌套推导与全内联推导，在 2 层和 3 层嵌套下都判定相等。修掉的 bug 写成了 L1、L2（见 frontend-model §5）。
- 用 MLIR Python bindings（22.0 开发版，2025-11-29，commit 9bae84b0）做真实验证：
  1. `tensor.extract_slice %t[0][0][1] : tensor<0xf32>` → 报错 "offset 0 is out-of-bounds: 0 >= 0"，证实了 §7 中引理找到的反例；
  2. LICM 会把 tensor.extract 提到可能 0 次迭代的循环之外，one-shot-bufferize 之后成为无条件的 memref.load。这是语义剖面加管线约束问题，见 frontend-model §3。

## 10 预言机检验（model/test_oracle.py、model/bv.py）
- 方法：随机生成 2–6 个 op 的 DAG，op 取自 add、add_nuw、sub、sub_nuw、mul、divui、remui、ult、select、常量；用 W 位的带毒值/UB 解释器穷举全部输入作为真值。
- 结果：
  - W=6，300 个图：Sem 精确的 82/82 与真值完全一致（Pre 与毒值集合都相等）；含抽象的 218/218 满足可靠方向。
  - 抽象带来的 Pre 损失：均值 9.6%，最大 100%（例如除数来自 mul 时，整张图都被拒绝）。
- QF_BV 层（W=4，200 个图）：Presburger 层只能做到可靠的 145 个用例，用 QF_BV 全部精确（145/145），其中 31 个可以推测执行。
- i64 规模试探：问“a/(a*b+1) 是否可推测”，z3 瞬间给出反例 a=2^64−1、b=1（乘积回绕后除数为 0）。
- select 的毒值传播不是“任一输入是毒值则输出是毒值”，为此 Schema 增加了 prop 字段，引用 {P:port}。
- 修复：isl 会打印成 3P_b 这样的并置形式，导致占位符替换失败（as_formula）。

## 11 下一步
- 把类型层并入 core，并让 core 正式变成双类（BV 值、ℤ 下标）；
- 语义覆盖层：评估复用 xdsl-smt 的覆盖面，补上 tensor、scf、memref；
- verifier 差分测试（需要 mlir bindings），以及 pass 白名单初稿。
