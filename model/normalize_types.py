"""把 MLIR CommonTypeConstraints.td 中的 211 个具名定义映射为规范化原子（NF）。
NF 语法：
  kind∈{..} | width∈S | sign=s | fmt=f | ranked | rank∈S | rank≥n
  | dim(i)∈S | static | shape=[..] | numel∈S/≥/≤ | scal(pattern)
  | strided | valsem | opaque(d,n) | elt:C | elts∀:C | ∧ ∨ ¬ | mult(?,*,**)
类别：ATOM 原子谓词，COMB 组合子/基础设施，PAT 模式定义（实例或参数化类）。
"""
import re, json, sys
src = open('inventory/IR_CommonTypeConstraints.td').read()
names = re.findall(r'^(?:def|class) ([A-Za-z0-9_]+)', src, re.M)

FMT8 = r'F8E\w+|F4E2M1FN|F6E\w+|BF16|TF32'
def nf(n):
    # ---- 组合子 / 基础设施
    comb = {'Type':'C 包装：pred+summary','TypeAlias':'≡ 原约束','DialectType':'C ∧ dialect=d',
      'Variadic':'mult=*','VariadicOfVariadic':'mult=**（段属性）','Optional':'mult=?',
      'BuildableType':'元：可构造（无约束语义）','SameBuildabilityAs':'元：可构造',
      'AnyTypeOf':'∨ Cᵢ','AllOfType':'∧ Cᵢ','ConfinedType':'C ∧ ∧Pᵢ',
      'ContainerType':'container(P) ∧ elt:C','ShapedContainerType':'shaped∧kindP ∧ elt:∨Cᵢ',
      'MixedContainerType':'container(P) ∧ elts∀:C','NormalizeIndex':'元：负索引归一 i<0 ⇒ i+rank',
      'OpaqueType':'opaque(d,n)'}
    if n in comb: return 'COMB', comb[n]
    # ---- 原子谓词
    atoms = {
      'IsVectorOfNonZeroRankTypePred':'kind=vector ∧ rank≥1','IsFixedVectorOfNonZeroRankTypePred':'kind=vector ∧ scal(none) ∧ rank≥1',
      'IsVectorOfAnyRankTypePred':'kind=vector','IsFixedVectorOfAnyRankTypePred':'kind=vector ∧ scal(none)',
      'IsVectorTypeWithAnyDimScalablePred':'kind=vector ∧ scal(any)','IsVectorTypeWithOnlyTrailingDimScalablePred':'kind=vector ∧ scal(trailing-only)',
      'IsVectorTypeWithAllDimsScalablePred':'kind=vector ∧ scal(all)','IsTensorTypePred':'kind∈{rtensor,utensor}',
      'IsMemRefTypePred':'kind=rmemref','IsTensorOrMemRefTypePred':'kind∈{rtensor,utensor,rmemref}',
      'IsUnrankedMemRefTypePred':'kind=umemref','IsUnrankedTensorTypePred':'kind=utensor','IsRankedTensorTypePred':'kind=rtensor',
      'IsBaseMemRefTypePred':'kind∈{rmemref,umemref}','IsShapedTypePred':'kind∈Shaped',
      'HasStaticShapePred':'ranked ∧ static','IsTupleTypePred':'kind=tuple','HasValueSemanticsPred':'valsem',
      'IsTokenTypePred':'kind=token','HasRankPred':'ranked','HasAnyRankOfPred':'rank∈S','HasRankGreaterOrEqualPred':'rank≥n',
      'IsVectorOfRankPred':'kind=vector ∧ rank∈S','IsFixedVectorOfRankPred':'kind=vector∧scal(none)∧rank∈S',
      'IsScalableVectorOfRankPred':'kind=vector∧scal(any)∧rank∈S','IsVectorOfLengthPred':'kind=vector ∧ numel∈S',
      'IsVectorOfMinLengthPred':'kind=vector ∧ numel≥n','IsVectorOfMaxLengthPred':'kind=vector ∧ numel≤n',
      'IsFixedVectorOfLengthPred':'kind=vector∧scal(none)∧numel∈S','IsScalableVectorOfLengthPred':'kind=vector∧scal(any)∧numel∈S',
      'IsNthDimSizeIsOneOfPred':'rank≥norm(n)+1 ∧ dim(norm(n))∈S','IsVectorOfShape':'kind=vector ∧ shape=[..]',
      'HasStridesPred':'kind=rmemref ∧ strided'}
    if n in atoms: return 'ATOM', atoms[n]
    # ---- 标量
    if n=='AnyType': return 'PAT','¬kind=token'
    if n in ('Token','NoneType','FunctionType','GraphType','Index','AnyComplex','AnyTuple','AnyFloat','AnyInteger'):
        return 'PAT', {'Token':'kind=token','NoneType':'kind=none','FunctionType':'kind=function','GraphType':'kind=graph',
          'Index':'kind=index','AnyComplex':'kind=complex','AnyTuple':'kind=tuple','AnyFloat':'kind=float','AnyInteger':'kind=int'}[n]
    m=re.fullmatch(r'(Any)?(I|SI|UI)(\d+)?',n)
    if m and (m.group(3) or m.group(2)): 
        sign={'I':'signless' if not m.group(1) else '*','SI':'signed','UI':'unsigned'}[m.group(2)]
        w=m.group(3) or 'w'
        return 'PAT', f'kind=int ∧ width∈{{{w}}}'+('' if sign=='*' else f' ∧ sign={sign}')
    if n in ('AnyIntOfWidths','SignlessIntOfWidths','SignedIntOfWidths','UnsignedIntOfWidths'):
        s={'AnyIntOfWidths':'','SignlessIntOfWidths':' ∧ sign=signless','SignedIntOfWidths':' ∧ sign=signed','UnsignedIntOfWidths':' ∧ sign=unsigned'}[n]
        return 'PAT','kind=int ∧ width∈S'+s
    if n in ('AnySignlessInteger','AnySignedInteger','AnyUnsignedInteger'):
        return 'PAT','kind=int ∧ sign='+{'AnySignlessInteger':'signless','AnySignedInteger':'signed','AnyUnsignedInteger':'unsigned'}[n]
    if n=='AnySignlessIntegerOrIndex': return 'PAT','(kind=int∧sign=signless) ∨ kind=index'
    if n=='AnyNonZeroBitwidthSignlessInteger': return 'PAT','kind=int∧sign=signless∧width≥1'
    if n=='AnyNonZeroBitwidthSignlessIntegerOrIndex': return 'PAT','(kind=int∧sign=signless∧width≥1) ∨ kind=index'
    if re.fullmatch(r'F(\d+)?',n) or n=='FloatOfWidths': return 'PAT','kind=float ∧ fmt∈IEEE(w)'
    if re.fullmatch(FMT8,n): return 'PAT',f'kind=float ∧ fmt={n}'
    if n=='Complex': return 'PAT','kind=complex ∧ elt:C'
    # ---- 向量
    vec = {'ValueSemanticsContainerOf':'valsem ∧ shaped ∧ elt:∨Cᵢ',
      'VectorOfNonZeroRankOf':'kind=vector∧rank≥1∧elt:C','FixedVectorOfNonZeroRankOf':'kind=vector∧scal(none)∧rank≥1∧elt:C',
      'VectorOfAnyRankOf':'kind=vector∧elt:C','FixedVectorOfAnyRank':'kind=vector∧scal(none)∧elt:C',
      'ScalableVectorOfAnyRank':'kind=vector∧scal(any)∧elt:C','VectorWithTrailingDimScalableOf':'kind=vector∧scal(trailing-only)∧elt:C',
      'VectorOfRank':'kind=vector∧rank∈S','FixedVectorOfRank':'kind=vector∧scal(none)∧rank∈S','ScalableVectorOfRank':'kind=vector∧scal(any)∧rank∈S',
      'VectorOfRankAndType':'kind=vector∧rank∈S∧elt:C','FixedVectorOfRankAndType':'kind=vector∧scal(none)∧rank∈S∧elt:C',
      'VectorOfLength':'kind=vector∧numel∈S','VectorOfMinLength':'kind=vector∧numel≥n','VectorOfMaxLength':'kind=vector∧numel≤n',
      'FixedVectorOfLength':'kind=vector∧scal(none)∧numel∈S','ScalableVectorOfLength':'kind=vector∧scal(any)∧numel∈S',
      'VectorOfLengthAndType':'kind=vector∧numel∈S∧elt:C','VectorOfRankAndLengthAndType':'kind=vector∧rank∈S∧numel∈S∧elt:C',
      'VectorOfMinMaxLengthAndType':'kind=vector∧numel∈[a,b]∧elt:C','FixedVectorOfShapeAndType':'kind=vector∧scal(none)∧shape=[..]∧elt:C',
      'FixedVectorOfLengthAndType':'kind=vector∧scal(none)∧numel∈S∧elt:C','FixedVectorOfRankAndLengthAndType':'kind=vector∧scal(none)∧rank∈S∧numel∈S∧elt:C',
      'ScalableVectorOfLengthAndType':'kind=vector∧scal(any)∧numel∈S∧elt:C','ScalableVectorOfRankAndLengthAndType':'kind=vector∧scal(any)∧rank∈S∧numel∈S∧elt:C',
      'ShapedTypeWithNthDimOfSize':'shaped∧rank≥norm(n)+1∧dim(norm(n))∈S','VectorWithTrailingDimScalableOfSizeAndType':'kind=vector∧scal(trailing-only)∧dim(-1)∈S∧elt:C',
      'AnyVectorOfNonZeroRank':'kind=vector∧rank≥1','AnyFixedVectorOfNonZeroRank':'kind=vector∧scal(none)∧rank≥1',
      'AnyVectorOfAnyRank':'kind=vector','AnyFixedVectorOfAnyRank':'kind=vector∧scal(none)','AnyScalableVectorOfAnyRank':'kind=vector∧scal(any)',
      'AnyShaped':'kind∈Shaped'}
    if n in vec: return 'PAT', vec[n]
    # ---- 张量
    m=re.fullmatch(r'(I1|I8|I16|I32|I64|Index|BF16|F16|F32|F64)(Tensor|MemRef)',n)
    if m: return 'PAT', ('kind∈{rtensor,utensor}' if m.group(2)=='Tensor' else 'kind=rmemref')+f' ∧ elt:{m.group(1)}'
    ten = {'UnrankedTensorOf':'kind=utensor∧elt:C∧P̄','RankedTensorOf':'kind=rtensor∧elt:C∧P̄','TensorOf':'kind∈{rtensor,utensor}∧elt:C∧P̄',
      'AnyTensor':'kind∈{rtensor,utensor}','Non0RankedTensorOf':'kind=rtensor∧rank≥1∧elt:C','AnyRankedTensor':'kind=rtensor',
      'AnyNon0RankedTensor':'kind=rtensor∧rank≥1','AnyUnrankedTensor':'kind=utensor','AnyNon0RankedOrUnrankedTensor':'kind=utensor ∨ (kind=rtensor∧rank≥1)',
      'TensorRankOf':'kind=rtensor∧rank∈S∧elt:C','StaticShapeTensorOf':'kind=rtensor∧static∧elt:C','AnyStaticShapeTensor':'kind=rtensor∧static',
      'UnrankedMemRefOf':'kind=umemref∧elt:C','AnyUnrankedMemRef':'kind=umemref','MemRefOf':'kind=rmemref∧elt:C',
      'Non0RankedMemRefOf':'kind=rmemref∧rank≥1∧elt:C','AnyMemRef':'kind=rmemref','AnyNon0RankedMemRef':'kind=rmemref∧rank≥1',
      'RankedOrUnrankedMemRefOf':'kind∈{rmemref,umemref}∧elt:C','AnyRankedOrUnrankedMemRef':'kind∈{rmemref,umemref}',
      'AnyNon0RankedOrUnrankedMemRef':'kind=umemref ∨ (kind=rmemref∧rank≥1)','MemRefRankOf':'kind=rmemref∧rank∈S∧elt:C',
      'StaticShapeMemRefOf':'kind=rmemref∧static∧elt:C','AnyStaticShapeMemRef':'kind=rmemref∧static',
      'StridedMemRefOf':'kind=rmemref∧strided∧elt:C','AnyStridedMemRef':'kind=rmemref∧strided',
      'AnyStridedMemRefOfRank':'kind=rmemref∧strided∧rank=n','StridedMemRefRankOf':'kind=rmemref∧strided∧rank∈S∧elt:C',
      'TupleOf':'kind=tuple∧elts∀:C','NestedTupleOf':'μX. kind=tuple∧elts∀:(C∨X)','TensorOrMemRef':'kind∈{rtensor,utensor,rmemref}∧elt:C',
      'TypeOrValueSemanticsContainer':'C ∨ (valsem∧shaped∧elt:C)','BoolLike':'I1 ∨ (valsem∧shaped∧elt:I1)',
      'SignlessIntegerLike':'Sl ∨ (valsem∧shaped∧elt:Sl), Sl=kind=int∧sign=signless',
      'SignlessIntegerOrIndexLike':'SlI ∨ (valsem∧shaped∧elt:SlI)','FloatLike':'Fl ∨ (valsem∧shaped∧elt:Fl), Fl=kind=float',
      'SignlessIntegerOrFloatLike':'SignlessIntegerLike ∨ FloatLike','SignlessIntegerOrIndexOrFloatLike':'SignlessIntegerOrIndexLike ∨ FloatLike'}
    if n in ten: return 'PAT', ten[n]
    if n=='0DTensorOf' or re.fullmatch(r'[1-4]DTensorOf',n): return 'PAT',f'kind=rtensor∧rank={n[0]}∧elt:C'
    return None, None

out, miss = {}, []
for n in names:
    c, f = nf(n)
    if c is None: miss.append(n)
    else: out[n] = {'class': c, 'nf': f}
cnt = {}
for v in out.values(): cnt[v['class']] = cnt.get(v['class'],0)+1
flags = {k: sorted(n for n,v in out.items() if k in v['nf']) for k in ('numel','strided','valsem','opaque','μX','scal(')}
json.dump(out, open('model/type_constraints_nf.json','w'), ensure_ascii=False, indent=1)
print(f'total={len(names)} mapped={len(out)} missing={miss}'); print(cnt)
for k,v in flags.items(): print(k, len(v))
