#!/bin/sh
# 拉取原型所需的 MLIR ODS 源文件（llvm-project main，Apache-2.0 WITH LLVM-exception），不纳入版本库。
# 用途：ods_import.py 解析 op 定义；normalize_types.py 统计类型约束。生产实现应改用 llvm-tblgen -dump-json。
set -e
REF=${LLVM_REF:-main}
B=https://raw.githubusercontent.com/llvm/llvm-project/$REF/mlir/include/mlir
mkdir -p inventory
for f in CommonTypeConstraints CommonAttrConstraints Constraints OpBase Properties Traits Interfaces OpAsmInterface RegionKindInterface SymbolInterfaces BuiltinTypeInterfaces BuiltinAttributeInterfaces; do
  curl -fsSL $B/IR/$f.td -o inventory/IR_$f.td
done
curl -fsSL $B/IR/OpDefinition.h -o inventory/IR_OpDefinition.h
curl -fsSL $B/Dialect/Arith/IR/ArithOps.td -o inventory/Arith_ArithOps.td
curl -fsSL $B/Dialect/Tensor/IR/TensorOps.td -o inventory/Tensor_TensorOps.td
