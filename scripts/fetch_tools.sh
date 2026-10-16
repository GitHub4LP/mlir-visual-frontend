#!/bin/sh
# 取与被测 MLIR Python 绑定同一提交的 llvm-tblgen（mlir_native_tools 轮子），供 model/odsjson.py 生成 ODS JSON。
# 绑定轮子自带编译时所用的 .td（mlir/_mlir_libs/include），两者同一提交，避免 .td 与 verifier 的版本偏差。
set -e
REV=${MLIR_WHEEL_REV:-22.0.0.2025112901+9bae84b0}
ARCH=$(uname -m)
mkdir -p tools && cd tools
curl -fsSL -o nt.whl "https://github.com/makslevental/wheels/releases/download/i/mlir_native_tools-$REV-py3-none-linux_$ARCH.whl"
python3 -m zipfile -e nt.whl . && rm nt.whl
chmod +x mlir_native_tools-*.data/data/bin/*
echo "tblgen: $(ls -d $PWD/mlir_native_tools-*.data/data/bin)/llvm-tblgen"
# 第 19 轮：linalg 命名结构化 op 的 .td 是构建期由 LinalgNamedStructuredOps.yaml 生成的，wheel 不带 ⇒ 取同一提交的 yaml，用同一轮子的 mlir-linalg-ods-yaml-gen 生成。
LLVM_COMMIT=${LLVM_COMMIT:-9bae84b01718e53495abf50958abc86ea45f16bb}
G=gen/mlir/Dialect/Linalg/IR; mkdir -p $G
curl -fsSL -o $G/LinalgNamedStructuredOps.yaml "https://raw.githubusercontent.com/llvm/llvm-project/$LLVM_COMMIT/mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml"
mlir_native_tools-*.data/data/bin/mlir-linalg-ods-yaml-gen $G/LinalgNamedStructuredOps.yaml --o-ods-decl=$G/LinalgNamedStructuredOps.yamlgen.td
echo "yamlgen: $PWD/$G/LinalgNamedStructuredOps.yamlgen.td"
