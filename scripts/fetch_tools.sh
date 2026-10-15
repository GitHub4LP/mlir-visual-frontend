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
