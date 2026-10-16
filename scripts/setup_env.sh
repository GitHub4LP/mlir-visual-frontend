#!/bin/sh
# 沙箱会重置 pip（第 17、18 轮都在跑测试途中发生）：每轮开始与长跑前执行。装与被测 MLIR 同一提交的 Python 绑定 + z3 + islpy。
set -e
REV=${MLIR_WHEEL_REV:-22.0.0.2025112901+9bae84b0}
ARCH=$(uname -m)
W="mlir_python_bindings-$REV-cp312-cp312-manylinux_2_35_$ARCH.whl"
python3 -c "import mlir.ir, z3, islpy" 2>/dev/null && { echo ok; exit 0; }
cd /tmp && curl -fsSL -o "$W" "https://github.com/makslevental/wheels/releases/download/i/$(python3 -c "import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))" "$W")"
pip install -q --user "$W" z3-solver islpy
python3 -c "import mlir.ir, z3; print('ok')"
