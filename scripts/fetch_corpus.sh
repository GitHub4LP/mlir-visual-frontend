#!/bin/sh
# 取 MLIR 自己的方言测试（mlir/test/Dialect/*），与 Python 绑定同一提交，供 model/corpus_test.py 做独立差分。语料不入库。
set -e
SHA=${MLIR_SHA:-9bae84b01718e53495abf50958abc86ea45f16bb}
mkdir -p corpus && cd corpus
for d in Arith Math Index Complex Tensor MemRef SCF Func ControlFlow Vector Shape UB; do
  mkdir -p $d
  curl -fsSL "https://api.github.com/repos/llvm/llvm-project/contents/mlir/test/Dialect/$d?ref=$SHA" \
    | python3 -c "import json,sys; [print(f['download_url']) for f in json.load(sys.stdin) if f['type']=='file' and f['name'].endswith('.mlir')]" \
    | while read u; do curl -fsSL -o "$d/$(basename "$u")" "$u"; done
  echo "$d $(ls $d | wc -l)"
done
