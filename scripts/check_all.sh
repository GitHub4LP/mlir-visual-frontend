#!/bin/bash
# 一键全量检验：bash scripts/check_all.sh [日志目录，缺省 logs/check]
# 2 CPU 约 3–4 小时；最多 2 段并发，每段 timeout 1150；环境丢失自动 setup 后重跑该段（≤2 次）。
# 结束时运行 scripts/check_summary.py：可靠性字段（误拒/误收(残差为空)/不一致/违例/漏判/未定）非零 ⇒ 硬失败；
# 仅数量阈值不足的已知 FAIL 列在 scripts/check_known.txt，不算失败。退出码 0 = 通过。
cd "$(dirname "$0")/.."; ROOT=$PWD; L=${1:-logs/check}; mkdir -p "$L"
python3 -c "import mlir.ir, z3" 2>/dev/null || bash scripts/setup_env.sh > "$L/setup.log" 2>&1
seg() { n=$1; shift; t=$1; shift
  for try in 0 1 2; do
    PYTHONPATH=model timeout 1150 python3 model/$t.py "$@" > "$L/$n.log" 2>&1
    grep -q "^VERDICT" "$L/$n.log" && break
    python3 -c "import mlir.ir, z3" 2>/dev/null && break
    flock "$L/.setup.lock" bash scripts/setup_env.sh >> "$L/setup.log" 2>&1
  done; echo "$(date +%T) done $n" >> "$L/status.txt"; }
while IFS= read -r line; do
  [ -z "$line" ] && continue; nm=${line%% *}
  grep -q "^VERDICT" "$L/$nm.log" 2>/dev/null && continue
  while [ "$(jobs -rp | wc -l)" -ge 2 ]; do wait -n; done
  seg $line & sleep 2
done < scripts/check_jobs.txt
wait
python3 scripts/check_summary.py "$L"
