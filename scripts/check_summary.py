"""汇总 check_all.sh 的日志：python3 scripts/check_summary.py 日志目录。退出码 0 = 通过。"""
import sys, re, pathlib
L = pathlib.Path(sys.argv[1]); here = pathlib.Path(__file__).parent
known = {tuple(l.split()[:2]) for l in (here/'check_known.txt').read_text().splitlines() if l.strip() and not l.startswith('#')}
jobs = [l.split()[0] for l in (here/'check_jobs.txt').read_text().splitlines() if l.strip()]
# 可靠性字段：任何非零即硬失败。“误收：残差非空”不计（残差非空即 Unknown 区，按定义交 MLIR）；“消融误收”是对照组，不计
HARD = [r'误拒 (\d+)', r'误收：残差为空 (\d+)', r'(?<!消融)误收 (\d+)', r'不一致 (\d+)', r'违例 (\d+)', r'漏判 (\d+)',
        r'未定 (\d+)，解释器', r'见证不成立 (\d+)', r'不可靠 (\d+)', r'推导异常 (\d+)', r'文件崩溃 (\d+)', r'增益误收 (\d+)']
bad, soft, n = [], [], 0
for j in jobs:
    f = L/f'{j}.log'
    vs = [l for l in f.read_text(errors='replace').splitlines() if l.startswith('VERDICT ')] if f.exists() else []
    if not vs: bad.append(f'{j}: 无 VERDICT（未跑完或环境失败）'); continue
    for v in vs:
        n += 1; tag = v.split()[1]
        for rx in (HARD if tag != 'L1' else [h for h in HARD if '误收' not in h]):   # L1 是 ODS 单层差分，误收按设计允许（上近似）

            for m in re.finditer(rx, v):
                if int(m.group(1)): bad.append(f'{j}: {v}')
        if ' FAIL ' in v and (j, tag) not in known: bad.append(f'{j}: 未登记的 FAIL：{v}')
        elif ' FAIL ' in v: soft.append(f'{j} {tag}')
print(f'判据行 {n}；已知数量阈值 FAIL {len(soft)}：{", ".join(soft)}')
for b in dict.fromkeys(bad): print('硬失败', b)
print('总判定', 'PASS' if not bad else 'FAIL'); sys.exit(1 if bad else 0)
