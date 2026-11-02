"""回归判定（第 44 轮）：测试打印统计之外，必须对每个被 clauses.md 引用的检验给出机器判定。
约定：每个判定一行 `VERDICT <标记> PASS|FAIL <说明>`；标记与 clauses.md“检验”列的标记相同；
main 末尾调用 finish()：有 FAIL ⇒ 退出码 1。clauses_test K7 静态核对每个“检验”引用都有对应的 check(…)。"""
import sys
SEEN, FAILS = [], []
def check(mark, ok, detail=''):
    ok = bool(ok); SEEN.append(mark)
    if not ok: FAILS.append(mark)
    print(f'VERDICT {mark} {"PASS" if ok else "FAIL"} {detail}', flush=True)
    return ok
def finish():
    print(f'VERDICTS {len(SEEN)} FAIL {len(FAILS)} {FAILS}', flush=True)
    sys.exit(1 if FAILS else 0)
