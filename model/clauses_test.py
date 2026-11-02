"""G3.1（第 36 轮）：frontend-model.md 的条款 ↔ spec/clauses.md 对照表 ↔ 检验 ↔ laws.md 等级，机械核对。
  K1 编号双向一致；K2 检验文件与标记存在；K3 条款声称 A/B 时所引定律无 C/✗（D 级定义允许，计数）；K4 A/B/C 定律反向完备；K5 状态合法。
  M2.7 回归：值层无拒绝权——必然 UB 的程序 MLIR 接受、结构层无诊断、推导不抛异常且 UB 恒真。
"""
import os, re, sys, collections
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
def rd(p): return open(os.path.join(ROOT, p), encoding='utf-8').read()

def rows(md, first):
    out = []
    for ln in md.splitlines():
        if ln.startswith('| ' + first) or re.match(r'\| ?' + first, ln):
            out.append([c.strip() for c in ln.strip().strip('|').split('|')])
    return out

def law_grades():
    g = {}
    for r in rows(rd('spec/laws.md'), 'L'):
        if len(r) < 4 or not re.fullmatch(r'L\d+\S*', r[0]): continue
        s = r[3].replace('*', '')
        if '→' in s: s = s.split('→', 1)[1]
        m = re.search(r'[ABCD✗]', s); g[r[0]] = m.group(0) if m else '?'
    return g

def law_ids(cell):
    return [x.strip() for x in re.split(r'[,，]', cell) if re.fullmatch(r'L\d+\S*', x.strip())]

def check_table():
    st = collections.Counter(); bad = []
    spec = rd('spec/frontend-model.md'); tab = rd('spec/clauses.md')
    ids_spec = re.findall(r'〔(M\d+\.\d+)〕', spec) + re.findall(r'\| (M\d+\.\d+) \|$', spec, re.M)
    rs = [r for r in rows(tab, 'M') if re.fullmatch(r'M\d+\.\d+', r[0])]
    ids_tab = [r[0] for r in rs]
    for i in set(ids_spec) - set(ids_tab): bad.append(f'K1 规范有、表无：{i}')
    for i in set(ids_tab) - set(ids_spec): bad.append(f'K1 表有、规范无：{i}')
    for i, c in collections.Counter(ids_tab).items():
        if c > 1: bad.append(f'K1 表内重复：{i}')
    st['条款'] = len(set(ids_tab))
    grades = law_grades(); cited = set()
    for r in rs:
        cid, tests, laws, lvl, status = r[0], r[3], r[4], r[5], r[6]
        st['状态/' + status] += 1
        if status not in ('检验', '部分', '元理论', '缺口'): bad.append(f'K5 {cid} 状态非法：{status}')
        if status in ('检验', '部分') and tests in ('', '—'): bad.append(f'K5 {cid} 声称{status}却无检验')
        for t in [x.strip() for x in tests.split(';') if x.strip() and x.strip() != '—']:
            f, _, mk = t.partition(':'); p = os.path.join(HERE, f + '.py')
            st['检验引用'] += 1
            if not os.path.exists(p): bad.append(f'K2 {cid} 文件不存在：{f}'); continue
            src_ = open(p, encoding='utf-8').read()
            if mk and mk not in src_: bad.append(f'K2 {cid} 标记不存在：{f}:{mk}')
            # K7（第 44 轮）：该检验是否有机器判定（verdict.check('标记', …)，失败 ⇒ 退出码 1）。此前多数测试只打印统计、退出码恒 0。
            if mk and re.search(r"check\(\s*'" + re.escape(mk), src_): st['K7 检验引用有机器判定'] += 1
            else: bad.append(f'K7 {cid} 引用无逐标记判定：{f}:{mk}')      # 第 47 轮：118/118 已逐标记，此后一律视为错误（文件级退出码不算）
        for L in law_ids(laws):
            cited.add(L); st['定律引用'] += 1
            if L not in grades: bad.append(f'K3 {cid} 定律不在账本：{L}'); continue
            if lvl in ('A', 'B') and grades[L] in ('C', '✗'): bad.append(f'K3 {cid} 声称 {lvl}，但 {L} 是 {grades[L]}')
            elif lvl in ('A', 'B') and grades[L] == 'D': st['A/B 条款依赖 D 级定义'] += 1      # 定义不是经验命题，允许，但计数
    unref = rd('spec/clauses.md').split('## 未引用', 1)[-1]
    excused = set(re.findall(r'L\d+\S*?(?=[、 |])', unref))
    for L, g in grades.items():
        if g in ('A', 'B', 'C') and L not in cited and L not in excused: bad.append(f'K4 {g} 级定律未被任何条款引用：{L}')
    st['A/B/C 定律'] = sum(g in 'ABC' for g in grades.values())
    return st, bad

def k6():
    """自检：核对器本身能抓出注入的错误（否则 0 报告无意义，L38）。"""
    global rd
    orig = rd; muts = [
        ('删规范编号', 'spec/frontend-model.md', '〔M3.5〕', ''),
        ('坏标记', 'spec/clauses.md', 'floatjit_test:F1 浮点定向', 'floatjit_test:F9 不存在'),
        ('C 级撑 A', 'spec/clauses.md', '| L7 | B | 检验 |', '| L6 | B | 检验 |'),
        ('非法状态', 'spec/clauses.md', '| B | 检验 | R5 21971/0 不符 |', '| B | 已完成 | R5 21971/0 不符 |'),
        ('漏引定律', 'spec/clauses.md', '| L11, L12 |', '| L11 |')]
    got = 0
    for nm, f, a, b in muts:
        def rd2(p, f=f, a=a, b=b):
            t = orig(p); return t.replace(a, b, 1) if p == f else t
        assert a in orig(f), nm
        rd = rd2
        try: got += bool(check_table()[1])
        finally: rd = orig
    return got, len(muts)

def m27():
    import mlir.ir as ir, irimport as I, structure as ST, z3
    import valimport_test as VT
    from compcorpus_test import _funcs
    res = collections.Counter()
    progs = [('divsi', 'arith.divsi %a, %c0 : i32'), ('remui', 'arith.remui %a, %c0 : i32'), ('divui', 'arith.divui %a, %c0 : i32')]
    for nm, body in progs:
        txt = f'func.func @f(%a: i32) -> i32 {{\n  %c0 = arith.constant 0 : i32\n  %r = {body}\n  return %r : i32\n}}'
        m = ir.Module.parse(txt, ir.Context()); res['MLIR 接受'] += bool(m.operation.verify())
        root = I.import_module(m); d = []; ST.check(root, d); res['结构层无诊断'] += not d
        fn = next(_funcs(root)); S, why = VT.derive_imported('f', fn)
        if S is None: res['推导失败/' + str(why)[:40]] += 1; continue
        res['推导不拒绝'] += not (S.type_diag or S.struct_diag)
        s = z3.Solver(); s.add(z3.Not(S.ubf) if not isinstance(S.ubf, bool) else z3.BoolVal(not S.ubf))
        res['UB 恒真'] += s.check() == z3.unsat
    ok = all(res[k] == len(progs) for k in ('MLIR 接受', '结构层无诊断', '推导不拒绝', 'UB 恒真'))
    return ok, dict(res)

if __name__ == '__main__':
    st, bad = check_table()
    print('── G3.1 对照表:', dict(st))
    for b in bad: print('   ', b)
    g, n = k6(); print(f'── K6 核对器自检：注入错误 {g}/{n} 被抓出'); bad += [] if g == n else ['K6 自检未全抓']
    ok, r = m27()
    print('── M2.7 值层无拒绝权:', r, 'OK' if ok else 'FAIL')
    import verdict as V
    V.check('K1–K6 对照表一致', not bad, f'问题 {len(bad)}')
    V.check('M2.7 值层无拒绝权', ok, str(r))
    V.finish()
