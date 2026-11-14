"""第 95 轮：SymbolUserOpInterface::verifySymbolUses 解释执行的差分（判据 Z，spec/closure.md §9.33）。

语料：corpus_ext/<方言>/*.mlir（缺省 Shard；SU_DIRS 环境变量可改），MLIR 9bae84b0 mlir/test/Dialect。按 '// -----' 分段。
每个 MLIR 接受的段：原样 + 变异（被引用 grid 的 shape、使用者的符号引用、使用者的整数 / 整数数组属性），
变异经 Python 绑定改属性后由 MLIR 自身 verify 判真值；模型对模块内每个使用者走 symtab.symuses_decide
（最近符号表解析 + cppverify.decide_symuses 解释执行 9bae84b0 源码）。

  Z1 完备：使用者的 completeness('symuses') 全部完备。
  Z2 不误拒：MLIR 接受 ⇒ 模型对每个使用者判过。
  Z3 不漏判：MLIR 的诊断以某使用者 verifySymbolUses（含其调用的辅助函数）里的字面量开头 ⇒ 模型对至少一个使用者判拒。
  Z4 无未定：完备 op 上未定 0、解释器 Python 异常 0。
  分母：变异比较 ≥ 200，其中 MLIR 拒绝 ≥ 50（判据写定于实现前）。
用法：PYTHONPATH=model python3 model/symuses_test.py [起 止]
"""
import os, sys, re, glob, copy, collections, multiprocessing as mp

DYN = -(2 ** 63)

def arr_variants(txt):
    m = re.fullmatch(r'array<(i\d+)(?::\s*(.*))?>', txt.strip())
    if not m: return []
    et, body = m.group(1), m.group(2)
    xs = [int(x) for x in (body or '').split(',') if x.strip()]
    fmt = lambda ys: f'array<{et}' + (': ' + ', '.join(map(str, ys)) if ys else '') + '>'
    out = []
    for i, x in enumerate(xs):
        for y in (x + 1, x - 1, 0, 1):
            if y != x: out.append(fmt(xs[:i] + [y] + xs[i + 1:]))
        if et == 'i64' and x != DYN: out.append(fmt(xs[:i] + [DYN] + xs[i + 1:]))
    if xs: out.append(fmt(xs[:-1])); out.append(fmt(xs + [xs[0]]))
    out.append(fmt(xs + [len(xs)]))
    return list(dict.fromkeys(out))

def int_variants(txt):
    m = re.fullmatch(r'(-?\d+)\s*:\s*(\w+)', txt.strip())
    if not m: return []
    v, t = int(m.group(1)), m.group(2)
    return [f'{y} : {t}' for y in (v + 1, v - 1, v + 2) if y != v]

def fork_verify(m):
    """MLIR verify 在子进程里跑（断言失败会杀死进程）。→ (接受?, 诊断表, 崩溃?)。崩溃按拒绝计（同 C1 口径）。"""
    import mlir.ir as ir, pickle
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            try: m.operation.verify(); res = (True, [])
            except ir.MLIRError as e: res = (False, [str(d.message) for d in e.error_diagnostics])
            except Exception as e: res = (False, [str(e)])
            os.write(w, pickle.dumps(res))
        finally: os._exit(0)
    os.close(w); data = b''
    import select, time
    t0 = time.time()
    while True:
        rd, _, _ = select.select([r], [], [], max(0.1, 60 - (time.time() - t0)))
        if not rd:
            os.kill(pid, 9); os.waitpid(pid, 0); os.close(r); return (None, [], False)   # 超时：不计入比较
        c = os.read(r, 65536)
        if not c: break
        data += c
    os.close(r); _, stt = os.waitpid(pid, 0)
    if not data: return (False, [], True)
    ok, d = pickle.loads(data); return (ok, d, False)

def run_file(path, lo_hi, q):
    import mlir.ir as ir, irimport as I, symtab as SY, cppverify as CV
    st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 8: ex[k].append(v)
    src = open(path).read()
    lits_cache = {}
    def lits(op):
        if op not in lits_cache:
            dia = op.split('.')[0]; cls, body = CV.symuses_source(dia, op)
            lits_cache[op] = CV.verify_literals(dia, op, src=(cls, body)) if body else frozenset()
        return lits_cache[op]
    comp = {}
    def complete(op):
        if op not in comp: comp[op] = CV.completeness(op.split('.')[0], op, 'symuses')[0]
        return comp[op]
    for k, ch in enumerate(I.split_chunks(src)):
        if not re.search(r'\S', I.strip_comments(ch)): continue
        ctx = ir.Context(); ctx.allow_unregistered_dialects = True; ctx.enable_multithreading(False)   # fork 前不得有线程池
        st['段'] += 1
        try: m = ir.Module.parse(ch, ctx)
        except Exception: st['段 MLIR 拒绝（跳过）'] += 1; continue
        users, grids, syms, allops = [], [], [], []
        def walk(op):
            for r in op.regions:
                for b in r.blocks:
                    for o in b.operations:
                        o = o.operation
                        nm_ = o.name; allops.append(o)
                        if 'sym_name' in o.attributes: syms.append(str(o.attributes['sym_name']).strip('"'))
                        if SY.is_user(nm_) and SY.has_symuses(nm_): users.append(o)
                        if nm_ == 'shard.grid': grids.append(o)
                        walk(o)
        walk(m.operation)
        if not users: continue
        st['含使用者的段'] += 1
        # 模型侧：只导入一次（原样合法模块；无效 IR 上 AsmState 会先跑 verifier 并可能断言崩溃），变异同样施加在导入图的属性文本上
        try: g0 = I.import_module(m)
        except Exception as e: st['导入失败'] += 1; note('导入失败', str(e)[:100]); continue
        def preorder(g):
            out = []
            def rec(n):
                for r in n.get('regions') or []:
                    for b in r:
                        for x in b['ops']: out.append(x); rec(x)
            rec(g); return out
        nodes0 = preorder(g0)
        if [n['op'] for n in nodes0] != [o.name for o in allops]: st['导入次序不符（跳过段）'] += 1; continue
        pos = {id(o): i for i, o in enumerate(allops)}
        upos = [allops.index(u) for u in users]; gpos = [allops.index(x) for x in grids]
        cases = [None]
        for gi_, g in enumerate(grids):
            for v in arr_variants(str(g.attributes['shape'])): cases.append(('grid', gi_, 'shape', v))
        for ui, u in enumerate(users):
            for an in list(u.attributes):
                if an in ('operandSegmentSizes', 'resultSegmentSizes'): continue      # 结构性属性（ODS 不变式层），不属符号义务
                a = str(u.attributes[an])
                if a.startswith('@'):
                    for v in ['@__absent__'] + [f'@{s}' for s in syms if f'@{s}' != a][:3]: cases.append(('user', ui, an, v))
                for v in arr_variants(a) + int_variants(a): cases.append(('user', ui, an, v))
        for case in cases:
            kind = '原样' if case is None else '变异'
            old = None
            if case is not None:
                tgt = (grids if case[0] == 'grid' else users)[case[1]]
                try: old = tgt.attributes[case[2]]; tgt.attributes[case[2]] = ir.Attribute.parse(case[3], ctx)
                except Exception: st['变异不可构造'] += 1; continue
            try:
                ok, diags, crash = fork_verify(m)
                if ok is None: st['MLIR 校验超时（不计入）'] += 1; continue
                if crash: st[f'{kind}/MLIR 崩溃（按拒绝计）'] += 1
                if case is None and not ok: st['原样被拒'] += 1; break
                g = copy.deepcopy(g0)
                if case is not None:
                    nd = preorder(g)[(gpos if case[0] == 'grid' else upos)[case[1]]]
                    p_ = nd['props']; p_['_attrs'] = dict(p_.get('_attrs') or {}); p_['_attrs'][case[2]] = case[3]
                    if case[2] in (p_.get('_props') or {}): p_['_props'] = dict(p_['_props']); p_['_props'][case[2]] = case[3]
                par = SY.parent_map(g)
                nodes = []
                def rec(n):
                    for r in n.get('regions') or []:
                        for b in r:
                            for x in b['ops']:
                                if SY.is_user(x['op']) and SY.has_symuses(x['op']): nodes.append(x)
                                rec(x)
                rec(g)
                verdicts = []; vmsgs = []
                for n in nodes:
                    msg = None
                    try: v, msg = SY.symuses_decide(g, n, par)
                    except CV.CppUnknown as e:
                        v = None; st['未定'] += 1
                        if complete(n['op']): st['未定/完备'] += 1; note('未定/完备', f"{n['op']}: {str(e)[:100]}")
                    verdicts.append((n['op'], v)); vmsgs.append(msg)
                st[f'{kind}比较'] += 1; st[f'{kind}/MLIR ' + ('接受' if ok else '拒绝')] += 1
                if ok:
                    bad = [o for o, v in verdicts if v is False]
                    if bad and all(m_ == '\x00ub-div0' for (o_, v_), m_ in zip(verdicts, vmsgs) if v_ is False):
                        st['平台 UB 桶（除零，MLIR 本机接受）'] += 1; note('平台 UB 桶', f"{os.path.basename(path)}#{k} {case}: {bad}")
                    elif bad: st['Z2 误拒'] += 1; note('Z2 误拒', f"{os.path.basename(path)}#{k} {case}: {bad}")
                else:
                    strip = lambda s_: re.sub(r"^'[\w.]+' op ", '', s_)
                    allit = set().union(*[lits(o) for o, _ in verdicts]) if verdicts else set()
                    su = any(any(strip(d).startswith(l[:40]) for l in allit) for d in diags)
                    if su:
                        st['MLIR 拒绝/诊断出自 verifySymbolUses'] += 1
                        if not any(v is False for _, v in verdicts):
                            if any(v is None for _, v in verdicts): st['Z3 未判（有未定）'] += 1
                            else: st['Z3 漏判'] += 1; note('Z3 漏判', f"{os.path.basename(path)}#{k} {case}: {diags[:1]}")
                        else: st['Z3 抓住'] += 1
                    else:
                        st['MLIR 拒绝/其他诊断'] += 1
                        if any(v is False for _, v in verdicts): st['其他诊断时模型也判拒'] += 1
            finally:
                if old is not None: tgt.attributes[case[2]] = old
    st['python 异常'] += CV.STATS.get('python 异常', 0)
    q.put((st, dict(ex)))

def main():
    files = []
    for d in os.environ.get('SU_DIRS', 'Shard').split(','): files += sorted(glob.glob(f'corpus_ext/{d}/*.mlir'))
    if len(sys.argv) >= 3: files = files[int(sys.argv[1]):int(sys.argv[2])]
    import symtab as SY, cppverify as CV, odsjson as O
    tot = collections.Counter(); exs = collections.defaultdict(list)
    users = sorted({op for d in os.environ.get('SU_DIRS', 'Shard').split(',') for dia in [d.lower()] if dia in O.DIALECTS for op in O.ops(dia) if SY.is_user(op) and SY.has_symuses(op)})
    inc = [op for op in users if not CV.completeness(op.split('.')[0], op, 'symuses')[0]]
    import json
    out = os.environ.get('SU_OUT')     # 分块运行：每个文件的计数落盘，SU_AGG=1 时只汇总
    if out: os.makedirs(out, exist_ok=True)
    for f in files:
        fj = out and os.path.join(out, os.path.basename(f) + '.json')
        if os.environ.get('SU_AGG'):
            if not os.path.exists(fj): st, ex = collections.Counter({'文件未跑': 1}), {}
            else: d_ = json.load(open(fj)); st, ex = collections.Counter(d_['st']), d_['ex']
        else:
            q = mp.Queue(); p = mp.Process(target=run_file, args=(f, None, q)); p.start()
            try: st, ex = q.get(timeout=900)
            except Exception: st, ex = collections.Counter({'文件崩溃': 1}), {}
            p.join(5)
            if p.is_alive(): p.kill()
            if p.exitcode not in (0, None) and not st: st = collections.Counter({'文件崩溃': 1})
            if fj: json.dump({'st': dict(st), 'ex': {k: list(v) for k, v in ex.items()}}, open(fj, 'w'), ensure_ascii=False)
        tot.update(st)
        for k, v in ex.items(): exs[k] += v
        print(os.path.basename(f), dict(st), flush=True)
    print('── 合计', dict(tot))
    for k, v in exs.items(): print(' 例', k, v[:6])
    cmp_, rej = tot['变异比较'], tot['变异/MLIR 拒绝']
    V = []
    V.append(('Z1 完备', not inc, f'使用者 {len(users)}，不完备 {len(inc)} {inc}'))
    ub = tot['平台 UB 桶（除零，MLIR 本机接受）']
    V.append(("Z2' 不误拒", ub <= cmp_ * 0.01 and tot['Z2 误拒'] == 0 and tot['文件崩溃'] == 0 and tot['文件未跑'] == 0 and cmp_ >= 200, f"比较 {tot['原样比较'] + cmp_}（变异 {cmp_}），MLIR 接受 {tot['原样/MLIR 接受'] + tot['变异/MLIR 接受']}，误拒 {tot['Z2 误拒']}，平台 UB 桶 {ub}（上限 {cmp_ * 0.01:.1f}），文件崩溃 {tot['文件崩溃']}"))
    V.append(('Z3 不漏判', tot['Z3 漏判'] == 0 and tot['Z3 未判（有未定）'] == 0 and rej >= 50, f"变异 MLIR 拒绝 {rej}，其中诊断出自 verifySymbolUses {tot['MLIR 拒绝/诊断出自 verifySymbolUses']}，抓住 {tot['Z3 抓住']}，漏判 {tot['Z3 漏判']}，有未定 {tot['Z3 未判（有未定）']}"))
    V.append(('Z4 无未定', tot['未定/完备'] == 0 and tot['python 异常'] == 0, f"完备 op 未定 {tot['未定/完备']}，未定合计 {tot['未定']}，python 异常 {tot['python 异常']}"))
    import verdict
    marks = {'Z1 完备': 'SU1 符号义务完备', "Z2' 不误拒": 'SU2 符号义务无误拒', 'Z3 不漏判': 'SU3 符号义务诊断必拒', 'Z4 无未定': 'SU4 符号义务无未定'}
    for nm, ok, msg in V:
        if nm == 'Z1 完备': verdict.check('SU1 符号义务完备', ok, msg)
        elif nm == "Z2' 不误拒": verdict.check('SU2 符号义务无误拒', ok, msg)
        elif nm == 'Z3 不漏判': verdict.check('SU3 符号义务诊断必拒', ok, msg)
        else: verdict.check('SU4 符号义务无未定', ok, msg)
    verdict.finish()

if __name__ == '__main__':
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
