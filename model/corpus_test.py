"""第 12 轮：用 MLIR 自己的测试集做独立差分（不经过我们的生成器、不经过 core 的参数解释）。

语料：mlir/test/Dialect/{Arith,Math,Index,Complex,Tensor,MemRef,SCF,Func,ControlFlow,Vector,Shape,UB}/*.mlir，
与 Python 绑定同一提交（scripts/fetch_corpus.sh）。按 '// -----' 分段，每段由 MLIR 解析并校验，作为真值。

  C1 导入忠实性：MLIR 接受的段，import → emit_generic → 再解析，与原 IR 的通用形式逐字相同。
  C2 合法 IR 上不误拒：对每个 func.func，结构层（structure.check）与符号层（自动签名，不开预言机）都不得报错。
     报错即模型缺陷（L29 与结构层义务在真实 IR 上的检验）。
  C3 MLIR 负例：校验失败的段，用 MLIR 诊断里 “see current operation” 的通用形式重建出错的 op，
     看符号层在 op 局部能否抓住；抓不住的按原因归类（C++ verify、依赖上下文、语法层错误）。
用法：PYTHONPATH=model python3 model/corpus_test.py [语料目录]
"""
import os, sys, re, glob, collections, multiprocessing as mp

def run_file(path, q):
    import mlir.ir as ir, irimport as I, structure as ST, core as C, symtab as SY, oracle as OR
    C.ORACLE_ON = False
    st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 6: ex[k].append(v)
    src = open(path).read()
    unreg = 'allow-unregistered-dialect' in src
    for k, ch in enumerate(I.split_chunks(src)):
        if not re.search(r'\S', I.strip_comments(ch)): continue
        ctx = ir.Context(); ctx.allow_unregistered_dialects = unreg
        st['段'] += 1; want_err = bool(I.expected(ch))
        try:
            m = ir.Module.parse(ch, ctx)
        except Exception as e:
            msg = str(e); st['MLIR 拒绝'] += 1; st['其中标了 expected-error'] += want_err
            cur = re.search(r'see current operation: (.*)', msg)
            first = re.search(r'error: "[^"]*":\d+:\d+: (.*)', msg)
            first = first.group(1) if first else msg.splitlines()[0]
            if not cur: st['C3 语法层错误（无 op 可重建）'] += 1; continue
            line = cur.group(1).strip()
            g = parse_generic_line(line)
            if g is None: st['C3 op 带区域/无法重建'] += 1; continue
            opn, tins, touts, attrs = g
            if attrs is None: st['C3 属性文本无法重建'] += 1; continue
            nd = ST.node('n0', opn, ins=[f'a{i}' for i in range(len(tins))], outs=[(f'r{i}', t) for i, t in enumerate(touts)],
                         props={'_attrs': attrs, '_types': tins})
            bs = [ST.blk('entry', [nd, ST.node('ret', 'func.return', ins=[])], args=[(f'a{i}', t) for i, t in enumerate(tins)])]
            import typesig as TSg
            bad = []
            for t in tins + touts:
                try: TSg.parse(t)
                except Exception: bad.append(t)
            if bad: st['C3 类型解析不支持（core 类型文法缺口）'] += 1; note('C3 类型解析不支持', f'{opn}: {bad[0][:60]}'); continue
            try:
                d = C.typecheck(bs, '() -> ()')
            except Exception as e2:
                st['C3 core 异常'] += 1; note('C3 core 异常', f'{opn}: {type(e2).__name__} {str(e2)[:80]}'); continue
            import odssig as G
            kind = (G.lookup(opn) or ('无签名',))[0]
            if d:
                st['C3 符号层抓住'] += 1; st[f'C3 抓住/{kind}'] += 1
                am = [m for _, m in d if m.startswith('attr:')]
                if am and not [m for _, m in d if not m.startswith('attr:')]:      # 只由属性层抓住：检查理由是否与 MLIR 的诊断一致
                    nm = re.findall(r"'(\w+)'", am[0])
                    same = any(f"'{x}'" in first or f"{x}" in first for x in nm) or '类型不同' in am[0] and 'same type' in first
                    st['C3 属性层抓住' + ('（理由与 MLIR 一致）' if same else '（理由不同）')] += 1
                    if not same: note('C3 属性层抓住但理由不同', f'{opn}: 我们 {am[0][:70]} ‖ MLIR {first[:80]}')
            else:
                st['C3 符号层未抓住'] += 1; st[f'C3 未抓住/{kind}'] += 1
                note(f'C3 未抓住/{kind}', f'{opn}: {first[:90]}')
            continue
        st['MLIR 接受'] += 1
        if want_err: st['接受但标了 expected-error（其它层的错误，如 pass）'] += 1
        # C1
        try:
            g = I.import_module(m)
            t = '\n'.join(I.emit_generic(g))
            ctx2 = ir.Context(); ctx2.allow_unregistered_dialects = unreg      # 新上下文：避免资源名在同一上下文里被改名
            m2 = ir.Module.parse(t, ctx2)
            a = m.operation.get_asm(print_generic_op_form=True, use_local_scope=True)
            b = m2.operation.get_asm(print_generic_op_form=True, use_local_scope=True)
            if '{-#' in ch: st['C1 含方言资源块（模块级数据，未导入）'] += 1
            if a == b: st['C1 往返逐字相同'] += 1
            else: st['C1 往返不同'] += 1; note('C1 往返不同', f'{os.path.basename(path)}#{k}')
        except Exception as e:
            st['C1 往返失败'] += 1; note('C1 往返失败', f'{os.path.basename(path)}#{k}: {str(e)[:120]}'); continue
        # C2
        try: mod_sd = list(ST.check(g))
        except Exception as e:
            st['C2 结构层异常'] += 1; note('C2 结构层异常', f'{os.path.basename(path)}#{k}: {type(e).__name__} {str(e)[:100]}'); mod_sd = None
        scope = {id(f): (sy, ma) for f, sy, ma in SY.func_scopes(g)}       # 第 16 轮：预言机按模块上下文发射
        for f in funcs(g):
            st['C2 函数'] += 1
            ftype = f['props']['_attrs'].get('function_type', '() -> ()')
            body = f['regions'][0] if f['regions'] and f['regions'][0] else None
            if body is None: st['C2 声明（无体）'] += 1; continue
            # 第 14 轮：结构层（含符号义务）对整个模块判定一次；此前把单个函数装进新模块，符号上下文丢失、且 S1–S4 在导入节点上从未触发
            sd = [x for x in mod_sd if x[0] in fids(f)] if mod_sd is not None else None
            try:
                sy, ma = scope.get(id(f), ({}, {}))
                # 第 16 轮：C2 打开预言机（此前 C2 一直关着它 ⇒ 有拒绝权的预言机层从未在真实 IR 上被检验过不误拒）；C3 仍关着（不让 MLIR 判自己）
                C.ORACLE_ON = True
                C.TYPE_WARN.clear(); td = C.typecheck(body, ftype, sy, SY.sym_of(f) or 'f', ma, unreg)
                C.ORACLE_ON = False
            except Exception as e:
                st['C2 符号层异常'] += 1; note('C2 符号层异常', f'{os.path.basename(path)}#{k}: {type(e).__name__} {str(e)[:100]}'); td = None
            if sd: st['C2 结构层误拒'] += 1; note('C2 结构层误拒', f'{os.path.basename(path)}#{k}: {sd[0]}'[:200])
            if C.TYPE_WARN: st['C2 人工签名提示（降级后不拒绝）'] += 1
            if td:
                st['C2 自动签名误拒'] += 1; note('C2 自动签名误拒', f'{os.path.basename(path)}#{k}: {td[0]}'[:200])
            if sd == [] and td == []: st['C2 两层都接受'] += 1
    # 第 16 轮：预言机可达性（防空检验：需要预言机的体有多少真的送到了 MLIR）
    st['C2 预言机调用'] = C.TC_STATS['预言机调用']
    st['C2 需预言机但不可发射'] = C.TC_STATS['预言机不可用→人工签名'] + C.TC_STATS['预言机不可用→仅符号层']
    for k, v in OR.WHY_NONE.items(): st['C2 不可发射：' + k[:40]] += v
    for d in C.TC_STATS.get('发射缺陷', []): st['C2 发射缺陷（MLIR 解析不了发射文本）'] += 1; note('C2 发射缺陷', f'{os.path.basename(path)}: {d}')
    q.put((dict(st), dict(ex)))

def fids(f):
    """函数 f 自身及其内部所有节点 / 块的诊断定位名。"""
    out = {f['id']}
    for r in f['regions']:
        for b in r:
            for m in b['ops']: out |= fids(m); out.add(f"{f['id']}/{b['name']}")
    return out

def funcs(n):
    if n['op'] == 'func.func': yield n; return
    for r in n['regions']:
        for b in r:
            for m in b['ops']: yield from funcs(m)

def parse_generic_line(line):
    """'%0 = "arith.addi"(%a, %b) <{…}> {…} : (i8, i8) -> i8' → (名字, 输入类型, 结果类型, 属性文本)。带区域的不重建。"""
    m = re.match(r'(?:[^=]*=\s*)?"([\w.$]+)"\((.*?)\)(\[.*?\])?\s*(.*)$', line)
    if not m or '({' in line: return None
    rest = m.group(4)
    tm = re.search(r':\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)\s*->\s*(.*)$', rest)
    if not tm: return None
    def split(s):
        out, d, cur = [], 0, ''
        for ch in s:
            if ch in '<([{': d += 1
            if ch in '>)]}': d -= 1
            if ch == ',' and d == 0: out.append(cur.strip()); cur = ''
            else: cur += ch
        if cur.strip(): out.append(cur.strip())
        return out
    outs = tm.group(2).strip()
    outs = split(outs[1:-1]) if outs.startswith('(') else [outs]
    raw = rest[:tm.start()].strip()
    return m.group(1), split(tm.group(1)), outs, raw_attrs(raw)

def raw_attrs(raw):
    """'<{a = 1 : i64}> {b}' → {'a': '1 : i64', 'b': 'unit'}：两个字典都交给 MLIR 的属性解析器（第 13 轮；此前 C3 不重建属性）。
    解析不了 ⇒ None（该例不参与属性层统计）。"""
    import mlir.ir as ir, attrsig as A
    out, rest = {}, raw.strip()
    try:
        while rest:
            if rest.startswith('<{'):
                d, k = 0, 0
                for k, ch in enumerate(rest):
                    d += ch in '<([{'; d -= ch in '>)]}'
                    if d == 0: break
                body, rest = rest[1:k], rest[k + 1:].strip()
            elif rest.startswith('{'):
                body, rest = rest, ''
            else: return None
            with A._ctx():
                da = ir.DictAttr(ir.Attribute.parse(body))
                for i in range(len(da)):
                    na = da[i]; v = str(na.attr); out[na.name] = 'unit' if v == 'unit' else v
    except Exception: return None
    return out

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else 'corpus'
    tot = collections.Counter(); exs = collections.defaultdict(list); per = {}
    ctxm = mp.get_context('fork')
    for path in sorted(glob.glob(f'{root}/*/*.mlir')):
        q = ctxm.Queue(); p = ctxm.Process(target=run_file, args=(path, q)); p.start(); p.join(120)
        if p.is_alive(): p.kill(); tot['文件超时'] += 1; continue
        if q.empty(): tot['文件崩溃（MLIR 断言）'] += 1; exs['文件崩溃'].append(path); continue
        st, ex = q.get(); tot.update(st)
        d = path.split('/')[-2]; per.setdefault(d, collections.Counter()).update(st)
        for k, v in ex.items(): exs[k] += v[:3]
    print('── 合计', dict(tot))
    for d, c in per.items():
        print('  ', d, {k: v for k, v in c.items() if k.startswith(('段', 'MLIR', 'C1 往返逐', 'C2 两', 'C2 结构层误', 'C2 自动签名误', 'C2 人工签名提示', 'C3 符号层'))})
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
