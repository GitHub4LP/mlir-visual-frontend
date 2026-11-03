"""第 59 轮：L10（复合 op 的参数约束 = 各使用点约束的合取）在真实 IR 上的外部检验（M1.4）。

此前 L10 只有 defs_test D3 的 16 个手写用例。这里把“一个参数在体内多处使用”直接放到 MLIR 语料上：
  语料中 MLIR 接受的模块里，取同一 ODS 属性约束（attrsig 的 cons）在 ≥ 2 个 op 实例上的使用点，组成一组 = 一个参数的多个使用点；
  候选值 v 取自全语料该约束下见过的属性文本，外加其它约束的文本（制造拒绝）。
  E1 定律本身（MLIR 自证，不经我们的实现）：MLIR(所有使用点都换成 v) 接受 ⇔ 每个使用点单独换成 v 时 MLIR 都接受。
     不成立 ⇒ 某处校验跨使用点读取同一属性，L10 须收窄。
  E2 我们的合取（attrsig.check 逐点求值后取合取）：判拒而 MLIR 接受 = 0（有拒绝权的层不得误拒）；另报精度。
  分层（运行前写定）：符号属性（sym_name、SymbolRefAttr 类）单列——符号唯一性本来就跨使用点，由符号层负责，不计入 E1 判据。
阈值（运行前写定）：E1 非符号层 组 ≥ 300、用例（组 × 候选）≥ 2000、违例 0；E2 误拒 0。
用法：PYTHONPATH=model python3 model/l10_test.py [语料块上限]
"""
import sys, os, glob, random, pickle, collections
import mlir.ir as ir
import irimport as I, attrsig as A, verdict as V

SYM = ('sym_name', 'sym_visibility')
SKIP = collections.Counter()
def is_sym(name, cons): return name in SYM or 'SymbolRef' in (cons or '') or 'SymbolName' in (cons or '')

def _walk(op):
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                o = o.operation; yield o; yield from _walk(o)

def sites(m):
    """[(op 序号, op 名, 属性名, cons, 文本)]：ODS 属性参数的使用点。"""
    out = []
    for i, o in enumerate(_walk(m.operation)):
        aa = A.attr_args(o.name)
        if not aa: continue
        names = {e['name']: e for e in aa}
        for na in o.attributes:
            k = na if isinstance(na, str) else na.name
            if k in names:
                try: out.append((i, o.name, k, names[k]['cons'], str(o.attributes[k])))
                except Exception: SKIP['绑定无法取属性'] += 1     # 绑定对部分 dense 属性（complex 元素）抛 TypeError
    return out

def _verify(ch, subs):
    with ir.Context() as c:
        c.enable_multithreading(False)
        m = ir.Module.parse(ch)
        ops = list(_walk(m.operation))
        try:
            for i, nm, v in subs:
                a = ir.Attribute.parse(v); ops[i].attributes[nm] = a
                # 绑定的 setter 对 property 做转换，类型不符时静默丢弃（不报错）；同样的文本交给解析器会被拒：
                # "invalid properties … Invalid attribute in property conversion"。所以读回不等 ⇒ MLIR 拒绝。
                if nm not in ops[i].attributes or str(ops[i].attributes[nm]) != str(a): return False
        except Exception: return None          # 文本本身无法解析为属性 ⇒ 用例无效
        try: return bool(m.operation.verify())
        except Exception: return False

def chunk_job(ch, groups):
    """groups: [(sites, cands)] ⇒ [(全换, [单换...], 候选, cons)]。"""
    out = []
    for ss, cands in groups:
        for v in cands:
            single = [_verify(ch, [(i, nm, v)]) for i, _, nm, _, _ in ss]
            allv = _verify(ch, [(i, nm, v) for i, _, nm, _, _ in ss])
            out.append((allv, single, v, ss[0][3], ss[0][2]))
    return out

def forked(fn, *a):
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r)
        try: data = pickle.dumps(fn(*a))
        except Exception as e: data = pickle.dumps(('err', str(e)[:80]))
        with os.fdopen(w, 'wb') as f: f.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, 'rb') as f: data = f.read()
    os.waitpid(pid, 0)
    return pickle.loads(data) if data else ('crash',)

def chunks(lim):
    n = 0
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read()
        if 'allow-unregistered-dialect' in src: continue
        for ch in I.split_chunks(src):
            if n >= lim: return
            if not ch.strip() or I.expected(ch): continue
            try:
                m = ir.Module.parse(ch, ir.Context())
                if not m.operation.verify(): continue
            except Exception: continue
            n += 1; yield ch, m

def run(lim, seed=59, gmax=4, ncand=6):
    rng = random.Random(seed)
    data = [(ch, sites(m)) for ch, m in chunks(lim)]
    pool = collections.defaultdict(set)
    for _, ss in data:
        for s in ss: pool[s[3]].add(s[4])
    allcons = sorted(pool)
    st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    for ch, ss in data:
        by = collections.defaultdict(list)
        for s in ss: by[(s[3], s[2])].append(s)
        groups = []
        for (cons, nm), g in by.items():
            if len(g) < 2: continue
            g = g[:gmax]
            own = sorted(pool[cons]); rng.shuffle(own)
            other = [rng.choice(sorted(pool[c])) for c in rng.sample(allcons, min(2, len(allcons))) if c != cons]
            groups.append((g, own[:ncand] + other))
        if not groups: continue
        res = forked(chunk_job, ch, groups)
        if isinstance(res, tuple): st['块失败/' + res[0]] += 1; continue
        st['块'] += 1
        seen_g = set()
        for allv, single, v, cons, nm in res:
            layer = '符号' if is_sym(nm, cons) else '非符号'
            if allv is None or any(x is None for x in single): st[f'{layer}/用例无效（值挂不上）'] += 1; continue
            st[f'{layer}/用例'] += 1
            if (cons, nm, ch[:60]) not in seen_g: seen_g.add((cons, nm, ch[:60])); st[f'{layer}/组'] += 1
            conj = all(single)
            st[f'{layer}/MLIR 接受' if allv else f'{layer}/MLIR 拒绝'] += 1
            if allv != conj:
                st[f'{layer}/E1 违例'] += 1; note(f'{layer} E1 违例', (cons, nm, v[:60], allv, single))
    return st, ex, data

def E2(data, lim_cases=4000, seed=59):
    """我们的合取 vs MLIR：在 E1 的同一类用例上，attrsig.check 逐点求值。只需判拒 ⇒ MLIR 拒（误拒 0）。"""
    rng = random.Random(seed + 1)
    pool = collections.defaultdict(set)
    for _, ss in data:
        for s in ss: pool[s[3]].add(s[4])
    allcons = sorted(pool)
    st = collections.Counter(); ex = []
    jobs = []
    for ch, ss in data:
        by = collections.defaultdict(list)
        for s in ss: by[(s[3], s[2])].append(s)
        for (cons, nm), g in by.items():
            if len(g) < 2 or is_sym(nm, cons): continue
            g = g[:4]
            cands = sorted(pool[cons])[:3] + [rng.choice(sorted(pool[rng.choice(allcons)])) for _ in range(3)]
            for v in cands:
                rej = any(x[0] == s[2] for s in g for x in A.check(s[1], {s[2]: v}))   # 只看被替换的属性（其余必需属性此处未提供）
                jobs.append((ch, g, v, rej))
    jobs = jobs[:lim_cases]
    bych = collections.defaultdict(list)
    for ch, g, v, rej in jobs: bych[ch].append((g, v, rej))
    for ch, js in bych.items():
        res = forked(lambda: [_verify(ch, [(s[0], s[2], v) for s in g]) for g, v, _ in js])
        if isinstance(res, tuple): st['块失败'] += 1; continue
        for (g, v, rej), ok in zip(js, res):
            if ok is None: st['用例无效'] += 1; continue
            st['用例'] += 1
            st[f'我们{"拒" if rej else "收"}/MLIR{"收" if ok else "拒"}'] += 1
            if rej and ok: ex.append((g[0][1], g[0][2], v[:60]))
    return st, ex

if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 1000   # 后台上限 1200 s；全量前定为 1000，阈值未改
    st, ex, data = run(lim)
    print('E1', dict(sorted(st.items())), '跳过', dict(SKIP))
    for k, v in ex.items(): print('  ', k, v)
    V.check('E1 L10 使用点合取（MLIR 自证）', st['非符号/组'] >= 300 and st['非符号/用例'] >= 2000 and st['非符号/E1 违例'] == 0,
            f"非符号：组 {st['非符号/组']}，用例 {st['非符号/用例']}（MLIR 接受 {st['非符号/MLIR 接受']}、拒绝 {st['非符号/MLIR 拒绝']}），违例 {st['非符号/E1 违例']}；"
            f"符号层（单列）：用例 {st['符号/用例']}，违例 {st['符号/E1 违例']}")
    s2, ex2 = E2(data)
    print('E2', dict(sorted(s2.items())), ex2[:6])
    V.check('E2 我们的合取不误拒', s2['用例'] >= 1000 and s2['我们拒/MLIR收'] == 0,
            f"用例 {s2['用例']}：我们拒且 MLIR 拒 {s2['我们拒/MLIR拒']}，我们收且 MLIR 拒 {s2['我们收/MLIR拒']}（精度缺口：关系约束、C++ verify），我们收 MLIR 收 {s2['我们收/MLIR收']}，误拒 {s2['我们拒/MLIR收']}")
    V.finish()
