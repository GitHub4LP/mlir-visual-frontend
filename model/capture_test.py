"""第 44 轮（L3 / M1.2）：“提取为复合 op”的端口推导在真实 IR 上以 MLIR verifier 为真值。
此前 L3 的证据是 core_test X2（一个手写例）与 structure_fuzz（随机 CFG 的支配/符号判定，不涉及提取）。
方法：语料中 MLIR 接受原样发射的函数，随机取任意嵌套层、任意块里的一段连续非终结 op [lo, hi)，用 core.extract_ports 求端口：
  · 被提取函数 @seg：参数 = 捕获（经 core.ren 改名），体 = 原段，返回 = 输出端口——单独发射交 MLIR（func 是 IsolatedFromAbove：
    漏掉的捕获 ⇒ “未声明的 SSA 值”/支配错误）；
  · 宿主：原段换成 func.call @seg（@seg 以私有声明给出）——漏掉的输出端口 ⇒ 后面的使用者引用了不存在的值。
发射时预置全部值的类型（emit_core types=），所以端口漏算不会被“无法发射”掩盖，而是交给 MLIR 判。
分类：
  P1 端口完备：两者都被 MLIR 接受；
  P2 端口错误：MLIR 诊断为未声明值/支配/类型不符（必须为 0）；
  P3 上下文约束：段内 op 依赖父 op（HasParent/ParentOneOf、AffineScope 维度合法性等）——提取改变了上下文，不是端口问题，
     但对前端是真约束：统计我们的结构层（预言机关）能否预先指出。
用法：PYTHONPATH=model python3 model/capture_test.py [函数上限] [每函数段数]
"""
import sys, re, glob, random, collections, copy
import verdict as V

PORT_ERR = re.compile(r'undeclared|does not dominate|use of value|expects different type|type mismatch|operand type|result type|number of (operands|results)|incorrect number', re.I)

def blocks_with_region(fn):
    """[(区域块表, 块下标, 嵌套深度)]：所有区域里的所有块。"""
    out = []
    def walk(region, d):
        for i, b in enumerate(region):
            out.append((region, i, d))
            for n in b['ops']:
                for r in n['regions']: walk(r, d + 1)
    walk(fn['regions'][0], 0); return out

def all_types(fn):
    t = {}
    def walk(region):
        for b in region:
            for v, ty in b['args']: t[v] = ty
            for n in b['ops']:
                for v, ty in n['outs']: t[v] = ty
                for r in n['regions']: walk(r)
    walk(fn['regions'][0]); return t

DUMP = []
P3LIST = []      # 第 50 轮：上下文约束段的明细（M1.2 上下文约束建模的真值样本）
SEGLOG = []      # 每个计入的段：(top op 名, 父 op 名, MLIR 结论 P1/P2/P3)
def ports_r43(block, lo, hi_, rest=()):
    """消融对照：第 43 轮及以前的端口算法（扁平名字、只看 ins、只扫本块后续 op）。"""
    seg = block['ops'][lo:hi_]; defined, used = {}, []
    def scan(ops):
        for n in ops:
            for v in n['ins']:
                if v not in defined and v not in used: used.append(v)
            for v, t in n['outs']: defined[v] = t
            for r in n['regions']:
                for b in r:
                    for v, t in b['args']: defined[v] = t
                    scan(b['ops'])
    scan(seg); later = set()
    def uses(ops):
        for n in ops:
            later.update(n['ins'])
            for r in n['regions']:
                for b in r: uses(b['ops'])
    uses(block['ops'][hi_:])
    return [v for v in used if v not in defined], [v for v in defined if v in later and any(v == o for n in seg for o, _ in n['outs'])]

PORTS = None
def scoped_types(fn, target):
    """target 块处可见的名字 → 类型（词法作用域：内层遮蔽外层；区域内整块可见）。导入的 IR 沿用 MLIR 文本名，兄弟区域会重名。"""
    def region_env(region, env):
        e = dict(env)
        for b in region:
            for v, t in b['args']: e[v] = t
            for n in b['ops']:
                for v, t in n['outs']: e[v] = t
        return e
    def walk(region, env):
        e = region_env(region, env)
        for b in region:
            if b is target: return e
            for n in b['ops']:
                for r in n['regions']:
                    got = walk(r, e)
                    if got is not None: return got
        return None
    return walk(fn['regions'][0], {})

def run(lim=300, per=3, seed=43, skip=0):
    import mlir.ir as ir, irimport as I, oracle as OR, core as C, structure as ST, ctxcons as X
    from compcorpus_test import _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list); o = OR.Oracle()
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    nf = 0; sk = 0
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        if nf >= lim: break
        src = open(path).read()
        if 'allow-unregistered-dialect' in src: continue
        for ch in I.split_chunks(src):
            if nf >= lim: break
            if not ch.strip() or I.expected(ch): continue
            try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
            except Exception: continue
            fns = list(_funcs(root))
            for fn in fns:
                if nf >= lim: break
                ft = fn['props']['_attrs'].get('function_type')
                if not ft or not fn['regions'] or not fn['regions'][0]: continue
                syms = {g['props']['_attrs'].get('sym_name', '').strip('"'): g for g in fns}
                base = OR.emit_core(fn['regions'][0], ft, name='f', symbols=syms)
                if base is None: st['D0 无法发射'] += 1; continue
                if o.verify(base)[0] != 'ok': st['D0 原样 MLIR 不接受（不计）'] += 1; continue
                if sk < skip: sk += 1; continue          # 第 51 轮：留出集（跳过前 skip 个合格函数）
                nf += 1; st['D0 函数'] += 1
                par = {}
                def _par(region, chain):
                    par[id(region)] = chain                      # 祖先链：直接父 op 在前
                    for b_ in region:
                        for n_ in b_['ops']:
                            for r_ in n_['regions']: _par(r_, [n_['op']] + chain)
                _par(fn['regions'][0], ['func.func'])
                cands = [(R, i, d) for R, i, d in blocks_with_region(fn) if len(R[i]['ops']) >= 2]
                if not cands: st['D1 无可提取段（块内非终结 op < 1）'] += 1; continue
                rich = [c for c in cands if c[2] > 0 or len(c[0]) > 1]          # 偏向嵌套区域与多块区域（端口问题主要出在这里）
                st['D0 函数/含嵌套或多块'] += bool(rich)
                for k in range(per):
                    R, bi, d = rng.choice(rich if rich and rng.random() < 0.8 else cands); b = R[bi]; nops = len(b['ops']) - 1      # 最后一个 op 当终结 op，不提取
                    lo = rng.randrange(nops); hi = rng.randint(lo + 1, nops)
                    try: caps, outs = (PORTS or C.extract_ports)(b, lo, hi, rest=[x for j, x in enumerate(R) if j != bi])
                    except C.NotExtractable as e: st['D1 不可提取（段内先用后定义）'] += 1; note('不可提取', str(e)[:80]); continue
                    types = scoped_types(fn, b)
                    if any(types.get(v) is None for v in caps + outs): st['D1 端口类型未知（不计）'] += 1; continue
                    st['段'] += 1; st[f'段/嵌套深度 {min(d, 3)}' + ('+' if d >= 3 else '')] += 1
                    st['段/多块区域'] += len(R) > 1
                    name = f'seg{nf}_{k}'
                    pm = {v: f'{name}_in{i}' for i, v in enumerate(caps)}
                    body = [C.ren(n, pm) for n in b['ops'][lo:hi]]
                    ret = {'id': f'{name}_ret', 'op': 'func.return', 'ins': [pm.get(v, v) for v in outs], 'outs': [], 'regions': [], 'succs': [], 'props': {'_attrs': {}}}
                    sft = '(' + ', '.join(types[v] for v in caps) + ') -> (' + ', '.join(types[v] for v in outs) + ')'
                    stys = dict(types); stys.update({pm[v]: types[v] for v in caps})   # 预置可见名字的类型：漏掉的捕获照样发射，由 MLIR 判
                    stxt = OR.emit_core([{'name': 'entry', 'args': [(pm[v], types[v]) for v in caps], 'ops': body + [ret]}], sft, name=name, symbols=syms, types=stys)
                    call = {'id': f'{name}_call', 'op': 'func.call', 'ins': list(caps), 'outs': [(v, types[v]) for v in outs], 'regions': [], 'succs': [],
                            'props': {'_attrs': {'callee': f'@{name}'}}}
                    host = copy.deepcopy(fn); HR = None
                    # 在深拷贝里找到同一个块（按块对象在原树中的路径定位：用 id 序列匹配）
                    key = [n['id'] for n in b['ops']]
                    for R2, i2, _ in blocks_with_region(host):
                        if [n['id'] for n in R2[i2]['ops']] == key: HR = R2[i2]; break
                    HR['ops'] = HR['ops'][:lo] + [call] + HR['ops'][hi:]
                    s2 = dict(syms); s2[name] = {'op': 'func.func', 'props': {'_attrs': {'function_type': sft, 'sym_name': f'"{name}"'}}, 'regions': []}
                    htxt = OR.emit_core(host['regions'][0], ft, name='f', symbols=s2, types={v: types[v] for v in outs})
                    if stxt is None or htxt is None: st['D2 无法发射（不计）'] += 1; note('无法发射', list(OR.WHY_NONE.items())[-1:]); continue
                    rs, ds = o.verify(stxt); rh, dh = o.verify(htxt)
                    if 'crash' in (rs, rh): st['MLIR 崩溃（不计）'] += 1; continue
                    tops, pop = b['ops'][lo:hi], par.get(id(R), ['?'])
                    cv, cw = X.move_check(tops, pop) if PORTS is None else ('ok', [])
                    st[f'C 上下文判定/{cv}'] += 1
                    if rs == 'ok' and rh == 'ok':
                        st['P1 端口完备（两者 MLIR 接受）'] += 1; SEGLOG.append(([n['op'] for n in tops], pop, 'P1', None)); st[f'C/P1/{cv}'] += 1
                        if cv == 'reject': note('C 误拒', cw[:1])
                        for w in (cw if cv == 'unknown' else [])[:1]: st['C/P1/unknown 原因/' + re.sub(r'（.*', '', w)[:60]] += 1
                        continue
                    msgs = [m for _, m in (ds if rs != 'ok' else []) + (dh if rh != 'ok' else [])]
                    if any(PORT_ERR.search(m) for m in msgs):
                        st['P2 端口错误'] += 1; note('P2', (name, 'seg' if rs != 'ok' else 'host', msgs[0][:160])); DUMP.append((name, path, stxt if rs != 'ok' else htxt, msgs[:2], caps, outs)); continue
                    st['P3 上下文约束（非端口）'] += 1; note('P3', msgs[0][:140])
                    st[f'C/P3/{cv}'] += 1; note('C/P3', (cv, cw[:1]))
                    SEGLOG.append(([n['op'] for n in tops], pop, 'P3', 'seg' if rs != 'ok' else 'host')); P3LIST.append((name, path, 'seg' if rs != 'ok' else 'host', pop, [n['op'] for n in tops], msgs[:1]))
                    C.ORACLE_ON = False
                    try: dd = C.typecheck([{'name': 'entry', 'args': [(pm[v], types[v]) for v in caps], 'ops': body + [ret]}], sft) if rs != 'ok' else None
                    except Exception: dd = None
                    finally: C.ORACLE_ON = True
                    st['P3 我们的层（预言机关）也指出'] += bool(dd)
    o.close()
    return st, ex

if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 1500     # 第 50 轮：默认 300 时只有 1 个函数含嵌套区域、上下文约束 0 段（检验形同虚设）
    per = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    st, ex = run(lim, per)
    print('── L3 提取端口 vs MLIR:', dict(sorted(st.items())))
    PORTS = ports_r43; sa, exa = run(lim, per); PORTS = None
    print('── 消融：第 43 轮端口算法:', {k: v for k, v in sorted(sa.items()) if k.startswith(('P', '段', 'D1'))}); print('   ', exa.get('P2', [])[:3])
    for k, v in ex.items(): print('   ', k, v)
    V.check('P 端口 vs MLIR', st['P2 端口错误'] == 0 and st['P1 端口完备（两者 MLIR 接受）'] >= 100,
            f"段 {st['段']}，完备 {st['P1 端口完备（两者 MLIR 接受）']}，端口错误 {st['P2 端口错误']}，上下文 {st['P3 上下文约束（非端口）']}")
    V.check('P 消融灵敏度', sa['P2 端口错误'] > 0, f"旧算法端口错误 {sa['P2 端口错误']}")
    p3 = st['P3 上下文约束（非端口）']
    V.check('C1 上下文拒绝 0 误拒', st['C/P1/reject'] == 0, f"P1 段中判拒 {st['C/P1/reject']}；判拒共 {st['C 上下文判定/reject']}")
    V.check('C2 上下文约束预先指出', p3 >= 10 and st['C/P3/reject'] + st['C/P3/unknown'] == p3,
            f"P3 {p3} 段：拒 {st['C/P3/reject']}、未知 {st['C/P3/unknown']}、漏 {st['C/P3/ok']}")
    V.check('C3 判 ok 的段 MLIR 接受', st['C/P3/ok'] == 0 and st['C 上下文判定/ok'] > 0,
            f"判 ok {st['C 上下文判定/ok']}，其中 MLIR 上下文失败 {st['C/P3/ok']}；P1 中判未知 {st['C/P1/unknown']}/{st['P1 端口完备（两者 MLIR 接受）']}（精度代价）")
    # 第 51 轮：读体判据按前 1500 个函数的未知原因收窄过，须在未参与收窄的留出集上再量（防止对样本过拟合）
    hs, hx = run(lim, per, seed=44, skip=lim)
    print('── 留出集（跳过前', lim, '个合格函数）:', {k: v for k, v in sorted(hs.items()) if k.startswith(('C/P', 'P1', 'P3', 'D0 函数', '段'))})
    for k in ('C/P3', 'C 误拒'): print('   ', k, hx.get(k, [])[:4])
    V.check('C6 留出集上下文判定', hs['D0 函数'] >= 300 and hs['C/P1/reject'] == 0 and hs['C/P3/ok'] == 0,
            f"函数 {hs['D0 函数']}，段 {hs['段']}；P1 判拒 {hs['C/P1/reject']}、判未知 {hs['C/P1/unknown']}/{hs['P1 端口完备（两者 MLIR 接受）']}；"
            f"P3 {hs['P3 上下文约束（非端口）']}：拒 {hs['C/P3/reject']}、未知 {hs['C/P3/unknown']}、漏 {hs['C/P3/ok']}")
    V.finish()
