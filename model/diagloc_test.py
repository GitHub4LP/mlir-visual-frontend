"""第 42 轮（L31 / M2.8 / M5.4，spec/oracle.md §归位）：诊断归位在真实 IR 上的统计。
第 36 轮之前 L31 只有 oracle_test O3：生成器上“诊断都落在某个节点上” 987/987——这个判据近乎恒真（emit 给每个 op 带 loc），
不能说明诊断落在**对的**节点上。这里改用真实 IR + 单点变异，外部真值是 MLIR verifier 给出的 Location：
  D0 基线：语料合法函数经 import → emit_core（每个 op 带 loc("节点")）→ MLIR 接受（否则不计）。
  D1 变异（每函数一处，记下被改节点 X）：结果类型改宽度（使用者的边类型随之改变，模拟前端改了一个端口类型）、
     cmpi 谓词越界、常量值与结果类型不符、把操作数换成另一个类型不同的已定义值。
  对 MLIR 拒绝的变异体统计：
     R1 MLIR 的诊断有 loc 且落在图中节点上（Location 回映在嵌套区域/导入节点上成立）；
     R2 MLIR 诊断节点 ∈ {X} ∪ 使用者(X)（“归位到编辑处或其直接后果”）；
     R3 我们自己的层（结构/符号/属性，预言机关）是否先抓住；抓住时诊断节点与 MLIR 的是否相同。
  另统计 MLIR 接受的变异（不计入归位率，但说明变异是否“真的错”）。
用法：PYTHONPATH=model python3 model/diagloc_test.py [函数上限]
"""
import sys, re, glob, random, collections, copy

INT = re.compile(r'^i(\d+)$')

def users(fn, v):
    out = set()
    def walk(blocks):
        for b in blocks:
            for n in b['ops']:
                if v in n['ins']: out.add(n['id'])
                for r in n['regions']: walk(r)
    walk(fn['regions'][0]); return out

def defs(fn):
    """[(节点, 结果名, 类型, 在入口块顶层?)]"""
    out = []
    for n in fn['regions'][0][0]['ops']:
        for o, t in n['outs']: out.append((n, o, t))
    return out

def mutate(fn, rng):
    """返回 (变异后的函数, 种类, 被改节点 id) 或 None。原地改深拷贝。"""
    f = copy.deepcopy(fn); top = f['regions'][0][0]['ops']
    kinds = ['restype', 'cmpi', 'const', 'operand']; rng.shuffle(kinds)
    for kind in kinds:
        if kind == 'restype':
            cands = [n for n in top if len(n['outs']) == 1 and INT.match(n['outs'][0][1]) and n['op'] not in ('func.return',) and users(f, n['outs'][0][0])]
            if not cands: continue
            n = rng.choice(cands); o, t = n['outs'][0]; w = int(INT.match(t).group(1))
            nt = 'i' + str(rng.choice([x for x in (1, 8, 16, 32, 64) if x != w]))
            n['outs'] = [(o, nt)]
            def fix(blocks):
                for b in blocks:
                    for m in b['ops']:
                        ty = (m.get('props') or {}).get('_types')
                        if ty: m['props']['_types'] = [nt if x == o else tt for x, tt in zip(m['ins'], ty)] + ty[len(m['ins']):]
                        for r in m['regions']: fix(r)
            fix(f['regions'][0]); return f, kind, n['id']
        if kind == 'cmpi':
            cands = [n for n in top if n['op'] == 'arith.cmpi']
            if not cands: continue
            n = rng.choice(cands); n['props']['_attrs']['predicate'] = '42 : i64'; return f, kind, n['id']
        if kind == 'const':
            cands = [n for n in top if n['op'] == 'arith.constant' and INT.match(n['outs'][0][1])]
            if not cands: continue
            n = rng.choice(cands); t = n['outs'][0][1]; w = int(INT.match(t).group(1))
            n['props']['_attrs']['value'] = f"1 : i{8 if w != 8 else 16}"; return f, kind, n['id']
        if kind == 'operand':
            ds = defs(f)
            for _ in range(20):
                n = rng.choice(top)
                if not n['ins'] or n['op'] == 'func.return': continue
                k = rng.randrange(len(n['ins'])); old = n['ins'][k]
                pos = top.index(n)
                earlier = [(o, t) for m in top[:pos] for o, t in m['outs']] + list(f['regions'][0][0]['args'])
                oldt = next((t for o, t in earlier if o == old), None)
                alt = [(o, t) for o, t in earlier if t != oldt and oldt is not None]
                if not alt: continue
                o, t = rng.choice(alt); n['ins'][k] = o
                ty = (n.get('props') or {}).get('_types')
                if ty and k < len(ty): ty[k] = t
                return f, kind, n['id']
    return None

def run(lim, seed=31):
    import mlir.ir as ir, irimport as I, oracle as OR, core as C
    from compcorpus_test import _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list); o = OR.Oracle()
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    nf = 0
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
                blocks = fn['regions'][0]; ft = fn['props']['_attrs'].get('function_type')
                if len(blocks) != 1 or not ft: continue
                syms = {g['props']['_attrs'].get('sym_name', '').strip('"'): g for g in fns}
                base = OR.emit_core(blocks, ft, name='f', symbols=syms)
                if base is None: st['D0 无法发射'] += 1; continue
                if o.verify(base)[0] != 'ok': st['D0 原样 MLIR 不接受（不计）'] += 1; continue
                nf += 1; st['D0 函数'] += 1
                mu = mutate(fn, rng)
                if mu is None: st['D1 无可变异处'] += 1; continue
                f2, kind, X = mu; st[f'D1 变异/{kind}'] += 1
                txt = OR.emit_core(f2['regions'][0], ft if kind != 'restype' else ft, name='f', symbols=syms)
                if txt is None: st['D1 变异体无法发射'] += 1; continue
                res, ds = o.verify(txt)
                if res == 'ok': st[f'MLIR 接受变异/{kind}'] += 1; continue
                if res == 'crash': st['MLIR 崩溃（不计）'] += 1; continue
                st['MLIR 拒绝'] += 1; st[f'MLIR 拒绝/{kind}'] += 1
                ids = set()
                def walk(bs):
                    for b in bs:
                        for n in b['ops']:
                            ids.add(n['id'])
                            for r in n['regions']: walk(r)
                walk(f2['regions'][0])
                locs = [l for l, _ in ds if l is not None]
                if not locs: st['R1 诊断无 loc（语法层）'] += 1; note('无 loc', f'{kind} {ds[:1]}'); continue
                st['R1 落在节点上'] += all(l in ids or l == 'f' for l in locs)
                Xn = next((n for n in _iter(f2) if n['id'] == X), None)
                near = {X} | (users(f2, Xn['outs'][0][0]) if Xn and Xn['outs'] else set())
                hit = locs[0] in near
                st['R2 MLIR 首诊断 ∈ 被改节点或其使用者'] += hit; st[f'R2/{kind} ' + ('命中' if hit else '未命中')] += 1
                if not hit: note('R2 未命中', f'{kind} X={X} MLIR@{locs[0]}: {ds[0][1][:90]}')
                C.ORACLE_ON = False
                try: d = C.typecheck(f2['regions'][0], ft)
                except Exception as e: d = None; st['R3 我们的层异常'] += 1
                finally: C.ORACLE_ON = True
                if d:
                    st['R3 我们的层先抓住'] += 1
                    st['R3 抓住且首诊断节点 = MLIR 的'] += d[0][0] == locs[0]
                    st['R3 抓住且首诊断 ∈ 被改节点或使用者'] += d[0][0] in near
                    if d[0][0] != locs[0]: note('R3 节点不同', f'{kind} 我们@{d[0][0]} {d[0][1][:60]} ‖ MLIR@{locs[0]} {ds[0][1][:60]}')
                elif d is not None: st['R3 我们的层未抓住（交给预言机）'] += 1
    o.close()
    return st, ex

def _iter(fn):
    def walk(bs):
        for b in bs:
            for n in b['ops']:
                yield n
                for r in n['regions']: yield from walk(r)
    return walk(fn['regions'][0])

if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    st, ex = run(lim)
    print('── 诊断归位（真实 IR + 单点变异）:'); [print(f'   {k}: {v}') for k, v in sorted(st.items())]
    for k, v in ex.items(): print('──', k); [print('    ', x) for x in v]
    rej = st['MLIR 拒绝'] - st['R1 诊断无 loc（语法层）']
    sys.exit(0 if rej > 0 and st['R1 落在节点上'] == rej else 1)
