"""第 20 轮：增量重验（spec/modular.md §9，L45）。

问题：前端每次编辑后不能重验整模块。隔离单元（L42）的预言机判定只依赖 emit_unit 的输入：
  单元自身子树、祖先外壳的属性、它引用的符号在最近符号表里的声明（func.func 只取 名字 + function_type）。
据此，编辑只需在“编辑点”局部算出受影响单元，不必发射全部单元：
  dirty     = 子树被改动的单元（含新增/删除）；编辑落在任何单元之外（模块级 op、外壳属性）⇒ 全部单元；
  dependent = 引用了 dirty 单元的名字（旧名与新名）、且该 dirty 单元的声明文本变了的单元——只比较 dirty 单元自身的声明，
              依赖靠一张反向索引（名字 → 引用它的单元），不读其它单元。
检验（真实 IR 变异，MLIR 9bae84b0）：
  I1 可靠：实际发射文本变化的单元 ⊆ 预测集（必须 0 违例）；
  I2 判定一致：增量判定（预测集重验 + 其余沿用底本判定）≡ 整模块 check_module；
  I3 代价：预测集 / 单元总数（模块单元数 ≥ 2 时）。
变异 = 第 15 轮逐义务变异（多为单元内局部编辑）+ 本轮的符号级编辑：改被调函数的 function_type、改名（会波及调用者）。
用法：PYTHONPATH=model python3 model/incr_test.py [每模块变异上限=8]
"""
import sys, re, glob, copy, random, collections
import mlir.ir as ir
import irimport as I, structure as ST, oracle as OR, core as C, obligations_test as Z

def _skel(n, uids):
    """模块骨架：单元换成占位符，其余原样（不含 loc）。骨架变了 ⇒ 编辑落在单元之外。"""
    if n['id'] in uids: return [f'<unit {n["id"]}>']
    out = [repr((n['op'], n['ins'], n['outs'], n['succs'], sorted(((n.get('props') or {}).get('_attrs') or {}).items())))]
    for r in n['regions']:
        out.append('{')
        for b in r:
            out.append(repr((b['name'], b.get('args'))))
            for m in b['ops']: out += _skel(m, uids)
        out.append('}')
    return out

def _decl(u):
    try: return OR._decl_full(u)
    except Exception: return '<err>'

def index(g):
    """→ {单元 id: (chain, u, 内容文本, 名字, 引用名集)}，反向索引 名字 → {单元 id}，骨架。"""
    us = {}; rev = collections.defaultdict(set)
    for chain, u in OR.units(g):
        refs = set()
        for m in OR._walk(u):
            for k, v in ((m.get('props') or {}).get('_attrs') or {}).items():
                if k != 'sym_name': refs.update(p[0] for p in OR._refpaths(v))
        us[u['id']] = (chain, u, '\n'.join(I.emit_generic(u)), OR._sym(u), refs)
        for r in refs: rev[r].add(u['id'])
    return us, rev, _skel(g, set(us))

def predict(A, B):
    """由编辑点局部推出受影响单元（只比较 dirty 单元自身）。A/B = index(底本)/index(变异)。"""
    ua, rev_a, ska = A; ub, rev_b, skb = B
    if ska != skb: return set(ub), 'outside'
    dirty = {k for k in set(ua) | set(ub) if (k in ua) != (k in ub) or ua[k][2] != ub[k][2]}
    dep = set()
    for k in dirty:
        da = _decl(ua[k][1]) if k in ua else None; db = _decl(ub[k][1]) if k in ub else None
        na = ua[k][3] if k in ua else None; nb = ub[k][3] if k in ub else None
        if da != db or na != nb:
            for nm in {na, nb} - {None}: dep |= rev_a.get(nm, set()) | rev_b.get(nm, set())
    return (dirty | dep) & set(ub), 'local'

def sym_edits(g, rnd):
    """符号级编辑：被引用的 func.func 改 function_type（结果加一个 i1）、改名。"""
    us, rev, _ = index(g); out = []
    for k, (chain, u, _, nm, _) in us.items():
        if u['op'] != 'func.func' or not nm or not rev.get(nm, set()) - {k}: continue
        for kind in ('改签名', '改名'):
            h = copy.deepcopy(g); v = next(m for m in OR._walk(h) if m['id'] == u['id'])
            a = v['props']['_attrs']
            if kind == '改签名':
                ft = a.get('function_type', '')
                m = re.fullmatch(r'\((.*)\) -> (.*)', ft)
                if not m: continue
                res = m.group(2).strip()
                res = '(i1)' if res == '()' else ('(' + res[1:-1] + ', i1)' if res.startswith('(') else f'({res}, i1)')
                a['function_type'] = f'({m.group(1)}) -> {res}'
            else: a['sym_name'] = f'"{nm}__renamed"'
            out.append((kind, 'func.func', h))
    rnd.shuffle(out); out = out[:4]
    h = copy.deepcopy(g); h.setdefault('props', {}).setdefault('_attrs', {})['test.incr'] = 'unit'
    out.append(('模块属性', 'builtin.module', h))
    return out

def main(cap=8):
    rnd = random.Random(20); orc = OR.get()
    st = collections.Counter(); cost = []; ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 6: ex[k].append(v)
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read(); unreg = 'allow-unregistered-dialect' in src
        for kk, ch in enumerate(I.split_chunks(src)):
            if not re.search(r'\S', I.strip_comments(ch)): continue
            ctx = ir.Context(); ctx.allow_unregistered_dialects = unreg
            try: g = I.import_module(ir.Module.parse(ch, ctx))
            except Exception: continue
            A = index(g)
            if not A[0]: continue
            st['模块'] += 1
            base = {k: C.check_unit(c, u, unreg) for k, (c, u, *_ ) in A[0].items()}
            if ST.check(g) or any(base.values()): st['底本不合法（跳过）'] += 1; continue
            for kind, par, h in list(Z.mutations(g, rnd, cap)) + sym_edits(g, rnd):
                B = index(h)
                P, where = predict(A, B)
                st['变异'] += 1; st['编辑在单元外' if where == 'outside' else '编辑在单元内'] += 1
                if kind in ('改签名', '改名'): st['符号级编辑'] += 1
                # I1：实际发射文本变化的单元（全量发射，只为检验）
                actual = set()
                for k, (c, u, *_ ) in B[0].items():
                    tb = OR.emit_unit(c, u); ta = OR.emit_unit(*A[0][k][:2]) if k in A[0] else None
                    if (tb and tb[0]) != (ta and ta[0]): actual.add(k)
                if actual - P: st['I1 违例'] += 1; note('I1 违例', f'{path}#{kk} {kind} {par}: {sorted(actual - P)[:3]}')
                # I2：增量判定 ≡ 整模块判定
                full = C.check_module(h, unreg)
                s = ST.check(h)
                inc = bool(s) or any(C.check_unit(*B[0][k][:2], unreg) if k in P else base.get(k) for k in B[0])
                if inc != bool(full): st['I2 不一致'] += 1; note('I2 不一致', f'{path}#{kk} {kind} {par} inc={inc} full={bool(full)}')
                if full: st['拒'] += 1
                dirty_only = bool(s) or any(C.check_unit(*B[0][k][:2], unreg) if k in P and B[0][k][2] != (A[0].get(k) or (0, 0, None))[2] else base.get(k) for k in B[0])
                if full and not dirty_only: st['I2′ 只有依赖单元能抓住（缺依赖规则会漏判）'] += 1
                # I4：只看单元层（去掉整模块结构/符号层），依赖规则是否必要——func.call 的跨单元义务平时被符号层（USES）先抓住，这里把它拿掉
                if kind in ('改签名', '改名', '模块属性'):
                    ou = {k: C.check_unit(*B[0][k][:2], unreg) for k in B[0]}
                    f4 = any(ou.values()); i4 = any(ou[k] if k in P else base.get(k) for k in B[0])
                    d4 = any(ou[k] if k in P and B[0][k][2] != (A[0].get(k) or (0, 0, None))[2] else base.get(k) for k in B[0])
                    if f4 != i4: st['I4 单元层增量不一致'] += 1; note('I4', f'{path}#{kk} {kind}')
                    if f4 and not d4: st['I4 只有依赖单元能抓住'] += 1
                    if f4: st['I4 单元层拒'] += 1
                st['I3 预测集超出实际（上近似余量，单元数）'] += len(P - actual); st['I3 实际变化单元数'] += len(actual); st['I3 预测单元数'] += len(P)
                if len(B[0]) >= 2 and where == 'local':
                    cost.append(len(P) / len(B[0])); st['多单元模块的局部编辑'] += 1
                    if kind in ('改签名', '改名') and len(P) > 1: st['符号级编辑波及调用者'] += 1
    st['I3 平均重验比例（多单元、局部编辑）'] = round(sum(cost) / len(cost), 4) if cost else None
    st['I3 重验比例 ≤ 0.5 的占比'] = round(sum(c <= 0.5 for c in cost) / len(cost), 4) if cost else None
    print('── I 增量重验', dict(st)); print('   单元', dict(C.MOD_STATS))
    for k, v in ex.items():
        print('  ', k)
        for x in v: print('     ', x)
    return st

if __name__ == '__main__':
    st = main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    assert not st['I1 违例'] and not st['I2 不一致'] and not st['I4 单元层增量不一致'], st
    print('通过')
