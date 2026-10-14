"""结构判定的随机差分测试：随机 CFG 函数 + 随机符号表，模型判定 vs 真实 MLIR verifier。
统计三类结果：一致 / 保守（模型拒绝、MLIR 接受）/ 不可靠（模型接受、MLIR 拒绝，必须为 0）。
用法：PYTHONPATH=model python3 model/structure_fuzz.py [用例数] [种子]
"""
import random, sys, collections
import structure as S
from structure import blk, node, M, F, ret

TYS = ['i32', 'i64']

SIGS = {}
def sig_of(rng, name):
    if name not in SIGS:
        SIGS[name] = ([rng.choice(TYS) for _ in range(rng.randint(0, 2))], [rng.choice(TYS) for _ in range(rng.randint(0, 1))])
    return SIGS[name]

def rand_func(rng, name, callees):
    nb = rng.randint(1, 4)
    names = ['entry'] + [f'b{i}' for i in range(1, nb)]
    fin, fout = sig_of(rng, name)
    bargs = {'entry': [(f'{name}_a{i}', t) for i, t in enumerate(fin)]}
    for b in names[1:]:
        bargs[b] = [(f'{name}_{b}_p{i}', rng.choice(TYS)) for i in range(rng.randint(0, 2))]
    pool = []                                    # (值, 类型)：全函数里定义的值，故意跨块取用以触发支配问题
    blocks = []
    uid = [0]
    def fresh(): uid[0] += 1; return f'{name}_v{uid[0]}'
    for b in names:
        ops = []
        local = list(bargs[b])
        for _ in range(rng.randint(0, 3)):
            r = rng.random()
            if r < 0.4:
                t = rng.choice(TYS); v = fresh(); ops.append(S.cst(v, 1, t)); local.append((v, t))
            elif r < 0.7 and (local or pool):
                src = rng.choice(local + pool if rng.random() < 0.3 else (local or pool))
                v = fresh(); ops.append(node(v, 'test.use', ins=[src[0]], outs=[(v, src[1])])); local.append((v, src[1]))
            elif callees:
                c = rng.choice(callees); cin, cout = sig_of(rng, c)
                args = []
                for t in cin:
                    cands = [v for v, tt in local if tt == t or rng.random() < 0.05]
                    if cands: args.append(rng.choice(cands))
                outs = [(fresh(), t if rng.random() > 0.05 else 'i32') for t in cout]
                ops.append(node(fresh(), 'func.call', callee='@' + c, ins=args, outs=outs)); local += outs
        pool += local
        def pick_args(target):
            out = []
            for _, t in bargs[target]:
                cands = [v for v, tt in (local if rng.random() < 0.85 else pool) if tt == t or rng.random() < 0.1]
                if not cands: cands = [v for v, _ in local] or [v for v, _ in pool]
                if not cands: return None
                out.append(rng.choice(cands))
            if rng.random() < 0.05 and out: out.pop()
            return out
        r = rng.random()
        term = None
        if r < 0.35 and nb > 1:
            tgt = rng.choice(names[1:] + (['nowhere'] if rng.random() < 0.03 else []))
            a = pick_args(tgt) if tgt in bargs else []
            if a is not None: term = node(fresh(), 'cf.br', succs=[(tgt, a)])
        elif r < 0.55 and nb > 1:
            conds = [v for v, t in local if t == 'i1']
            c = S.cst(fresh(), 1, 'i1'); ops.append(c)
            t1, t2 = rng.choice(names[1:]), rng.choice(names[1:])
            a1, a2 = pick_args(t1), pick_args(t2)
            if a1 is not None and a2 is not None:
                term = node(fresh(), 'cf.cond_br', ins=[c['id']], succs=[(t1, a1), (t2, a2)])
        if term is None and rng.random() > 0.03:
            vals = []
            for t in fout:
                cands = [v for v, tt in local if tt == t] or ([v for v, _ in local] if rng.random() < 0.3 else [])
                if not cands and rng.random() < 0.5: cands = [v for v, tt in pool if tt == t]
                if cands: vals.append(rng.choice(cands))
            term = ret(fresh(), ins=vals)
        if term is not None: ops.append(term)
        blocks.append(blk(b, ops, bargs[b]))
    ft = '(' + ', '.join(fin) + ') -> ' + ('(' + ', '.join(fout) + ')' if len(fout) != 1 else fout[0])
    return F(name, blocks, ftype=ft, args=bargs['entry'], vis=rng.choice(['public', 'private']))

def rand_program(rng):
    SIGS.clear()
    top = [f'f{i}' for i in range(rng.randint(1, 3))]
    inner = [f'g{i}' for i in range(rng.randint(0, 2))]
    names_top = top + (rng.sample(top, 1) if rng.random() < 0.05 else [])     # 偶发重名
    funcs = []
    for i, n in enumerate(names_top):
        callees = top + (inner if rng.random() < 0.15 else [])               # 偶发引用嵌套表里的符号
        f = rand_func(rng, n, callees); f['id'] = f'{n}_{i}'; funcs.append(f)
    if inner:
        ifs = [rand_func(rng, n, inner + (top if rng.random() < 0.15 else [])) for n in inner]
        funcs.append(node('mod', 'builtin.module', sym='m', regions=[[blk('entry', ifs)]]))
    return M(*funcs)

if __name__ == '__main__':
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    rng = random.Random(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    stat = collections.Counter(); obl = collections.Counter(); unsound = []; cons = []
    for k in range(N):
        g = rand_program(rng)
        okm, d = S.verdict_model(g)
        okr, msg = S.verdict_mlir(g)
        if okm and okr: stat['都接受'] += 1
        elif not okm and not okr: stat['都拒绝'] += 1
        elif okm: stat['不可靠'] += 1; unsound.append((S.emit(g), msg))
        else: stat['保守'] += 1; cons.append(d)
        for _, o, _ in d: obl[o] += 1
    print(dict(stat)); print('触发的义务：', dict(obl))
    for src, m in unsound[:2]: print('── 不可靠反例\n', src, '\n', m)
    for d in cons[:3]: print('── 保守', d[:2])
