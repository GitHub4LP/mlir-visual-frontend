"""值层审计扩面（第 23 轮，spec/values.md）。真值只取 MLIR 自己：ODS（属性账）、JIT 执行（O0/O3，index 位宽 64 与 32）、
MLIR 利用毒值/UB 的变换（canonicalize + int-range-optimizations、sccp）。

  S0 语义属性账：值层覆盖的每个 arith op，ODS 的每个属性都必须被语义读取（经 core.ATTR_MAP 译成 core 参数），或登记为语义无关。
     反向：core 读取而 MLIR 没有的参数（第 11 轮 xori+nuw 一类）也列出。附回溯：第 20 轮 / 第 22 轮的 reads 表在这本账上会报什么。
  V5 JIT 定向 + 随机：移位、divsi/remsi/remui、exact、trunci 标志，类型 i8/i16/i32/i64/index；index 在 IW=64 与 IW=32
     （降低时 index-bitwidth=32）下分别对照。模型给出确定值而 JIT 不同 ⇒ 模型比 MLIR 更“有定义”（不可靠）。
     对照：FLAGS_EXT=False（第 22 轮前的语义）应在同一批程序上出现反例——证明本检验会“利用”被测语义（第 21 轮教训）。
  V6 翻译确认：随机程序（含新 op 与标志）交 canonicalize + int-range-optimizations，读回后在每个语义参数取值
     （IW ∈ {32, 64} × remsi(INT_MIN,-1) ∈ {LangRef, UB}）下判 refines(原, 结果)。
  V7 scf + sccp：loop_test 的循环程序经 sccp + canonicalize 后 JIT，与模型对原程序给出的确定值比较。
用法：PYTHONPATH=model python3 model/values_test.py [V5 程序数] [V6 用例数] [V7 程序数]
"""
import random, sys, ctypes, collections, z3
import mlir.ir as ir
from mlir.passmanager import PassManager
import core as C, loop_test as LT
from core import blk, node

NEW = ['arith.shli', 'arith.shrui', 'arith.shrsi', 'arith.divui', 'arith.divsi', 'arith.remui', 'arith.remsi']
OLD = ['arith.addi', 'arith.subi', 'arith.muli', 'arith.andi', 'arith.ori', 'arith.xori']
FLAGS_OF = {'arith.addi': ('nsw', 'nuw'), 'arith.subi': ('nsw', 'nuw'), 'arith.muli': ('nsw', 'nuw'), 'arith.shli': ('nsw', 'nuw'),
            'arith.shrui': ('exact',), 'arith.shrsi': ('exact',), 'arith.divui': ('exact',), 'arith.divsi': ('exact',), 'arith.trunci': ('nsw', 'nuw')}

# ───────────────────────── S0
IRRELEVANT = {('ub.poison', 'value'): '毒值属性的种类（PoisonAttrInterface）；结果恒为毒值，与种类无关'}   # (op, ODS 属性) → 理由
def S0(reads_override=None):
    import odssig as G
    out = collections.OrderedDict(); gaps, extra = [], []
    for op, sch in sorted(C.REG.items()):
        if not op.startswith(('arith.', 'ub.')) or not hasattr(sch, 'reads'): continue
        lk = G.lookup(op)
        if lk is None: out[op] = '无 ODS'; continue
        reads = (reads_override or {}).get(op, sch.reads)
        if reads is None: out[op] = '未对账'; continue
        need = set()
        for nm, *_ in lk[1]['attrs']:
            if (op, nm) in IRRELEVANT: continue
            if nm not in C.ATTR_MAP: gaps.append((op, nm, '无译名')); continue
            for k in C.ATTR_MAP[nm]:
                need.add(k)
                if k not in reads: gaps.append((op, nm, k))
        for k in set(reads) - need - {'type'}: extra.append((op, k))
        out[op] = 'ok'
    return out, gaps, extra
R20 = {'arith.addi': {'nuw'}, 'arith.subi': {'nuw'}, 'arith.muli': {'nuw'}, 'arith.divui': set(), 'arith.trunci': set()}   # 第 20 轮的语义读到的参数
R22 = {'arith.divui': set(), 'arith.trunci': set()}                                                                       # 第 22 轮

# ───────────────────────── 通用发射（通用形式）与读回
def attr_text(n):
    pr, op, ats = n.get('props', {}), n['op'], []
    if op == 'arith.constant': ats.append(f'value = {pr["value"]} : {pr["type"]}')
    if op == 'arith.cmpi': ats.append(f'predicate = {pr["predicate"]} : i64')
    fl = [f for f in ('nsw', 'nuw') if pr.get(f)]
    if fl: ats.append(f'overflowFlags = #arith.overflow<{", ".join(fl)}>')
    if pr.get('exact'): ats.append('isExact')
    return f'<{{{", ".join(ats)}}}>' if ats else ''
def emit(blocks, ins, rty):
    ty = dict(ins); v = lambda x: '%' + x; out = []
    for n in blocks[0]['ops']:
        for o, t in n['outs']: ty[o] = t
        if n['op'] == 'func.return': out.append(f'  return {v(n["ins"][0])} : {rty}'); continue
        sig = '(' + ', '.join(ty[x] for x in n['ins']) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')'
        out.append(f'  {v(n["outs"][0][0])} = "{n["op"]}"(' + ', '.join(v(x) for x in n['ins']) + f') {attr_text(n)} : {sig}')
    args = ', '.join(f'%{a}: {t}' for a, t in ins)
    return f'func.func @main({args}) -> {rty} attributes {{llvm.emit_c_interface}} {{\n' + '\n'.join(out) + '\n}'
def read_back(func, ins):
    """MLIR → core：属性经 core.ATTR_MAP 翻译；未知属性 ⇒ KeyError（不静默丢弃）。"""
    names = {}; cnt = [0]; ops = []
    for i, a in enumerate(func.regions[0].blocks[0].arguments): names[a.get_name()] = ins[i][0]
    def vid(val):
        k = val.get_name()
        if k not in names: cnt[0] += 1; names[k] = f'm{cnt[0]}'
        return names[k]
    for op in func.regions[0].blocks[0].operations:
        o = op.operation; name = o.name
        if name not in C.REG and name != 'func.return': raise KeyError(name)
        outs = [(vid(r), str(r.type)) for r in o.results]; xs = [vid(x) for x in o.operands]; pr = {}
        for na in o.attributes:
            k = na.name if hasattr(na, 'name') else na; a = o.attributes[k]
            if k == 'value': pr['value'] = ir.IntegerAttr(a).value; pr['type'] = str(o.results[0].type)
            elif k == 'predicate': pr['predicate'] = ir.IntegerAttr(a).value
            elif k == 'overflowFlags': pr.update({f: True for f in ('nsw', 'nuw') if f in str(a)})
            elif k == 'isExact': pr['exact'] = True
            else: raise KeyError(f'{name}.{k}')
        ops.append(node(f'n{cnt[0]}_{len(ops)}', name, ins=xs, outs=outs, props=pr)); cnt[0] += 1
    return [blk('entry', ops, args=list(ins))]

# ───────────────────────── V5 JIT
INS64 = [('a', 'i64'), ('b', 'i64'), ('c', 'i64')]
def width(T, iw): return iw if T == 'index' else int(T[1:])
def consts(T, iw):
    w = width(T, iw)
    return [0, 1, 2, 3, 4, w - 1, w, w + 1, -1, -(1 << (w - 1)), (1 << (w - 1)) - 1, 7, 16]
def gen5(rng, t, iw):
    T = rng.choice(['i8', 'i16', 'i32', 'i64', 'index', 'index'])
    k = [0]; ops = []
    def fresh(): k[0] += 1; return f'v{k[0]}'
    xs = []
    for a in ('a', 'b', 'c'):
        if T == 'i64': xs.append(a); continue
        v = fresh(); ops.append(node(v, 'arith.index_cast' if T == 'index' else 'arith.trunci', ins=[a], outs=[(v, T)])); xs.append(v)
    def cst():
        v = fresh(); ops.append(C.cst(v, rng.choice(consts(T, iw)), T)); return v
    pick = lambda: rng.choice(xs[-3:]) if rng.random() < 0.7 else rng.choice(xs)
    if rng.random() < 0.5:                                          # 定向：LLVM/MLIR 会利用标志的形状（见 spec/values.md §3）
        x = pick(); s = cst(); v = fresh(); w = fresh(); kind = rng.randrange(5)
        if kind == 0: ops += [C.op2(v, rng.choice(['arith.divui', 'arith.divsi']), x, s, T, exact=True)]; xs.append(v)
        elif kind == 1:
            op = rng.choice(['arith.shrui', 'arith.shrsi']); ops += [C.op2(v, op, x, s, T, exact=True), C.op2(w, 'arith.shli', v, s, T)]; xs += [v, w]
        elif kind == 2:
            f = rng.choice(['nuw', 'nsw']); ops += [C.op2(v, 'arith.shli', x, s, T, **{f: True}),
                                                    C.op2(w, 'arith.shrui' if f == 'nuw' else 'arith.shrsi', v, s, T)]; xs += [v, w]
        elif kind == 3 and T != 'index' and width(T, iw) > 8:
            f = rng.choice(['nuw', 'nsw']); ops += [node(v, 'arith.trunci', ins=[x], outs=[(v, 'i8')], props={f: True}),
                                                    node(w, 'arith.extui' if f == 'nuw' else 'arith.extsi', ins=[v], outs=[(w, T)])]; xs.append(w)
        else: ops += [C.op2(v, rng.choice(['arith.remsi', 'arith.divsi']), x, s, T)]; xs.append(v)
    for _ in range(rng.randint(1, 4)):
        op = rng.choice(NEW + OLD[:3]); v = fresh()
        y = cst() if rng.random() < 0.6 else pick()
        fl = {f: True for f in FLAGS_OF.get(op, ()) if rng.random() < 0.4}
        ops.append(C.op2(v, op, pick(), y, T, **fl)); xs.append(v)
    r = xs[-1]
    if T != 'i64':
        v = fresh(); ops.append(node(v, 'arith.index_cast' if T == 'index' else 'arith.extsi', ins=[r], outs=[(v, 'i64')])); r = v
    return T, [blk('entry', ops + [C.ret(f'ret{t}', r)], args=INS64)]

def pipe(iw):
    o = f'{{index-bitwidth={iw}}}'
    return f'builtin.module(convert-arith-to-llvm{o},convert-func-to-llvm{o},reconcile-unrealized-casts)'
def jit(src, inputs, iw, opt):
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(src); assert m.operation.verify()
        PassManager.parse(pipe(iw)).run(m.operation)
        ee = ExecutionEngine(m, opt_level=opt); res = []
        for xs in inputs:
            a = [(ctypes.c_int64 * 1)(x) for x in xs]; r = (ctypes.c_int64 * 1)(0)
            ee.invoke('main', *a, r); res.append(r[0] % (1 << 64))
        return res

def V5(n, rng, iw, flags=True):
    C.set_iw(iw); C.FLAGS_EXT = flags
    st = collections.Counter()
    for t in range(n):
        T, g = gen5(rng, t, iw)
        try: G = C.derive(f'V5_{iw}_{flags}_{t}', INS64, [('q', 'i64')], g)
        except C.ValueUnknown: st['值层拒绝'] += 1; continue
        src = emit(g, INS64, 'i64')
        ins = [[rng.choice([0, 1, 2, 3, 5, 7, 8, 64, 100, 255, -1, -128, 1 << 31, 1 << 40, (1 << 63) - 1, -(1 << 63), rng.randrange(-1 << 63, 1 << 63)])
                for _ in range(3)] for _ in range(4)]
        model = [LT.model_eval(G, xs) for xs in ins]
        for opt in (0, 3):
            outs = jit(src, ins, iw, opt)
            for xs, mv, o in zip(ins, model, outs):
                st['输入'] += 1
                if mv[0] != 'val': st[f'模型 {mv[0]}'] += 1; continue
                if mv[1] == o: st['一致'] += 1
                else:
                    st['反例（模型有定义而 MLIR 不同）'] += 1
                    if st['反例（模型有定义而 MLIR 不同）'] <= 2 and flags: print('  ✗', T, f'O{opt}', xs, mv[1], o, '\n', src)
    C.set_iw(64); C.FLAGS_EXT = True
    return dict(st)

# ───────────────────────── V6 翻译确认
PIPE6 = 'builtin.module(func.func(canonicalize,int-range-optimizations,canonicalize))'
def gen6(rng, t):
    T = rng.choice(['i8', 'index']); ins = [('a', T), ('b', T)]
    k = [0]; ops = []; xs = ['a', 'b']
    def fresh(): k[0] += 1; return f'q{t}_{k[0]}'
    for _ in range(rng.randint(2, 6)):
        v = fresh(); r = rng.random()
        if r < 0.3: ops.append(C.cst(v, rng.choice(consts(T, 64) + [8, 64, 63, 100]), T)); xs.append(v); continue
        op = rng.choice(NEW + OLD)
        fl = {f: True for f in FLAGS_OF.get(op, ()) if rng.random() < 0.5}
        ops.append(C.op2(v, op, rng.choice(xs[-3:]), rng.choice(xs), T, **fl)); xs.append(v)
    if rng.random() < 0.5:                                         # 与常量比较：int-range-optimizations 折叠比较的形状
        c = fresh(); b = fresh(); s = fresh()
        ops += [C.cst(b, rng.choice(consts(T, 64)), T), C.op2(c, 'arith.cmpi', xs[-1], b, 'i1', predicate=rng.randrange(10)),
                node(s, 'arith.select', ins=[c, 'a', 'b'], outs=[(s, T)])]; xs.append(s)
    return T, ins, [blk('entry', ops + [C.ret(f'r{t}', xs[-1])], args=ins)]
SETTINGS = [(64, 'langref'), (64, 'ub'), (32, 'langref'), (32, 'ub')]
def V6(n, rng):
    st = collections.Counter()
    for t in range(n):
        T, ins, g = gen6(rng, t)
        with ir.Context():
            src = emit(g, ins, T); m = ir.Module.parse(src); assert m.operation.verify()
            PassManager.parse(PIPE6).run(m.operation)
            try: h = read_back(m.body.operations[0], ins)
            except KeyError as e: st[f'读回失败 {e}'] += 1; continue
            tgt = str(m); changed = tgt.count('\n') < src.count('\n')
        st['用例'] += 1; st['有变化'] += changed
        for iw, rm in SETTINGS:
            C.set_iw(iw); C.REMSI_MIN = rm
            G = C.derive(f'V6s_{t}_{iw}{rm}', ins, [('q', T)], g); H = C.derive(f'V6h_{t}_{iw}{rm}', ins, [('q', T)], h)
            u0 = C.STATS['unknown']; ok = C.refines(G, H)
            key = f'IW={iw}, remsi={rm}'
            if ok: st[f'{key}: 精化'] += 1
            elif C.STATS['unknown'] > u0: st[f'{key}: 求解器未定'] += 1
            else:
                st[f'{key}: 反例'] += 1
                if st[f'{key}: 反例'] <= 1: print(f'  ✗ {key}\n', src, '\n  →\n', tgt)
    C.set_iw(64); C.REMSI_MIN = 'langref'
    return dict(st)

# ───────────────────────── V8 IW=32（WASM32 一类目标）：MLIR 的折叠是否保持语义；arith 与 index 方言对照
IXD = {'arith.addi': 'add', 'arith.subi': 'sub', 'arith.muli': 'mul', 'arith.divui': 'divu', 'arith.divsi': 'divs', 'arith.remui': 'remu',
       'arith.remsi': 'rems', 'arith.shli': 'shl', 'arith.shrui': 'shru', 'arith.shrsi': 'shrs', 'arith.andi': 'and', 'arith.ori': 'or', 'arith.xori': 'xor'}
def gen8(rng, t):
    k = [0]; ops = []
    def fresh(): k[0] += 1; return f'w{k[0]}'
    xs = []
    for a in ('a', 'b'):
        v = fresh(); ops.append(node(v, 'arith.index_cast', ins=[a], outs=[(v, 'index')])); xs.append(v)
    if rng.random() < 0.6:                    # 定向：常量子表达式在 32 位上溢出（折叠器才会动它；第 21 轮教训：随机生成触发不到）
        c1, c2, v = fresh(), fresh(), fresh()
        ops += [C.cst(c1, rng.choice([(1 << 31) - 1, -(1 << 31), 1 << 30, 65536, 3 << 29]), 'index'),
                C.cst(c2, rng.choice([1, 2, 3, 65536, (1 << 31) - 1, -1]), 'index'),
                C.op2(v, rng.choice(['arith.addi', 'arith.subi', 'arith.muli', 'arith.shli']), c1, c2, 'index')]
        xs += [c1, c2, v]
        if rng.random() < 0.5:                # 折叠后的常量再进入比较/除法/右移：32 位与 64 位的结果在这里分叉
            w = fresh(); c3 = fresh()
            ops += [C.cst(c3, rng.choice([2, 3, 7, 1]), 'index'), C.op2(w, rng.choice(['arith.divsi', 'arith.remsi', 'arith.shrsi', 'arith.divui']), v, c3, 'index')]
            xs += [c3, w]
    directed = len(xs) > 2
    for _ in range(rng.randint(0, 3) if directed else rng.randint(2, 5)):
        v = fresh()
        if rng.random() < 0.45: ops.append(C.cst(v, rng.choice([0, 1, 2, 3, 31, 32, -1, (1 << 31) - 1, -(1 << 31), 1 << 30, 65535]), 'index'))
        else: ops.append(C.op2(v, rng.choice(list(IXD)), rng.choice(xs[-3:]), rng.choice(xs), 'index'))
        xs.append(v)
    c, z, s = fresh(), fresh(), fresh()
    tgt = xs[-1] if not directed or rng.random() < 0.5 else xs[4] if len(xs) > 4 else xs[-1]
    ops += [C.cst(z, rng.choice([0, 1, -1, (1 << 31) - 1]), 'index'), C.op2(c, 'arith.cmpi', tgt, z, 'i1', predicate=rng.randrange(10)),
            node(s, 'arith.select', ins=[c, 'a', 'b'], outs=[(s, 'i64')])]
    return [blk('entry', ops + [C.ret(f'r{t}', s)], args=INS64[:2])]
def emit_index_dialect(blocks):
    """同一程序，index 上的运算改用 index 方言（其折叠器只在 32 位与 64 位结果一致时折叠）。"""
    out = []
    for n in blocks[0]['ops']:
        o = n['outs'][0][0] if n['outs'] else None; op, pr = n['op'], n.get('props', {})
        if op == 'func.return': out.append(f'  return %{n["ins"][0]} : i64')
        elif op == 'arith.index_cast': out.append(f'  %{o} = index.casts %{n["ins"][0]} : i64 to index')
        elif op == 'arith.constant' and pr['type'] == 'index': out.append(f'  %{o} = index.constant {pr["value"]}')
        elif op == 'arith.cmpi': out.append(f'  %{o} = index.cmp {C.PRED[pr["predicate"]]}(%{n["ins"][0]}, %{n["ins"][1]})')
        elif op in IXD: out.append(f'  %{o} = index.{IXD[op]} %{n["ins"][0]}, %{n["ins"][1]}')
        elif op == 'arith.select': out.append(f'  %{o} = arith.select %{n["ins"][0]}, %{n["ins"][1]}, %{n["ins"][2]} : i64')
        else: raise KeyError(op)
    return 'func.func @main(%a: i64, %b: i64) -> i64 attributes {llvm.emit_c_interface} {\n' + '\n'.join(out) + '\n}'
def pipe_ix(iw):
    o = f'{{index-bitwidth={iw}}}'
    return f'builtin.module(convert-index-to-llvm{o},convert-arith-to-llvm{o},convert-func-to-llvm{o},reconcile-unrealized-casts)'
def V8(n, rng):
    st = collections.Counter(); global pipe
    base_pipe = pipe
    for t in range(n):
        g = gen8(rng, t)
        ins = [[rng.choice([0, 1, 2, 5, -1, (1 << 31) - 1, -(1 << 31), 1 << 31, 1 << 32, rng.randrange(-1 << 31, 1 << 31)]), rng.choice([0, 1, 7, -1])] for _ in range(4)]
        mv = {}
        for iw in (32, 64):
            C.set_iw(iw)
            try: G = C.derive(f'V8_{t}_{iw}', INS64[:2], [('q', 'i64')], g); mv[iw] = [LT.model_eval(G, xs) for xs in ins]
            except C.ValueUnknown: mv[iw] = [('?',)] * len(ins)
        C.set_iw(64)
        for dia, src in (('arith', emit(g, INS64[:2], 'i64')), ('index', emit_index_dialect(g))):
            pipe = base_pipe if dia == 'arith' else pipe_ix
            o32 = jit(src, ins, 32, 0)
            if dia == 'arith':
                for m, o in zip(mv[32], o32):
                    if m[0] == 'val': st['模型@32 = JIT@32'] += m[1] == o; st['模型@32 ≠ JIT@32'] += m[1] != o
            for P in ('canonicalize', 'int-range-optimizations', 'sccp'):
                with ir.Context():
                    mm = ir.Module.parse(src); PassManager.parse(f'builtin.module(func.func({P}))').run(mm.operation); tgt = str(mm)
                for iw in (32, 64):
                    a = o32 if iw == 32 else jit(src, ins, 64, 0); b = jit(tgt, ins, iw, 0)
                    for m, x, y in zip(mv[iw], a, b):
                        if m[0] != 'val': continue
                        st[f'{dia}/{P}/IW={iw}: 输入'] += 1; st[f'{dia}/{P}/IW={iw}: 误编译'] += x != y
        pipe = base_pipe
    C.set_iw(64)
    return dict(st)

# ───────────────────────── V9 index 方言 CmpOp::canonicalize：cmp(pred, sub(x, y), 0) → cmp(pred, x, y)（V8 中 index/canonicalize 的误编译归因于此）
def V9():
    global pipe
    base = pipe; pipe = pipe_ix; out = {}
    ins = [[0, 2], [2, 0], [-(1 << 63), 1], [1, -(1 << 63)], [5, 5], [-(1 << 31), 1]]
    for pred in C.PRED:
        src = ('func.func @main(%a: i64, %b: i64) -> i64 attributes {llvm.emit_c_interface} {\n  %x = index.casts %a : i64 to index\n'
               '  %y = index.casts %b : i64 to index\n  %d = index.sub %x, %y\n  %z = index.constant 0\n'
               f'  %p = index.cmp {pred}(%d, %z)\n  %r = arith.select %p, %a, %b : i64\n  return %r : i64\n}}')
        with ir.Context():
            m = ir.Module.parse(src); PassManager.parse('builtin.module(func.func(canonicalize))').run(m.operation); tgt = str(m)
        bad = {iw: sum(x != y for x, y in zip(jit(src, ins, iw, 0), jit(tgt, ins, iw, 0))) for iw in (64, 32)}
        out[pred] = ('改写' if 'index.sub' not in tgt else '未改写', bad)
    pipe = base
    return out

# ───────────────────────── V7 scf + sccp
PIPE7 = 'builtin.module(sccp,canonicalize,cse)'
def V7(n, rng):
    st = collections.Counter()
    for t in range(n):
        g = LT.gen(rng, t); G = C.derive(f'V7_{t}', LT.INS, [('q', 'i64')], g)
        src = LT.emit(g)
        with ir.Context():
            m = ir.Module.parse(src); PassManager.parse(PIPE7).run(m.operation); tgt = str(m)
        st['程序'] += 1; st['含循环'] += any(x['op'] == 'scf.for' for x in g[0]['ops']); st['sccp 后变小'] += tgt.count('\n') < src.count('\n')
        ins = [[rng.choice([0, 1, 2, 3, 200, -1, 1 << 40, (1 << 63) - 1, -(1 << 63), rng.randrange(-1 << 63, 1 << 63)]) for _ in range(3)] for _ in range(4)]
        o0, o1 = LT.jit(src, ins), LT.jit(tgt, ins)
        for xs, a, b in zip(ins, o0, o1):
            mv = LT.model_eval(G, xs); st['输入'] += 1
            if mv[0] != 'val': st[f'模型 {mv[0]}'] += 1; continue
            st['模型=原程序 JIT'] += mv[1] == a; st['模型=sccp 后 JIT'] += mv[1] == b
            if mv[1] != b: st['反例'] += 1
    return dict(st)

if __name__ == '__main__':
    n5 = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    n6 = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    n7 = int(sys.argv[3]) if len(sys.argv) > 3 else 100
    out, gaps, extra = S0()
    print('── S0 语义属性账:', dict(collections.Counter(out.values())), '缺口', gaps, 'core 多出的参数', extra)
    print('   回溯：第 20 轮的 reads ⇒ 缺口', S0(R20)[1])
    print('   回溯：第 22 轮的 reads ⇒ 缺口', S0(R22)[1])
    for iw in (64, 32):
        print(f'── V5 JIT（IW={iw}）:', V5(n5, random.Random(5 + iw), iw))
        print(f'   对照 FLAGS_EXT=False（第 22 轮前语义）:', V5(n5, random.Random(5 + iw), iw, flags=False))
    print('── V6 canonicalize + int-range-optimizations 翻译确认:', V6(n6, random.Random(6)))
    print('── V7 scf + sccp:', V7(n7, random.Random(7)))
    print('── V8 IW=32 下 MLIR 折叠是否保持语义（arith vs index 方言）:', V8(n7, random.Random(8)))
    print('── V9 index 方言 cmp(sub(x,y),0) 规范化（每个谓词：是否改写，{IW: 误编译输入数}）:', V9())
    print('── 求解器 unknown:', C.STATS['unknown'])
