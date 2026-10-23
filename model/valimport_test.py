"""第 25 轮：值层吃导入的 MLIR（spec/values.md §8，L53）。
  A0 属性账：值层覆盖的每个 op（core.REG 中有 ODS 的），ODS 的每个属性 ∈ 翻译 ∪ 语义无关 ∪ 拒绝（⇒ 未知），列出拒绝项。
  A1 语料覆盖：MLIR 测试集中值层覆盖的 op 节点，翻译 / 未知（按原因）。
  A2 语料 JIT：语料里单块、标量整数/index 参数与单个结果、可完全翻译且值层给出事实的函数——
     值层（irimport → valimport → core.derive）vs 同一提交 MLIR JIT（IW=64），随机输入；UB/毒值/守卫不成立的输入跳过。
  A3 切片定向：static/动态混合的 extract_slice/insert_slice（MLIR 文本 → 解析 → 导入），值层 vs JIT。
  A4 变异灵敏度：翻译里故意把静态偏移当 0（static_zero）/ cmpi 谓词编号异或 1（pred），A2/A3 必须能发现。
     （丢 nsw 不可测：nsw 只把值变成毒值，而毒值输入本就跳过——毒值的精化自由度不能用执行来检验。）
用法：PYTHONPATH=model python3 model/valimport_test.py [A3 程序数] [文件数上限]
"""
import sys, re, glob, random, ctypes, collections, multiprocessing as mp
import z3

CT = {'i1': ctypes.c_bool, 'i8': ctypes.c_int8, 'i16': ctypes.c_int16, 'i32': ctypes.c_int32, 'i64': ctypes.c_int64, 'index': ctypes.c_int64}
PIPE = ('builtin.module(one-shot-bufferize{bufferize-function-boundaries},func.func(buffer-deallocation-simplification),'
        'convert-bufferization-to-memref,expand-strided-metadata,lower-affine,convert-scf-to-cf,finalize-memref-to-llvm,'
        'convert-arith-to-llvm,convert-index-to-llvm,convert-ub-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)')
def W(t): return 64 if t == 'index' else int(t[1:])
def consts(t):
    w = W(t)
    if w == 1: return [0, 1]
    return [0, 1, 2, 3, 5, 7, -1, -2, w - 1, w, (1 << (w - 1)) - 1, -(1 << (w - 1))]

def A0():
    import core as C, odssig as G, valimport as V
    out = collections.Counter(); refuse = []
    for op in sorted(C.REG):
        lk = G.lookup(op)
        if lk is None: continue
        for nm, *_ in lk[1]['attrs']:
            if (op, nm) in V.TRANSLATED: out['翻译'] += 1
            elif (op, nm) in V.IRRELEVANT: out['语义无关'] += 1
            elif nm in V.STRUCTURAL: out['结构'] += 1
            else: out['拒绝（⇒ 未知）'] += 1; refuse.append(f'{op}.{nm}')
    for op in ('scf.for', 'scf.if', 'scf.yield', 'func.return'):
        for nm, *_ in G.lookup(op)[1]['attrs']: out['拒绝（⇒ 未知）'] += 1; refuse.append(f'{op}.{nm}')
    return out, refuse

def model_eval(S, ins_ty, xs):
    """S：core.derive 的结果；返回 ('skip', 原因) 或 ('val', 无符号值)。"""
    import core as C
    pairs = []
    for (h, hp), t, x in zip(S.holes, ins_ty, xs):
        pairs.append((h, z3.IntVal(x) if t == 'index' else z3.BitVecVal(x % (1 << W(t)), W(t)))); pairs.append((hp, z3.BoolVal(False)))
    sub = lambda f: z3.simplify(z3.substitute(f, pairs)) if pairs else z3.simplify(f)
    if not all(z3.is_true(sub(f)) for _, f in S.guards): return 'skip', '守卫'
    if not z3.is_false(sub(S.ubf)): return 'skip', 'UB'
    v, p = S.outv[0]
    if not z3.is_false(sub(p)): return 'skip', '毒值'
    vv = sub(v)
    if z3.is_int_value(vv): return 'val', vv.as_long() % (1 << 64)
    if z3.is_bv_value(vv): return 'val', vv.as_long()
    return 'skip', '未定'

def jit_run(module_text, fname, ins_ty, rty, inputs):
    import mlir.ir as ir, os
    from mlir.passmanager import PassManager
    from mlir.execution_engine import ExecutionEngine
    with ir.Context():
        m = ir.Module.parse(module_text)
        for op in m.body.operations:
            if op.operation.name == 'func.func': op.operation.attributes['llvm.emit_c_interface'] = ir.UnitAttr.get()
        PassManager.parse(PIPE).run(m.operation)
        lib = os.path.join(os.path.dirname(ir.__file__), '_mlir_libs', 'libmlir_c_runner_utils.so')
        ee = ExecutionEngine(m, opt_level=0, shared_libs=[lib])
        res = []
        for xs in inputs:
            a = [ctypes.pointer(CT[t](x if t != 'i1' else bool(x & 1))) for t, x in zip(ins_ty, xs)]
            r = CT[rty](0); ee.invoke(fname, *a, ctypes.pointer(r))
            res.append(int(r.value) % (1 << W(rty)))
        return res

def eq(t, mv, jv): return mv % (1 << W(t)) == jv % (1 << W(t))

def derive_imported(name, fn, flip=None):
    import core as C, valimport as V
    blocks = fn['regions'][0]; ft = fn['props']['_attrs']['function_type']
    try: bs = V.to_core(blocks, flip)
    except V.Unknown as e: return None, str(e)
    ins_ty = [t for _, t in blocks[0]['args']]
    m = re.fullmatch(r'\((.*)\) -> \(?([^()]*?)\)?', ft)
    outs = [x.strip() for x in m.group(2).split(',')] if m and m.group(2).strip() else []
    old = C.ORACLE_ON; C.ORACLE_ON = False
    try: S = C.derive(name, [(f'p{i}', t) for i, t in enumerate(ins_ty)], [(f'r{i}', t) for i, t in enumerate(outs)], bs)
    except Exception as e: return None, f'core 异常 {type(e).__name__}: {str(e)[:80]}'
    finally: C.ORACLE_ON = old
    if not S.value_known: return None, '值层未知（类型/结构/ValueUnknown）'
    return S, None

def run_file(path, q, seed):
    import mlir.ir as ir, irimport as I, core as C, valimport as V
    from compcorpus_test import _walk, _funcs
    rng = random.Random(seed); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 4: ex[k].append(v)
    src = open(path).read()
    if 'allow-unregistered-dialect' in src: q.put((st, dict(ex))); return
    covered = set(C.REG) | {'scf.for', 'scf.if', 'scf.yield'}
    for ci, ch in enumerate(I.split_chunks(src)):
        if not ch.strip() or I.expected(ch): continue
        try: m = ir.Module.parse(ch, ir.Context()); root = I.import_module(m)
        except Exception: continue
        for n in _walk(root):                                                         # A1
            if n['op'] not in covered: continue
            st['A1 覆盖节点'] += 1
            try: V.to_core([dict(name='b', args=[], ops=[n])]); st['A1 翻译'] += 1
            except V.Unknown as e: st['A1 未知/' + re.sub(r'\s.*', '', str(e).split('：', 1)[-1])[:40]] += 1
        for fi, fn in enumerate(_funcs(root)):                                        # A2
            blocks = fn['regions'][0]
            if len(blocks) != 1 or not fn['props']['_attrs'].get('function_type'): continue
            if any(x['op'] not in covered and x['op'] != 'func.return' for x in _walk(fn) if x is not fn): continue
            ins_ty = [t for _, t in blocks[0]['args']]
            ft = fn['props']['_attrs']['function_type']
            rty = re.fullmatch(r'\(.*\) -> \(?([\w]+)\)?', ft)
            if not rty or rty.group(1) not in CT or any(t not in CT for t in ins_ty): continue
            rty = rty.group(1)
            st['A2 候选函数'] += 1
            S, why = derive_imported(f'imp{seed}_{ci}_{fi}', fn)
            if S is None:
                st['A2 未知/' + why.split('：')[0][:40]] += 1
                if why.startswith('core 异常'): note('core 异常', f'{path} {why}')
                continue
            st['A2 值层有事实'] += 1
            inputs = [[rng.choice(consts(t)) for t in ins_ty] for _ in range(8)]
            ev = [model_eval(S, ins_ty, xs) for xs in inputs]
            run = [(xs, r[1]) for xs, r in zip(inputs, ev) if r[0] == 'val']
            for r in ev:
                if r[0] == 'skip': st['A2 跳过/' + r[1]] += 1
            if not run: continue
            name = fn['props']['_attrs']['sym_name'].strip('"')
            ftxt = next((str(o.operation) for o in m.body.operations if o.operation.name == 'func.func' and
                         str(o.operation.attributes['sym_name']).strip('"') == name), None)
            try: outs = jit_run(ftxt, name, ins_ty, rty, [xs for xs, _ in run])   # 只发射该函数：同块里的其他函数可能无法降级
            except Exception as e: st['A2 JIT 不可用'] += 1; note('JIT 不可用', f'{path}:{name} {str(e)[:100]}'); continue
            st['A2 函数（已执行）'] += 1
            for (xs, mv), jv in zip(run, outs):
                st['A2 输入'] += 1
                if eq(rty, mv, jv): st['A2 一致'] += 1
                else: st['A2 不一致'] += 1; note('A2 不一致', f'{path}:{name} {xs} 模型 {mv} JIT {jv}')
            for flip in ('pred', 'static_zero'):                                    # A4（只在本函数上）
                S2, _ = derive_imported(f'imp{seed}_{ci}_{fi}_{flip}', fn, flip)
                if S2 is None: continue
                for (xs, _), jv in zip(run, outs):
                    r = model_eval(S2, ins_ty, xs)
                    if r[0] == 'val': st[f'A4 {flip} 输入'] += 1; st[f'A4 {flip} 发现'] += not eq(rty, r[1], jv)
    q.put((st, dict(ex)))

# ───────────────────────── A3 定向切片
def gen3(rng):
    """6 元素 tensor（from_elements a,b,c,…）上的 extract_slice / insert_slice，静态/动态混合，再 extract 出 i64。"""
    L = ['  %t = tensor.from_elements %a, %b, %c, %a, %b, %c : tensor<6xi64>',
         '  %ia = arith.index_cast %a : i64 to index', '  %ib = arith.index_cast %b : i64 to index', '  %ic = arith.index_cast %c : i64 to index']
    def ix(lo, hi):
        if rng.random() < 0.5: return str(rng.randrange(lo, hi)), None
        return None, rng.choice(['%ia', '%ib', '%ic'])
    def sl(off, size):
        o = off[0] if off[0] is not None else off[1]; s = size[0] if size[0] is not None else size[1]
        return f'[{o}][{s}][1]'
    k = rng.random(); cur, cty = '%t', 'tensor<6xi64>'
    if k < 0.6:
        off, size = ix(0, 4), ix(1, 4)
        rt = f'tensor<{size[0]}xi64>' if size[0] is not None else 'tensor<?xi64>'
        L.append(f'  %s = tensor.extract_slice %t{sl(off, size)} : tensor<6xi64> to {rt}'); cur, cty = '%s', rt
    else:
        n = rng.randrange(1, 4)
        L.append(f'  %src = tensor.from_elements ' + ', '.join(rng.choice(['%c', '%b']) for _ in range(n)) + f' : tensor<{n}xi64>')
        off = ix(0, 6 - n + 1)
        L.append(f'  %s = tensor.insert_slice %src into %t{sl(off, (str(n), None))} : tensor<{n}xi64> into tensor<6xi64>'); cur = '%s'
    j = rng.choice(['0', '1', '2', '%ia', '%ic'])
    if j.isdigit(): L.append(f'  %j = arith.constant {j} : index'); j = '%j'
    L.append(f'  %e = tensor.extract {cur}[{j}] : {cty}')
    return ('func.func @main(%a: i64, %b: i64, %c: i64) -> i64 {\n' + '\n'.join(L) + '\n  return %e : i64\n}')

def A3(n, seed=7):
    import mlir.ir as ir, irimport as I
    rng = random.Random(seed); st = collections.Counter(); bad = []
    for t in range(n):
        txt = gen3(rng)
        try: m = ir.Module.parse(txt, ir.Context())
        except Exception as e: st['不合法（MLIR 拒绝，跳过）'] += 1; continue
        fn = I.import_module(m)['regions'][0][0]['ops'][0]
        S, why = derive_imported(f'a3_{t}', fn)
        if S is None: st['未知/' + why[:30]] += 1; continue
        st['程序'] += 1
        inputs = [[rng.randrange(0, 4) for _ in range(3)] for _ in range(6)]
        ev = [model_eval(S, ['i64'] * 3, xs) for xs in inputs]
        run = [(xs, r[1]) for xs, r in zip(inputs, ev) if r[0] == 'val']
        st['跳过（守卫/UB/毒值）'] += len(inputs) - len(run)
        if not run: continue
        outs = jit_run(txt, 'main', ['i64'] * 3, 'i64', [xs for xs, _ in run])
        for (xs, mv), jv in zip(run, outs):
            st['输入'] += 1; ok = eq('i64', mv, jv); st['一致' if ok else '不一致'] += 1
            if not ok and len(bad) < 3: bad.append((txt, xs, mv, jv))
        S2, _ = derive_imported(f'a3_{t}_z', fn, 'static_zero')
        if S2 is not None:
            for (xs, _), jv in zip(run, outs):
                r = model_eval(S2, ['i64'] * 3, xs)
                if r[0] == 'val': st['A4 static_zero 输入'] += 1; st['A4 static_zero 发现'] += not eq('i64', r[1], jv)
    return st, bad

if __name__ == '__main__':
    n3 = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9
    out, refuse = A0()
    print('── A0 属性账:', dict(out)); print('   拒绝（⇒ 未知）:', refuse)
    st, bad = A3(n3)
    print('── A3 切片定向:', dict(st))
    for b in bad: print('  ✗', b[1:], '\n', b[0])
    tot = collections.Counter(); exs = collections.defaultdict(list); ctxm = mp.get_context('fork')
    for i, path in enumerate(sorted(glob.glob('corpus/*/*.mlir'))[:lim]):
        q = ctxm.Queue(); pr = ctxm.Process(target=run_file, args=(path, q, i)); pr.start()
        try: s, ex = q.get(timeout=600)
        except Exception: pr.kill(); tot['文件超时/崩溃'] += 1; print('  ! 超时/崩溃', path, flush=True); continue
        pr.join(); tot.update(s)
        for k, v in ex.items(): exs[k] += v[:2]
    print('── A1/A2/A4 语料'); [print(f'   {k}: {v}') for k, v in sorted(tot.items())]
    for k, v in exs.items(): print('──', k); [print('    ', x) for x in v[:8]]
