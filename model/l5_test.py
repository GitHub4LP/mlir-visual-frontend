"""第 52 轮：L5（needs_ssacfg 的体不能内联进 Graph 区域）的外部证据，以及 Graph 区域的结构义务。

MLIR 的规则（9bae84b0）：
  · Verifier.cpp verifyOnEntrance：已注册 op 的 Graph 区域“0 或 1 块”（不依赖 SingleBlock trait）；
  · Dominance.cpp / RegionKindInterface.cpp mayBeGraphRegion：Graph 区域与未注册 op 的区域不查支配。
所以 L5 的正确判据是“顶层区域多于 1 块”，而不是第 51 轮及以前的“体内任何处有后继”（嵌套 op 自己的区域内联后种类不变）。

L5a 源码审计：ODS 中带 Graph 区域的 op 都同时带 SingleBlock（规则在 MLIR 里有两道，结论一致）。
L5b 真实 IR：语料中 MLIR 接受的函数，把体内联进 builtin.module（Graph 区域；参数换成 ub.poison、单块时去掉 func.return），
    交 MLIR：needs_ssacfg ⇔ “0 or 1 blocks” 诊断。消融：旧判据的误报/漏报。
L5c 定向：Graph 区域里先用后定义（gpu.module、builtin.module、未注册 op）MLIR 接受，结构层不得报 R5（有拒绝权的层不得误拒）；
    多块 Graph 区域结构层报 R7、MLIR 拒绝；嵌套 cf 的函数体内联后 MLIR 接受、needs_ssacfg 为假；无跳转的不可达块 needs_ssacfg 为真。
用法：PYTHONPATH=model python3 model/l5_test.py [函数上限]
"""
import sys, re, glob, collections
import mlir.ir as ir
import irimport as I, oracle as OR, structure as ST, odsjson as O, symtab as SY, verdict as V
from compcorpus_test import _funcs

GRAPH_DIAG = 'to have 0 or 1 blocks'

def _ops(op):
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                o = o.operation; yield o; yield from _ops(o)

def to_graph(gtxt, single):
    """通用形式的 func.func 文本 → 通用形式的 builtin.module（Graph 区域）文本；不认识的形状返回 None。"""
    m = re.match(r'"func\.func"\(\)(.*?)\(\{\n', gtxt, re.S)
    if not m: return None
    rest = gtxt[m.end():]
    lines = rest.split('\n'); pre = []
    if lines and re.match(r'\s*\^bb0\(', lines[0]):
        hdr = lines.pop(0).strip()
        inner = hdr[len('^bb0('):hdr.rindex('):')]
        pos = [x.start() for x in re.finditer(r'%[\w$.-]+: ', inner)]
        for i, p in enumerate(pos):
            seg = inner[p:pos[i + 1] if i + 1 < len(pos) else len(inner)].rstrip().rstrip(',')
            nm, ty = seg.split(': ', 1)
            pre.append(f'  {nm} = "ub.poison"() <{{value = #ub.poison}}> : () -> {ty}')
    body = '\n'.join(lines)
    if single: body = re.sub(r'(?m)^\s*"func\.return"\(.*$\n?', '', body)
    tail = body.rstrip()
    if not tail.endswith('}) : () -> ()') and not re.search(r'\}\)\s*(\{.*\})?\s*:\s*\(\)\s*->\s*\(\)\s*$', tail): return None
    body = re.sub(r'\}\)\s*(\{.*\})?\s*:\s*\(\)\s*->\s*\(\)\s*$', '}) : () -> ()', tail)
    return 'module {\n"builtin.module"() ({\n' + '\n'.join(pre) + ('\n' if pre else '') + body + '\n}'

def corpus(lim):
    o = OR.Oracle(); st = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 3: ex[k].append(v)
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
            core = {}
            for f in _funcs(root):
                nm = f['props']['_attrs'].get('sym_name', '').strip('"'); core.setdefault(nm, []).append(f)
            for op in _ops(m.operation):
                if nf >= lim: break
                if op.name != 'func.func' or not op.regions[0].blocks: continue
                nm = str(op.attributes['sym_name']).strip('"')
                if len(core.get(nm, [])) != 1: st['跳过：同名函数或未导入'] += 1; continue
                blocks = core[nm][0]['regions'][0]
                g = op.get_asm(print_generic_op_form=True)
                if o.verify('module {\n' + g + '\n}')[0] != 'ok': st['跳过：单独发射 MLIR 不接受（引用外部符号等）'] += 1; continue
                single = len(op.regions[0].blocks) == 1
                t = to_graph(g, single)
                if t is None: st['跳过：通用形式无法改写'] += 1; continue
                nf += 1; st['函数'] += 1
                r, msgs = o.verify(t); diag = ' '.join(str(x[1]) for x in msgs)
                mlir_multi = r != 'ok' and GRAPH_DIAG in diag
                pred = ST.derive_struct(blocks).needs_ssacfg; old = ST.needs_ssacfg_v51(blocks)
                st[f'函数/顶层多块 {not single}'] += 1
                st['L5b 判据一致'] += pred == mlir_multi
                if pred != mlir_multi: note('L5b 不一致', (nm, pred, r, diag[:120]))
                if not mlir_multi:
                    st[f'内联后 MLIR/{r}'] += 1
                    if r != 'ok': note(f'内联后 MLIR 其它拒绝', diag[:120])
                st['消融：旧判据一致'] += old == mlir_multi
                if old and not mlir_multi: st['消融：旧判据误报'] += 1; note('旧误报', nm)
                if not old and mlir_multi: st['消融：旧判据漏报'] += 1; note('旧漏报', nm)
    return st, ex

def audit():
    n, bad = 0, []
    for dia in O.DIALECTS:
        try: ops = O.ops(dia)
        except Exception: continue
        for op in ops:
            it = SY.interfaces(op) or set()
            if {'HasOnlyGraphRegion', 'GraphRegionNoTerminator'} & set(it):
                n += 1
                if 'SingleBlock' not in it: bad.append(op)
                if ST.st(op).rkind != 'Graph': bad.append(op + '（结构层未判为 Graph）')
    return n, bad

UBD = '''  %1 = "arith.addi"(%0, %0) : (i32, i32) -> i32
  %0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
'''
DIRECTED = {   # 名字：(文本, 允许未注册, 期望 MLIR 接受)
  'gpu.module 先用后定义': ('gpu.module @m {\n' + UBD + '}', False, True),
  'builtin.module 先用后定义': ('module { "builtin.module"() ({\n' + UBD + '}) : () -> () }', False, True),
  '未注册 op 区域先用后定义': ('func.func @f() {\n "test.foo"() ({\n' + UBD + '  "test.end"() : () -> ()\n }) : () -> ()\n return\n}', True, True),
}

def directed():
    o = OR.Oracle(); res = {}
    for k, (t, u, want) in DIRECTED.items():
        r = o.verify(t, unreg=u)[0]
        c = ir.Context(); c.allow_unregistered_dialects = u
        try: d = ST.check(I.import_module(ir.Module.parse(t, c)))
        except Exception as e: d = [('parse', 'X', str(e)[:80])]
        res[k] = (r == 'ok') == want and (not d) == want, f"MLIR {r}，结构层 {[x[1] for x in d]}"
    # MLIR 解析即拒的情形无法经绑定导入，结构层由 core 节点直接构造
    nd, blk = ST.node, ST.blk
    ex = ST.M(ST.F('f', [blk('entry', [nd('e', 'scf.execute_region', regions=[[blk('entry', [
        nd('ad', 'arith.addi', ins=['a', 'a'], outs=[('s', 'i32')]), ST.cst('a', 1), nd('y', 'scf.yield')])]]), ST.ret()])]))
    d = ST.check(ex)
    t = 'func.func @f() {\n "scf.execute_region"() ({\n' + UBD + '  "scf.yield"() : () -> ()\n }) : () -> ()\n return\n}'
    r = o.verify(t)[0]
    res['SSACFG 区域先用后定义（须拒）'] = 'R5' in {x[1] for x in d} and r != 'ok', f"结构层 {sorted({x[1] for x in d})}，MLIR {r}"

    host = ST.M(nd('g', 'gpu.module', sym='m', regions=[[blk('entry', [nd('c0', 'arith.constant', outs=[('a', 'i32')], props={'value': 1})]),
                                                          blk('b1', [nd('c1', 'arith.constant', outs=[('b', 'i32')], props={'value': 1})])]]))
    d = ST.check(host)
    mt = '"gpu.module"() <{sym_name = "m"}> ({\n  %a = "arith.constant"() <{value = 1 : i32}> : () -> i32\n^bb1:\n  %b = "arith.constant"() <{value = 1 : i32}> : () -> i32\n}) : () -> ()'
    r, msgs = o.verify(mt)
    res['多块 Graph 区域'] = 'R7' in {x[1] for x in d} and r != 'ok', f"结构层 {sorted({x[1] for x in d})}，MLIR {r} {str(msgs[:1])[:90]}"
    # 判据：嵌套 cf ⇒ 不需要 SSACFG；无跳转的不可达块 ⇒ 需要
    fn = '''func.func @f(%c: i1) -> i32 {
  %r = "scf.execute_region"() ({
    "cf.cond_br"(%c)[^bb1, ^bb2] <{operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:
    %x = "arith.constant"() <{value = 1 : i32}> : () -> i32
    "scf.yield"(%x) : (i32) -> ()
  ^bb2:
    %y = "arith.constant"() <{value = 2 : i32}> : () -> i32
    "scf.yield"(%y) : (i32) -> ()
  }) : () -> i32
  return %r : i32
}'''
    unr = '''func.func @g() {
  return
^bb1:
  return
}'''
    for k, t, want in (('嵌套 cf ⇒ 不需要 SSACFG', fn, False), ('无跳转不可达块 ⇒ 需要 SSACFG', unr, True)):
        m = ir.Module.parse(t, ir.Context()); f = next(iter(_funcs(I.import_module(m))))
        op = next(x for x in _ops(m.operation) if x.name == 'func.func')
        g = to_graph(op.get_asm(print_generic_op_form=True), len(op.regions[0].blocks) == 1)
        r, msgs = o.verify(g); multi = r != 'ok' and GRAPH_DIAG in ' '.join(str(x[1]) for x in msgs)
        p, old = ST.derive_struct(f['regions'][0]).needs_ssacfg, ST.needs_ssacfg_v51(f['regions'][0])
        res[k] = p == want == multi and old != want and (want or r == 'ok'), f"新判据 {p}，旧判据 {old}，内联后 MLIR {r}"
    global ABL; ABL = sum(1 for k in ('嵌套 cf ⇒ 不需要 SSACFG', '无跳转不可达块 ⇒ 需要 SSACFG') if res[k][0])
    return res

ABL = 0
if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    n, bad = audit()
    print('── L5a', n, bad)
    V.check('L5a Graph 区域 op 的审计', n >= 5 and not bad, f"ODS 中 Graph 区域 op {n} 个，全部带 SingleBlock 且结构层判为 Graph；问题 {bad}")
    dr = directed()
    for k, (ok, msg) in dr.items(): print('   ', k, ok, msg)
    V.check('L5c 定向', all(ok for ok, _ in dr.values()), '; '.join(f"{k}: {'✓' if ok else '✗'}" for k, (ok, _) in dr.items()))
    st, ex = corpus(lim)
    print('── L5b', dict(sorted(st.items())))
    for k, v in ex.items(): print('   ', k, v)
    nm = st['函数/顶层多块 True']
    V.check('L5b 内联进 Graph 区域 vs MLIR', st['函数'] >= 2000 and nm >= 5 and st['L5b 判据一致'] == st['函数'],
            f"函数 {st['函数']}（顶层多块 {nm}，语料仅此数）：判据一致 {st['L5b 判据一致']}；单块内联后 MLIR 接受 {st['内联后 MLIR/ok']}、其它拒绝 {st['函数'] - nm - st['内联后 MLIR/ok']}（memref.alloca 需 AutomaticAllocationScope 祖先，builtin.module 不是——上下文约束，非 L5）")
    V.check('L5b 消融灵敏度', ABL == 2,
            f"旧判据在定向例上错 {ABL}/2（嵌套 cf 误报、不可达块漏报）；语料中旧判据误报 {st['消融：旧判据误报']}、漏报 {st['消融：旧判据漏报']}")
    V.finish()
