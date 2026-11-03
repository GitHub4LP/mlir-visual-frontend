"""第 57 轮：Graph 区域里的副作用次序（L56，M1.2 的未覆盖项）。

Graph 区域没有执行次序。体内有副作用的 op 内联进 Graph 区域以后，MLIR 照样接受（第 52 轮 L5b：单块函数体内联进 builtin.module，
MLIR 接受 2570 个），但谁先执行无定义。structure.graph_order：Graph（或可能为 Graph 的 '?'）区域中可能有副作用的 op ⇒ 未知并警告，从不拒绝。
副作用分类 structure.effect_class 自下而上取 ODS（NoMemoryEffect/MemoryEffects<[]> ⇒ none；RecursiveMemoryEffects ⇒ 看体；
其余 ⇒ may，过近似）。

G1 定向：各种 Graph 区域（builtin.module、gpu.module、scf.forall.in_parallel）里有/无副作用的体，MLIR 的判定与我们的结论。
G2 消融：去掉递归 / Symbol 豁免 / C++ MemoryEffectOpInterface 规则，各自至少错一个定向例。
G3 外部对照（分类器的可靠方向）：语料中每个无区域、有结果的 op，在原处克隆一份（结果不用），跑 MLIR canonicalize；
   贪心驱动只删 wouldOpBeTriviallyDead（无副作用）的 op。我们判 none 而 MLIR 保留 ⇒ 分类器低估副作用（不可靠）。
   我们判 may 而 MLIR 删除 ⇒ 过近似（只损精度）。
阈值（运行前写定）：G1 全对；G2 每项消融至少错 1 例；G3 检验 op 名 ≥ 150、实例 ≥ 500，“判 none 而 MLIR 保留”= 0。
用法：PYTHONPATH=model python3 model/graphorder_test.py [语料块上限]
"""
import sys, os, glob, pickle, collections
import mlir.ir as ir
import irimport as I, oracle as OR, structure as ST, verdict as V

DIRECTED = {   # 名字：(MLIR 文本, 期望 MLIR 接受, 期望 graph_order 的结论条数下限/恰为 0)
  '模块里的 func 体有 store（Symbol 豁免）': ('''module {
  func.func @f(%m: memref<4xi32>, %v: i32) {
    %c0 = arith.constant 0 : index
    memref.store %v, %m[%c0] : memref<4xi32>
    return
  }
}''', True, 0),
  '内联进 builtin.module 的体有 alloc+store': ('''module {
  "builtin.module"() ({
    %c0 = arith.constant 0 : index
    %v = arith.constant 7 : i32
    %m = memref.alloc() : memref<4xi32>
    memref.store %v, %m[%c0] : memref<4xi32>
    %x = memref.load %m[%c0] : memref<4xi32>
  }) : () -> ()
}''', True, 3),
  '内联进 builtin.module 的纯体': ('''module {
  "builtin.module"() ({
    %a = arith.constant 7 : i32
    %b = arith.addi %a, %a : i32
    %c = arith.divui %b, %a : i32
  }) : () -> ()
}''', True, 0),
  'Graph 区域里 scf.if 体内有 store（递归）': ('''module {
  "builtin.module"() ({
    %c0 = arith.constant 0 : index
    %t = arith.constant true
    %v = arith.constant 7 : i32
    %m = memref.alloc() : memref<4xi32>
    scf.if %t {
      memref.store %v, %m[%c0] : memref<4xi32>
    }
  }) : () -> ()
}''', True, 2),
  'Graph 区域里 linalg.generic 写 memref（C++ getEffects）': ('''#id = affine_map<(d0) -> (d0)>
module {
  "builtin.module"() ({
    %m = memref.alloc() : memref<4xi32>
    %v = arith.constant 7 : i32
    linalg.generic {indexing_maps = [#id], iterator_types = ["parallel"]} outs(%m : memref<4xi32>) {
    ^bb0(%o: i32):
      linalg.yield %v : i32
    }
  }) : () -> ()
}''', True, 2),
  'gpu.module 里 gpu.func 有 store（Symbol 豁免）': ('''module attributes {gpu.container_module} {
  gpu.module @k {
    gpu.func @f(%m: memref<4xi32>, %v: i32) kernel {
      %c0 = arith.constant 0 : index
      memref.store %v, %m[%c0] : memref<4xi32>
      gpu.return
    }
  }
}''', True, 0),
  'in_parallel 两个重叠的 parallel_insert_slice': ('''func.func @f(%t: tensor<4xi32>, %s: tensor<2xi32>) -> tensor<4xi32> {
  %r = scf.forall (%i) in (2) shared_outs(%o = %t) -> (tensor<4xi32>) {
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %s into %o[0] [2] [1] : tensor<2xi32> into tensor<4xi32>
      tensor.parallel_insert_slice %s into %o[1] [2] [1] : tensor<2xi32> into tensor<4xi32>
    }
  }
  return %r : tensor<4xi32>
}''', True, 2),
  'in_parallel 里放纯 op（MLIR 拒；我们不拒，交预言机）': ('''func.func @f(%t: tensor<4xi32>) -> tensor<4xi32> {
  %r = scf.forall (%i) in (2) shared_outs(%o = %t) -> (tensor<4xi32>) {
    scf.forall.in_parallel {
      %c = arith.constant 1 : i32
    }
  }
  return %r : tensor<4xi32>
}''', False, 0),
}

def _check(root, effect=None, notexec=None):
    sv = (ST.effect_class, ST._not_executed_here)
    if effect: ST.effect_class = effect
    if notexec: ST._not_executed_here = notexec
    try: return ST.graph_order(root)
    finally: ST.effect_class, ST._not_executed_here = sv

def G1(abl=None):
    o = OR.Oracle(); res = {}
    for k, (t, want_ok, want_n) in DIRECTED.items():
        r = o.verify(t)[0]
        c = ir.Context()
        if want_ok:
            try: root = I.import_module(ir.Module.parse(t, c))
            except Exception as e: res[k] = (False, f'导入失败 {e}'[:100]); continue
        else: root = BUILT[k]()          # MLIR 解析即拒，绑定无法导入 ⇒ 由节点直接构造同形的图
        f = _check(root, *(abl or ()))
        d = [x for x in ST.check(root) if x[1] != 'R2'] if want_ok else []
        okn = len(f) >= want_n if want_n else len(f) == 0
        res[k] = ((r == 'ok') == want_ok and okn and not d, f"MLIR {r}，未知 {len(f)} {[x[2] for x in f][:4]}，结构层拒 {[x[1] for x in d]}")
    return res

def _inpar_pure():
    nd, blk = ST.node, ST.blk
    ip = nd('ip', 'scf.forall.in_parallel', regions=[[blk('entry', [nd('c', 'arith.constant', outs=[('c', 'i32')], props={'value': 1})])]])
    fa = nd('fa', 'scf.forall', ins=['t'], outs=[('r', 'tensor<4xi32>')], regions=[[blk('entry', [ip], args=[('i', 'index'), ('o', 'tensor<4xi32>')])]])
    return ST.M(ST.F('f', [blk('entry', [fa, ST.ret(ins=['r'])], args=[('t', 'tensor<4xi32>')])]))
BUILT = {'in_parallel 里放纯 op（MLIR 拒；我们不拒，交预言机）': _inpar_pure}

ORIG_EFFECT = ST.effect_class
def _abl_norec(op):
    c = ORIG_EFFECT(op); return 'none' if c == 'recursive' else c
def _abl_nosym(op): return False
def _abl_nocpp(op):
    import odsjson as O
    c = ORIG_EFFECT(op)
    if c != 'may': return c
    dia = op.split('.')[0]
    if dia not in O.DIALECTS or op not in O.ops(dia): return c
    d = O.records(dia); found = set()
    def walk(t):
        x = d.get(O.ref(t))
        if not isinstance(x, dict): return
        sc = x.get('!superclasses') or []
        if 'TraitList' in sc:
            for u in x.get('traits') or []: walk(u)
            return
        if 'MemoryEffects' in sc: found.add('none' if not (x.get('effects') or []) else 'may')
        elif O.ref(t) == 'RecursiveMemoryEffects': found.add('recursive')
    for t in d[O.ops(dia)[op]].get('traits', []): walk(t)
    return 'may' if 'may' in found else 'recursive' if 'recursive' in found else 'none' if 'none' in found else 'may'

def _walk(op):
    for r in op.regions:
        for b in r.blocks:
            for o in b.operations:
                o = o.operation; yield o; yield from _walk(o)

def _count(m, name): return sum(1 for o in _walk(m.operation) if o.name == name)

def _chunk_job(ch, per_chunk=10):
    from mlir.passmanager import PassManager
    out = []
    with ir.Context() as ctx:
        ctx.enable_multithreading(False)
        m = ir.Module.parse(ch)
        if not m.operation.verify(): return out
        cand, seen = [], set()
        for i, o in enumerate(_walk(m.operation)):
            if o.name in seen or len(o.regions) or len(o.results) == 0 or o.name in ('func.func', 'builtin.module'): continue
            if o.parent is None or o.parent.name not in ('func.func',): continue      # 只取直接在函数体里的（SSACFG），克隆后支配不变
            seen.add(o.name); cand.append((i, o.name))
            if len(cand) >= per_chunk: break
        m0 = ir.Module.parse(ch); PassManager.parse('builtin.module(canonicalize)').run(m0.operation)
        for i, nm in cand:
            m1 = ir.Module.parse(ch)
            tgt = [o for j, o in enumerate(_walk(m1.operation)) if j == i][0]
            tgt.clone(ip=ir.InsertionPoint(tgt))
            if not m1.operation.verify(): out.append((nm, 'skip')); continue
            PassManager.parse('builtin.module(canonicalize)').run(m1.operation)
            out.append((nm, 'kept' if _count(m1, nm) > _count(m0, nm) else 'erased'))
    return out

def forked(fn, *a):
    r, w = os.pipe(); pid = os.fork()
    if pid == 0:
        os.close(r)
        try: data = pickle.dumps(fn(*a))
        except Exception as e: data = pickle.dumps(('err:' + type(e).__name__ + ':' + str(e).split(chr(10))[0][:50],))
        with os.fdopen(w, 'wb') as f: f.write(data)
        os._exit(0)
    os.close(w)
    with os.fdopen(r, 'rb') as f: data = f.read()
    os.waitpid(pid, 0)
    return pickle.loads(data) if data else ('crash',)

def G3(lim):
    st = collections.Counter(); byname = collections.defaultdict(collections.Counter); nch = 0
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        if nch >= lim: break
        src = open(path).read()
        if 'allow-unregistered-dialect' in src: continue
        for ch in I.split_chunks(src):
            if nch >= lim: break
            if not ch.strip() or I.expected(ch) or 'func.func' not in ch: continue
            nch += 1
            res = forked(_chunk_job, ch)
            if isinstance(res, tuple): st['块：' + res[0][:30]] += 1; continue
            st['块'] += 1
            for nm, k in res:
                if k == 'skip': st['克隆后不合法'] += 1; continue
                c = ORIG_EFFECT(nm); byname[nm][(c, k)] += 1; st[f'{c}/{k}'] += 1
    return st, byname

if __name__ == '__main__':
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    r1 = G1()
    for k, (ok, m) in r1.items(): print('  G1', '✓' if ok else '✗', k, m)
    V.check('G1 Graph 区域副作用定向', all(ok for ok, _ in r1.values()), f"{sum(ok for ok, _ in r1.values())}/{len(r1)}")
    ab = {}
    for nm, a in (('不递归', (_abl_norec, None)), ('无 Symbol 豁免', (None, _abl_nosym)), ('不看 C++ MemoryEffectOpInterface', (_abl_nocpp, None))):
        rr = G1(a); ab[nm] = [k for k, (ok, _) in rr.items() if not ok]
        print('  G2', nm, ab[nm])
    V.check('G2 消融灵敏度', all(len(v) >= 1 for v in ab.values()), '; '.join(f"{k} 错 {len(v)}" for k, v in ab.items()))
    st, byname = G3(lim)
    print('  G3', dict(sorted(st.items())))
    bad = {n: dict(c) for n, c in byname.items() if c[('none', 'kept')]}
    prec = sorted(n for n, c in byname.items() if c[('may', 'erased')])
    print('  G3 判 none 而 MLIR 保留', bad)
    print('  G3 判 may 而 MLIR 删除（过近似）', len(prec), prec[:40])
    ninst = sum(sum(c.values()) for c in byname.values())
    V.check('G3 副作用分类 vs MLIR 死代码删除', len(byname) >= 150 and ninst >= 500 and not bad,
            f"op 名 {len(byname)}，实例 {ninst}：none 删除 {st['none/erased']}、none 保留 {st['none/kept']}、may 保留 {st['may/kept']}、may 删除 {st['may/erased']}（过近似 op 名 {len(prec)}）")
    V.finish()
