"""第 37 轮（M5.3，spec/frontend-model §5）：透明内联保持 AutomaticAllocationScope。
  K0 ODS 对账：SCOPE_ALLOC / SCOPE_OPS 与 tblgen 记录一致。
  K1 值：小迭代下，包 alloca_scope 的内联 ≡ 直接计算（JIT）。
  K2 A/B：宿主是非作用域区域（scf.while 体、scf.if 体）时，大迭代下不包 ⇒ 栈耗尽崩溃，包 ⇒ 正常且值正确；
     宿主是作用域（scf.for 体）时，不包也崩溃：scf-to-cf 不按 trait 每次迭代释放（见 K5、closure §8）。
  K3 逃逸：体内 alloca 且有 memref 输出 ⇒ emit_core 返回 None 并记 WHY（不静默发射悬垂指针）。
  K5 包之后再跑 MLIR canonicalize：宿主为作用域 op（scf.for）时，alloca_scope 被内联/提升到 scf.for（依 trait 语义正确），
     而 scf-to-cf 不为每次迭代 stacksave/restore ⇒ 大迭代崩溃。这是 MLIR trait 与降低之间的已知不一致（只记录，不作模型缺陷）。
  K4 MLIR 自己的 inline pass（func.call → 内联）在同一宿主下的行为，作参照记录（不作为真值）。
"""
import os, sys, re, collections, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))

BODY = '''func.func @body(%x: i32) -> i32 {
  %m = memref.alloca() : memref<4096xi32>
  %c0 = arith.constant 0 : index
  %c7 = arith.constant 7 : index
  memref.store %x, %m[%c0] : memref<4096xi32>
  %v = memref.load %m[%c0] : memref<4096xi32>
  %one = arith.constant 1 : i32
  memref.store %one, %m[%c7] : memref<4096xi32>
  %w = memref.load %m[%c7] : memref<4096xi32>
  %r = arith.addi %v, %w : i32
  return %r : i32
}'''
HOSTS = {
 'while': '''func.func @main(%x: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r:2 = scf.while (%i = %c0, %acc = %x) : (index, i32) -> (index, i32) {
    %c = arith.cmpi slt, %i, %n : index
    scf.condition(%c) %i, %acc : index, i32
  } do {
  ^bb0(%i: index, %acc: i32):
    %y = func.call @body(%acc) : (i32) -> i32
    %i2 = arith.addi %i, %c1 : index
    scf.yield %i2, %y : index, i32
  }
  return %r#1 : i32
}''',
 'for': '''func.func @main(%x: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %x) -> (i32) {
    %y = func.call @body(%acc) : (i32) -> i32
    scf.yield %y : i32
  }
  return %r : i32
}''',
 'if_in_while': '''func.func @main(%x: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t = arith.constant true
  %r:2 = scf.while (%i = %c0, %acc = %x) : (index, i32) -> (index, i32) {
    %c = arith.cmpi slt, %i, %n : index
    scf.condition(%c) %i, %acc : index, i32
  } do {
  ^bb0(%i: index, %acc: i32):
    %y = scf.if %t -> (i32) {
      %z = func.call @body(%acc) : (i32) -> i32
      scf.yield %z : i32
    } else {
      scf.yield %acc : i32
    }
    %i2 = arith.addi %i, %c1 : index
    scf.yield %i2, %y : index, i32
  }
  return %r#1 : i32
}'''}
ESC = '''func.func @body(%x: i32) -> memref<4xi32> {
  %m = memref.alloca() : memref<4xi32>
  return %m : memref<4xi32>
}'''

def _imp(txt):
    import mlir.ir as ir, irimport as I
    from compcorpus_test import _funcs
    m = ir.Module.parse(txt, ir.Context()); return list(_funcs(I.import_module(m)))

def build(host, body=BODY, name='cmp_alloca'):
    """宿主中 func.call @body → 复合 op（透明）。返回 (宿主块, 函数类型)。"""
    import core as C
    fb = _imp(body)[0]; bb = fb['regions'][0]
    ft = fb['props']['_attrs']['function_type']; outs = re.fullmatch(r'\((.*)\) -> \(?([^()]*?)\)?', ft).group(2).split(',')
    old = C.ORACLE_ON; C.ORACLE_ON = False
    try: C.derive(name, list(bb[0]['args']), [(f'o{i}', t.strip()) for i, t in enumerate(outs)], bb, 'transparent')
    finally: C.ORACLE_ON = old
    fh = [f for f in _imp('module {\n' + body + '\n' + host + '\n}') if f['props']['_attrs'].get('sym_name', '').strip('"') == 'main'][0]
    def fix(blocks):
        for b in blocks:
            for i, n in enumerate(b['ops']):
                if n['op'] == 'func.call': b['ops'][i] = dict(id=n['id'], op=name, ins=n['ins'], outs=n['outs'], regions=[], succs=[], props={})
                for r in n['regions']: fix(r)
    hb = fh['regions'][0]; fix(hb)
    return hb, fh['props']['_attrs']['function_type']

def emit(host, scope=True, body=BODY):
    import core as C, oracle as O
    C.ALLOCA_SCOPE = scope
    try:
        hb, ft = build(host, body); return O.emit_core(hb, ft, name='main')
    finally: C.ALLOCA_SCOPE = True

def _child(txt, x, n, q):
    import valimport_test as VT
    try: q.put(('ok', VT.jit_run(txt, 'main', ['i32', 'index'], 'i32', [[x, n]])[0]))
    except Exception as e: q.put(('err', str(e)[:100]))
def run(txt, x, n, timeout=120):
    ctx = mp.get_context('fork'); q = ctx.Queue(); p = ctx.Process(target=_child, args=(txt, x, n, q)); p.start(); p.join(timeout)
    if p.is_alive(): p.kill(); return ('timeout', None)
    if p.exitcode != 0: return ('crash', p.exitcode)
    return q.get()

def K0():
    import odsjson as J, core as C
    alloc, scope = set(), set()
    for dia in J.DIALECTS:
        try: d = J.records(dia)
        except Exception: continue
        def eff(k): return isinstance(d.get(k), dict) and 'AutomaticAllocationScopeResource' in str(d[k].get('resource', ''))
        for opn, n in J.ops(dia).items():
            for a, _ in d[n]['results']['args']:
                a = J.ref(a)
                if a and isinstance(d.get(a), dict) and any(eff(J.ref(x)) for x in d[a].get('decorators', [])): alloc.add(opn)
            if any(J.ref(t) == 'AutomaticAllocationScope' for t in d[n]['traits']): scope.add(opn)
    return alloc == C.SCOPE_ALLOC and scope == C.SCOPE_OPS, dict(alloc=sorted(alloc), scope=sorted(scope))

def K4(host, x, n):
    """MLIR 参照：原样（func.call）与 MLIR inline pass 之后。"""
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    src = 'module {\n' + BODY + '\n' + host + '\n}'
    with ir.Context():
        m = ir.Module.parse(src); PassManager.parse('builtin.module(inline)').run(m.operation); inl = str(m)
    return run(src, x, n), run(inl, x, n), 'memref.alloca_scope' in inl

if __name__ == '__main__':
    import oracle as O
    ok0, d0 = K0(); print('── K0 ODS 对账:', 'OK' if ok0 else 'FAIL', d0)
    st = collections.Counter(); bad = []
    BIG, SMALL = 200000, 5
    for h, txt in HOSTS.items():
        on, off = emit(txt, True), emit(txt, False)
        if on is None or off is None: bad.append(f'{h} 发射失败 {O.WHY_NONE}'); continue
        st['发射/含 alloca_scope'] += 'memref.alloca_scope' in on
        exp = (7 + SMALL) & 0xffffffff
        for nm, t in (('包', on), ('不包', off)):
            r = run(t, 7, SMALL); st[f'K1 {h}/{nm} 小迭代 {"正确" if r == ("ok", exp) else r}'] += 1
        exp = (7 + BIG) & 0xffffffff
        for nm, t in (('包', on), ('不包', off)):
            r = run(t, 7, BIG); st[f'K2 {h}/{nm} 大迭代 {"正确" if r == ("ok", exp) else r[0]}'] += 1
        a, b, wrapped = K4(txt, 7, BIG)
        st[f'K4 {h}/MLIR func.call {a[0] if a[0] != "ok" else ("正确" if a[1] == exp else a)}'] += 1
        st[f'K4 {h}/MLIR inline 后 {b[0] if b[0] != "ok" else ("正确" if b[1] == exp else b)}，含 alloca_scope={wrapped}'] += 1
    import mlir.ir as ir
    from mlir.passmanager import PassManager
    for h, txt in HOSTS.items():
        on = emit(txt, True)
        with ir.Context():
            m = ir.Module.parse(on); PassManager.parse('builtin.module(canonicalize)').run(m.operation); c = str(m)
        r = run(c, 7, BIG)
        st[f'K5 {h}/包+canonicalize：scope {"保留" if "alloca_scope" in c else "被消去"}，大迭代 {"正确" if r == ("ok", (7 + BIG) & 0xffffffff) else r[0]}'] += 1
    O.WHY_NONE.clear()
    e = emit(HOSTS['for'].replace('%y = func.call @body(%acc) : (i32) -> i32', '%mm = func.call @body(%acc) : (i32) -> memref<4xi32>\n    %y = arith.addi %acc, %acc : i32'), True, ESC)
    st[f'K3 逃逸 ⇒ {"None" if e is None else "发射了"} {dict(O.WHY_NONE)}'] += 1
    for k in sorted(st): print('  ', k, st[k])
    for b in bad: print('  ', b)
    must = [f'K2 {h}/包 大迭代 正确' for h in HOSTS] + [f'K1 {h}/包 小迭代 正确' for h in HOSTS]
    sys.exit(0 if ok0 and not bad and all(st[k] for k in must) and e is None else 1)
