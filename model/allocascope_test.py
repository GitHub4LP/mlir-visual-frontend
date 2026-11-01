"""第 37 轮（M5.3，spec/frontend-model §5）：透明内联保持 AutomaticAllocationScope。
  K0 ODS 对账：SCOPE_ALLOC / SCOPE_OPS 与 tblgen 记录一致。
  K1 值：小迭代下，包 alloca_scope 的内联 ≡ 直接计算（JIT）。
  K2 A/B：宿主是非作用域区域（scf.while 体、scf.if 体）时，大迭代下不包 ⇒ 栈耗尽崩溃，包 ⇒ 正常且值正确；
     宿主是作用域（scf.for 体）时，不包也崩溃：scf-to-cf 不按 trait 每次迭代释放（见 K5、closure §8）。
  K3 逃逸：体内 alloca 且有 memref 输出 ⇒ emit_core 返回 None 并记 WHY（不静默发射悬垂指针）。
  K5 包之后再跑 MLIR canonicalize：宿主为作用域 op（scf.for）时，alloca_scope 被内联/提升到 scf.for（依 trait 语义正确），
     而 scf-to-cf 不为每次迭代 stacksave/restore ⇒ 大迭代崩溃。这是 MLIR trait 与降低之间的已知不一致（只记录，不作模型缺陷）。
  K6 第 38 轮：前端管线在优化之后、方言降低之前跑 lowering.lower_alloca_scopes（作用域 ⇒ stacksave/stackrestore）：
     包+canonicalize+恢复、以及直接写在 scf.for 体内的 alloca（无复合 op），大迭代均应正确；值不变。
  K9 D2：复合 op 体内有 scf.if + alloca（包成 alloca_scope 后体内有控制流），宿主 scf.while：标准管线降不下去（校验失败），
     经 lower_alloca_scopes 后大迭代正确。
  K7 语料安全性：对每个可解析、可校验的语料段跑恢复，恢复后仍须通过 MLIR verifier；报包裹数与未处理原因。
  K4 MLIR 自己的 inline pass（func.call → 内联）在同一宿主下的行为，作参照记录（不作为真值）。
"""
import os, sys, re, collections, multiprocessing as mp
HERE = os.path.dirname(os.path.abspath(__file__))
def LWrun(m):
    import lowering as LW
    st, why = LW.lower_alloca_scopes(m); return sum(st.values()), why

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

def _k7_child(seg, q):
    import mlir.ir as ir, lowering as LW
    try:
        with ir.Context() as ctx:
            ctx.allow_unregistered_dialects = False
            m = ir.Module.parse(seg)
            if not m.operation.verify(): q.put(('skip', 0, [])); return
            k, why = LWrun(m)
            q.put(('ok' if m.operation.verify() else 'bad', k, why))
    except Exception as e: q.put(('skip', 0, []))
def K7(root=os.path.join(HERE, '..', 'corpus')):
    st = collections.Counter(); ctx = mp.get_context('fork')
    for dp, _, fs in os.walk(root):
        for f in sorted(fs):
            if not f.endswith('.mlir') or 'invalid' in f: continue
            for seg in open(os.path.join(dp, f)).read().split('// -----'):
                if 'scf.for' not in seg and 'scf.parallel' not in seg: continue
                q = ctx.Queue(); p = ctx.Process(target=_k7_child, args=(seg, q)); p.start(); p.join(60)
                if p.is_alive(): p.kill(); st['超时'] += 1; continue
                if p.exitcode != 0: st['MLIR 崩溃（不计）'] += 1; continue
                r, k, why = q.get()
                st[{'ok': '恢复后 verify 通过', 'bad': '恢复后 verify 失败', 'skip': '原段不可解析/不可校验（不计）'}[r]] += 1
                st['包裹的循环'] += k
                for w in why: st['未处理：' + w] += 1
    st.setdefault('恢复后 verify 失败', 0)
    return st

def gen(rng):
    """K8 生成器（补充证据，不作真值；真值是 JIT 原样 vs 恢复）：嵌套 scf.for / scf.parallel / scf.if / scf.while，alloca 位置随机。"""
    cnt = [0]
    def v(): cnt[0] += 1; return f'%v{cnt[0]}'
    def alloca_use(acc, ind):
        m, a, b = v(), v(), v(); sz = rng.choice([4, 64, 1024])
        return [f'{ind}{m} = memref.alloca() : memref<{sz}xi32>', f'{ind}memref.store {acc}, {m}[%c1] : memref<{sz}xi32>',
                f'{ind}{a} = memref.load {m}[%c1] : memref<{sz}xi32>', f'{ind}{b} = arith.addi {a}, %one : i32'], b
    def body(acc, ind, depth):
        L = []; cur = acc
        for _ in range(rng.randint(1, 3)):
            k = rng.choice(['alloca', 'alloca', 'arith'] + (['for', 'if', 'while', 'par'] if depth < 2 else []))
            if k == 'alloca': ls, cur = alloca_use(cur, ind); L += ls
            elif k == 'arith': r = v(); L.append(f'{ind}{r} = arith.muli {cur}, %three : i32'); cur = r
            elif k == 'for':
                r, i, a = v(), v(), v(); n = rng.randint(1, 3)
                inner, y = body(a, ind + '  ', depth + 1)
                L += [f'{ind}{r} = scf.for {i} = %c0 to %c{n} step %c1 iter_args({a} = {cur}) -> (i32) {{'] + inner + [f'{ind}  scf.yield {y} : i32', f'{ind}}}']; cur = r
            elif k == 'if':
                r = v(); inner, y = body(cur, ind + '  ', depth + 1)
                L += [f'{ind}{r} = scf.if %t -> (i32) {{'] + inner + [f'{ind}  scf.yield {y} : i32', f'{ind}}} else {{', f'{ind}  scf.yield {cur} : i32', f'{ind}}}']; cur = r
            elif k == 'while':
                r, i, a, i2, a2, c, i3 = v(), v(), v(), v(), v(), v(), v(); n = rng.randint(1, 3)
                inner, y = body(a2, ind + '  ', depth + 1)
                L += [f'{ind}{r}:2 = scf.while ({i} = %c0, {a} = {cur}) : (index, i32) -> (index, i32) {{',
                      f'{ind}  {c} = arith.cmpi slt, {i}, %c{n} : index', f'{ind}  scf.condition({c}) {i}, {a} : index, i32',
                      f'{ind}}} do {{', f'{ind}^bb0({i2}: index, {a2}: i32):'] + inner + [f'{ind}  {i3} = arith.addi {i2}, %c1 : index', f'{ind}  scf.yield {i3}, {y} : index, i32', f'{ind}}}']
                cur = f'{r}#1'
            else:                                                  # scf.parallel：每次迭代写自己的格子，之后求和（顺序降低下无竞争）
                i, j, s0, r, a, b2 = v(), v(), v(), v(), v(), v(); n = rng.randint(1, 3)
                inner, y = body(cur, ind + '  ', depth + 1)
                L += [f'{ind}scf.parallel ({i}) = (%c0) to (%c{n}) step (%c1) {{'] + inner + [f'{ind}  {b2} = arith.index_cast {i} : index to i32', f'{ind}  {a} = arith.addi {y}, {b2} : i32',
                      f'{ind}  memref.store {a}, %buf[{i}] : memref<4xi32>', f'{ind}  scf.reduce', f'{ind}}}']
                L += [f'{ind}{r} = scf.for {j} = %c0 to %c{n} step %c1 iter_args({s0} = {cur}) -> (i32) {{']
                ld, ad = v(), v()
                L += [f'{ind}  {ld} = memref.load %buf[{j}] : memref<4xi32>', f'{ind}  {ad} = arith.addi {s0}, {ld} : i32', f'{ind}  scf.yield {ad} : i32', f'{ind}}}']; cur = r
        return L, cur
    inner, y = body('%acc', '    ', 0)
    return '\n'.join(['func.func @main(%x: i32, %n: index) -> i32 {', '  %c0 = arith.constant 0 : index', '  %c1 = arith.constant 1 : index',
        '  %c2 = arith.constant 2 : index', '  %c3 = arith.constant 3 : index', '  %one = arith.constant 1 : i32', '  %three = arith.constant 3 : i32',
        '  %t = arith.constant true', '  %buf = memref.alloca() : memref<4xi32>',
        '  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %x) -> (i32) {'] + inner + [f'    scf.yield {y} : i32', '  }', '  return %r : i32', '}'])

def K8(N=60, seed=0):
    import random, mlir.ir as ir, lowering as LW
    from mlir.passmanager import PassManager
    rng = random.Random(seed); st = collections.Counter()
    for _ in range(N):
        src = gen(rng)
        with ir.Context():
            try: m = ir.Module.parse(src)
            except Exception as e: st['生成不合法（不计）'] += 1; continue
            PassManager.parse('builtin.module(canonicalize)').run(m.operation); canon = str(m)
            k, why = LWrun(m); ok = m.operation.verify(); rest = str(m)
        st['程序'] += 1; st['恢复包裹的循环'] += k; st['恢复后 verify 失败'] += not ok
        for w in why: st['未处理：' + w] += 1
        for x, n in ((3, 4), (-5, 2)):
            a, b = run(canon, x, n), run(rest, x, n)
            st['值比较'] += 1
            if a[0] != 'ok': st[f'原样 {a[0]}（不计）'] += 1; continue
            st['一致' if a == b else '不一致'] += 1
        big = run(rest, 1, 20000); st[f'恢复后 2 万次外层迭代 {big[0]}'] += 1
        big0 = run(canon, 1, 20000); st[f'原样 2 万次外层迭代 {big0[0]}'] += 1
        if big[0] == 'ok' and big0[0] == 'ok': st['大迭代值一致' if big == big0 else '大迭代值不一致'] += 1
    return st

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
    import lowering as LW
    def restored(txt, canon=True):
        with ir.Context():
            m = ir.Module.parse(txt)
            if canon: PassManager.parse('builtin.module(canonicalize)').run(m.operation)
            k, why = LWrun(m); ok = m.operation.verify(); return str(m), k, why, ok
    for h, txt in HOSTS.items():
        c, k, why, ok = restored(emit(txt, True))
        for nn in (SMALL, BIG):
            r = run(c, 7, nn)
            st[f'K6 {h}/包+canonicalize+恢复 n={nn}：{"正确" if r == ("ok", (7 + nn) & 0xffffffff) else r[0]}，恢复包裹 {k}，verify={ok}'] += 1
    direct = '''func.func @main(%x: i32, %n: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %x) -> (i32) {
    %m = memref.alloca() : memref<4096xi32>
    memref.store %acc, %m[%c0] : memref<4096xi32>
    %v = memref.load %m[%c0] : memref<4096xi32>
    %one = arith.constant 1 : i32
    %y = arith.addi %v, %one : i32
    scf.yield %y : i32
  }
  return %r : i32
}'''
    for nm, t in (('原样', direct), ('恢复', restored(direct, False)[0])):
        r = run(t, 7, BIG); st[f'K6 直接 scf.for 体内 alloca/{nm}：{"正确" if r == ("ok", (7 + BIG) & 0xffffffff) else r[0]}'] += 1
    BODY_IF = BODY.replace('  %r = arith.addi %v, %w : i32\n  return %r : i32', '''  %t = arith.constant true
  %r = scf.if %t -> (i32) {
    %q = arith.addi %v, %w : i32
    scf.yield %q : i32
  } else {
    scf.yield %v : i32
  }
  return %r : i32''')
    assert BODY_IF != BODY
    on = emit(HOSTS['while'], True, BODY_IF)
    r0 = run(on, 7, SMALL); st[f'K9 体内 scf.if：标准管线 {r0[0]}'] += 1
    c, k, why, ok = restored(on, False)
    for nn in (SMALL, BIG):
        r = run(c, 7, nn); st[f'K9 体内 scf.if：降低后 n={nn} {"正确" if r == ("ok", (7 + nn) & 0xffffffff) else r[0]}'] += 1
    k7 = K7(); print('── K7 语料:', dict(k7))
    k8 = K8(); print('── K8 生成器 原样 vs 恢复（JIT）:', dict(k8))
    O.WHY_NONE.clear()
    e = emit(HOSTS['for'].replace('%y = func.call @body(%acc) : (i32) -> i32', '%mm = func.call @body(%acc) : (i32) -> memref<4xi32>\n    %y = arith.addi %acc, %acc : i32'), True, ESC)
    st[f'K3 逃逸 ⇒ {"None" if e is None else "发射了"} {dict(O.WHY_NONE)}'] += 1
    for k in sorted(st): print('  ', k, st[k])
    for b in bad: print('  ', b)
    must = [k for k in st if k.startswith('K6') and ('恢复' in k) and '正确' in k]
    k6ok = len(must) == 2 * len(HOSTS) + 1 and st[f'K9 体内 scf.if：降低后 n={BIG} 正确'] and k7['恢复后 verify 失败'] == 0 and k8['不一致'] == 0 and k8['恢复后 verify 失败'] == 0 and k8['一致'] > 0 and k8['大迭代值不一致'] == 0 and k8['恢复后 2 万次外层迭代 ok'] == k8['程序']
    must = [f'K2 {h}/包 大迭代 正确' for h in HOSTS] + [f'K1 {h}/包 小迭代 正确' for h in HOSTS]
    sys.exit(0 if ok0 and not bad and all(st[k] for k in must) and e is None and k6ok else 1)
