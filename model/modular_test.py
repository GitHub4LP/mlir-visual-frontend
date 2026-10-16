"""第 16 轮：预言机按模块上下文发射（spec/modular.md）。

动机（防空检验）：
  · 第 12–15 轮的 C2“误拒 0”一直在 ORACLE_ON=False 下跑：有拒绝权的预言机层从未在真实 IR 上被检验过不误拒；
  · 真实语料 2706 个函数体里 738 个（27%）到不了预言机：导入节点的分段属性没被认出 537、insert_slice 被当成 core 简写 42、
    func.call 26、未导入 ODS 的方言（linalg/affine…）约 130。
  第 16 轮把导入节点改为逐字发射，函数体内引用的符号从最近符号表补成私有声明，模块属性原样带上（oracle.emit_core）。
检验：
  M1 单元：core 原生 func.call 有/无模块上下文；调用类型错 ⇒ 预言机拒并归位到调用节点。
  M2 体级预言机 ≡ 整模块 verifier（真实 IR 变异差分）：以 MLIR 接受的模块为底，做第 15 轮的逐义务变异 + T1（体内 op 结果类型改变），
     比较 前端（结构+符号+类型层+体级预言机）与 MLIR 对整个变异模块的判定。误拒必须 0；漏判按种类分账；
     并单列“无预言机时漏判、体级预言机抓住”的数目（预言机按模块发射的贡献）。
用法：PYTHONPATH=model python3 model/modular_test.py [每模块变异上限=16]
"""
import sys, re, glob, collections, random
import mlir.ir as ir
import irimport as I, structure as ST, symtab as SY, oracle as OR, core as C, obligations_test as Z, odssig as G, odsjson as O

def m1():
    node, blk = ST.node, ST.blk
    callee = node('g', 'func.func', sym='g', ftype='(i32) -> i32', vis='private', regions=[[blk('e', [node('r', 'func.return', ins=['x'])], args=[('x', 'i32')])]])
    def body(argty):
        return [blk('entry', [node('c1', 'func.call', callee='@g', ins=['a'], outs=[('y', 'i32')]), node('ret', 'func.return', ins=['y'])], args=[('a', argty)])]
    syms = {'g': callee}
    out = {}
    out['无上下文 ⇒ 不发射'] = OR.emit_core(body('i32'), '(i32) -> i32', 'f') is None
    t = OR.emit_core(body('i32'), '(i32) -> i32', 'f', syms)
    out['有上下文 ⇒ 发射'] = t is not None
    out['好 ⇒ ok'] = OR.get().verify(t)[0] == 'ok'
    st, ds = OR.get().verify(OR.emit_core(body('f32'), '(f32) -> i32', 'f', syms))
    out['调用实参类型错 ⇒ 拒并归位 c1'] = st == 'reject' and any(n == 'c1' for n, _ in ds)
    rec = [blk('entry', [node('c1', 'func.call', callee='@f', ins=['a'], outs=[('y', 'i32')]), node('ret', 'func.return', ins=['y'])], args=[('a', 'i32')])]
    out['自递归不需声明'] = OR.get().verify(OR.emit_core(rec, '(i32) -> i32', 'f'))[0] == 'ok'
    return out

SYNTAX = r"(use of undeclared SSA value|definition of SSA value|expected (?!'[\w.]+' as terminator)|custom op|redefinition of|undefined symbol alias|use of value .* expects different type)"   # 第 18 轮：隐式终结子的解析期校验是真拒绝，不是发射伪影

def in_func(h, par):
    """变异点（父 op 名）是否出现在某个 func.func 体内——体级预言机只发射 func.func 体。"""
    return any(n['op'] == par for f, _, _ in SY.func_scopes(h) for n, _ in Z.walk(f) if n is not f)

def judge(h, unreg, oracle):
    out = [('struct',) + tuple(x) for x in ST.check(h)]
    if out: return out
    C.ORACLE_ON = oracle
    for f, sy, ma in SY.func_scopes(h):
        C.TYPE_WARN.clear()
        try: out += [('type', x) for x in C.typecheck(f['regions'][0], f['props']['_attrs'].get('function_type'), sy, SY.sym_of(f) or 'f', ma, unreg)]
        except Exception: pass
    C.ORACLE_ON = False
    return out

MISS = collections.Counter()
UNIT = True      # 第 17 轮：M3 = 整模块按隔离单元判定（core.check_module）；False 复现第 16 轮的 M2（只发射 func.func 体）

def main(cap=16):
    print('── M1', m1())
    Z.TYPES = True; rnd = random.Random(16); orc = OR.Oracle()
    st = collections.Counter(); byk = collections.Counter(); ex = collections.defaultdict(list)
    def note(k, v):
        if len(ex[k]) < 6: ex[k].append(v)
    for path in sorted(glob.glob('corpus/*/*.mlir')):
        src = open(path).read(); unreg = 'allow-unregistered-dialect' in src
        for k, ch in enumerate(I.split_chunks(src)):
            if not re.search(r'\S', I.strip_comments(ch)): continue
            ctx = ir.Context(); ctx.allow_unregistered_dialects = unreg
            try: g = I.import_module(ir.Module.parse(ch, ctx))
            except Exception: continue
            st['模块'] += 1
            if (C.check_module(g, unreg) if UNIT else judge(g, unreg, True)): st['底本误拒'] += 1; note('底本误拒', f'{path}#{k}'); continue
            if orc.verify('\n'.join(I.emit_generic(g)), unreg)[0] != 'ok': continue
            for kind, par, h in Z.mutations(g, rnd, cap):
                s_, ds = orc.verify('\n'.join(I.emit_generic(h)), unreg)
                if s_ == 'parse':                    # Module.parse 解析即校验：只有词法/SSA 名字层的解析错误算发射伪影，其余（含块无终结子、支配）都是校验拒绝
                    if not any(re.match(SYNTAX, m) for _, m in ds): s_ = 'reject'
                    else: st['变异发射伪影（跳过）'] += 1; note('发射伪影 ' + kind, f"{par}: {ds[0][1][:120] if ds else ''}"); continue
                rej = s_ != 'ok'; st['变异'] += 1
                q0 = C.TC_STATS['预言机调用']
                sym = judge(h, unreg, False)
                full = sym or (C.check_module(h, unreg) if UNIT else judge(h, unreg, True))
                key = ('拒' if full else '收') + '/' + ('拒' if rej else '收'); byk[(kind, key)] += 1
                if full and not rej: st['误拒'] += 1; note('误拒 ' + kind, f'{path}#{k} {par}: {full[0]}'[:220])
                if rej and not full and not UNIT and not in_func(h, par): st['漏判：不在 func.func 体内（体级预言机不覆盖）'] += 1
                if rej and not full: st['漏判'] += 1; byk[(kind + ' 漏判父 op', par)] += 1; note('漏判 ' + kind, f"{par}: {' | '.join(x for _, x in ds)[:150]}")
                if rej and full: st['两者都拒'] += 1
                if not rej and not full: st['两者都收'] += 1
                if rej and not sym: st['无预言机漏判'] += 1
                if rej and not sym and full: st['其中体级预言机抓住'] += 1
                if rej and not sym and full:                 # 第 18 轮：符号层自身漏判按“预言机诊断落在的 op”及其导入账归因
                    nid = full[0][1][0] if full[0][0] == 'oracle' else None
                    op = next((m['op'] for m in OR._walk(h) if m['id'] == nid), '?')
                    lk = G.lookup(op) if op != '?' else None
                    MISS[(op, lk[0] if lk else ('未导入方言' if op.split('.')[0] not in O.DIALECTS else '未知 op'))] += 1
                    if lk and lk[0] == 'auto': note('符号层漏判落在 auto op', f"{kind} {par} → {op}: {full[0][1][1][:160]}")
    print('── M3 单元级预言机 vs 整模块 verifier' if UNIT else '── M2 体级预言机 vs 整模块 verifier', dict(st)); print('   单元', dict(C.MOD_STATS)); print('   预言机（真值）', orc.stats); print('   发射', dict(sorted(OR.WHY_NONE.items(), key=lambda x: -x[1])[:12]))
    for kk, v in sorted(byk.items(), key=lambda x: (x[0][0], -x[1])):
        if '漏判父' not in kk[0] or v >= 3: print('     ', kk, v)
    for kk, v in ex.items(): print('──', kk); [print('    ', x) for x in v]
    print('── 符号层自身漏判归因（op, 导入账）'); [print('     ', kk, v) for kk, v in MISS.most_common(40)]
    agg = collections.Counter()
    for (op, c), v in MISS.items(): agg[c] += v
    print('   按导入账', dict(agg))
    orc.close()

if __name__ == '__main__':
    if '--m2' in sys.argv: UNIT = False; sys.argv.remove('--m2')
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
