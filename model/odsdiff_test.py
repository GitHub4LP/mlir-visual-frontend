"""第 31 轮：ODS 导入（odsjson/odssig，自写的 tblgen JSON 解释）对照 mlir-tblgen 自己的 C++ 解释（mlir::tblgen::Operator，经 eudsl-tblgen 绑定）。
动机：odssig 从 llvm-tblgen -dump-json 的原始记录**重新实现**了 mlir-tblgen 的一部分语义：
  端口多重性（Variadic/Optional/VariadicOfVariadic）、属性与 property 是否必需、区域与后继个数，
  以及 OpDefinitionsGen 是否自动生成 inferReturnTypes（Operator::allResultTypesKnown）。
这些重实现此前只有语料间接验证（L28/L29）。mlir::tblgen::Operator 是 MLIR 生成 verifier/builder 时用的同一份 C++ 代码。
版本：eudsl-tblgen wheel 用较新的 LLVM 构建，解析的是本 wheel（9bae84b0）自带的 .td。所以一处分歧要么是 odssig 的缺陷，
要么是 mlir::tblgen 版本漂移，须逐条归因，不能直接当作缺陷。
用法：PYTHONPATH=model python3 model/odsdiff_test.py
"""
import os, collections
import eudsl_tblgen as T
import odsjson as O, odssig as G

def mult(c):
    return 'varvar' if c.is_variadic_of_variadic() else 'variadic' if c.is_variadic() else 'optional' if c.is_optional() else 'one'

def tg_ops(dialect):
    tds = O.DIALECTS[dialect]; tds = [tds] if isinstance(tds, str) else tds
    inc = [O.SITE] + ([O.GEN] if os.path.isdir(O.GEN) else [])
    out = {}
    for td in tds:
        rk = T.RecordKeeper(); rk.parse_td(f'{O.SITE}/mlir/Dialect/{td}.td', inc)
        main = os.path.realpath(f'{O.SITE}/mlir/Dialect/{td}.td')
        for r in rk.get_all_derived_definitions('Op'):
            op = T.Operator(r); nm = op.get_operation_name()
            if not nm.startswith(dialect + '.'): continue
            ins = [(o.name, mult(o)) for o in (op.get_operand(i) for i in range(op.get_num_operands()))]
            outs = [(o.name, mult(o)) for o in (op.get_result(i) for i in range(op.get_num_results()))]
            attrs = {}
            for i in range(op.get_num_native_attributes()):
                a = op.get_attribute(i)
                attrs[a.name] = not (a.attr.is_optional() or a.attr.has_default_value())
            nprop = op.get_num_args() - op.get_num_operands() - op.get_num_native_attributes()
            for i in range(nprop):
                p = op.get_property(i); attrs[p.name] = not p.prop.has_default_value()
            out[nm] = dict(ins=ins, outs=outs, attrs=attrs, regions=op.get_num_regions(), successors=op.get_num_successors(),
                           infer=op.all_result_types_known() and op.get_num_results() > 0)
    return out

def ours(dialect, nm):
    S = G.sig(dialect, nm)
    return dict(ins=[(n or '', m) for n, _, m, _ in S['ins']], outs=[(n or '', m) for n, _, m, _ in S['outs']],
                attrs={n: req for n, _, _, req in S['attrs']}, regions=S['regions'], successors=S['successors'],
                infer=S['infer'] is not None)

def run():
    st = collections.Counter(); diff = collections.defaultdict(list)
    for dia in O.DIALECTS:
        try: tg = tg_ops(dia)
        except Exception as e: st['方言解析失败'] += 1; diff['方言解析失败'].append((dia, str(e)[:80])); continue
        mine = set(O.ops(dia))
        for nm in sorted(set(tg) | mine):
            if nm not in tg: st['只在 odsjson'] += 1; diff['只在 odsjson'].append(nm); continue
            if nm not in mine: st['只在 tblgen'] += 1; diff['只在 tblgen'].append(nm); continue
            st['op'] += 1
            try: a = ours(dia, nm)
            except Exception as e: st['odssig 异常'] += 1; diff['odssig 异常'].append((nm, str(e)[:60])); continue
            b = tg[nm]
            for k in ('ins', 'outs', 'attrs', 'regions', 'successors', 'infer'):
                if a[k] == b[k]: st[k + ' 一致'] += 1
                else: st[k + ' 分歧'] += 1; diff[k].append((nm, a[k], b[k]))
    return st, diff

if __name__ == '__main__':
    st, diff = run()
    print('── T1 odssig vs mlir::tblgen::Operator:', dict(st))
    for k, v in diff.items():
        print(f'   {k}: {len(v)}')
        for x in v[:8]: print('      ', x)
