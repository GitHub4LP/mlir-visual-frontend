"""第 12 轮：真实 MLIR IR → core 图（导入），以及 core 图 → MLIR 通用形式（回发）。

用途：
  1. 覆盖性：core 的图结构（块、块参数、区域、后继、属性）能否无损表示真实 IR —— 往返 parse∘emit∘import 与原 IR 的通用形式逐字相同（C1）。
  2. 独立差分：用 MLIR 自己的测试集（mlir/test/Dialect/**，与绑定同一提交）而不是我们自己的生成器，检查
     结构层 + 符号层在合法 IR 上不误拒（C2），以及在 MLIR 的负例（expected-error）上能抓住多少（C3）。

导入只读 MLIR 的对象模型（Python 绑定），属性一律以 MLIR 的文本原样保存在 props['_attrs']，不翻译成 core 参数：
core 参数集合比 MLIR 大（第 11 轮 xori+nuw），导入方向绝不能经过 core 的参数解释。
后继实参的划分：cf.br 全给唯一后继；cf.cond_br / cf.switch 按分段属性；其它带后继的 op 记 '_succ_opaque'（只保留扁平操作数）。
"""
import re
from structure import node, blk

def _segs(a):
    m = re.match(r'array<i32(?::\s*([-\d,\s]*))?>', a or '')
    return [int(x) for x in m.group(1).split(',')] if m and m.group(1) else []

_ID = [0]
def import_op(op, st, bnames):
    """MLIR Operation → core 节点（递归）。st: AsmState；bnames: Block → 名字（按出现顺序分配）。"""
    import mlir.ir as ir
    nm = lambda v: v.get_name(st)
    attrs = {na.name if hasattr(na, 'name') else na: None for na in op.attributes}
    out_a = {}
    for k in attrs:
        try: out_a[k] = str(op.attributes[k])
        except Exception: raise NotImplementedError(f'Python 绑定无法读出属性 {k}（如复数 dense）')
    attrs = out_a
    ins = [nm(v) for v in op.operands]
    outs = [(nm(r), str(r.type)) for r in op.results]
    regions = []
    for r in op.regions:
        bs = []
        for b in r.blocks:
            bn = bname(b, bnames)
            bs.append(blk(bn, [import_op(o.operation, st, bnames) for o in b.operations],
                          args=[(nm(a), str(a.type)) for a in b.arguments]))
        regions.append(bs)
    succs = []; props = {'_attrs': attrs, '_types': [str(v.type) for v in op.operands]}
    if len(op.successors):
        sb = [bname(b, bnames) for b in op.successors]
        if op.name == 'cf.br': succs = [(sb[0], ins)]; ins = []
        elif op.name == 'cf.cond_br':
            s = _segs(attrs.get('operandSegmentSizes')); a, t = ins[:s[0]], ins[s[0]:]
            succs = [(sb[0], t[:s[1]]), (sb[1], t[s[1]:s[1] + s[2]])]; ins = a
        elif op.name == 'cf.switch':
            s = _segs(attrs.get('operandSegmentSizes')); cs = _segs(attrs.get('case_operand_segments'))
            flag, dfl, rest = ins[:s[0]], ins[s[0]:s[0] + s[1]], ins[s[0] + s[1]:]
            succs = [(sb[0], dfl)]; k = 0
            for b, c in zip(sb[1:], cs): succs.append((b, rest[k:k + c])); k += c
            ins = flag
        else:
            succs = [(b, []) for b in sb]; props['_succ_opaque'] = True
    _ID[0] += 1
    return node(f'{op.name}#{_ID[0]}', op.name, ins=ins, outs=outs, props=props, regions=regions, succs=succs)

def bname(b, bnames):
    """Block 没有稳定的 Python 身份：用相等比较找回已登记的同一个块。"""
    for k, n in bnames:
        if k == b: return n
    bnames.append((b, f'b{len(bnames)}')); return bnames[-1][1]

def import_module(m):
    import mlir.ir as ir
    st = ir.AsmState(m.operation, use_local_scope=True); bnames = []
    return import_op(m.operation, st, bnames)

# ── core 图（导入得到的）→ MLIR 通用形式。属性原样回发；值名、块名照用。
def emit_generic(n, ind='', loc=False):
    out = []
    a = n['props'].get('_attrs', {})
    ins = list(n['ins']); tys = list(n['props'].get('_types', []))
    succ = ''
    if n['succs']:
        if not n['props'].get('_succ_opaque'):
            ins = ins + [v for _, vs in n['succs'] for v in vs]
        succ = '[' + ', '.join('^' + b for b, _ in n['succs']) + ']'
    regs = []
    for r in n['regions']:
        lines = []
        for b in r:
            lines.append(f'{ind}^{b["name"]}(' + ', '.join(f'{v}: {t}' for v, t in b['args']) + '):')
            for m in b['ops']: lines += emit_generic(m, ind + '  ', loc)
        regs.append('{\n' + '\n'.join(lines) + f'\n{ind}}}')
    groups = []                                           # %x#0, %x#1 → %x:2（MLIR 的多结果组名）
    for v, _ in n['outs']:
        b = v.split('#')[0]
        if groups and groups[-1][0] == b and '#' in v: groups[-1][1] += 1
        else: groups.append([b, 1])
    res = ', '.join(b if k == 1 and not any(v.startswith(b + '#') for v, _ in n['outs']) else f'{b}:{k}' for b, k in groups)
    out.append(ind + (res + ' = ' if res else '') + f'"{n["op"]}"(' + ', '.join(ins) + ')' + succ
               + ((' (' + ', '.join(regs) + ')') if regs else '')
               + ((' {' + ', '.join(f'{k} = {v}' for k, v in a.items()) + '}') if a else '')
               + ' : (' + ', '.join(tys) + ') -> (' + ', '.join(t for _, t in n['outs']) + ')' + (f' loc("{n["id"]}")' if loc else ''))
    return out

def split_chunks(text):
    """lit 的 --split-input-file 约定：以 '// -----' 分段。"""
    return re.split(r'^// -----\s*$', text, flags=re.M)

def expected(chunk):
    return re.findall(r'expected-error\s*(@[-+\w]+)?\s*\{\{(.*?)\}\}', chunk)

def strip_comments(chunk):
    return '\n'.join(l for l in chunk.splitlines() if not l.lstrip().startswith('//'))
