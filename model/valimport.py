"""第 25 轮：导入的 MLIR（irimport，属性原样存 _attrs）→ 值层的 core 记法（spec/values.md §8）。

此前值层只吃 core 自己的记法（生成器、编辑），导入的 IR 从未进入值层：
  arith 节点带 _attrs ⇒ reads 守卫一律 ValueUnknown（没有错，但也没有事实）；
  tensor/memref 的 reads=None（未对账）⇒ static_offsets 等属性被无视，_extract_slice 按位置取操作数 ⇒ z3 异常/错位。
原则（L51 推广到导入，L53）：每个 MLIR 属性要么被翻译（TRANSLATED，带适用条件），要么登记为语义无关（IRRELEVANT，带理由），
否则该节点 ⇒ 未知（返回原因，不给事实）。条件不满足（步长≠1、降秩、insert 尺寸与源形状不符…）同样 ⇒ 未知。
"""
import copy, re

KDYN = -9223372036854775808
# (op, MLIR 属性) → 处理方式；values_test.S0 / valimport_test.A0 与 ODS 对账
TRANSLATED = {
    ('arith.constant', 'value'): '整数/index 字面量 → value + type；浮点等 ⇒ 未知',
    ('arith.cmpi', 'predicate'): '谓词编号 → predicate',
    **{(f'arith.{o}', 'overflowFlags'): '#arith.overflow<…> → nsw/nuw' for o in ('addi', 'subi', 'muli', 'shli', 'trunci')},
    **{(f'arith.{o}', 'isExact'): 'isExact → exact' for o in ('divui', 'divsi', 'shrui', 'shrsi')},
    **{(f'tensor.{o}', a): 'static_* 与动态操作数合并为 core 的全动态记法（静态项物化为 index 常量）；步长须全为 1、不降秩'
       for o in ('extract_slice', 'insert_slice') for a in ('static_offsets', 'static_sizes', 'static_strides')},
}
IRRELEVANT = {
    ('ub.poison', 'value'): '毒值属性的种类；结果恒为毒值',
    ('memref.load', 'nontemporal'): '缓存提示，不影响语义（取 false 以外的值也不影响值层：读到的内容本就是自由变量）',
    ('memref.store', 'nontemporal'): '同上',
}
STRUCTURAL = {'operandSegmentSizes', 'resultSegmentSizes'}
# 未登记（⇒ 未知）：scf.for.unsignedCmp（无符号比较的循环界，core 的循环摘要只有有符号）、memref.*.alignment（未对齐 ⇒ UB，值层未建模）

class Unknown(Exception): pass

def _arr(s):
    m = re.fullmatch(r'array<i64(?::\s*(.*))?>', s.strip())
    if not m: raise Unknown(f'无法解析数组属性 {s}')
    return [int(x) for x in m.group(1).split(',')] if m.group(1) else []
def _segs(s):
    m = re.fullmatch(r'array<i32:\s*(.*)>', s.strip())
    if not m: raise Unknown(f'无法解析 operandSegmentSizes {s}')
    return [int(x) for x in m.group(1).split(',')]
def _int_lit(s):
    m = re.fullmatch(r'(-?\d+|true|false)\s*(?::\s*(\S+))?', s.strip())
    if not m: raise Unknown(f'非整数字面量 {s}')
    v = m.group(1); return (1 if v == 'true' else 0 if v == 'false' else int(v)), m.group(2) or 'i1'
def _dense(s, ty):
    """dense<…> : tensor<…> 的整数字面量（splat 或嵌套列表）→ 行优先展平表；十六进制/资源/浮点/复数 ⇒ Unknown。嵌套结构须与形状一致。"""
    m = re.fullmatch(r'dense<(.*)>\s*:\s*(\S+)', s.strip(), re.S)
    if not m or m.group(2) != ty: raise Unknown(f'tensor 常量 {s[:40]}')
    body = m.group(1).strip()
    if not re.fullmatch(r'[\[\]\s,\-\d]+|true|false', body): raise Unknown(f'tensor 常量非整数字面量 {body[:30]}')
    if body in ('true', 'false'): return [1 if body == 'true' else 0]
    import json
    try: v = json.loads(body)
    except Exception: raise Unknown(f'tensor 常量无法解析 {body[:30]}')
    if isinstance(v, int): return [v]
    dims = _rank(ty)
    def flat(x, k):
        if k == len(dims):
            if not isinstance(x, int): raise Unknown('tensor 常量嵌套与形状不符')
            return [x]
        if not isinstance(x, list) or (dims[k] is not None and len(x) != dims[k]): raise Unknown('tensor 常量嵌套与形状不符')
        return [y for e in x for y in flat(e, k + 1)]
    return flat(v, 0)
def _rank(ty):
    m = re.fullmatch(r'tensor<((?:[0-9?]+x)*)\w+>', ty)
    if not m: raise Unknown(f'非有秩 tensor {ty}')
    return [None if d == '?' else int(d) for d in m.group(1).split('x')[:-1]]

def to_core(blocks, flip=None):
    """返回 core 记法的块（深拷贝）。不能安全翻译 ⇒ 抛 Unknown(原因)。flip：变异灵敏度用的错误翻译开关（static_zero：静态偏移当 0；pred：cmpi 谓词编号异或 1；drop_nsw：丢 nsw）。"""
    k = [0]
    def const(v):
        k[0] += 1; o = f'%vi_c{k[0]}'
        return {'id': f'vi{k[0]}', 'op': 'arith.constant', 'ins': [], 'outs': [(o, 'index')], 'regions': [], 'succs': [],
                'props': {'value': v, 'type': 'index'}}, o
    def blk(b):
        out = []
        for n in b['ops']: out.extend(node(n))
        return dict(name=b.get('name', 'entry'), args=list(b['args']), ops=out)
    def node(n):
        op = n['op']; at = dict((n.get('props') or {}).get('_attrs') or {})
        m = {'id': n['id'], 'op': op, 'ins': list(n['ins']), 'outs': list(n['outs']), 'succs': list(n.get('succs') or []),
             'regions': [[blk(b) for b in r] for r in n['regions']], 'props': {}}
        if m['succs']: raise Unknown(f'{op}：后继（多块 CFG）')
        if any(re.search(r'\bi0\b', t) for t in [t for _, t in n['outs']] + list((n.get('props') or {}).get('_types') or [])):
            raise Unknown(f'{op}：i0（零宽整数）值层未建模')
        pre = []
        tys = [t for _, t in n['outs']] + list((n.get('props') or {}).get('_types') or [])
        if op.startswith(('arith.', 'ub.')) and any(re.match(r'(vector|memref)<', t) for t in tys):
            raise Unknown(f'{op}：逐元素作用于 vector/memref 未建模（core 的 arith 只有标量与 ranked tensor 语义）')
        if op.startswith(('arith.', 'ub.')) and any(t.startswith('tensor<') and not re.fullmatch(r'tensor<(?:(?:\d+|\?)x)*(?:i\d+|index)>', t) for t in tys):
            raise Unknown(f'{op}：逐元素作用于非整数元素/无秩/带 encoding 的 tensor 未建模')     # 第 33 轮：ranked 整数 tensor 交给 core.elementwise
        for a in list(at):
            if a in STRUCTURAL or (op, a) in IRRELEVANT: continue
            if (op, a) not in TRANSLATED: raise Unknown(f'{op}.{a}：属性未登记（不静默忽略）')
        p = m['props']
        if op == 'arith.constant' and n['outs'][0][1].startswith('tensor<'):
            p.update(value=0, dense=_dense(at['value'], n['outs'][0][1]))     # 第 33 轮
        elif op == 'arith.constant':
            v, ty = _int_lit(at['value'])
            if ty != n['outs'][0][1] and not (ty == 'i1' and n['outs'][0][1] == 'i1'): raise Unknown(f'常量类型 {ty}')
            p.update(value=v, type=ty)
        elif op == 'arith.cmpi': p['predicate'] = _int_lit(at['predicate'])[0] ^ (1 if flip == 'pred' else 0)
        elif 'overflowFlags' in at:
            fl = re.fullmatch(r'#arith\.overflow<([\w, ]*)>', at['overflowFlags'].strip())
            if not fl: raise Unknown(f'溢出标志 {at["overflowFlags"]}')
            for f in fl.group(1).replace(' ', '').split(','):
                if f == 'nsw':
                    if flip != 'drop_nsw': p['nsw'] = True
                elif f == 'nuw': p['nuw'] = True
                elif f not in ('none', ''): raise Unknown(f'未知溢出标志 {f}')
        if 'isExact' in at: p['exact'] = True
        if op in ('tensor.extract_slice', 'tensor.insert_slice'):
            ins = n['ins']; src_ty = n['props']['_types'][0]
            seg = _segs(at['operandSegmentSizes'])
            nfix = 1 if op == 'tensor.extract_slice' else 2
            fixed, rest = ins[:nfix], ins[nfix:]
            dyn = [rest[:seg[nfix]], rest[seg[nfix]:seg[nfix] + seg[nfix + 1]], rest[seg[nfix] + seg[nfix + 1]:]]
            stat = [_arr(at['static_offsets']), _arr(at['static_sizes']), _arr(at['static_strides'])]
            vals = []
            for sv, dv in zip(stat, dyn):
                it = iter(dv); row = []
                for x in sv:
                    if x == KDYN: row.append(next(it))
                    else:
                        if flip == 'static_zero' and sv is stat[0]: x = 0
                        c, o = const(x); pre.append(c); row.append(o)
                vals.append(row)
            # 第 34b 轮：全形式（步长、降/升秩、insert 的尺寸）翻译到 core 的 *.g；降秩掩码按 MLIR computeRankReductionMask 的贪心匹配
            big = _rank(src_ty if op == 'tensor.extract_slice' else n['props']['_types'][1])
            small = _rank(n['outs'][0][1] if op == 'tensor.extract_slice' else src_ty)
            if not (len(stat[0]) == len(stat[1]) == len(stat[2]) == len(big)): raise Unknown(f'{op}：偏移/尺寸/步长个数与秩不符')
            szs = [None if x == KDYN else x for x in stat[1]]; drop, j = [], 0
            for kk, sk in enumerate(szs):
                if j < len(small) and small[j] == sk: j += 1
                elif sk == 1: drop.append(kk)
                else: raise Unknown(f'{op}：降秩掩码无法匹配')
            if j != len(small): raise Unknown(f'{op}：降秩掩码无法匹配')
            m['op'] = op + '.g'; p['drop'] = tuple(drop)
            m['ins'] = fixed + vals[0] + vals[1] + vals[2]
        if op == 'arith.constant' or op.startswith(('arith.', 'tensor.', 'memref.', 'ub.')): pass
        elif op in ('scf.for', 'scf.if', 'scf.yield', 'func.return'): pass
        else: raise Unknown(f'{op}：值层未覆盖')
        return pre + [m]
    for b in blocks:
        if any(re.search(r'\bi0\b', t) for _, t in b['args']): raise Unknown('块参数 i0：值层未建模')
    return [blk(b) for b in blocks]
