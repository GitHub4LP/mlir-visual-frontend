"""迷你 ODS 导入器：解析 .td 中的 class/def 继承链，取出 arguments/results/traits。
这只是原型。生产实现应改用 `llvm-tblgen -dump-json`，因为它会完整展开模板参数。
这里的近似：traits 取整条继承链上出现的 trait 标识的并集（arith/tensor 的写法都是 traits # [...]，所以这个近似对它们成立）；
arguments/results 取继承链上最先出现的一个（def 自身的 let 优先）。
"""
import re

def _headers(src):
    out = {}
    for m in re.finditer(r'^(class|def)\s+(\w+)\s*', src, re.M):
        i = m.end()
        if src[i:i+1] == '<':                                   # 跳过带嵌套 <> 的模板参数表
            d = 0
            while True:
                d += {'<': 1, '>': -1}.get(src[i], 0); i += 1
                if d == 0: break
        while src[i:i+1].isspace(): i += 1
        if src[i:i+1] != ':': continue
        i += 1; start, depth = i, 0
        while i < len(src):                                    # 读到顶层的 { 或 ; 为止
            c = src[i]
            if c in '<([': depth += 1
            elif c in '>)]': depth -= 1
            elif c in '{;' and depth <= 0: break
            i += 1
        head = src[start:i]
        body = ''
        if i < len(src) and src[i] == '{':
            d, j = 0, i
            while j < len(src):
                d += {'{': 1, '}': -1}.get(src[j], 0); j += 1
                if d == 0: break
            body = src[i:j]
        pm = re.match(r'\s*(\w+)', head)
        if not pm: continue
        parent = pm.group(1)
        out[m.group(2)] = dict(head=head, body=body, parent=parent)
    return out

def _dag(text, kind):
    m = re.search(kind + r'\s*=\s*\(' + ('ins' if kind == 'arguments' else 'outs') + r'(.*?)\)\s*;', text, re.S) \
        or re.search(('Arguments' if kind == 'arguments' else 'Results') + r'<\(\s*' + ('ins' if kind == 'arguments' else 'outs') + r'(.*?)\)>', text, re.S)
    if not m: return None
    items = re.findall(r'([\w<>\[\]",:\s]+?):\$(\w+)', m.group(1))
    return [(n, re.sub(r'\s+', '', t.strip(', '))) for t, n in items]

def import_op(src, name, known_traits):
    H = _headers(src); traits, args, res, chain, cur = set(), None, None, [], name
    while cur in H:
        h = H[cur]; chain.append(cur); text = h['head'] + h['body']
        traits |= {t for t in re.findall(r'\b(\w+)\b', h['head']) if t in known_traits}
        args = args or _dag(text, 'arguments'); res = res or _dag(text, 'results')
        cur = h['parent']
    return dict(name=name, chain=chain, traits=sorted(traits), args=args, results=res)

KNOWN = {'Pure', 'Commutative', 'SameOperandsAndResultType', 'ConditionallySpeculatable', 'AttrSizedOperandSegments',
         'OffsetSizeAndStrideOpInterface', 'NoMemoryEffect', 'AlwaysSpeculatable', 'Idempotent', 'Involution',
         'SameOperandsAndResultShape', 'ConstantLike', 'NoTerminator', 'SingleBlockImplicitTerminator'}

if __name__ == '__main__':
    import sys, json
    for path, ops in [('inventory/Arith_ArithOps.td', ['Arith_AddIOp', 'Arith_SubIOp', 'Arith_DivUIOp']),
                      ('inventory/Tensor_TensorOps.td', ['Tensor_DimOp', 'Tensor_ExtractOp', 'Tensor_ExtractSliceOp'])]:
        src = open(path).read()
        for op in ops: print(json.dumps(import_op(src, op, KNOWN), ensure_ascii=False))
