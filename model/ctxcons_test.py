"""第 50 轮：ctxcons 的定向检验（真实 MLIR verifier 为真值）。
  C4 HasParent 拒绝权：op 放到 ODS HasParent 允许集之外 ⇒ 我们判 reject 且 MLIR 拒绝；放在允许集内 ⇒ 我们不判 reject 且 MLIR 接受。
     （trait 的校验是 MLIR 原生的 OpTrait::HasParent::Impl::verifyTrait，见 inventory/IR_OpDefinition.h；与 odssig 的 auto 级同类。）
  C5 导入账：全部已导入 op 中 HasParent 类名可解析的比例（不可解析 ⇒ 未知，不拒绝）。
"""
import sys
import verdict as V, ctxcons as X, oracle as OR, odsjson as O

CASES = [   # (op, 新父, MLIR 文本)
    ('memref.atomic_yield', 'func.func', 'func.func @f(%x: f32) {\n  "memref.atomic_yield"(%x) : (f32) -> ()\n}'),
    ('tensor.yield', 'func.func', 'func.func @f(%x: f32) {\n  "tensor.yield"(%x) : (f32) -> ()\n}'),
    ('scf.yield', 'func.func', 'func.func @f() {\n  "scf.yield"() : () -> ()\n}'),
    ('func.return', 'scf.execute_region', 'func.func @f() {\n  scf.execute_region {\n    "func.return"() : () -> ()\n  }\n  return\n}'),
    ('scf.reduce.return', 'func.func', 'func.func @f(%x: f32) {\n  "scf.reduce.return"(%x) : (f32) -> ()\n}'),
    ('scf.yield', 'scf.execute_region', 'func.func @f() {\n  scf.execute_region {\n    "scf.yield"() : () -> ()\n  }\n  return\n}'),
    ('func.return', 'func.func', 'func.func @f() {\n  "func.return"() : () -> ()\n}'),
    ('tensor.yield', 'tensor.generate', 'func.func @f(%n: index) -> tensor<?xf32> {\n  %t = tensor.generate %n {\n  ^bb0(%i: index):\n    %c = arith.constant 0.0 : f32\n    "tensor.yield"(%c) : (f32) -> ()\n  } : tensor<?xf32>\n  return %t : tensor<?xf32>\n}'),
]

if __name__ == '__main__':
    o = OR.Oracle(); agree, rows = 0, []
    for op, par, txt in CASES:
        cv, why = X.move_check([{'op': op, 'regions': []}], ['?', 'func.func'], par)
        r, d = o.verify(txt)
        ok = (cv == 'reject') == (r != 'ok')
        agree += ok; rows.append((op, par, cv, r, (d[0][1][:70] if d else '')))
    o.close()
    for x in rows: print('   ', x)
    V.check('C4 HasParent 拒绝权', agree == len(CASES) and sum(r[2] == 'reject' for r in rows) >= 4, f'{agree}/{len(CASES)} 与 MLIR 一致')
    tot, unres, withp = 0, [], 0
    for dia in O.DIALECTS:
        try: ops = O.ops(dia)
        except Exception: continue
        for op in ops:
            try: ap = X.allowed_parents(op)
            except Exception as e: ap = 'unresolved'
            tot += 1; withp += ap is not None; unres += [op] if ap == 'unresolved' else []
    print('── C5 HasParent 导入:', {'op': tot, '有 HasParent': withp, '不可解析': len(unres)}, unres[:8])
    V.check('C5 HasParent 导入账', withp > 0 and len(unres) <= withp // 5, f'op {tot}，有 HasParent {withp}，不可解析（⇒ 未知）{len(unres)}')
    V.finish()
