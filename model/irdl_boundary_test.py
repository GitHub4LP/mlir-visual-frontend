"""第 24 轮：IRDL 对 builtin 类型的表达边界（L48 的证据，单独成测；崩溃在隔离子进程里发生，不影响其余用例）。
每个用例 = (说明, 约束定义行, 操作数约束, 实例类型, 期望)；期望 ∈ ok / reject / load / crash（本提交 9bae84b0 的实测行为）。
期望不符即失败：MLIR 升级后 IRDL 能力变化时，这里先报出来，再更新 irdl_export 的安全子集。
用法：PYTHONPATH=model python3 model/irdl_boundary_test.py
"""
import irdl_export as X

def dia(lines, opds):
    return 'irdl.dialect @vf {\n  irdl.operation @b {\n' + '\n'.join('    ' + l for l in lines) + f'\n    irdl.operands({opds})\n    irdl.results()\n  }}\n}}'
def inst(T): return X.instance('b', T, [])

ANY = ['%v = irdl.any']; I32 = ['%v = irdl.is i32']
CASES = [
    ('any 接受整数', ANY, 'p0: %v', ['i32'], 'ok'),
    ('any 接受 tensor', ANY, 'p0: %v', ['tensor<2xf32>'], 'ok'),
    ('is 精确：i32', I32, 'p0: %v', ['i32'], 'ok'),
    ('is 精确：拒绝 i64', I32, 'p0: %v', ['i64'], 'reject'),
    ('同一变量 ⇒ 相等：接受', ANY, 'p0: %v, p1: %v', ['i32', 'i32'], 'ok'),
    ('同一变量 ⇒ 相等：拒绝', ANY, 'p0: %v, p1: %v', ['i32', 'i64'], 'reject'),
    ('base integer：接受 i8', ['%v = irdl.base "!builtin.integer"'], 'p0: %v', ['i8'], 'ok'),
    ('base integer：拒绝 f32', ['%v = irdl.base "!builtin.integer"'], 'p0: %v', ['f32'], 'reject'),
    ('base tensor：接受', ['%v = irdl.base "!builtin.tensor"'], 'p0: %v', ['tensor<2xf32>'], 'ok'),
    ('base tensor：拒绝 vector', ['%v = irdl.base "!builtin.tensor"'], 'p0: %v', ['vector<2xf32>'], 'reject'),
    ('any_of(is, is)：接受', ['%a = irdl.is i32', '%b = irdl.is i64', '%v = irdl.any_of(%a, %b)'], 'p0: %v', ['i64'], 'ok'),
    ('any_of(is, is)：拒绝', ['%a = irdl.is i32', '%b = irdl.is i64', '%v = irdl.any_of(%a, %b)'], 'p0: %v', ['i16'], 'reject'),
    ('any_of(base, base) ⇒ MLIR UNREACHABLE', ['%a = irdl.base "!builtin.integer"', '%b = irdl.base "!builtin.tensor"', '%v = irdl.any_of(%a, %b)'],
     'p0: %v', ['i8'], 'crash'),
    ('base ranked_tensor ⇒ 加载失败', ['%v = irdl.base "!builtin.ranked_tensor"'], 'p0: %v', ['tensor<2xf32>'], 'load'),
]

if __name__ == '__main__':
    V = X.get(); bad = 0; got = {}
    for name, lines, opds, T, want in CASES:
        r = V.verify(dia(lines, opds), inst(T))[0]
        got[r] = got.get(r, 0) + 1
        ok = r == want; bad += not ok
        print(f'  {"✓" if ok else "✗"} {name}: 实测 {r}，期望 {want}')
    V.close()
    print(f'── {len(CASES)} 例，不符 {bad}；分布 {got}')
    raise SystemExit(bad)
