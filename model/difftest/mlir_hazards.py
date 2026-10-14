from mlir.ir import Context, Module
from mlir.passmanager import PassManager
def run(src, pipe=None):
    with Context() as c:
        c.emit_error_diagnostics = True
        try:
            m = Module.parse(src)
            if pipe: PassManager.parse(pipe).run(m.operation)
            return str(m)
        except Exception as e: return 'ERROR: ' + str(e)[:600]
print('1:', run('''func.func @f(%t: tensor<0xf32>) -> tensor<0xf32> {
  %s = tensor.extract_slice %t[0][0][1] : tensor<0xf32> to tensor<0xf32>
  return %s : tensor<0xf32> }'''))
print('2:', run('''func.func @g(%t: tensor<?xf32>, %k: index, %n: index) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %z = arith.constant 0.0 : f32
  %r = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %z) -> f32 {
    %e = tensor.extract %t[%k] : tensor<?xf32>
    %a = arith.addf %acc, %e : f32
    scf.yield %a : f32
  }
  return %r : f32 }''', 'builtin.module(func.func(loop-invariant-code-motion),one-shot-bufferize{bufferize-function-boundaries})'))
