module {
func.func @fib__i64(%n: i64) -> (i64) attributes {llvm.emit_c_interface} {
  %lb = arith.constant 0 : index loc("fib:lb")
  %st = arith.constant 1 : index loc("fib:st")
  %ub = arith.index_cast %n : i64 to index loc("fib:ub")
  %a0 = arith.constant 0 : i64 loc("fib:a0")
  %b0 = arith.constant 1 : i64 loc("fib:b0")
  %loop, %loop_r1 = scf.for %i = %lb to %ub step %st iter_args(%a = %a0, %b = %b0) -> (i64, i64) {
    %t = arith.addi %a, %b : i64 loc("fib:t")
    scf.yield %b, %t : i64, i64
  } loc("fib:loop")
  func.return %loop : i64
}
func.func @fact__i64(%n: i64) -> (i64) attributes {llvm.emit_c_interface} {
  %one = arith.constant 1 : i64 loc("fact:one")
  %le = arith.cmpi sle, %n, %one : i64 loc("fact:le")
  %if = scf.if %le -> (i64) {
    scf.yield %one : i64
  } else {
    %m = arith.subi %n, %one : i64 loc("fact:m")
    %r = func.call @fact__i64(%m) : (i64) -> (i64) loc("fact:r")
    %p = arith.muli %n, %r : i64 loc("fact:p")
    scf.yield %p : i64
  } loc("fact:if")
  func.return %if : i64
}
func.func @sumsq__i64(%n: i64) -> (i64) attributes {llvm.emit_c_interface} {
  %buf = memref.alloca() : memref<1xi64> loc("sumsq:buf")
  %z = arith.constant 0 : index loc("sumsq:z")
  %c0 = arith.constant 0 : i64 loc("sumsq:c0")
  memref.store %c0, %buf[%z] : memref<1xi64> loc("sumsq:init")
  %lb = arith.constant 0 : index loc("sumsq:lb")
  %st = arith.constant 1 : index loc("sumsq:st")
  %ub = arith.index_cast %n : i64 to index loc("sumsq:ub")
  scf.for %i = %lb to %ub step %st {
    %v = memref.load %buf[%z] : memref<1xi64> loc("sumsq:v")
    %ii = arith.index_cast %i : index to i64 loc("sumsq:ii")
    %sq = arith.muli %ii, %ii : i64 loc("sumsq:sq")
    %s = arith.addi %v, %sq : i64 loc("sumsq:s")
    memref.store %s, %buf[%z] : memref<1xi64> loc("sumsq:w")
  } loc("sumsq:loop")
  %res = memref.load %buf[%z] : memref<1xi64> loc("sumsq:res")
  func.return %res : i64
}
func.func @axpy__f64_f64_f64(%a: f64, %x: f64, %y: f64) -> (f64) {
  %m = arith.mulf %a, %x : f64 loc("axpy:m")
  %s = arith.addf %m, %y : f64 loc("axpy:s")
  func.return %s : f64
}
func.func @axpy_test__f64_f64_f64(%a: f64, %x: f64, %y: f64) -> (f64) attributes {llvm.emit_c_interface} {
  %r = func.call @axpy__f64_f64_f64(%a, %x, %y) : (f64, f64, f64) -> (f64) loc("axpy_test:r")
  func.return %r : f64
}
func.func @axpy__f32_f32_f32(%a: f32, %x: f32, %y: f32) -> (f32) {
  %m = arith.mulf %a, %x : f32 loc("axpy:m")
  %s = arith.addf %m, %y : f32 loc("axpy:s")
  func.return %s : f32
}
func.func @axpy_f32__f32_f32_f32(%a: f32, %x: f32, %y: f32) -> (f32) attributes {llvm.emit_c_interface} {
  %r = func.call @axpy__f32_f32_f32(%a, %x, %y) : (f32, f32, f32) -> (f32) loc("axpy_f32:r")
  func.return %r : f32
}
}