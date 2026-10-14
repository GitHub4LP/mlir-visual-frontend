func.func @f(%x: i4) -> i4 {
  %w = arith.constant 5 : i4
  %r = arith.shrui %x, %w : i4
  func.return %r : i4
}
