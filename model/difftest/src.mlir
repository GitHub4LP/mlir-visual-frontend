func.func @f(%x: i4) -> i4 {
  %c = arith.constant 0 : i4
  %r = arith.andi %x, %c : i4
  func.return %r : i4
}
