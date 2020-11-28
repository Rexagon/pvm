# PVM
Strange stack VM prototype

### Types
```
num:
  | int -> 4 byte integer
  | short -> 2 byte integer
  | byte

flags:
  | Z - result is zero
  | S - result is negative
  | V - overflow occurred
  | C - carry
```

### Commands

```
t : I => int, S => short, _ => byte
f : Z => flag Z, S => flag S, V => flag V, C => flag C
num : integer_type(t)

- NOP
- HLT

- PUSH(t) value, (...) -> (..., num value)
- POP(t), (..., num x) -> (...)
- SWAP(t), (..., num a, num b) -> (..., num b, num a)
- ROR(t), (..., num a, num b, num c) -> (..., num c, num a, num b)
- ROL(t), (..., num a, num b, num c) -> (..., num b, num c, num a)

- CALL, (..., addr x) -> (...), s15=x, PC=PC+x
- JMP, (..., addr x) -> (...), PC=PC+x
- RET, PC=s15
- BRANCH(t), (..., num v, addr x) -> (...), (v > 0 ? NOP : JMP x)
- BRANCHNOT(t), (..., num v, addr x) -> (...), (v == 0 ? NOP : JMP x)
- FBRANCH(f), (..., addr x) -> (...), (flags(f) == 1 ? NOP : JMP x)
- FBRANCHNOT(f), (..., addr x) -> (...), (flags(f) != 1 ? NOP : JMP x)

- DUP(t), (..., num a) -> (..., num a, num a)
- PEEK(t) rel, (..., s0 - rel: value, ...) -> (..., s0 - rel: value, ..., value), 1 <= rel <= 16

- CLRFLAGS, clear flags
- ADD(C)(t), (..., num a, num b) -> (..., num sum), sum = (a + b + ?C)
- ADD(C)L(t), (..., num a, num b) -> (..., num (high sum), num (low sum)), sum = (a + b + ?C)
- SUB(C)(t), (..., num a, num b) -> (..., num (a - b - ?C))
- INC(t), (..., num x) -> (..., num (x + 1))
- DEC(t), (..., num x) -> (..., num (x - 1))
- MUL(t), (..., num x, num y) -> (..., num m), m = (x * y)
- MULL(t), (..., num x, num y) -> (..., num (high m), num (low m)), m = (x * y)
- NEGATE(t), (..., num a) -> (..., num (- a))
- DIV(t), (..., num x, num y) -> (..., num (x / y))
- MOD(t), (..., num x, num y) -> (..., num (x mod y))
- DIVMOD(t), (..., num x, num y) -> (..., num (x / y), num (x mod y))

- AND(t), (..., num x, num y) -> (..., num (x & y))
- OR(t), (..., num x, num y) -> (..., num (x | y))
- XOR(t), (..., num x, num y) -> (..., num (x ^ y))
- NOT(t), (..., num x) -> (..., num (~x))
- MIN(t), (..., num x, num y) -> (..., num min(x, y))
- MAX(t), (..., num x, num y) -> (..., num max(x, y))
- ABS(t), (..., num x) -> (..., num abs(x))
- SGN(t), (..., num x) -> (..., byte (x < 0 ? -1 : (x == 0 ? 0 : 1)))
- RSHIFT(t), (..., num x, num n) -> (..., num (x >> n))
- LSHIFT(t), (..., num x, num n) -> (..., num (x << n))

- LT, (..., num x, num y) -> (..., byte (x < y))
- LE, (..., num x, num y) -> (..., byte (x <= y))
- GT, (..., num x, num y) -> (..., byte (x > y))
- GE, (..., num x, num y) -> (..., byte (x >= y))
- EQ, (..., num x, num y) -> (..., byte (x == y))
- NEQ, (..., num x, num y) -> (..., byte (x == y))
```