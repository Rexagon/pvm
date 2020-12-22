main:
  ; make arrays
  PUSHI make_array_1
  CALL
  PUSHI make_array_2
  CALL

  ; compare arrays length
  DUP
  PUSH 2
  MUL
  PUSH 1
  ADD
  LOAD

  PEEK 1
  NEQ
  PUSHI end
  BRANCH

  ; push counter
  DUP

  ; push sum
  PUSHI 0

loop:
  ; push array length as short int
  PUSH 0
  PEEK 5

  ; check exit condition
  EQ
  PUSHI end
  BRANCH

  ; calculate first array offset
  PUSH 0        ; N*2
  PEEK 6
  PUSHS 2
  MULS
  PUSH 0        ; i*2
  PEEK 7
  PUSHS 2
  MULS

  PUSHS 5       ; N*2+i*2+5
  ADDS
  ADDS
  LOADS

  ; calculate second array offset
  PUSH 0        ; i*2
  PEEK 7
  PUSHS 2
  MULS

  PUSHS 6       ; i*2+6
  ADDS
  LOADS

  ; multiply and sum numbers
  MULLS
  ADDI

  ; decrement counter
  ROLS
  DEC
  RORS

  ; goto loop
  PUSHI loop
  JMP loop

end:
  HLT


make_array_1:
  PUSHS 1
  PUSHS 2
  PUSHS 3
  PUSHS 4
  PUSHS 16
  PUSH 5      ; array length
  RET

make_array_2:
  PUSHS 5
  PUSHS 6
  PUSHS 7
  PUSHS 8
  PUSHS 1
  PUSH 5      ; array length
  RET
