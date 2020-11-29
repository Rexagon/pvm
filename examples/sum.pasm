main:
  PUSHI make_array
  CALL

sum_loop:
  DUP
  PUSH 1
  EQ
  PUSHI end
  BRANCH

  DEC
  PUSHI sum_loop
  JMP

end:
  HLT


make_array:
  PUSHI 123
  PUSHI 5345
  PUSHI 345
  PUSHI 1345
  PUSH 4      ; array length
  RET
