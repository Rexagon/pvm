s1:           ; length register
  PUSHI 0

s2:           ; low sum bits
  PUSHI 0
s3:           ; high sum bits
  PUSHI 0

main:
  PUSHI make_array
  CALL
  POP s1      ; save array length

sum_loop:
  PUSH s1
  DEC
  PUSHI end
  BRANCH      ; check all elements are processed
  POP s1

  ADDL        ; sum numbers

  PUSH s3     ; sum with low result bits
  ADD
  POP s3

  PUSH s2     ; sum with high result bits
  ADDC
  POP s2

  PUSHI sum_loop
  JMP

end:
  PUSH s2
  PUSH s3
  HLT


make_array:
  PUSHI 123
  PUSHI 5345
  PUSHI 345
  PUSHI 1345
  PUSHI 4      ; array length
  RET
