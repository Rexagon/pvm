main:
  PUSHI make_array
  CALL
  PUSHI 0   ; high 4 bytes of result

sum_loop:
  PEEKI 4
  PUSHI 1
  EQI
  PUSHI end
  BRANCH

  ERORI
  ERORI
  ADDLI
  RORI
  ADDI
  SWAPI
  RORI

  PEEKI 4
  DECI
  SWAPI
  ROLI
  POPI

  PUSHI sum_loop
  JMP

end:
  SWAPI
  POPI
  SWAPI
  HLT


make_array:
  PUSHI 2147483658
  PUSHI 2147483659
  PUSHI 2147483660
  PUSHI 2147483661
  PUSHI 4      ; array length
  RET
