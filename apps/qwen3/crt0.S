# Bare-metal startup for the Qwen3 app. Zeroes .bss, sets the stack to the
# app's reserved top of RAM, calls main, and reports its return value through
# the semihosting exit convention (a7 = 93).

    .section .text.crt0, "ax", @progbits
    .globl _start
    # STACK_TOP from qwen3.h (assembler sources cannot include the C header)
    .equ    STACK_TOP, 0xC0000000
_start:
    la      t0, __bss_start
    la      t1, __bss_end
1:  bgeu    t0, t1, 2f
    sd      zero, 0(t0)
    addi    t0, t0, 8
    j       1b
2:  li      sp, STACK_TOP
    call    main
    li      a7, 93
    ecall
