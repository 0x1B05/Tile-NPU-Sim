# Bare-metal startup for guest C programs. Sets the stack to 0x88000000
# inside the runner's RAM, calls main, and reports its return value through
# the ebreak/a0 convention. Linked first (see link.ld) so execution enters here.

    .section .text.crt0, "ax", @progbits
    .globl _start
_start:
    li      sp, 0x88000000
    call    main
    ebreak
