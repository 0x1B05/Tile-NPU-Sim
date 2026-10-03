# Bare-metal startup for guest C programs. Sets the stack to the top of the
# runner's 128 MiB RAM, calls main, and reports its return value through the
# ebreak/a0 convention. Linked first (see link.ld) so execution enters here.

    .section .text.crt0, "ax", @progbits
    .globl _start
_start:
    li      sp, 0x88000000
    call    main
    ebreak
