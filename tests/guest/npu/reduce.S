#include "npu.inc"

# Self-checking reduce image: programs configs and descriptors, then runs an
# areduce.rows.sum (f32) and a vreduce.argmax (NaN candidate wins). The
# areduce result is stored back and compared word-wise (lwu/subw); the
# argmax scalar outputs are compared in registers. Ends with ebreak: a0 = 0
# pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # ac1: dtype = f32 (4), 2x2; am1: row_stride = 8, col_stride = 4 (bytes)
    CFG_SETI  5, 1, 0, 4, 1
    CFG_SETI  5, 1, 1, 2, 1
    CFG_SETI  5, 1, 2, 2, 1
    CFG_SETI  5, 3, 1, 8, 1
    CFG_SETI  5, 3, 2, 4, 1
    # vc0: dtype = f32, len = 2; vm0: stride = 4
    CFG_SETI  6, 1, 0, 4, 0
    CFG_SETI  6, 1, 1, 2, 0
    CFG_SETI  6, 3, 1, 4, 0
    # vc1: dtype = f32, len = 3; vm1: stride = 4
    CFG_SETI  6, 1, 0, 4, 1
    CFG_SETI  6, 1, 1, 3, 1
    CFG_SETI  6, 3, 1, 4, 1

    # Stage 1 - areduce.rows.sum v0, a1 over
    #   {1.5, -2.5}, {3.0, 4.5} -> {-1.0, 7.5}
    lla       x3, src_m
    CFG_REG   3, 0, 0, 0, 3, 1
    ALOAD     1
    AREDUCE   0, 0, 1
    lla       x3, dstbuf
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_sum
    li        x6, 2
    li        a0, 1
1:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 1b

    # Stage 2 - vreduce.argmax x5, x6, v1 over {1.0, 7.0, NaN} ->
    #   x5 = 2, x6 = 0x7FC00000 (NaN candidate wins)
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    VREDUCE_ARGMAX 5, 6, 1
    li        a0, 2
    li        x7, 2
    bne       x5, x7, fail
    li        x7, 0x7FC00000
    bne       x6, x7, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_m:   .word 0x3FC00000, 0xC0200000, 0x40400000, 0x40900000
exp_sum: .word 0xBF800000, 0x40F00000
src_v:   .word 0x3F800000, 0x40E00000, 0x7FC00000
dstbuf:  .space 8
