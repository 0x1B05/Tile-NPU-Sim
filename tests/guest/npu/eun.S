#include "npu.inc"

# Self-checking E_UNARY image: programs configs and descriptors, then runs
# not/abs/neg and the approximate f32 functions in place over vectors loaded
# from memory. Every stage stores the result back and compares it against an
# expected array word by word (subw keeps the comparison in the low 32 bits,
# immune to lw/li extension differences). Ends with ebreak: a0 = 0 pass,
# a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # vc0: dtype = f32 (4), len = 4; vm0: stride = 4
    CFG_SETI  6, 1, 0, 4, 0
    CFG_SETI  6, 1, 1, 4, 0
    CFG_SETI  6, 3, 1, 4, 0
    # tc0: dtype = i8 (0), rows = 1, cols = 4; tm0: row_stride = 4, col = 1
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 1, 0
    CFG_SETI  5, 0, 2, 4, 0
    CFG_SETI  5, 2, 1, 4, 0
    CFG_SETI  5, 2, 2, 1, 0
    # ac0: dtype = i32 (2), rows = 1, cols = 2; am0: row_stride = 8, col = 4
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 1, 0
    CFG_SETI  5, 1, 2, 2, 0
    CFG_SETI  5, 3, 1, 8, 0
    CFG_SETI  5, 3, 2, 4, 0

    # vexp2.approx: {3, -inf, +0, 1} -> {8, +0, 1, 2}
    lla       x3, src_f1
    CFG_REG   3, 1, 0, 0, 3, 0
    VLOAD     0
    VEXP2     0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_f1
    li        x6, 4
    li        a0, 1
    FENCE_MEM
1:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 1b

    # vrcp.approx: {4, -2, +inf, -0} -> {0.25, -0.5, +0, -inf}
    lla       x3, src_f2
    CFG_REG   3, 1, 0, 0, 3, 0
    VLOAD     0
    VRCP      0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_f2
    li        x6, 4
    li        a0, 2
    FENCE_MEM
2:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 2b

    # vrsqrt.approx: {0.25, 4, +inf, +0} -> {2, 0.5, +0, +inf}
    lla       x3, src_f3
    CFG_REG   3, 1, 0, 0, 3, 0
    VLOAD     0
    VRSQRT    0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_f3
    li        x6, 4
    li        a0, 3
    FENCE_MEM
3:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 3b

    # tnot: {0x0F, 0xF0, 0x55, 0xAA} -> {0xF0, 0x0F, 0xAA, 0x55}
    lla       x3, src_b
    CFG_REG   2, 0, 0, 0, 3, 0
    TLOAD     0
    TNOT      0, 0
    lla       x3, dstbuf32
    CFG_REG   2, 0, 0, 0, 3, 0
    TSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_b
    li        x6, 4
    li        a0, 5
    FENCE_MEM
5:  lbu       x5, 0(x3)
    lbu       x7, 0(x4)
    sub       x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 1
    addi      x4, x4, 1
    addi      x6, x6, -1
    bnez      x6, 5b

    # aneg: {5, -5} -> {-5, 5}
    lla       x3, src_n
    CFG_REG   3, 0, 0, 0, 3, 0
    ALOAD     0
    ANEG      0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_n
    li        x6, 2
    li        a0, 6
    FENCE_MEM
6:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 6b

    # aabs: {-7, INT32_MIN} -> {7, INT32_MIN}
    lla       x3, src_a
    CFG_REG   3, 0, 0, 0, 3, 0
    ALOAD     0
    AABS      0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_a
    li        x6, 2
    li        a0, 7
    FENCE_MEM
7:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 7b

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_f1: .word 0x40400000, 0xFF800000, 0x00000000, 0x3F800000
exp_f1: .word 0x41000000, 0x00000000, 0x3F800000, 0x40000000
src_f2: .word 0x40800000, 0xC0000000, 0x7F800000, 0x80000000
exp_f2: .word 0x3E800000, 0xBF000000, 0x00000000, 0xFF800000
src_f3: .word 0x3E800000, 0x40800000, 0x7F800000, 0x00000000
exp_f3: .word 0x40000000, 0x3F000000, 0x00000000, 0x7F800000
src_b:  .byte 0x0F, 0xF0, 0x55, 0xAA
exp_b:  .byte 0xF0, 0x0F, 0xAA, 0x55
    .balign 4
src_n:  .word 5, 0xFFFFFFFB
exp_n:  .word 0xFFFFFFFB, 5
src_a:  .word 0xFFFFFFF9, 0x80000000
exp_a:  .word 7, 0x80000000
dstbuf32: .space 16
