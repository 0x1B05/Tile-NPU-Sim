#include "npu.inc"

# Self-checking E_R4 image: programs configs and descriptors, then runs a
# Tile select (u8 mask, any nonzero lane picks A) and a Vec32 fmadd (f32,
# single rounding). Every stage stores the result back and compares it
# against an expected array (bytes with lbu, words with subw to stay immune
# to lw/li extension differences). Ends with ebreak: a0 = 0 pass, a0 = N
# identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0/tc2/tc3: dtype = i8 (0), 2x2; tc1 (mask): dtype = u8 (1), 2x2
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 2, 0
    CFG_SETI  5, 0, 0, 1, 1
    CFG_SETI  5, 0, 1, 2, 1
    CFG_SETI  5, 0, 2, 2, 1
    CFG_SETI  5, 0, 0, 0, 2
    CFG_SETI  5, 0, 1, 2, 2
    CFG_SETI  5, 0, 2, 2, 2
    CFG_SETI  5, 0, 0, 0, 3
    CFG_SETI  5, 0, 1, 2, 3
    CFG_SETI  5, 0, 2, 2, 3
    # tm0..tm3: row_stride = 2, col_stride = 1
    CFG_SETI  6, 0, 1, 2, 0
    CFG_SETI  6, 0, 2, 1, 0
    CFG_SETI  6, 0, 1, 2, 1
    CFG_SETI  6, 0, 2, 1, 1
    CFG_SETI  6, 0, 1, 2, 2
    CFG_SETI  6, 0, 2, 1, 2
    CFG_SETI  6, 0, 1, 2, 3
    CFG_SETI  6, 0, 2, 1, 3
    # vc0..vc3: dtype = f32 (4), len = 2; vm0..vm3: stride = 4
    CFG_SETI  5, 2, 0, 4, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  6, 2, 1, 4, 0
    CFG_SETI  5, 2, 0, 4, 1
    CFG_SETI  5, 2, 1, 2, 1
    CFG_SETI  6, 2, 1, 4, 1
    CFG_SETI  5, 2, 0, 4, 2
    CFG_SETI  5, 2, 1, 2, 2
    CFG_SETI  6, 2, 1, 4, 2
    CFG_SETI  5, 2, 0, 4, 3
    CFG_SETI  5, 2, 1, 2, 3
    CFG_SETI  6, 2, 1, 4, 3

    # Stage 1 - tselect t0, t1, t2, t3: M = {0xFF, 0, 0, 1} over
    #   A = {1, 2, 3, 4}, B = {10, 20, 30, 40} -> {1, 20, 30, 4}
    lla       x3, src_m
    CFG_REG   0, 1, 0, 0, 3, 1
    TLOAD     1
    lla       x3, src_a
    CFG_REG   0, 1, 0, 0, 3, 2
    TLOAD     2
    lla       x3, src_b
    CFG_REG   0, 1, 0, 0, 3, 3
    TLOAD     3
    TSELECT   0, 1, 2, 3
    lla       x3, dstbuf
    CFG_REG   0, 1, 0, 0, 3, 0
    TSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_sel
    li        x6, 4
    li        a0, 1
    FENCE_MEM
1:  lbu       x5, 0(x3)
    lbu       x7, 0(x4)
    sub       x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 1
    addi      x4, x4, 1
    addi      x6, x6, -1
    bnez      x6, 1b

    # Stage 2 - vfmadd v0, v1, v2, v3:
    #   {1.5, -2.0} * {4.0, 0.5} + {0.25, 1.0} -> {6.25, 0.0}
    lla       x3, src_fa
    CFG_REG   2, 1, 0, 0, 3, 1
    VLOAD     1
    lla       x3, src_fb
    CFG_REG   2, 1, 0, 0, 3, 2
    VLOAD     2
    lla       x3, src_fc
    CFG_REG   2, 1, 0, 0, 3, 3
    VLOAD     3
    VFMADD    0, 1, 2, 3
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_fma
    li        x6, 2
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

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_m:  .byte 0xFF, 0, 0, 1
src_a:  .byte 1, 2, 3, 4
src_b:  .byte 10, 20, 30, 40
exp_sel: .byte 1, 20, 30, 4
    .balign 4
src_fa: .word 0x3FC00000, 0xC0000000
src_fb: .word 0x40800000, 0x3F000000
src_fc: .word 0x3E800000, 0x3F800000
exp_fma: .word 0x40C80000, 0x00000000
dstbuf: .space 16
