#include "npu.inc"

# Self-checking CVT image: programs configs and descriptors, then widens an
# i8 Tile into an i32 Acc (twiden), converts the Acc in place to f32 (acvt),
# and converts a Vec32 (vcvt). Every stage stores the result back and
# compares it against an expected array (bytes with lbu, words with
# lwu/subw). Ends with ebreak: a0 = 0 pass, a0 = N identifies the failing
# check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc1: dtype = i8 (0), 2x2; tm1: row_stride = 2, col_stride = 1 (bytes)
    CFG_SETI  5, 0, 0, 0, 1
    CFG_SETI  5, 0, 1, 2, 1
    CFG_SETI  5, 0, 2, 2, 1
    CFG_SETI  5, 2, 1, 2, 1
    CFG_SETI  5, 2, 2, 1, 1
    # ac0: dtype = i32 (2), 2x2; am0: row_stride = 8, col_stride = 4 (bytes)
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 2, 0
    CFG_SETI  5, 1, 2, 2, 0
    CFG_SETI  5, 3, 1, 8, 0
    CFG_SETI  5, 3, 2, 4, 0
    # vc0/vc1: dtype = i32 (2), len = 2; vm0/vm1: stride = 4
    CFG_SETI  6, 1, 0, 2, 0
    CFG_SETI  6, 1, 1, 2, 0
    CFG_SETI  6, 3, 1, 4, 0
    CFG_SETI  6, 1, 0, 2, 1
    CFG_SETI  6, 1, 1, 2, 1
    CFG_SETI  6, 3, 1, 4, 1

    # Stage 1 - twiden.i8.i32 a0, t1 over {-1, 2, -3, 4} (i8) ->
    #   {0xFFFFFFFF, 2, 0xFFFFFFFD, 4} (i32)
    lla       x3, src_t
    CFG_REG   2, 0, 0, 0, 3, 1
    TLOAD     1
    TWIDEN    0, 1
    lla       x3, dstbuf
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf
    lla       x4, exp_wide
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

    # Stage 2 - acvt.i32.f32 a0, a0 (in place) over the widened values ->
    #   {-1.0f, 2.0f, -3.0f, 4.0f}
    ACVT      0, 0
    lla       x3, dstbuf
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf
    lla       x4, exp_cvt
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

    # Stage 3 - vcvt.i32.f32 v0, v1 over {7, -8} -> {7.0f, -8.0f}
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    VCVT      0, 1
    lla       x3, dstbuf
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_vcvt
    li        x6, 2
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

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_t:   .byte 0xFF, 2, 0xFD, 4
    .balign 4
exp_wide: .word 0xFFFFFFFF, 2, 0xFFFFFFFD, 4
exp_cvt: .word 0xBF800000, 0x40000000, 0xC0400000, 0x40800000
src_v:   .word 7, 0xFFFFFFF8
exp_vcvt: .word 0x40E00000, 0xC1000000
dstbuf:  .space 16
