#include "npu.inc"

# Self-checking E_BCAST image: programs configs and descriptors, then runs an
# Acc byrow broadcast add (i32) and an Acc bycol broadcast mul (i32).
# Every stage stores the result back and compares it against an expected
# array (words with subw to stay immune to lw/li extension differences).
# Ends with ebreak: a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # ac0/ac1: dtype = i32 (2), rows = 2, cols = 2; am0/am1: row = 8, col = 4
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 2, 0
    CFG_SETI  5, 1, 2, 2, 0
    CFG_SETI  5, 3, 1, 8, 0
    CFG_SETI  5, 3, 2, 4, 0
    CFG_SETI  5, 1, 0, 2, 1
    CFG_SETI  5, 1, 1, 2, 1
    CFG_SETI  5, 1, 2, 2, 1
    CFG_SETI  5, 3, 1, 8, 1
    CFG_SETI  5, 3, 2, 4, 1
    # vc0: dtype = i32 (2), len = 2; vm0: stride = 4
    CFG_SETI  6, 1, 0, 2, 0
    CFG_SETI  6, 1, 1, 2, 0
    CFG_SETI  6, 3, 1, 4, 0

    # Stage 1 - aaddv.byrow a0, a1, v0 (i32): row i takes v0[i]
    #   {{6, 7}, {8, 9}} with v0 = {2, 3} -> {{8, 9}, {11, 12}}
    lla       x3, src_a
    CFG_REG   3, 0, 0, 0, 3, 1
    ALOAD     1
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 0
    VLOAD     0
    AADDV_BYROW 0, 1, 0
    lla       x3, dstbuf
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf
    lla       x4, exp_a_add
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

    # Stage 2 - amulv.bycol a0, a1, v0 (i32): column j takes v0[j]
    #   {{6, 7}, {8, 9}} with v0 = {2, 3} -> {{12, 21}, {16, 27}}
    lla       x3, src_a
    CFG_REG   3, 0, 0, 0, 3, 1
    ALOAD     1
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 0
    VLOAD     0
    AMULV_BYCOL 0, 1, 0
    lla       x3, dstbuf
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf
    lla       x4, exp_a
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

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_a:  .word 6, 7, 8, 9
exp_a_add: .word 8, 9, 11, 12
exp_a:  .word 12, 21, 16, 27
src_v:  .word 2, 3
dstbuf: .space 16
