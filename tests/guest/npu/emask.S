#include "npu.inc"

# Self-checking E_MASK image: programs configs and descriptors, then runs an
# Acc tril mask (i32, xDelta = 0, fill5 = -1) and a Vec32 tail mask (i32,
# xLen = 3, fill5 = +1). Every stage stores the result back and compares it
# against an expected array (words with lwu/subw to stay immune to lw/li
# extension differences). Ends with ebreak: a0 = 0 pass, a0 = N identifies
# the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # ac0/ac1: dtype = i32 (2), 3x3
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 3, 0
    CFG_SETI  5, 1, 2, 3, 0
    CFG_SETI  5, 1, 0, 2, 1
    CFG_SETI  5, 1, 1, 3, 1
    CFG_SETI  5, 1, 2, 3, 1
    # am0/am1: row_stride = 12, col_stride = 4 (strides are in bytes)
    CFG_SETI  5, 3, 1, 12, 0
    CFG_SETI  5, 3, 2, 4, 0
    CFG_SETI  5, 3, 1, 12, 1
    CFG_SETI  5, 3, 2, 4, 1
    # vc0/vc1: dtype = i32 (2), len = 4; vm0/vm1: stride = 4
    CFG_SETI  6, 1, 0, 2, 0
    CFG_SETI  6, 1, 1, 4, 0
    CFG_SETI  6, 3, 1, 4, 0
    CFG_SETI  6, 1, 0, 2, 1
    CFG_SETI  6, 1, 1, 4, 1
    CFG_SETI  6, 3, 1, 4, 1

    # Stage 1 - amask.tril a0, a1, x2, -1 with xDelta = 0 over
    #   1..9 -> {1, -1, -1}, {4, 5, -1}, {7, 8, 9}
    lla       x3, src_m
    CFG_REG   3, 0, 0, 0, 3, 1
    ALOAD     1
    li        x2, 0
    AMASK_TRIL 0, 1, 2, 2
    lla       x3, dstbuf
    CFG_REG   3, 0, 0, 0, 3, 0
    ASTORE    0
    lla       x3, dstbuf
    lla       x4, exp_tril
    li        x6, 9
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

    # Stage 2 - vmask.tail v0, v1, x2, +1 with xLen = 3 over
    #   {10, 20, 30, 40} -> {10, 20, 30, 1}
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    li        x2, 3
    VMASK_TAIL 0, 1, 2, 1
    lla       x3, dstbuf
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_vtail
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
src_m:   .word 1, 2, 3, 4, 5, 6, 7, 8, 9
exp_tril: .word 1, 0xFFFFFFFF, 0xFFFFFFFF, 4, 5, 0xFFFFFFFF, 7, 8, 9
src_v:   .word 10, 20, 30, 40
exp_vtail: .word 10, 20, 30, 1
dstbuf:  .space 36
