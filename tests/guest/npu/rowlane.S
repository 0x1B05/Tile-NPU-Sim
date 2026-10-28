#include "npu.inc"

# Self-checking ROW_MOVE/LANE_MOVE image: programs configs and descriptors,
# then exercises Acc <-> Vec32 row insert/extract and lane
# extract/insert/broadcast (immediate and scalar index forms).
# Every stage stores results back to memory and checks them with scalar
# loads. Ends with ebreak: a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # ac0: dtype = i32 (2), rows = 2, cols = 2
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 2, 0
    CFG_SETI  5, 1, 2, 2, 0
    # vc0/vc1: dtype = i32 (2), len = 2
    CFG_SETI  5, 2, 0, 2, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 0, 2, 1
    CFG_SETI  5, 2, 1, 2, 1
    # am0: base = srcbuf32, row_stride = 8, col_stride = 4
    lla       x3, srcbuf32
    CFG_REG   1, 1, 0, 0, 3, 0
    CFG_SETI  6, 1, 1, 8, 0
    CFG_SETI  6, 1, 2, 4, 0
    # vm0/vm1: base = dstbuf32, stride = 4
    lla       x3, dstbuf32
    CFG_REG   2, 1, 0, 0, 3, 0
    CFG_SETI  6, 2, 1, 4, 0
    CFG_REG   2, 1, 0, 0, 3, 1
    CFG_SETI  6, 2, 1, 4, 1

    # aextract.row v0, a0[x9]: a0 = {1, 2, 3, 4} (2x2), x9 = 1 -> v0 = {3, 4}.
    ALOAD     0
    li        x9, 1
    ROW_MOVE  1, 1, 9, 0, 0   # aextract.row v0, a0[x9]
    VSTORE    0
    lla       x3, dstbuf32
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 3
    li        a0, 1
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 4
    bne       x4, x5, fail

    # ainsert.row a0[0], v0: row 0 becomes {3, 4} as well.
    AINSROW   0, 0, 0
    lla       x3, dstbuf32
    CFG_REG   1, 1, 0, 0, 3, 0   # am0 base = dstbuf32 (strides still 8/4)
    ASTORE    0
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 3
    li        a0, 2
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 4
    bne       x4, x5, fail

    # vextract x5, v0[0] = 3; vinsert v0[x9], x6 with x9 = 1, x6 = -1 writes
    # the low 32 bits (0xFFFFFFFF) into lane 1; vbroadcast v1, v0[0] = {3, 3}.
    VEXTRACT  5, 0, 0
    li        x7, 3
    li        a0, 3
    bne       x5, x7, fail
    li        x6, -1
    li        x9, 1
    LANE_MOVE 1, 1, 9, 6, 0   # vinsert v0[x9], x6
    VSTORE    0
    lla       x3, dstbuf32
    FENCE_MEM
    lwu       x4, 4(x3)
    li        x5, 0xFFFFFFFF
    bne       x4, x5, fail
    VBROADCAST 1, 0, 0
    VSTORE    1
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 3
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    bne       x4, x5, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
srcbuf32: .word 1, 2, 3, 4
dstbuf32: .space 16
