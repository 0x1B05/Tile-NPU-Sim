#include "npu.inc"

# Self-checking ROW_MOVE/LANE_MOVE image: programs configs and descriptors,
# then exercises row insert/extract between Tile <-> Vec8 and Acc <-> Vec32,
# plus lane extract/insert/broadcast (immediate and scalar index forms).
# Every stage stores results back to memory and checks them with scalar
# loads. Ends with ebreak: a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0: dtype = i8 (0), rows = 2, cols = 4
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 4, 0
    # bc0/bc1: dtype = i8 (0), len = 4
    CFG_SETI  6, 0, 0, 0, 0
    CFG_SETI  6, 0, 1, 4, 0
    CFG_SETI  6, 0, 0, 0, 1
    CFG_SETI  6, 0, 1, 4, 1
    # ac0: dtype = i32 (2), rows = 2, cols = 2
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 2, 0
    CFG_SETI  5, 1, 2, 2, 0
    # vc0/vc1: dtype = i32 (2), len = 2
    CFG_SETI  6, 1, 0, 2, 0
    CFG_SETI  6, 1, 1, 2, 0
    CFG_SETI  6, 1, 0, 2, 1
    CFG_SETI  6, 1, 1, 2, 1
    # tm0: base = srcbuf, row_stride = 4, col_stride = 1
    lla       x3, srcbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    CFG_SETI  5, 2, 1, 4, 0
    CFG_SETI  5, 2, 2, 1, 0
    # am0: base = srcbuf32, row_stride = 8, col_stride = 4
    lla       x3, srcbuf32
    CFG_REG   3, 0, 0, 0, 3, 0
    CFG_SETI  5, 3, 1, 8, 0
    CFG_SETI  5, 3, 2, 4, 0
    # bm0/bm1: base = dstbuf, stride = 1
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 0
    CFG_SETI  6, 2, 1, 1, 0
    CFG_REG   2, 1, 0, 0, 3, 1
    CFG_SETI  6, 2, 1, 1, 1
    # vm0/vm1: base = dstbuf32, stride = 4
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    CFG_SETI  6, 3, 1, 4, 0
    CFG_REG   3, 1, 0, 0, 3, 1
    CFG_SETI  6, 3, 1, 4, 1

    # textract.row b0, t0[1]: t0 = {1..8} (2x4), so b0 = {5, 6, 7, 8}.
    TLOAD     0
    TEXROW    0, 0, 1
    BSTORE    0
    lla       x3, dstbuf
    lwu       x4, 0(x3)
    li        x5, 0x08070605
    li        a0, 1
    bne       x4, x5, fail

    # tinsert.row t0[0], b0: row 0 becomes {5, 6, 7, 8} as well.
    TINSROW   0, 0, 0
    lla       x3, dstbuf2
    CFG_REG   2, 0, 0, 0, 3, 0   # tm0 base = dstbuf2 (strides still 4/1)
    TSTORE    0
    lwu       x4, 0(x3)
    li        x5, 0x08070605
    li        a0, 2
    bne       x4, x5, fail

    # aextract.row v0, a0[x9]: a0 = {1, 2, 3, 4} (2x2), x9 = 1 -> v0 = {3, 4}.
    ALOAD     0
    li        x9, 1
    ROW_MOVE  1, 1, 1, 9, 0, 0   # aextract.row v0, a0[x9]
    VSTORE    0
    lla       x3, dstbuf32
    lwu       x4, 0(x3)
    li        x5, 3
    li        a0, 5
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 4
    bne       x4, x5, fail

    # ainsert.row a0[0], v0: row 0 becomes {3, 4} as well.
    AINSROW   0, 0, 0
    lla       x3, dstbuf32
    CFG_REG   3, 0, 0, 0, 3, 0   # am0 base = dstbuf32 (strides still 8/4)
    ASTORE    0
    lwu       x4, 0(x3)
    li        x5, 3
    li        a0, 6
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 4
    bne       x4, x5, fail

    # vextract x5, v0[0] = 3; vinsert v0[x9], x6 with x9 = 1, x6 = -1 writes
    # the low 32 bits (0xFFFFFFFF) into lane 1; vbroadcast v1, v0[0] = {3, 3}.
    VEXTRACT  5, 0, 0
    li        x7, 3
    li        a0, 7
    bne       x5, x7, fail
    li        x6, -1
    li        x9, 1
    LANE_MOVE 1, 1, 1, 9, 6, 0   # vinsert v0[x9], x6
    VSTORE    0
    lla       x3, dstbuf32
    lwu       x4, 4(x3)
    li        x5, 0xFFFFFFFF
    bne       x4, x5, fail
    VBROADCAST 1, 0, 0
    VSTORE    1
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
srcbuf:   .byte 1, 2, 3, 4, 5, 6, 7, 8
    .balign 4
srcbuf32: .word 1, 2, 3, 4
dstbuf:   .space 4
dstbuf2:  .space 8
dstbuf32: .space 16
