#include "npu.inc"

# Self-checking INIT_MOVE image: programs configs and descriptors, then
# exercises fill/fillx/copy on the tile, acc and vec8 domains and ttranspose
# (including in-place). Every stage stores results back to memory and checks
# them with scalar loads; the transpose config update is read back with
# cfg.get. Ends with ebreak: a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0/tc1: dtype = i8 (0), rows = 2, cols = 3; tc2: i8 (transpose output)
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 3, 0
    CFG_SETI  5, 0, 0, 0, 1
    CFG_SETI  5, 0, 1, 2, 1
    CFG_SETI  5, 0, 2, 3, 1
    CFG_SETI  5, 0, 0, 0, 2
    # ac0: dtype = i32 (2), rows = 1, cols = 2
    CFG_SETI  5, 1, 0, 2, 0
    CFG_SETI  5, 1, 1, 1, 0
    CFG_SETI  5, 1, 2, 2, 0
    # bc0: dtype = u8 (1), len = 4
    CFG_SETI  6, 0, 0, 1, 0
    CFG_SETI  6, 0, 1, 4, 0
    # tm0: base = dstbuf, row_stride = 3, col_stride = 1
    lla       x3, dstbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    CFG_SETI  5, 2, 1, 3, 0
    CFG_SETI  5, 2, 2, 1, 0

    # tfill t0, -3: every effective lane takes the low 8 bits (0xFD).
    TFILL     0, -3
    TSTORE    0
    lla       x3, dstbuf
    lwu       x4, 0(x3)
    li        x5, 0xFDFDFDFD
    li        a0, 1
    bne       x4, x5, fail
    lhu       x4, 4(x3)
    li        x5, 0xFDFD
    bne       x4, x5, fail

    # tfillx t1, x6 (0x7F); tcopy t0, t1 moves it across registers.
    li        x6, 0x7F
    TFILLX    1, 6
    TCOPY     0, 1
    TSTORE    0
    lla       x3, dstbuf
    lwu       x4, 0(x3)
    li        x5, 0x7F7F7F7F
    li        a0, 2
    bne       x4, x5, fail
    lhu       x4, 4(x3)
    li        x5, 0x7F7F
    bne       x4, x5, fail

    # afillx a0, x6: 32-bit lanes keep the low 32 bits of the scalar value.
    li        x6, 0xCAFEF00D
    AFILLX    0, 6
    lla       x3, dstbuf2
    CFG_REG   3, 0, 0, 0, 3, 0   # am0 base = dstbuf2
    CFG_SETI  5, 3, 1, 8, 0      # am0 row_stride = 8
    CFG_SETI  5, 3, 2, 4, 0      # am0 col_stride = 4
    ASTORE    0
    lwu       x4, 0(x3)
    li        a0, 3
    bne       x4, x6, fail
    lwu       x4, 4(x3)
    bne       x4, x6, fail

    # bfill b0, 0x5A on the vec8 domain.
    BFILL     0, 0x5A
    lla       x3, dstbuf3
    CFG_REG   2, 1, 0, 0, 3, 0   # bm0 base = dstbuf3
    CFG_SETI  6, 2, 1, 1, 0      # bm0 stride = 1
    BSTORE    0
    lwu       x4, 0(x3)
    li        x5, 0x5A5A5A5A
    li        a0, 4
    bne       x4, x5, fail

    # tload t0 = {1..6} (2x3), then ttranspose t2, t0 -> 3x2:
    #   1 2 3        1 4
    #   4 5 6  -->   2 5
    #                3 6
    lla       x3, srcbuf
    CFG_REG   2, 0, 0, 0, 3, 0   # tm0 base = srcbuf (strides still 3/1)
    TLOAD     0
    TTRANSPOSE 2, 0
    lla       x3, dstbuf4
    CFG_REG   2, 0, 0, 0, 3, 2   # tm2 base = dstbuf4
    CFG_SETI  5, 2, 1, 2, 2      # tm2 row_stride = 2
    CFG_SETI  5, 2, 2, 1, 2      # tm2 col_stride = 1
    TSTORE    2
    lwu       x4, 0(x3)
    li        x5, 0x05020401     # bytes {1, 4, 2, 5}
    li        a0, 5
    bne       x4, x5, fail
    lhu       x4, 4(x3)
    li        x5, 0x0603         # bytes {3, 6}
    bne       x4, x5, fail
    CFG_REG   0, 0, 2, 1, 2, 5   # cfg.get x5, tc2, rows
    li        x7, 3
    bne       x5, x7, fail
    CFG_REG   0, 0, 2, 2, 2, 5   # cfg.get x5, tc2, cols
    li        x7, 2
    bne       x5, x7, fail

    # In-place: ttranspose t0, t0 turns the 2x3 {1..6} into the same 3x2.
    TTRANSPOSE 0, 0
    lla       x3, dstbuf4
    CFG_REG   2, 0, 0, 0, 3, 0   # tm0 base = dstbuf4
    CFG_SETI  5, 2, 1, 2, 0      # tm0 row_stride = 2
    TSTORE    0
    lwu       x4, 0(x3)
    li        x5, 0x05020401
    li        a0, 6
    bne       x4, x5, fail
    lhu       x4, 4(x3)
    li        x5, 0x0603
    bne       x4, x5, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
srcbuf:  .byte 1, 2, 3, 4, 5, 6
    .balign 4
dstbuf:  .space 8
dstbuf2: .space 8
dstbuf3: .space 4
dstbuf4: .space 8
