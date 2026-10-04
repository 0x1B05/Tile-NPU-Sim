#include "npu.inc"

# Self-checking M_BLK image: programs configs and descriptors with
# cfg.seti/cfg.setx, then exercises block load/store, a transposed tload,
# row load/store (immediate and scalar row forms), and a gather through an
# index vector loaded from memory. Every stage stores results back to memory
# and checks them with scalar loads; the descriptor auto-increment is read
# back with cfg.get. Ends with ebreak: a0 = 0 pass, a0 = N identifies the
# failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0..tc2: dtype = i8 (0), rows = 2, cols = 2
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 2, 0
    CFG_SETI  5, 0, 0, 0, 1
    CFG_SETI  5, 0, 1, 2, 1
    CFG_SETI  5, 0, 2, 2, 1
    CFG_SETI  5, 0, 0, 0, 2
    CFG_SETI  5, 0, 1, 2, 2
    CFG_SETI  5, 0, 2, 2, 2
    # vc2: dtype = i32 (2), len = 2 (gather index vector)
    CFG_SETI  6, 1, 0, 2, 2
    CFG_SETI  6, 1, 1, 2, 2
    # tm0: base = srcbuf, row_stride = 2, col_stride = 1
    lla       x3, srcbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 2, 1, 0
    # tm1: base = srcbuf + 2 (row 1 of the source), col_stride = 1
    lla       x3, srcbuf + 2
    CFG_REG   2, 0, 0, 0, 3, 1
    CFG_SETI  5, 2, 2, 1, 1
    # tm2: base = gthbuf, row_stride = 1, col_stride = 2
    lla       x3, gthbuf
    CFG_REG   2, 0, 0, 0, 3, 2
    CFG_SETI  5, 2, 1, 1, 2
    CFG_SETI  5, 2, 2, 2, 2
    # vm2: base = idxbuf, stride = 4
    lla       x3, idxbuf
    CFG_REG   3, 1, 0, 0, 3, 2
    CFG_SETI  6, 3, 1, 4, 2

    # Block round trip: tload t0 from srcbuf, tstore t0 to dstbuf.
    TLOAD     0
    lla       x3, dstbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    TSTORE    0
    lw        x4, 0(x3)
    li        x5, 0x04030201  # bytes {1, 2, 3, 4}
    li        a0, 1
    bne       x4, x5, fail

    # Transposed tload: transform = 1 swaps the address indices.
    CFG_SETI  5, 2, 3, 1, 0
    lla       x3, srcbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    TLOAD     0
    CFG_SETI  5, 2, 3, 0, 0
    lla       x3, dstbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    TSTORE    0
    lw        x4, 0(x3)
    li        x5, 0x04020301  # bytes {1, 3, 2, 4}
    li        a0, 2
    bne       x4, x5, fail

    # Row access: tloadr t1, #1 loads the memory row into register row 1;
    # tstorer t1, x9 (x9 = 1) stores register row 1 back to rowbuf.
    TLOADR    1, 1
    lla       x3, rowbuf
    CFG_REG   2, 0, 0, 0, 3, 1
    li        x9, 1
    TSTORERX  1, 9
    lhu       x4, 0(x3)
    li        x5, 0x0403  # bytes {3, 4}
    li        a0, 3
    bne       x4, x5, fail

    # Gather: vload v2 = {1, 0}, then tload.gather t2, v2 picks rows
    # {base + 2} = {30, 40} and {base + 0} = {10, 20} from gthbuf.
    VLOAD     2
    TGATHER   2, 2
    lla       x3, gthdst
    CFG_REG   2, 0, 0, 0, 3, 2
    CFG_SETI  5, 2, 1, 2, 2   # store back with row_stride = 2, col_stride = 1
    CFG_SETI  5, 2, 2, 1, 2
    TSTORE    2
    lw        x4, 0(x3)
    li        x5, 0x140a281e  # bytes {30, 40, 10, 20}
    li        a0, 4
    bne       x4, x5, fail

    # Auto-increment along rows: base advances by rows * row_stride = 4.
    lla       x3, srcbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 4, 1, 0
    TLOAD     0
    CFG_REG   2, 0, 2, 0, 0, 5  # cfg.get x5, tm0, base_addr
    lla       x6, srcbuf + 4
    li        a0, 5
    bne       x5, x6, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
srcbuf: .byte 1, 2, 3, 4
idxbuf: .word 1, 0
gthbuf: .byte 10, 20, 30, 40
dstbuf: .space 4
rowbuf: .space 2
    .balign 4
gthdst: .space 4
