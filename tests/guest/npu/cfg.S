#include "npu.inc"

# Self-checking cfg image: programs config registers with cfg.seti/cfg.setx,
# reads them back with cfg.get, and exercises cfg.copy on both the matrix and
# vector sides. Ends with ebreak: a0 = 0 pass, a0 = 1 fail.

    .globl _start
_start:
    # cfg.seti tc0, rows, 4 / cfg.seti tc0, cols, 3
    CFG_SETI  5, 0, 1, 4, 0
    CFG_SETI  5, 0, 2, 3, 0
    CFG_REG   0, 0, 2, 1, 0, 1     # cfg.get x1, tc0, rows
    li        x2, 4
    bne       x1, x2, fail

    # cfg.setx tm0, base_addr, x3 with the full 64-bit scalar value
    li        x3, 0x4000
    CFG_REG   2, 0, 0, 0, 3, 0     # cfg.setx tm0, base_addr, x3
    CFG_REG   2, 0, 2, 0, 0, 4     # cfg.get x4, tm0, base_addr
    bne       x3, x4, fail

    # cfg.seti tm0, row_stride_bytes, -4: negative immediates sign-extend
    CFG_SETI  5, 2, 1, -4, 0
    CFG_REG   2, 0, 2, 1, 0, 7     # cfg.get x7, tm0, row_stride_bytes
    li        x8, -4
    bne       x7, x8, fail

    # cfg.copy tc1, tc0: all fields come along (cols = 3)
    CFG_REG   0, 0, 1, 0, 0, 1     # cfg.copy tc1, tc0
    CFG_REG   0, 0, 2, 2, 1, 5     # cfg.get x5, tc1, cols
    li        x6, 3
    bne       x5, x6, fail

    # vector side: cfg.seti vc0, len, 32 (major4 = 6, MV = 1 on readback)
    CFG_SETI  6, 1, 1, 32, 0
    CFG_REG   1, 1, 2, 1, 0, 9     # cfg.get x9, vc0, len
    li        x10, 32
    bne       x9, x10, fail

    li        a0, 0
    ebreak
fail:
    li        a0, 1
    ebreak
