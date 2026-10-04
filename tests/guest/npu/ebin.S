#include "npu.inc"

# Self-configuring E_BIN image: programs tc0..tc4 as 2x2 i8 with cfg.seti,
# then runs tsub / tmax / taddx over host-poked tile data. Ends with ebreak;
# the host driver asserts the destination tiles.

    .globl _start
_start:
    # tc0..tc4: dtype = i8 (0), rows = 2, cols = 2
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 2, 0
    CFG_SETI  5, 0, 0, 0, 1
    CFG_SETI  5, 0, 1, 2, 1
    CFG_SETI  5, 0, 2, 2, 1
    CFG_SETI  5, 0, 0, 0, 2
    CFG_SETI  5, 0, 1, 2, 2
    CFG_SETI  5, 0, 2, 2, 2
    CFG_SETI  5, 0, 0, 0, 3
    CFG_SETI  5, 0, 1, 2, 3
    CFG_SETI  5, 0, 2, 2, 3
    CFG_SETI  5, 0, 0, 0, 4
    CFG_SETI  5, 0, 1, 2, 4
    CFG_SETI  5, 0, 2, 2, 4

    li        x2, 5
    TSUB      0, 1, 2     # tsub t0, t1, t2 (wrap)
    TMAX      3, 1, 2     # tmax t3, t1, t2
    TADDX     4, 1, 2     # taddx t4, t1, x2
    ebreak
