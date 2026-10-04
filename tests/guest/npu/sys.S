#include "npu.inc"

# Self-checking SYS image: fills a Tile, interleaves the three fences with a
# store and a word comparison, then terminates with kernel.end (a0 = 0 after
# the checks pass, a0 = N identifies the failing check). The host driver
# asserts the kernel_end stop status.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0: dtype = i8 (0), rows = 2, cols = 2; tm0: row_stride = 2, col = 1
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 2, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 2, 1, 0

    # tfill t0, 0x5A; the fences must not disturb the surrounding operations.
    TFILL     0, 0x5A
    FENCE_MEM
    lla       x3, dstbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    TSTORE    0
    FENCE_SA
    FENCE_ALL
    lla       x3, dstbuf
    lwu       x4, 0(x3)
    li        x5, 0x5A5A5A5A
    li        a0, 1
    bne       x4, x5, fail

    li        a0, 0
    KERNEL_END
fail:
    ebreak

    .section .data
    .balign 4
dstbuf:  .space 4
