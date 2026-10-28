#include "npu.inc"

# Self-checking MMA image: programs configs and descriptors, loads a
# shared 2x3 left Tile and a 2x3 right Tile in N x K order, then runs
# mma.nt.zero, mma.nt.acc (accumulating onto the first result), and the
# Tile-row vmma.nt.zero and vmma.nt.acc forms over the same left Tile. Results are
# stored back and compared word-wise. Ends with ebreak: a0 = 0 pass, a0 = N
# identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0: dtype = i8 (0), rows = 2, cols = 3; tm0: row_stride = 3, col = 1
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 3, 0
    CFG_SETI  6, 0, 1, 3, 0
    CFG_SETI  6, 0, 2, 1, 0
    # tc2: i8, 2x3; tm2: row_stride = 3, col_stride = 1
    CFG_SETI  5, 0, 0, 0, 2
    CFG_SETI  5, 0, 1, 2, 2
    CFG_SETI  5, 0, 2, 3, 2
    CFG_SETI  6, 0, 1, 3, 2
    CFG_SETI  6, 0, 2, 1, 2
    # ac2: dtype = i32 (2), 2x2; am2: row_stride = 8, col_stride = 4
    CFG_SETI  5, 1, 0, 2, 2
    CFG_SETI  5, 1, 1, 2, 2
    CFG_SETI  5, 1, 2, 2, 2
    CFG_SETI  6, 1, 1, 8, 2
    CFG_SETI  6, 1, 2, 4, 2
    # vc4: dtype = i32 (2), len = 2; vm4: stride = 4
    CFG_SETI  5, 2, 0, 2, 4
    CFG_SETI  5, 2, 1, 2, 4
    CFG_SETI  6, 2, 1, 4, 4

    # Load the operands: t0 = {{1,2,3},{4,-5,6}},
    # t2 = {{7,8,9},{10,11,12}}.
    lla       x3, src_a
    CFG_REG   0, 1, 0, 0, 3, 0
    TLOAD     0
    lla       x3, src_b_nt
    CFG_REG   0, 1, 0, 0, 3, 2
    TLOAD     2

    # Stage 1 - mma.nt.zero.i8.i32 a2, t0, t2 -> {{50, 68}, {42, 57}}
    MMA_NT_ZERO 2, 0, 2
    lla       x3, dstbuf
    CFG_REG   1, 1, 0, 0, 3, 2
    ASTORE    2
    lla       x3, dstbuf
    lla       x4, exp_mma1
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

    # Stage 2 - mma.nt.acc.i8.i32 a2, t0, t2 accumulates
    # the same product onto stage 1 -> {{100, 136}, {84, 114}}
    MMA_NT_ACC 2, 0, 2
    lla       x3, dstbuf
    CFG_REG   1, 1, 0, 0, 3, 2
    ASTORE    2
    lla       x3, dstbuf
    lla       x4, exp_mma2
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

    # Stage 3 - vmma.nt.zero.i8.i32 v4, t0[1], t2 reads {4, -5, 6}
    # and writes {42, 57}.
    VMMA_NT_ZERO 4, 0, 1, 2
    lla       x3, dstbuf2
    CFG_REG   2, 1, 0, 0, 3, 4
    VSTORE    4
    lla       x3, dstbuf2
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 42
    li        a0, 3
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 57
    bne       x4, x5, fail

    # Stage 4 - vmma.nt.acc.i8.i32 v4, t0[0], t2 adds
    # {50, 68} from row 0 -> {92, 125}.
    VMMA_NT_ACC 4, 0, 0, 2
    lla       x3, dstbuf2
    CFG_REG   2, 1, 0, 0, 3, 4
    VSTORE    4
    lla       x3, dstbuf2
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 92
    li        a0, 4
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 125
    bne       x4, x5, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_a:    .byte 1, 2, 3, 4, 0xFB, 6
    .balign 4
src_b_nt: .byte 7, 8, 9, 10, 11, 12
    .balign 4
exp_mma1: .word 50, 68, 42, 57
exp_mma2: .word 100, 136, 84, 114
dstbuf:   .space 16
dstbuf2:  .space 8
