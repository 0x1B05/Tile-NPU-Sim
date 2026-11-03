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
    # vc6/vc7: dtype = f32 (4), len = 2
    CFG_SETI  5, 2, 0, 4, 6
    CFG_SETI  5, 2, 1, 2, 6
    CFG_SETI  5, 2, 0, 4, 7
    CFG_SETI  5, 2, 1, 2, 7
    # vc8/vc9: f32 col scales (len 2) and scalar row scale (len 1)
    CFG_SETI  5, 2, 0, 4, 8
    CFG_SETI  5, 2, 1, 2, 8
    CFG_SETI  5, 2, 0, 4, 9
    CFG_SETI  5, 2, 1, 1, 9

    # Load the operands: t0 = {{1,2,3},{4,-5,6}},
    # t2 = {{7,8,9},{10,11,12}}.
    lla       x3, src_a
    CFG_REG   0, 1, 0, 0, 3, 0
    TLOAD     0
    lla       x3, src_b_nt
    CFG_REG   0, 1, 0, 0, 3, 2
    TLOAD     2

    # Stage 1 - mma.nt.zero a2, t0, t2 -> {{50, 68}, {42, 57}}
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

    # Stage 2 - mma.nt.acc a2, t0, t2 accumulates
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

    # Stage 3 - vmma.nt.zero v4, t0[1], t2 reads {4, -5, 6}
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

    # Stage 4 - vmma.nt.acc v4, t0[0], t2 adds
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

    # Stage 5 - mma.nt.dequant.zero a2, t0, t2, v6, v7:
    # product {{50,68},{42,57}}, vCol {2,3}, vRow {0.5,0.25}
    # -> {{50,102},{21,42.75}}.
    li        x5, 0x40000000     # 2.0
    VINSERT   6, 0, 5
    li        x5, 0x40400000     # 3.0
    VINSERT   6, 1, 5
    li        x5, 0x3F000000     # 0.5
    VINSERT   7, 0, 5
    li        x5, 0x3E800000     # 0.25
    VINSERT   7, 1, 5
    li        a0, 5
    MMA_NT_DEQUANT_ZERO 2, 0, 2, 6, 7
    lla       x3, dstbuf
    CFG_REG   1, 1, 0, 0, 3, 2
    ASTORE    2
    lla       x3, dstbuf
    lla       x4, exp_dequant
    li        x6, 4
    FENCE_MEM
5:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 5b

    # Stage 6 - vmma.nt.dequant.zero v4, t0[1], t2, v8/v9:
    # product {42,57}, vCol {2,3}, vRow[0] = 0.5 -> {42.0, 85.5}.
    li        x5, 0x40000000     # 2.0
    VINSERT   8, 0, 5
    li        x5, 0x40400000     # 3.0
    VINSERT   8, 1, 5
    li        x5, 0x3F000000     # 0.5
    VINSERT   9, 0, 5
    li        a0, 6
    VMMA_NT_DEQUANT_ZERO 4, 0, 1, 2, 8
    lla       x3, dstbuf2
    CFG_REG   2, 1, 0, 0, 3, 4
    VSTORE    4
    lla       x3, dstbuf2
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 0x42280000     # 42.0
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 0x42AB0000     # 85.5
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
exp_dequant: .word 0x42480000, 0x42CC0000, 0x41A80000, 0x422B0000
exp_vdequant: .word 0x42280000, 0x42AB0000
dstbuf:   .space 16
dstbuf2:  .space 8
