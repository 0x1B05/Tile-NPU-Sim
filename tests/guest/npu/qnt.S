#include "npu.inc"

# Self-checking QNT image: programs configs and descriptors, then runs a
# dynamic tquant (RNE + saturation, all-zero row -> scale +0.0), a tdequant
# that consumes the tquant outputs, a vquant with an immediate scale lane,
# and an external-scale tquant with distinct source and scale register
# numbers. Results are stored back and compared word-wise. Ends with ebreak:
# a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # tc0: dtype = i8 (0), rows = 2, cols = 2
    CFG_SETI  5, 0, 0, 0, 0
    CFG_SETI  5, 0, 1, 2, 0
    CFG_SETI  5, 0, 2, 2, 0
    # ac0/ac1: dtype = f32 (4), rows = 2, cols = 2
    CFG_SETI  5, 1, 0, 4, 0
    CFG_SETI  5, 1, 1, 2, 0
    CFG_SETI  5, 1, 2, 2, 0
    CFG_SETI  5, 1, 0, 4, 1
    CFG_SETI  5, 1, 1, 2, 1
    CFG_SETI  5, 1, 2, 2, 1
    # tm0: row_stride = 2, col_stride = 1 (bytes)
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 2, 1, 0
    # am0/am1: row_stride = 8, col_stride = 4 (bytes)
    CFG_SETI  5, 3, 1, 8, 0
    CFG_SETI  5, 3, 2, 4, 0
    CFG_SETI  5, 3, 1, 8, 1
    CFG_SETI  5, 3, 2, 4, 1
    # vc0: dtype = f32, len = 2; vm0: stride = 4
    CFG_SETI  6, 1, 0, 4, 0
    CFG_SETI  6, 1, 1, 2, 0
    CFG_SETI  6, 3, 1, 4, 0
    # vc1: dtype = f32, len = 2; vm1: stride = 4
    CFG_SETI  6, 1, 0, 4, 1
    CFG_SETI  6, 1, 1, 2, 1
    CFG_SETI  6, 3, 1, 4, 1
    # vc2: dtype = f32, len = 1; vm2: stride = 4
    CFG_SETI  6, 1, 0, 4, 2
    CFG_SETI  6, 1, 1, 1, 2
    CFG_SETI  6, 3, 1, 4, 2
    # vc3: dtype = f32, len = 2; vm3: stride = 4
    CFG_SETI  6, 1, 0, 4, 3
    CFG_SETI  6, 1, 1, 2, 3
    CFG_SETI  6, 3, 1, 4, 3
    # bc0: dtype = i8 (0), len = 2; bm0: stride = 1
    CFG_SETI  6, 0, 0, 0, 0
    CFG_SETI  6, 0, 1, 2, 0
    CFG_SETI  6, 2, 1, 1, 0

    # Stage 1 - tquant.rows.q8s32 t0, v0, a0 over
    #   {127.0, -63.5}, {0.0, 0.0} -> t0 bytes {0x7F, 0xC0, 0, 0},
    #   v0 = {1.0, +0.0} (RNE(-63.5) = -64; the zero row divides by +0.0,
    #   giving NaN quotients that quantize to 0).
    lla       x3, src_m
    CFG_REG   3, 0, 0, 0, 3, 0
    ALOAD     0
    TQUANT    0, 0, 0
    lla       x3, dstbuf
    CFG_REG   2, 0, 0, 0, 3, 0
    TSTORE    0
    lla       x3, dstbuf2
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 0x0000C07F     # bytes {0x7F, 0xC0, 0, 0}
    li        a0, 1
    bne       x4, x5, fail
    lla       x3, dstbuf2
    lwu       x4, 0(x3)
    li        x5, 0x3F800000     # 1.0
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    bnez      x4, fail           # row1 scale is +0.0

    # Stage 2 - tdequant.rows.f32 a1, t0, v0 (consuming stage 1) ->
    #   {127.0, -64.0}, {0.0, 0.0}
    lla       x3, dstbuf3
    CFG_REG   3, 0, 0, 0, 3, 1
    TDEQUANT  1, 0, 0
    ASTORE    1
    lla       x3, dstbuf3
    lla       x4, exp_tdq
    li        x6, 4
    li        a0, 2
    FENCE_MEM
1:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 1b

    # Stage 3 - vquant.q8s32 b0, v2[0], v1 over {127.0, 63.5} ->
    #   b0 bytes {0x7F, 0x40} (RNE(63.5) = 64), v2[0] = 1.0
    lla       x3, src_v
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    VQUANT    0, 2, 1, 0
    lla       x3, dstbuf4
    CFG_REG   2, 1, 0, 0, 3, 0
    BSTORE    0
    lla       x3, dstbuf5
    CFG_REG   3, 1, 0, 0, 3, 2
    VSTORE    2
    lla       x3, dstbuf4
    lhu       x4, 0(x3)
    li        x5, 0x407F         # bytes {0x7F, 0x40}
    li        a0, 3
    bne       x4, x5, fail
    lla       x3, dstbuf5
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 0x3F800000     # 1.0
    bne       x4, x5, fail

    # Stage 4 - tquant.rows.scale.q8s32 t0, a0, v3 over stage 1's input
    #   with scales {2.0, 1.0} -> bytes {64, -32, 0, 0} (RNE).
    #   Distinct source/scale numbers catch swapped macro operands;
    #   external scales must remain unchanged.
    lla       x3, dstbuf6
    CFG_REG   3, 1, 0, 0, 3, 3
    li        a0, 4
    li        x5, 0x40000000     # 2.0
    VINSERT   3, 0, 5
    li        x5, 0x3F800000     # 1.0
    VINSERT   3, 1, 5
    TQUANTX   0, 0, 3
    TSTORE    0
    VSTORE    3
    lla       x3, dstbuf
    FENCE_MEM
    lwu       x4, 0(x3)
    li        x5, 0x0000E040     # bytes {64, -32, 0, 0}
    bne       x4, x5, fail
    lla       x3, dstbuf6
    lwu       x4, 0(x3)
    li        x5, 0x40000000     # 2.0
    bne       x4, x5, fail
    lwu       x4, 4(x3)
    li        x5, 0x3F800000     # 1.0
    bne       x4, x5, fail

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_m:   .word 0x42FE0000, 0xC27E0000, 0x00000000, 0x00000000
src_v:   .word 0x42FE0000, 0x427E0000
exp_tdq: .word 0x42FE0000, 0xC2800000, 0x00000000, 0x00000000
dstbuf:  .space 4
dstbuf2: .space 8
dstbuf3: .space 16
dstbuf4: .space 4
dstbuf5: .space 4
dstbuf6: .space 8
