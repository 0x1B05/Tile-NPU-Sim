#include "npu.inc"

# Self-checking reduce image: programs configs and descriptors, then runs an
# areduce.rows.sum/argmax and vreduce.argmax cases (NaN candidates are ignored;
# an all-NaN source falls back to index 0 / NaN value). Results are stored back
# from the vector domain and compared word-wise (lwu/subw). Ends with ebreak:
# a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # ac1: dtype = f32 (4), 2x2; am1: row_stride = 8, col_stride = 4 (bytes)
    CFG_SETI  5, 1, 0, 4, 1
    CFG_SETI  5, 1, 1, 2, 1
    CFG_SETI  5, 1, 2, 2, 1
    CFG_SETI  6, 1, 1, 8, 1
    CFG_SETI  6, 1, 2, 4, 1
    # vc0: dtype = f32, len = 2; vm0: stride = 4
    CFG_SETI  5, 2, 0, 4, 0
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  6, 2, 1, 4, 0
    # vc1: dtype = f32, len = 3; vm1: stride = 4
    CFG_SETI  5, 2, 0, 4, 1
    CFG_SETI  5, 2, 1, 3, 1
    CFG_SETI  6, 2, 1, 4, 1
    # vc2/vc3: dtype = u32/f32, len = 3 (vreduce.argmax vIdx/vVal)
    CFG_SETI  5, 2, 0, 3, 2
    CFG_SETI  5, 2, 1, 3, 2
    CFG_SETI  5, 2, 0, 4, 3
    CFG_SETI  5, 2, 1, 3, 3
    # vm2/vm3: stride = 4
    CFG_SETI  6, 2, 1, 4, 2
    CFG_SETI  6, 2, 1, 4, 3

    # Stage 1 - areduce.rows.sum v0, a1 over
    #   {1.5, -2.5}, {3.0, 4.5} -> {-1.0, 7.5}
    lla       x3, src_m
    CFG_REG   1, 1, 0, 0, 3, 1
    ALOAD     1
    AREDUCE   0, 0, 1
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf
    lla       x4, exp_sum
    li        x6, 2
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

    # Stage 2 - vreduce.argmax v2, v3, v1 over {1.0, 7.0, NaN} ->
    #   v2 = {1, 1, 1} (u32), v3 = {7.0, 7.0, 7.0} (NaN candidate is ignored)
    lla       x3, src_v
    CFG_REG   2, 1, 0, 0, 3, 1
    VLOAD     1
    VREDUCE_ARGMAX 2, 3, 1
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 2
    addi      x3, x3, 12
    CFG_REG   2, 1, 0, 0, 3, 3   # vm3 base = dstbuf + 12 (vIdx and vVal side by side)
    VSTORE    2
    VSTORE    3
    lla       x3, dstbuf
    li        a0, 2
    FENCE_MEM
    lwu       x5, 0(x3)
    li        x7, 1
    bne       x5, x7, fail
    lwu       x5, 4(x3)
    bne       x5, x7, fail
    lwu       x5, 12(x3)
    li        x7, 0x40E00000
    bne       x5, x7, fail
    lwu       x5, 16(x3)
    bne       x5, x7, fail

    # Stage 3 - all-NaN source falls back to index 0 and a NaN value
    lla       x3, src_nan
    CFG_REG   2, 1, 0, 0, 3, 1
    VLOAD     1
    VREDUCE_ARGMAX 2, 3, 1
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 2
    addi      x3, x3, 12
    CFG_REG   2, 1, 0, 0, 3, 3
    VSTORE    2
    VSTORE    3
    lla       x3, dstbuf
    li        a0, 3
    FENCE_MEM
    lwu       x5, 0(x3)
    bnez      x5, fail
    lwu       x5, 4(x3)
    bnez      x5, fail
    lwu       x5, 12(x3)
    li        x7, 0x7FC00000
    bne       x5, x7, fail
    lwu       x5, 16(x3)
    bne       x5, x7, fail

    # Stage 4 - areduce.rows.argmax v0, v1, a1 over
    #   {1.5, -2.5}, {3.0, 4.5} -> indices {0, 1}, values {1.5, 4.5}
    CFG_SETI  5, 2, 0, 3, 0       # vc0: u32, len 2 (vIdx)
    CFG_SETI  5, 2, 1, 2, 0
    CFG_SETI  5, 2, 0, 4, 1       # vc1: f32, len 2 (vVal)
    CFG_SETI  5, 2, 1, 2, 1
    AREDUCE_ARGMAX 0, 1, 1
    lla       x3, dstbuf
    CFG_REG   2, 1, 0, 0, 3, 0
    VSTORE    0
    addi      x3, x3, 8
    CFG_REG   2, 1, 0, 0, 3, 1
    VSTORE    1
    lla       x3, dstbuf
    lla       x4, exp_arga
    li        x6, 4
    li        a0, 4
    FENCE_MEM
4:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 4b

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_m:   .word 0x3FC00000, 0xC0200000, 0x40400000, 0x40900000
exp_sum: .word 0xBF800000, 0x40F00000
exp_arga: .word 0, 1, 0x3FC00000, 0x40900000
src_v:   .word 0x3F800000, 0x40E00000, 0x7FC00000
src_nan: .word 0x7FC00000, 0x7FC00000, 0x7FC00000
dstbuf:  .space 24
