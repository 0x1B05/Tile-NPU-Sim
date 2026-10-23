#include "npu.inc"

# Self-checking E_CMP image: programs configs and descriptors, then runs
# register/scalar comparisons on Vec32 (i32 and f32), producing u32 masks
# (all-ones / zero). Every stage stores the mask back and compares it against
# an expected array word by word (subw keeps the comparison in the low 32
# bits, immune to lw/li extension differences). Source registers are reused
# across stages when a stage does not overwrite them. Ends with ebreak:
# a0 = 0 pass, a0 = N identifies the failing check.
#
# Buffer addresses go through lla (pc-relative): the image is loaded at
# 0x80000000, where medlow's %hi/%lo expansion would sign-extend bit 31.

    .globl _start
_start:
    # vc0: dtype = u32 (3), len = 4 (mask); vm0: stride = 4
    CFG_SETI  6, 1, 0, 3, 0
    CFG_SETI  6, 1, 1, 4, 0
    CFG_SETI  6, 3, 1, 4, 0
    # vc1/vc2: dtype = i32 (2), len = 4; vm1/vm2: stride = 4
    CFG_SETI  6, 1, 0, 2, 1
    CFG_SETI  6, 1, 1, 4, 1
    CFG_SETI  6, 3, 1, 4, 1
    CFG_SETI  6, 1, 0, 2, 2
    CFG_SETI  6, 1, 1, 4, 2
    CFG_SETI  6, 3, 1, 4, 2

    # Stage 1 - vcmp.lt v0, v1, v2 (i32):
    #   {-1, 5, INT32_MIN, 7} < {1, 5, 0, INT32_MIN}
    lla       x3, src_i1
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    lla       x3, src_i2
    CFG_REG   3, 1, 0, 0, 3, 2
    VLOAD     2
    VCMPLT    0, 1, 2
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_lt
    li        x6, 4
    li        a0, 1
1:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 1b

    # Stage 2 - vcmp.ge v0, v1, v2 (i32), reusing the stage-1 sources
    VCMPGE    0, 1, 2
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_ge
    li        x6, 4
    li        a0, 2
2:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 2b

    # Stage 3 - vcmpx.eq v0, v1, x5 (scalar form; only the low 32 bits read)
    li        x5, 0xffffffff00000005
    VCMPXEQ   0, 1, 5
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_eqx
    li        x6, 4
    li        a0, 3
3:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 3b

    # Stage 4 - vcmp.unord v0, v1, v2 (f32):
    #   {1.5, NaN, +inf, -0} vs {1.5, +0, NaN, 1.0}
    CFG_SETI  6, 1, 0, 4, 1
    CFG_SETI  6, 1, 0, 4, 2
    lla       x3, src_f1
    CFG_REG   3, 1, 0, 0, 3, 1
    VLOAD     1
    lla       x3, src_f2
    CFG_REG   3, 1, 0, 0, 3, 2
    VLOAD     2
    VCMPUNORD 0, 1, 2
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_unord
    li        x6, 4
    li        a0, 4
4:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 4b

    # Stage 5 - vcmp.le v0, v1, v2 (f32), reusing the stage-4 sources
    VCMPLE    0, 1, 2
    lla       x3, dstbuf32
    CFG_REG   3, 1, 0, 0, 3, 0
    VSTORE    0
    lla       x3, dstbuf32
    lla       x4, exp_le
    li        x6, 4
    li        a0, 5
5:  lwu       x5, 0(x3)
    lwu       x7, 0(x4)
    subw      x5, x5, x7
    bnez      x5, fail
    addi      x3, x3, 4
    addi      x4, x4, 4
    addi      x6, x6, -1
    bnez      x6, 5b

    li        a0, 0
    ebreak
fail:
    ebreak

    .section .data
    .balign 4
src_i1: .word 0xFFFFFFFF, 5, 0x80000000, 7
src_i2: .word 1, 5, 0, 0x80000000
exp_lt: .word 0xFFFFFFFF, 0, 0xFFFFFFFF, 0
exp_ge: .word 0, 0xFFFFFFFF, 0, 0xFFFFFFFF
exp_eqx: .word 0, 0xFFFFFFFF, 0, 0
src_f1: .word 0x3FC00000, 0x7FC00000, 0x7F800000, 0x80000000
src_f2: .word 0x3FC00000, 0x00000000, 0x7FC00000, 0x3F800000
exp_unord: .word 0, 0xFFFFFFFF, 0xFFFFFFFF, 0
exp_le: .word 0xFFFFFFFF, 0, 0, 0xFFFFFFFF
dstbuf32: .space 16
