# NPU instruction encoding macros, mirroring the NpuR3 field layout in
# src/decode/encoding.h:
#   [31:28] major4  [27:25] sel3  [24:21] funct4  [20:17] reserved4
#   [16:12] src2    [11:7]  src1  [6:2]   dest    [1:0] q
# Correctness is pinned by the known-good words in tests/simulator_test.cpp
# (0x00002080 = tadd t0, t1, t2), which stay as raw constants on purpose.

# E_BIN: major4 = 0, reserved4 = 0, q = 00; sel3 = {dom2, BX}.
.macro E_BIN sel3, funct4, dest, src1, src2
    .word ((\sel3 << 25) | (\funct4 << 21) | (\src2 << 12) | (\src1 << 7) | (\dest << 2))
.endm

# E_BIN add on the tile domain with BX=0: sel3 = 0b000, funct4 = 0.
.macro TADD dest, src1, src2
    E_BIN 0, 0, \dest, \src1, \src2
.endm

# I format (cfg.seti), mirroring NpuI in src/decode/encoding.h:
#   [31:28] major4 (5 = matrix side, 6 = vector side)  [27:26] sel2
#   [25:23] field3  [22:7] imm16  [6:2] C5  [1:0] q = 10
.macro CFG_SETI major4, sel2, field3, imm, c5
    .word ((\major4 << 28) | (\sel2 << 26) | (\field3 << 23) | ((\imm & 0xffff) << 7) | (\c5 << 2) | 2)
.endm

# R4 format (CFG_REG), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 7  [27:26] sel2  [25] MV  [24:22] funct3
#   [21:17] field5  [16:12] reserved(0)  [11:7] S5  [6:2] D5  [1:0] q = 10
# funct3: 0 = setx (S5 = xS, D5 = C), 1 = copy (S5 = CS, D5 = CD),
#         2 = get (S5 = C, D5 = xD)
.macro CFG_REG sel2, mv, funct3, field5, s5, d5
    .word ((7 << 28) | (\sel2 << 26) | (\mv << 25) | (\funct3 << 22) | (\field5 << 17) | (\s5 << 7) | (\d5 << 2) | 2)
.endm
