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

# E_BIN subtract on the tile domain with BX=0: funct4 = 1.
.macro TSUB dest, src1, src2
    E_BIN 0, 1, \dest, \src1, \src2
.endm

# E_BIN max on the tile domain with BX=0: funct4 = 5.
.macro TMAX dest, src1, src2
    E_BIN 0, 5, \dest, \src1, \src2
.endm

# E_BIN add on the tile domain with a scalar source (BX=1): src2 names xS.
.macro TADDX dest, src1, xs
    E_BIN 1, 0, \dest, \src1, \xs
.endm

# M_BLK block and gather forms (R2 format), mirroring NpuR2 in
# src/decode/encoding.h:
#   [31:28] major4 = 0000  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:12] reserved(0)    [11:7] S5 (gather index vS)  [6:2] D5  [1:0] q = 01
# funct3 = {form2, LS}: 000/001 = block load/store, 010 = gather load.
# dom2: 0 = tile, 1 = acc, 2 = b (vec8), 3 = v (vec32).
.macro M_BLK dom2, funct3, d5, s5
    .word ((\dom2 << 26) | (\funct3 << 22) | (\s5 << 7) | (\d5 << 2) | 1)
.endm

# M_BLK row form (MR format), mirroring NpuMR in src/decode/encoding.h:
#   [31:28] major4 = 0000  [27] TA  [26] 0  [25] IS  [24:22] funct3
#   [21:17] reserved(0)  [16:12] index5  [11:7] reserved(0)
#   [6:2] reg5  [1:0] q = 01
# funct3: 100 = row load, 101 = row store. IS = 0: index5 is an immediate
# row number; IS = 1: index5 names xS holding the row number.
.macro M_ROW ta, is, funct3, index5, reg5
    .word ((\ta << 27) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\reg5 << 2) | 1)
.endm

.macro TLOAD d5
    M_BLK 0, 0, \d5, 0
.endm

.macro TSTORE d5
    M_BLK 0, 1, \d5, 0
.endm

# tload.gather: rows indexed by vS lane values.
.macro TGATHER d5, vs
    M_BLK 0, 2, \d5, \vs
.endm

.macro VLOAD d5
    M_BLK 3, 0, \d5, 0
.endm

.macro TLOADR reg5, index5
    M_ROW 0, 0, 4, \index5, \reg5
.endm

# Row store with the row number in a scalar register.
.macro TSTORERX reg5, xs
    M_ROW 0, 1, 5, \xs, \reg5
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

.macro ASTORE d5
    M_BLK 1, 1, \d5, 0
.endm

.macro BSTORE d5
    M_BLK 2, 1, \d5, 0
.endm

# INIT_MOVE fill (I format), mirroring NpuI in src/decode/encoding.h:
#   [31:28] major4 = 2  [27:26] dom2  [25:23] funct3 = 000
#   [22:7] imm16  [6:2] D5  [1:0] q = 01
.macro INIT_FILL dom2, d5, imm
    .word ((2 << 28) | (\dom2 << 26) | ((\imm & 0xffff) << 7) | (\d5 << 2) | 1)
.endm

# INIT_MOVE R2 forms, mirroring NpuR2 in src/decode/encoding.h:
#   [31:28] major4 = 2  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:12] reserved(0)  [11:7] S5  [6:2] D5  [1:0] q = 01
# funct3: 1 = fillx (S5 = xS), 2 = copy (S5 = same-domain source),
#         3 = ttranspose (Tile only).
.macro INIT_R2 dom2, funct3, d5, s5
    .word ((2 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\s5 << 7) | (\d5 << 2) | 1)
.endm

.macro TFILL d5, imm
    INIT_FILL 0, \d5, \imm
.endm

.macro TFILLX d5, xs
    INIT_R2 0, 1, \d5, \xs
.endm

.macro TCOPY d5, s5
    INIT_R2 0, 2, \d5, \s5
.endm

.macro TTRANSPOSE d5, s5
    INIT_R2 0, 3, \d5, \s5
.endm

.macro AFILLX d5, xs
    INIT_R2 1, 1, \d5, \xs
.endm

.macro BFILL d5, imm
    INIT_FILL 2, \d5, \imm
.endm
