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

.macro BLOAD d5
    M_BLK 2, 0, \d5, 0
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

.macro ALOAD d5
    M_BLK 1, 0, \d5, 0
.endm

.macro VSTORE d5
    M_BLK 3, 1, \d5, 0
.endm

# ROW_MOVE (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 3  [27] EW  [26] 0  [25] IS  [24:22] funct3
#   [21:17] reserved(0)  [16:12] index5  [11:7] S5  [6:2] D5  [1:0] q = 01
# EW = 0: Tile <-> Vec8; EW = 1: Acc <-> Vec32.
# funct3: 0 = insert.row (D5 = matrix, S5 = vector),
#         1 = extract.row (D5 = vector, S5 = matrix).
# IS = 0: index5 is an immediate row number; IS = 1: index5 names xS.
.macro ROW_MOVE ew, is, funct3, index5, s5, d5
    .word ((3 << 28) | (\ew << 27) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\s5 << 7) | (\d5 << 2) | 1)
.endm

.macro TINSROW d5, idx, s5
    ROW_MOVE 0, 0, 0, \idx, \s5, \d5
.endm

.macro TEXROW d5, s5, idx
    ROW_MOVE 0, 0, 1, \idx, \s5, \d5
.endm

.macro AINSROW d5, idx, s5
    ROW_MOVE 1, 0, 0, \idx, \s5, \d5
.endm

.macro AEXROW d5, s5, idx
    ROW_MOVE 1, 0, 1, \idx, \s5, \d5
.endm

# LANE_MOVE (R4): major4 = 4, same layout as ROW_MOVE.
# EW = 0: Vec8; EW = 1: Vec32.
# funct3: 0 = extract (D5 = xD, S5 = vector), 1 = insert (D5 = vector,
#         S5 = xS), 2 = broadcast (D5/S5 vectors).
.macro LANE_MOVE ew, is, funct3, index5, s5, d5
    .word ((4 << 28) | (\ew << 27) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\s5 << 7) | (\d5 << 2) | 1)
.endm

.macro BEXTRACT xd, s5, idx
    LANE_MOVE 0, 0, 0, \idx, \s5, \xd
.endm

.macro BINSERT d5, idx, xs
    LANE_MOVE 0, 0, 1, \idx, \xs, \d5
.endm

.macro BBROADCAST d5, s5, idx
    LANE_MOVE 0, 0, 2, \idx, \s5, \d5
.endm

.macro VEXTRACT xd, s5, idx
    LANE_MOVE 1, 0, 0, \idx, \s5, \xd
.endm

.macro VINSERT d5, idx, xs
    LANE_MOVE 1, 0, 1, \idx, \xs, \d5
.endm

.macro VBROADCAST d5, s5, idx
    LANE_MOVE 1, 0, 2, \idx, \s5, \d5
.endm

# E_UNARY (R2), mirroring NpuR2 in src/decode/encoding.h:
#   [31:28] major4 = 1  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:12] reserved(0)  [11:7] S5  [6:2] D5  [1:0] q = 00
# funct3: 0 not, 1 abs, 2 neg, 3 exp2.approx, 4 rcp.approx, 5 rsqrt.approx,
#         6 silu.approx (7 reserved).
.macro E_UNARY dom2, funct3, d5, s5
    .word ((1 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\s5 << 7) | (\d5 << 2))
.endm

.macro TNOT d5, s5
    E_UNARY 0, 0, \d5, \s5
.endm

.macro AABS d5, s5
    E_UNARY 1, 1, \d5, \s5
.endm

.macro ANEG d5, s5
    E_UNARY 1, 2, \d5, \s5
.endm

.macro VEXP2 d5, s5
    E_UNARY 3, 3, \d5, \s5
.endm

.macro VRCP d5, s5
    E_UNARY 3, 4, \d5, \s5
.endm

.macro VRSQRT d5, s5
    E_UNARY 3, 5, \d5, \s5
.endm

.macro VSILU d5, s5
    E_UNARY 3, 6, \d5, \s5
.endm

# E_CMP (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 2  [27:26] dom2  [25] BX  [24:22] funct3 (cond)
#   [21:17] reserved(0)  [16:12] B5/xS  [11:7] A5  [6:2] M5  [1:0] q = 00
# funct3: 0 eq, 1 ne (int), 2 lt, 3 ge (int), 4 le (f32), 5 unord (f32).
.macro E_CMP dom2, bx, funct3, m5, a5, b5
    .word ((2 << 28) | (\dom2 << 26) | (\bx << 25) | (\funct3 << 22) | (\b5 << 12) | (\a5 << 7) | (\m5 << 2))
.endm

.macro ACMPLT m5, a5, b5
    E_CMP 1, 0, 2, \m5, \a5, \b5
.endm

.macro ACMPGE m5, a5, b5
    E_CMP 1, 0, 3, \m5, \a5, \b5
.endm

.macro ACMPXEQ m5, a5, xs
    E_CMP 1, 1, 0, \m5, \a5, \xs
.endm

.macro VCMPUNORD m5, a5, b5
    E_CMP 3, 0, 5, \m5, \a5, \b5
.endm

.macro VCMPLE m5, a5, b5
    E_CMP 3, 0, 4, \m5, \a5, \b5
.endm

# E_BCAST (R3), mirroring NpuR3 in src/decode/encoding.h:
#   [31:28] major4 = 5  [27] TA  [26] 0  [25] DIR  [24:21] funct4
#   [20:17] reserved(0)  [16:12] S5  [11:7] A5  [6:2] D5  [1:0] q = 00
# TA: 0 = Tile + Vec8, 1 = Acc + Vec32. DIR: 0 = byrow, 1 = bycol.
.macro E_BCAST ta, dir, funct4, d5, a5, s5
    .word ((5 << 28) | (\ta << 27) | (\dir << 25) | (\funct4 << 21) | (\s5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

.macro TADDB_ROW d5, a5, s5
    E_BCAST 0, 0, 0, \d5, \a5, \s5
.endm

.macro AMULV_COL d5, a5, s5
    E_BCAST 1, 1, 2, \d5, \a5, \s5
.endm

# E_R4 (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 3  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:17] arg5 (mask M / addend C)  [16:12] B5  [11:7] A5  [6:2] D5
#   [1:0] q = 00
# funct3: 0 select, 1 fmadd (Acc/Vec32 only).
.macro E_R4 dom2, funct3, arg5, d5, a5, b5
    .word ((3 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\arg5 << 17) | (\b5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

# select: asm operand order D, M, A, B; arg5 carries M.
.macro TSELECT d5, m5, a5, b5
    E_R4 0, 0, \m5, \d5, \a5, \b5
.endm

# fmadd: asm operand order D, A, B, C; arg5 carries C.
.macro VFMADD d5, a5, b5, c5
    E_R4 3, 1, \c5, \d5, \a5, \b5
.endm

# E_MASK (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 4  [27:26] dom2  [25] FS  [24:22] funct3
#   [21:17] arg5 (fill5 / xFill)  [16:12] B5 (xBound)  [11:7] A5 (S)
#   [6:2] D5  [1:0] q = 00
# funct3: 0 tail, 1 tril, 2 triu (matrix-only). FS = 0: arg5 is a fill5
# code; FS = 1: arg5 names the xFill Scalar register.
.macro E_MASK dom2, fs, funct3, arg5, b5, a5, d5
    .word ((4 << 28) | (\dom2 << 26) | (\fs << 25) | (\funct3 << 22) | (\arg5 << 17) | (\b5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

# tril: asm operand order D, S, xDelta, fill5.
.macro AMASK_TRIL d5, s5, xd, fill5
    E_MASK 1, 0, 1, \fill5, \xd, \s5, \d5
.endm

# vector tail: asm operand order D, S, xLen, fill5.
.macro VMASK_TAIL d5, s5, xl, fill5
    E_MASK 3, 0, 0, \fill5, \xl, \s5, \d5
.endm
