# NPU guest bindings: config field numbers, sel2/dtype codes, scalar register
# numbers, and instruction encoding macros. Shared by tests/guest and
# apps/qwen3. Field layouts mirror src/decode/encoding.h; the CF_* numbers
# follow the manual's field3 table. Correctness is pinned by the known-good
# words in tests/simulator_test.cpp (0x00002080 = tadd t0, t1, t2), which
# stay as raw constants on purpose.

# Config field numbers (field3 / field5).
.equ CF_DTYPE,    0   # TC/AC/BC/VC dtype; TM/AM/BM/VM base_addr
.equ CF_ROWS,     1   # TC/AC rows; TM/AM row_stride_bytes
.equ CF_COLS,     2   # TC/AC cols; TM/AM col_stride_bytes; BM/VM flags
.equ CF_LAYOUT,   3   # TC/AC layout; TM/AM transform
.equ CF_ARITH,    4   # TC arith_mode; TM/AM flags
.equ CF_LEN,      1   # BC/VC len; BM/VM stride_bytes
.equ CF_BASE,     0
.equ CF_RSTRIDE,  1
.equ CF_CSTRIDE,  2
.equ CF_FLAGS,    4
.equ CF_STRIDE,   1
.equ CF_VFLAGS,   2

# Config register sel2 codes: matrix side TC/AC/TM/AM, vector side BC/VC/BM/VM.
.equ SEL_TC, 0
.equ SEL_AC, 1
.equ SEL_TM, 2
.equ SEL_AM, 3
.equ SEL_BC, 0
.equ SEL_VC, 1
.equ SEL_BM, 2
.equ SEL_VM, 3

# DType codes.
.equ DT_I8,  0
.equ DT_U8,  1
.equ DT_I32, 2
.equ DT_U32, 3
.equ DT_F32, 4

# Scalar register numbers for NPU instruction operands (xS/xD fields take
# numbers, not ABI names).
.equ X_ZERO, 0
.equ X_RA,   1
.equ X_SP,   2
.equ X_T0,   5
.equ X_T1,   6
.equ X_T2,   7
.equ X_S0,   8
.equ X_S1,   9
.equ X_A0,   10
.equ X_A1,   11
.equ X_A2,   12
.equ X_A3,   13
.equ X_A4,   14
.equ X_A5,   15
.equ X_A6,   16
.equ X_A7,   17
.equ X_S2,   18
.equ X_S3,   19
.equ X_S4,   20
.equ X_S5,   21
.equ X_S6,   22
.equ X_T3,   28
.equ X_T4,   29
.equ X_T5,   30
.equ X_T6,   31

# E_BIN: major4 = 0, reserved4 = 0, q = 00; sel3 = {dom2, BX}.
#   [31:28] major4  [27:25] sel3  [24:21] funct4  [20:17] reserved4
#   [16:12] src2    [11:7]  src1  [6:2]   dest    [1:0] q
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

# Vec32 elementwise forms (E_BIN, dom2 = 3).
.macro VADD d5, a5, b5
    E_BIN 6, 0, \d5, \a5, \b5
.endm
.macro VSUB d5, a5, b5
    E_BIN 6, 1, \d5, \a5, \b5
.endm
.macro VMUL d5, a5, b5
    E_BIN 6, 2, \d5, \a5, \b5
.endm
.macro VMAX d5, a5, b5
    E_BIN 6, 5, \d5, \a5, \b5
.endm
.macro VADDX d5, a5, xs
    E_BIN 7, 0, \d5, \a5, \xs
.endm
.macro VSUBX d5, a5, xs
    E_BIN 7, 1, \d5, \a5, \xs
.endm
.macro VMULX d5, a5, xs
    E_BIN 7, 2, \d5, \a5, \xs
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
#   [31:28] major4 = 0000  [27] 0  [26] EW  [25] IS  [24:22] funct3
#   [21:17] reserved(0)  [16:12] index5  [11:7] reserved(0)
#   [6:2] reg5  [1:0] q = 01
# EW: 0 = Tile, 1 = Acc. funct3: 100 = row load, 101 = row store. IS = 0: index5 is an immediate
# row number; IS = 1: index5 names xS holding the row number.
.macro M_ROW ew, is, funct3, index5, reg5
    .word ((\ew << 26) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\reg5 << 2) | 1)
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
#   cfg.setx/get: [21:20] reserved2 [19:17] field3 [16:12] N5/reserved5
#   cfg.copy/setw: [21:17] reserved5 [16:12] N5; cfg.getw: [21:12] reserved10
#   [11:7] S5  [6:2] D5  [1:0] q = 10
# funct3: 0 = setx (S5 = xS, D5 = C), 1 = copy (S5 = CS, D5 = CD),
#         2 = get (S5 = C, D5 = xD)
# N5 ([16:12]) is the last destination offset N; omitted N defaults to 0.
.macro CFG_REG sel2, mv, funct3, field5, s5, d5, n=0
    .word ((7 << 28) | (\sel2 << 26) | (\mv << 25) | (\funct3 << 22) | (\field5 << 17) | (\n << 12) | (\s5 << 7) | (\d5 << 2) | 2)
.endm

# cfg.seti on the matrix side (major4 = 5).
.macro SETIM sel2, field, imm, c5
    CFG_SETI 5, \sel2, \field, \imm, \c5
.endm

# cfg.seti on the vector side (major4 = 6).
.macro SETIV sel2, field, imm, c5
    CFG_SETI 6, \sel2, \field, \imm, \c5
.endm

# cfg.setw C, xS[, N] (CFG_REG funct3 = 011): writes xS[31:0] as the config
# word into C through C+N; N is the last destination offset in [16:12].
# cfg.getw xD, C (funct3 = 100) reads one config word into xD.
.macro SETW sel2, mv, c5, xs, n=0
    CFG_REG \sel2, \mv, 3, 0, \xs, \c5, \n
.endm

.macro GETW sel2, mv, xd, c5
    CFG_REG \sel2, \mv, 4, 0, \c5, \xd
.endm

# Config words for cfg.setw (row-major, wrap): matrix rows[7:0] cols[15:8]
# dtype[19:16]; vector len[7:0] dtype[11:8]. xd is an ABI register name.
.macro LI_CWM xd, dtype, rows, cols
    li      \xd, ((\dtype << 16) | (\cols << 8) | \rows)
.endm

.macro LI_CWV xd, dtype, len
    li      \xd, ((\dtype << 8) | \len)
.endm

# cfg.setx: C[field] = x[xS] for C through C+N. MV = 0 matrix side,
# 1 vector side.
.macro SETX sel2, mv, field, c5, xs, n=0
    CFG_REG \sel2, \mv, 0, \field, \xs, \c5, \n
.endm

# cfg.copy: broadcast config cs into configs cd through cd+N.
.macro CFGCOPY sel2, mv, cd, cs, n=0
    CFG_REG \sel2, \mv, 1, 0, \cs, \cd, \n
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

# FILL_IMM (I format), mirroring NpuI in src/decode/encoding.h:
#   [31:28] major4 = 1  [27:26] dom2  [25:23] funct3 = 000
#   [22:7] imm16  [6:2] D5  [1:0] q = 01
.macro FILL_IMM dom2, d5, imm
    .word ((1 << 28) | (\dom2 << 26) | ((\imm & 0xffff) << 7) | (\d5 << 2) | 1)
.endm

# REG_MOVE R2 forms, mirroring NpuR2 in src/decode/encoding.h:
#   [31:28] major4 = 2  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:12] reserved(0)  [11:7] S5  [6:2] D5  [1:0] q = 01
# funct3: 1 = fillx (S5 = xS), 2 = copy (S5 = same-domain source).
.macro REG_MOVE dom2, funct3, d5, s5
    .word ((2 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\s5 << 7) | (\d5 << 2) | 1)
.endm

.macro TFILL d5, imm
    FILL_IMM 0, \d5, \imm
.endm

.macro TFILLX d5, xs
    REG_MOVE 0, 1, \d5, \xs
.endm

.macro TCOPY d5, s5
    REG_MOVE 0, 2, \d5, \s5
.endm

.macro AFILLX d5, xs
    REG_MOVE 1, 1, \d5, \xs
.endm

.macro BFILL d5, imm
    FILL_IMM 2, \d5, \imm
.endm

# Vec32 fill forms.
.macro VFILL d5, imm
    FILL_IMM 3, \d5, \imm
.endm
.macro VFILLX d5, xs
    REG_MOVE 3, 1, \d5, \xs
.endm

.macro ALOAD d5
    M_BLK 1, 0, \d5, 0
.endm

.macro VSTORE d5
    M_BLK 3, 1, \d5, 0
.endm

# ROW_MOVE (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 3  [27] 0  [26] EW  [25] IS  [24:22] funct3
#   [21:17] reserved(0)  [16:12] index5  [11:7] S5  [6:2] D5  [1:0] q = 01
# EW = 0: Tile <-> Vec8; EW = 1: Acc <-> Vec32.
# funct3: 0 = insert.row (D5 = matrix, S5 = vector),
#         1 = extract.row (D5 = vector, S5 = matrix).
# IS = 0: index5 is an immediate row number; IS = 1: index5 names xS.
.macro ROW_MOVE ew, is, funct3, index5, s5, d5
    .word ((3 << 28) | (\ew << 26) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\s5 << 7) | (\d5 << 2) | 1)
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
    .word ((4 << 28) | (\ew << 26) | (\is << 25) | (\funct3 << 22) | (\index5 << 12) | (\s5 << 7) | (\d5 << 2) | 1)
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
#         6/7 reserved.
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

.macro VABS d5, s5
    E_UNARY 3, 1, \d5, \s5
.endm

# E_CMP (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 2  [27:26] dom2  [25] BX  [24:22] funct3 (cond)
#   [21:17] reserved(0)  [16:12] B5/xS  [11:7] A5  [6:2] M5  [1:0] q = 00
# funct3: 0 eq, 1 ne (int), 2 lt, 3 ge (int), 4 le (f32), 5 unord (f32).
.macro E_CMP dom2, bx, funct3, m5, a5, b5
    .word ((2 << 28) | (\dom2 << 26) | (\bx << 25) | (\funct3 << 22) | (\b5 << 12) | (\a5 << 7) | (\m5 << 2))
.endm

.macro VCMPLT m5, a5, b5
    E_CMP 3, 0, 2, \m5, \a5, \b5
.endm

.macro VCMPGE m5, a5, b5
    E_CMP 3, 0, 3, \m5, \a5, \b5
.endm

.macro VCMPXEQ m5, a5, xs
    E_CMP 3, 1, 0, \m5, \a5, \xs
.endm

.macro VCMPUNORD m5, a5, b5
    E_CMP 3, 0, 5, \m5, \a5, \b5
.endm

.macro VCMPLE m5, a5, b5
    E_CMP 3, 0, 4, \m5, \a5, \b5
.endm

# E_BCAST (R3), mirroring NpuR3 in src/decode/encoding.h:
#   [31:28] major4 = 5  [27] 0  [26] EW  [25] DIR  [24:21] funct4
#   [20:17] reserved(0)  [16:12] S5  [11:7] A5  [6:2] D5  [1:0] q = 00
# EW: 0 = reserved, 1 = Acc + Vec32. DIR: 0 = byrow, 1 = bycol.
.macro E_BCAST ew, dir, funct4, d5, a5, s5
    .word ((5 << 28) | (\ew << 26) | (\dir << 25) | (\funct4 << 21) | (\s5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

.macro AADDV_ROW d5, a5, s5
    E_BCAST 1, 0, 0, \d5, \a5, \s5
.endm

.macro AMULV_COL d5, a5, s5
    E_BCAST 1, 1, 2, \d5, \a5, \s5
.endm

.macro AMULV_ROW d5, a5, s5
    E_BCAST 1, 0, 2, \d5, \a5, \s5
.endm

# E_R4 (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 3  [27:26] dom2  [25] 0  [24:22] funct3
#   [21:17] M5/C5 (mask M / addend C)  [16:12] B5  [11:7] A5  [6:2] D5
#   [1:0] q = 00
# funct3: 0 select, 1 fmadd (Acc/Vec32 only).
.macro E_R4 dom2, funct3, m5_c5, d5, a5, b5
    .word ((3 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\m5_c5 << 17) | (\b5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

# select: asm operand order D, M, A, B; M5/C5 carries M.
.macro TSELECT d5, m5, a5, b5
    E_R4 0, 0, \m5, \d5, \a5, \b5
.endm

# fmadd: asm operand order D, A, B, C; M5/C5 carries C.
.macro VFMADD d5, a5, b5, c5
    E_R4 3, 1, \c5, \d5, \a5, \b5
.endm

# E_MASK (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 4  [27:26] dom2  [25] FS  [24:22] funct3
#   [21:17] xBound  [16:12] fill5 / xFill  [11:7] A5 (S)
#   [6:2] D5  [1:0] q = 00
# funct3: 0 tail, 1 tril, 2 triu (matrix-only). FS = 0: [16:12] is a fill5
# code; FS = 1: [16:12] names the xFill Scalar register.
.macro E_MASK dom2, fs, funct3, xb5, fill5, a5, d5
    .word ((4 << 28) | (\dom2 << 26) | (\fs << 25) | (\funct3 << 22) | (\xb5 << 17) | (\fill5 << 12) | (\a5 << 7) | (\d5 << 2))
.endm

# tril: asm operand order D, S, xDelta, fill5.
.macro AMASK_TRIL d5, s5, xd, fill5
    E_MASK 1, 0, 1, \xd, \fill5, \s5, \d5
.endm

# vector tail: asm operand order D, S, xLen, fill5.
.macro VMASK_TAIL d5, s5, xl, fill5
    E_MASK 3, 0, 0, \xl, \fill5, \s5, \d5
.endm

# CVT (R2), mirroring NpuR2 in src/decode/encoding.h:
#   [31:28] major4 = 3  [27:26] dom2 (dst domain)  [25] 0  [24:22] funct3
#   [21:12] reserved(0)  [11:7] S5  [6:2] D5  [1:0] q = 10
# funct3: 0 = i32 -> f32 (acvt/vcvt), 1 = i8 -> i32 widen (twiden, Acc dst).
.macro CVT dom2, funct3, d5, s5
    .word ((3 << 28) | (\dom2 << 26) | (\funct3 << 22) | (\s5 << 7) | (\d5 << 2) | 2)
.endm

.macro ACVT d5, s5
    CVT 1, 0, \d5, \s5
.endm

.macro VCVT d5, s5
    CVT 3, 0, \d5, \s5
.endm

.macro TWIDEN d5, s5
    CVT 1, 1, \d5, \s5
.endm

# AREDUCE (R2), mirroring NpuR2 in src/decode/encoding.h:
#   [31:28] major4 = 1  [27:25] reserved(0)  [24:22] funct3
#   [21:12] reserved(0)  [11:7] S5 (aS)  [6:2] D5 (vD)  [1:0] q = 10
# funct3: 0 sum, 1 max, 2 min, 3 sumsq, 4 argmax, 5 absmax.
.macro AREDUCE funct3, vd, as
    .word ((1 << 28) | (\funct3 << 22) | (\as << 7) | (\vd << 2) | 2)
.endm

# VREDUCE (R2): major4 = 2, D5 = xD; same layout otherwise.
.macro VREDUCE funct3, xd, vs
    .word ((2 << 28) | (\funct3 << 22) | (\vs << 7) | (\xd << 2) | 2)
.endm

# vreduce.argmax (R4 two-destination variant): xVal5 in the src2 slot,
# reserved3/reserved5 = 0, D5 = xIndex.
.macro VREDUCE_ARGMAX xi, xv, vs
    .word ((2 << 28) | (4 << 22) | (\xv << 12) | (\vs << 7) | (\xi << 2) | 2)
.endm

# QNT (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 4 (0100)  [27:26] dom2  [25] IS  [24:22] funct3
#   [21:17] vScale5  [16:12] index5  [11:7] S5  [6:2] D5  [1:0] q = 10
# dom2: 0 tquant (Tile dst), 1 tdequant (Acc dst), 2 vquant (Vec8 dst),
#       3 bdequant (Vec32 dst). funct3: 0 = quant/dequant, 1 = tquant with
# external scale (Tile dst only). index5/IS carry the vector-form scale
# lane and are reserved (0) for the matrix forms.
.macro QNT dom2, is, funct3, vscale5, index5, s5, d5
    .word ((4 << 28) | (\dom2 << 26) | (\is << 25) | (\funct3 << 22) | (\vscale5 << 17) | (\index5 << 12) | (\s5 << 7) | (\d5 << 2) | 2)
.endm

# tquant.rows.q8s32: asm operand order tD, vScale, aS.
.macro TQUANT d5, vscale, as
    QNT 0, 0, 0, \vscale, 0, \as, \d5
.endm

# tquant.rows.scale.q8s32: asm operand order tD, aS, vScale.
.macro TQUANTX d5, as, vscale
    QNT 0, 0, 1, \vscale, 0, \as, \d5
.endm

# tdequant.rows.f32: asm operand order aD, tS, vScale.
.macro TDEQUANT d5, s5, vscale
    QNT 1, 0, 0, \vscale, 0, \s5, \d5
.endm

# vquant.q8s32: asm operand order bD, vScale[lane], vS.
.macro VQUANT d5, vscale, s5, lane
    QNT 2, 0, 0, \vscale, \lane, \s5, \d5
.endm

# bdequant.f32: asm operand order vD, bS, vScale[lane].
.macro BDEQUANT d5, s5, vscale, lane
    QNT 3, 0, 0, \vscale, \lane, \s5, \d5
.endm

# MMA_BDOT (R4), mirroring NpuR4 in src/decode/encoding.h:
#   [31:28] major4 = 0000  [27:25] reserved3  [24:22] funct3
#   [21:17] reserved5  [16:12] B5 (tB)  [11:7] A5 (tA/bA)  [6:2] D5 (aD/vD)
#   [1:0] q = 10
# funct3: 000 mma.nn.zero, 001 mma.nn.acc, 010/011 reserved,
#         100 bdot.nn, 101 bdot.nn.acc, 110 bdot.nt, 111 bdot.nt.acc.
.macro MMA_BDOT funct3, d5, a5, b5
    .word ((\funct3 << 22) | (\b5 << 12) | (\a5 << 7) | (\d5 << 2) | 2)
.endm

.macro MMA_NN_ZERO d5, a5, b5
    MMA_BDOT 0, \d5, \a5, \b5
.endm

.macro MMA_NN_ACC d5, a5, b5
    MMA_BDOT 1, \d5, \a5, \b5
.endm

.macro BDOT_NT d5, a5, b5
    MMA_BDOT 6, \d5, \a5, \b5
.endm

.macro BDOT_NN_ACC d5, a5, b5
    MMA_BDOT 5, \d5, \a5, \b5
.endm

.macro BDOT_NT_ACC d5, a5, b5
    MMA_BDOT 7, \d5, \a5, \b5
.endm

# SYS (Z format):
#   [31:28] major4 = 1000  [27:25] funct3  [24:2] reserved23  [1:0] q = 10
# funct3: 000 fence.mem, 001 fence.sa, 010 fence.all, 011 kernel.end.
.macro SYS funct3
    .word ((8 << 28) | (\funct3 << 25) | 2)
.endm

.macro FENCE_MEM
    SYS 0
.endm

.macro FENCE_SA
    SYS 1
.endm

.macro FENCE_ALL
    SYS 2
.endm

.macro KERNEL_END
    SYS 3
.endm
