# Qwen3 kernel macros: config field numbers and instruction forms beyond
# tests/guest/npu.inc. Field numbers follow the manual's field3 table and
# src/decode/encoding.h layouts.

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

# cfg.seti on the matrix side (major4 = 5).
.macro SETIM sel2, field, imm, c5
    CFG_SETI 5, \sel2, \field, \imm, \c5
.endm

# cfg.seti on the vector side (major4 = 6).
.macro SETIV sel2, field, imm, c5
    CFG_SETI 6, \sel2, \field, \imm, \c5
.endm

# cfg.setx: C[field] = x[xS]. MV = 0 matrix side, 1 vector side.
.macro SETX sel2, mv, field, c5, xs
    CFG_REG \sel2, \mv, 0, \field, \xs, \c5
.endm

# ---- Vec32 elementwise forms (E_BIN, dom2 = 3) -------------------------
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

# ---- Vec32 init / unary -------------------------------------------------
.macro VFILL d5, imm
    INIT_FILL 3, \d5, \imm
.endm
.macro VFILLX d5, xs
    INIT_R2 3, 1, \d5, \xs
.endm
.macro VABS d5, s5
    E_UNARY 3, 1, \d5, \s5
.endm

# tquant with an external per-row scale (QNT funct3 = 1).
.macro TQUANTX d5, vscale, as
    QNT 0, 0, 1, \vscale, 0, \as, \d5
.endm

# mma.nt forms.
.macro MMA_NT_ZERO d5, a5, b5
    MMA_BDOT 2, \d5, \a5, \b5
.endm
