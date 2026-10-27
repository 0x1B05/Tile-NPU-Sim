# Qwen3 NPU kernels. C ABI (rv64im lp64): args a0..a7, return a0.
# qwen3_ = Qwen3 NPU kernel.
# Dimensions are multiples of 32 unless noted; f32 data is 4-byte aligned.
# NPU register/descriptor usage is local to each kernel; scalar s-registers
# are preserved. Vector loads/stores bind descriptor vmN to register vN, so
# every streamed vector keeps its own descriptor.

    .include "npu.inc"
#include "dims.h"

    .bss
    .align 4
    .globl qwen3_accbuf
qwen3_accbuf: .skip VOCAB_CAP * 4

    .bss
    .align 3
    .globl qwen3_debug_qtile
qwen3_debug_qtile: .skip 8

    .text

# ---------------------------------------------------------------------------
# void qwen3_fence_mem(void)
#
# C-callable fence.mem: scalar readers of NPU-stored memory (cache_append,
# snapshot readers) call this before their loads.
# ---------------------------------------------------------------------------
    .globl qwen3_fence_mem
qwen3_fence_mem:
    FENCE_MEM
    ret

# ---------------------------------------------------------------------------
# uint32_t qwen3_quant_act(const float *x, uint32_t k)
#
# Quantizes x (f32[k], zero-padded by the caller to ceil(k/1024)*1024) with a
# uniform symmetric scale into ceil(k/1024) 32x32 i8 tile panels t2..t4
# (k <= 3072); zero x padding makes the padded tile rows contribute nothing.
# Returns the f32 activation scale bits. The panels stay resident in t2..t4
# for qwen3_matvec_q.
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_quant_act
qwen3_quant_act:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 2
    LI_CWV  t6, DT_F32, 1
    SETW    SEL_VC, 1, 30, X_T6     # vc30: len-1 store of the activation scale

    # activation absmax over the padded panels (v0 max, v1 chunk)
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    VFILL   0, 0
    addi    t6, a1, 1023
    srli    t6, t6, 10          # panels = ceil(k/1024)
    slli    t6, t6, 5           # padded chunk count = panels * 32
.Lqa_amax:
    VLOAD   1
    VABS    1, 1
    VMAX    0, 0, 1
    addi    t6, t6, -1
    bnez    t6, .Lqa_amax
    # v2 = absmax * (1/127): the uniform scale reduces into the len-1
    # scratch v30, broadcasts back as aquantx's external scale, and leaves
    # the vector domain through a one-lane store.
    VREDUCE 1, 30, 0            # v30[0] = absmax
    li      t1, 0x3C010204      # f32 bits of 1/127
    VMULX   30, 30, X_T1        # v30[0] = scale
    VBROADCAST 2, 30, 0         # v2 = scale in every lane
    addi    sp, sp, -16
    SETX    SEL_VM, 1, CF_BASE, 30, X_SP
    SETIV   SEL_VM, CF_STRIDE, 4, 30
    SETIV   SEL_VM, CF_VFLAGS, 0, 30
    VSTORE  30
    FENCE_MEM                   # the len-1 store must be visible to the lw
    lw      t2, 0(sp)           # x_scale bits reach scalars only via memory
    addi    sp, sp, 16

    # quantize the panels into t2..t4 (32x32 i8)
    LI_CWM  t3, DT_F32, 32, 32
    SETW    SEL_AC, 0, 0, X_T3
    SETX    SEL_AM, 0, CF_BASE, 0, X_A0
    SETIM   SEL_AM, CF_RSTRIDE, 128, 0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 1, 0        # inc along rows: += 32 * 128
    LI_CWM  t3, DT_I8, 32, 32
    SETW    SEL_TC, 0, 2, X_T3, 2
    addi    t3, a1, 1023
    srli    t3, t3, 10          # panels
    li      t4, 0
.Lqa_quant:
    ALOAD   0
    beqz    t4, .Lqa_q0
    li      t5, 1
    beq     t4, t5, .Lqa_q1
    AQUANTX 4, 0, 2
    j       .Lqa_next
.Lqa_q0:
    AQUANTX 2, 0, 2
    j       .Lqa_next
.Lqa_q1:
    AQUANTX 3, 0, 2
.Lqa_next:
    addi    t4, t4, 1
    bltu    t4, t3, .Lqa_quant

    # debug hook: dump the last quantized activation tile when requested
    la      t0, qwen3_debug_qtile
    ld      t1, 0(t0)
    beqz    t1, .Lqa_nodump
    SETX    SEL_TM, 0, CF_BASE, 2, X_T1
    SETIM   SEL_TM, CF_RSTRIDE, 32, 2
    SETIM   SEL_TM, CF_CSTRIDE, 1, 2
    SETIM   SEL_TM, CF_FLAGS, 0, 2
    TSTORE  2
.Lqa_nodump:
    mv      a0, t2
    ret

# ---------------------------------------------------------------------------
# void qwen3_matvec_q(const int8_t *w, const float *w_scales, float *out,
#                  uint32_t n, uint32_t k, uint32_t xs_bits)
#
# out[j] = xs * w_scales[j] * sum_k quant(x)[k] * W[j,k]
# W is stored as [n/32][k/32][32][32] tiles in [n][k] order, with per-row f32
# scales. quant(x) is the tile panels t2..t4 left resident by qwen3_quant_act;
# nothing between the two calls may clobber them. Cross-tile accumulation runs
# in an i32 memory buffer (qwen3_accbuf). n % 32 == 0, 32 | k, k <= 3072.
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_matvec_q
qwen3_matvec_q:
    addi    sp, sp, -48
    sd      s2, 0(sp)
    sd      s3, 8(sp)
    sd      s4, 16(sp)
    sd      s5, 24(sp)
    sd      s6, 32(sp)

    mv      s6, a5              # x_scale bits
    mv      a5, a4              # k
    mv      a4, a3              # n
    mv      a3, a2              # out

    # shared vector configs (len 32)
    LI_CWV  t0, DT_I32, 32
    SETW    SEL_VC, 1, 4, X_T0, 1
    LI_CWV  t0, DT_F32, 32
    SETW    SEL_VC, 1, 6, X_T0
    SETW    SEL_VC, 1, 8, X_T0

    # phase B configs: one [n][k] weight tile in t1 (tile row_stride = 32)
    LI_CWM  t0, DT_I8, 32, 32
    SETW    SEL_TC, 0, 1, X_T0
    SETIM   SEL_TM, CF_RSTRIDE, 32, 1
    SETIM   SEL_TM, CF_CSTRIDE, 1, 1
    SETIM   SEL_TM, CF_FLAGS, 0, 1
    LI_CWV  t0, DT_I8, 32
    SETW    SEL_BC, 1, 0, X_T0, 31
    la      t0, qwen3_accbuf
    SETX    SEL_VM, 1, CF_BASE, 4, X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, 4
    SETIV   SEL_VM, CF_VFLAGS, 0, 4
    SETX    SEL_VM, 1, CF_BASE, 6, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 6
    SETIV   SEL_VM, CF_VFLAGS, 1, 6
    SETX    SEL_VM, 1, CF_BASE, 8, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 8
    SETIV   SEL_VM, CF_VFLAGS, 1, 8

    # per panel: texrows once; per n-block: tload/bdot/vadd over the panel's
    # valid activation rows (k/32 per panel) into accbuf
    addi    s4, a5, 1023
    srli    s4, s4, 10          # panels
    li      s5, 0
.Lmv_panel:
    beqz    s5, .Lmv_tex0
    li      t5, 1
    beq     s5, t5, .Lmv_tex1
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    TEXROW  \R, 4, \R
    .endr
    j       .Lmv_texdone
.Lmv_tex0:
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    TEXROW  \R, 2, \R
    .endr
    j       .Lmv_texdone
.Lmv_tex1:
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    TEXROW  \R, 3, \R
    .endr
.Lmv_texdone:
    # t4 = valid activation rows in this panel: min(32, k/32 - panel*32)
    srli    t4, a5, 5
    slli    t5, s5, 5
    sub     t4, t4, t5
    li      t5, 32
    bltu    t4, t5, 1f
    mv      t4, t5
1:
    srli    s3, a4, 5           # n/32 blocks
    li      s2, 0
.Lmv_nblock:
    # tm1 base = w + (nb*kblocks + panel*32)*1024; advance one kblock per
    # loaded tile, since the tiled layout is [nblock][kblock][32][32].
    srli    t0, a5, 5
    mul     t0, s2, t0
    slli    t0, t0, 10
    add     t0, a0, t0
    slli    t1, s5, 15
    add     t0, t0, t1
    li      t1, 1024
    la      t2, qwen3_accbuf
    slli    t3, s2, 7
    add     t2, t2, t3
    SETX    SEL_VM, 1, CF_BASE, 4, X_T2
    beqz    s5, .Lmv_fresh
    VLOAD   4                   # continue accumulation across panels
    j       .Lmv_inner
.Lmv_fresh:
    VFILL   4, 0
.Lmv_inner:
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    li      t5, \R
    bgeu    t5, t4, .Lmv_inner_done
    SETX    SEL_TM, 0, CF_BASE, 1, X_T0
    TLOAD   1
    BDOT_NT_ACC 4, \R, 1
    add     t0, t0, t1
    .endr
.Lmv_inner_done:
    VSTORE  4
    addi    s2, s2, 1
    bltu    s2, s3, .Lmv_nblock
    addi    s5, s5, 1
    bltu    s5, s4, .Lmv_panel

    # epilogue: i32 -> f32, per-row weight scales, activation scale
    SETIV   SEL_VM, CF_VFLAGS, 1, 4       # accbuf descriptor now advances
    la      t0, qwen3_accbuf
    SETX    SEL_VM, 1, CF_BASE, 4, X_T0
    srli    s3, a4, 5
    li      s2, 0
.Lmv_epi:
    SETIV   SEL_VC, CF_DTYPE, DT_I32, 4
    VLOAD   4
    VCVT    4, 4                          # i32 -> f32, vc4 becomes f32
    VLOAD   6                             # 32 weight scales
    VMUL    4, 4, 6
    VMULX   4, 4, X_S6
    REG_MOVE 3, 2, 8, 4                    # vcopy v8, v4
    VSTORE  8
    addi    s2, s2, 1
    bltu    s2, s3, .Lmv_epi

    ld      s2, 0(sp)
    ld      s3, 8(sp)
    ld      s4, 16(sp)
    ld      s5, 24(sp)
    ld      s6, 32(sp)
    addi    sp, sp, 48
    FENCE_MEM                   # out (incl. k_buf/v_buf) is read by scalar code
    ret

# ---------------------------------------------------------------------------
# void qwen3_quant_rows(const float *x, uint32_t k, int8_t *xq, float *xs,
#                       uint32_t rows)
#
# Chunked sibling of qwen3_quant_act: x is [rows, k] f32 row-major (rows are
# tokens, up to 32). Pass 1 folds per-row absmax over the k/32 chunks
# (areduce.absmax + vmax) and stores xs = absmax * (1/127) as rows f32 scales;
# pass 2 quantizes each rows x 32 chunk with the per-row external scales into
# xq, [rows, k] i8 row-major. Rows whose absmax is 0 quantize to 0 (NaN
# quotient rule).
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_quant_rows
qwen3_quant_rows:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 1
    SETX    SEL_VC, 1, CF_LEN, 0, X_A4, 1
    LI_CWM  t6, DT_F32, 32, 32
    SETW    SEL_AC, 0, 0, X_T6
    SETX    SEL_AC, 0, CF_ROWS, 0, X_A4
    SETX    SEL_AM, 0, CF_BASE, 0, X_A0
    slli    t0, a1, 2
    SETX    SEL_AM, 0, CF_RSTRIDE, 0, X_T0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 2, 0        # inc along cols: += 32 * 4

    # pass 1: v0 = per-row absmax over all chunks (v1 = chunk absmax)
    srli    t6, a1, 5
    ALOAD   0
    AREDUCE 5, 0, 0
    addi    t6, t6, -1
    beqz    t6, .Lqr_scale
.Lqr_amax:
    ALOAD   0
    AREDUCE 5, 1, 0
    VMAX    0, 0, 1
    addi    t6, t6, -1
    bnez    t6, .Lqr_amax
.Lqr_scale:
    li      t0, 0x3C010204      # f32 bits of 1/127
    VMULX   0, 0, X_T0
    SETX    SEL_VM, 1, CF_BASE, 0, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 0, 0
    VSTORE  0                   # xs scales out; v0 stays for pass 2

    # pass 2: quantize chunks to xq (t0, tm0 inc along cols: += 32 * 1)
    LI_CWM  t6, DT_I8, 32, 32
    SETW    SEL_TC, 0, 0, X_T6
    SETX    SEL_TC, 0, CF_ROWS, 0, X_A4
    SETX    SEL_AM, 0, CF_BASE, 0, X_A0   # rewind x
    SETX    SEL_TM, 0, CF_BASE, 0, X_A2
    SETX    SEL_TM, 0, CF_RSTRIDE, 0, X_A1
    SETIM   SEL_TM, CF_CSTRIDE, 1, 0
    SETIM   SEL_TM, CF_FLAGS, 2, 0
    srli    t6, a1, 5
.Lqr_quant:
    ALOAD   0
    AQUANTX 0, 0, 0
    TSTORE  0
    addi    t6, t6, -1
    bnez    t6, .Lqr_quant
    ret

# ---------------------------------------------------------------------------
# void qwen3_matmul_q(const int8_t *w, const float *w_scales, const int8_t *xq,
#                  const float *xs, float *out, uint32_t n, uint32_t k,
#                  uint32_t rows)
#
# Chunked sibling of qwen3_matvec_q: out[t,j] = xs[t] * w_scales[j] *
# sum_c xq[t,c] * W[j,c] for t in [0,rows), computing one up-to-32-row token
# chunk. xq is [rows, k] i8 row-major (qwen3_quant_rows output), xs its rows
# row scales. Per 32-column block of out: t1 holds one [n][k] weight tile from
# the [n/32][k/32][32][32] layout and mma.nt accumulates the k/32 chunks in
# acc a0, then acc -> f32, * ws (bycol), * xs (byrow), stored via am8
# (auto-advances one block per store). Full groups of eight nblocks keep
# accelerators a0..a7 and reuse one xq tile across the group.
# n % 32 == 0, 32 | k, k <= 3072.
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_matmul_q
qwen3_matmul_q:
    addi    sp, sp, -48
    sd      s2, 0(sp)
    sd      s3, 8(sp)
    sd      s4, 16(sp)
    sd      s5, 24(sp)
    sd      s6, 32(sp)
    sd      s7, 40(sp)

    mv      s6, a2              # xq base
    mv      s7, a0              # weight base

    LI_CWM  t0, DT_I8, 32, 32
    SETW    SEL_TC, 0, 0, X_T0, 1
    SETX    SEL_TC, 0, CF_ROWS, 0, X_A7
    SETX    SEL_TM, 0, CF_RSTRIDE, 0, X_A6
    SETIM   SEL_TM, CF_CSTRIDE, 1, 0
    SETIM   SEL_TM, CF_FLAGS, 2, 0      # inc along cols: += 32 * 1
    SETIM   SEL_TM, CF_RSTRIDE, 32, 1
    SETIM   SEL_TM, CF_CSTRIDE, 1, 1
    SETIM   SEL_TM, CF_FLAGS, 0, 1
    LI_CWM  t0, DT_I32, 32, 32
    SETW    SEL_AC, 0, 0, X_T0, 8
    SETX    SEL_AC, 0, CF_ROWS, 0, X_A7, 8
    SETX    SEL_AM, 0, CF_BASE, 8, X_A4
    slli    t0, a5, 2
    SETX    SEL_AM, 0, CF_RSTRIDE, 8, X_T0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 8
    SETIM   SEL_AM, CF_FLAGS, 2, 8      # inc along cols: += 32 * 4
    LI_CWV  s3, DT_F32, 32
    SETW    SEL_VC, 1, 29, X_S3, 1    # vc29 (ws stream), vc30 (xs)
    SETX    SEL_VC, 1, CF_LEN, 30, X_A7
    SETX    SEL_VM, 1, CF_BASE, 29, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 29
    SETIV   SEL_VM, CF_VFLAGS, 1, 29    # ws stream, one block per load
    SETX    SEL_VM, 1, CF_BASE, 30, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 30
    SETIV   SEL_VM, CF_VFLAGS, 0, 30
    VLOAD   30                          # xs stays resident in v30

    srli    s3, a5, 5           # n/32 blocks
    srli    s4, a6, 5           # k/32 chunks
    li      s2, 0               # first nblock of the group
    li      t0, 32
    bne     a7, t0, .Lmm_nblock  # grouping only pays off for a full xq tile
.Lmm_group:
    sub     t0, s3, s2
    li      t1, 8
    bltu    t0, t1, .Lmm_nblock    # fewer than eight blocks remain

    # Full group: load xq once per k chunk and keep eight accumulators.
    SETX    SEL_TM, 0, CF_BASE, 0, X_S6
    mul     t0, s2, s4
    slli    t0, t0, 10
    add     t0, s7, t0              # first weight tile of this nblock group
    slli    t3, s4, 10              # next nblock stride within this kblock
    TLOAD   0
    mv      t4, t0
    .irp G,0,1,2,3,4,5,6,7
    SETX    SEL_TM, 0, CF_BASE, 1, X_T4
    TLOAD   1
    MMA_NT_ZERO \G, 0, 1
    add     t4, t4, t3
    .endr
    addi    t0, t0, 1024            # next kblock for the group base
    li      s5, 1
.Lmm_group_kloop:
    bgeu    s5, s4, .Lmm_group_epi
    TLOAD   0
    mv      t4, t0
    .irp G,0,1,2,3,4,5,6,7
    SETX    SEL_TM, 0, CF_BASE, 1, X_T4
    TLOAD   1
    MMA_NT_ACC \G, 0, 1
    add     t4, t4, t3
    .endr
    addi    t0, t0, 1024
    addi    s5, s5, 1
    j       .Lmm_group_kloop
.Lmm_group_epi:
    .irp G,0,1,2,3,4,5,6,7
    VLOAD   29
    SETIM   SEL_AC, CF_DTYPE, DT_I32, 8
    ACVT    8, \G
    AMULV_BYCOL 8, 8, 29
    AMULV_BYROW 8, 8, 30
    ASTORE  8
    .endr
    addi    s2, s2, 8
    bltu    s2, s3, .Lmm_group
    j       .Lmm_done

.Lmm_nblock:
    # Tail: fewer than eight output blocks remain.
    SETX    SEL_TM, 0, CF_BASE, 0, X_S6
    mul     t0, s2, s4
    slli    t0, t0, 10
    add     t0, s7, t0
    li      t1, 1024
    SETX    SEL_TM, 0, CF_BASE, 1, X_T0
    TLOAD   0
    TLOAD   1
    MMA_NT_ZERO 0, 0, 1
    add     t0, t0, t1
    li      s5, 1
.Lmm_tail_kloop:
    bgeu    s5, s4, .Lmm_tail_epi
    SETX    SEL_TM, 0, CF_BASE, 1, X_T0
    TLOAD   0
    TLOAD   1
    MMA_NT_ACC 0, 0, 1
    add     t0, t0, t1
    addi    s5, s5, 1
    j       .Lmm_tail_kloop
.Lmm_tail_epi:
    VLOAD   29
    SETIM   SEL_AC, CF_DTYPE, DT_I32, 8
    ACVT    8, 0
    AMULV_BYCOL 8, 8, 29
    AMULV_BYROW 8, 8, 30
    ASTORE  8
    addi    s2, s2, 1
    bltu    s2, s3, .Lmm_nblock
.Lmm_done:
    ld      s2, 0(sp)
    ld      s3, 8(sp)
    ld      s4, 16(sp)
    ld      s5, 24(sp)
    ld      s6, 32(sp)
    ld      s7, 40(sp)
    addi    sp, sp, 48
    FENCE_MEM                   # out (incl. k_buf/v_buf) is read by scalar code
    ret

# ---------------------------------------------------------------------------
# void qwen3_rmsnorm(float *x, const float *gamma, float *out, uint32_t n,
#                 uint32_t eps_bits, uint32_t inv_n_bits)
# out = x * rsqrt(mean(x^2) + eps) * gamma, n % 32 == 0.
# ---------------------------------------------------------------------------
    .globl qwen3_rmsnorm
qwen3_rmsnorm:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 4
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    VFILL   0, 0
    srli    t6, a3, 5
.Lrn_sum:
    VLOAD   1
    VFMADD  0, 1, 1, 0
    addi    t6, t6, -1
    bnez    t6, .Lrn_sum
    VREDUCE 0, 2, 0                       # v2 = sumsq broadcast
    VMULX   2, 2, X_A5                    # mean = sum * (1/n)
    VADDX   2, 2, X_A4                    # + eps
    VRSQRT  2, 2
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0   # rewind x
    SETX    SEL_VM, 1, CF_BASE, 3, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 3
    SETIV   SEL_VM, CF_VFLAGS, 1, 3
    SETX    SEL_VM, 1, CF_BASE, 4, X_A2
    SETIV   SEL_VM, CF_STRIDE, 4, 4
    SETIV   SEL_VM, CF_VFLAGS, 1, 4
    srli    t6, a3, 5
.Lrn_out:
    VLOAD   1
    VLOAD   3
    VMUL    3, 3, 2
    VMUL    1, 1, 3
    REG_MOVE 3, 2, 4, 1                    # vcopy v4, v1
    VSTORE  4
    addi    t6, t6, -1
    bnez    t6, .Lrn_out
    ret

# ---------------------------------------------------------------------------
# void qwen3_rope(float *x, const float *cos_h, const float *sin_h, uint32_t hd)
# In-place rotate-half RoPE on one head. cos_h/sin_h point at the position's
# row (hd/2 f32 each). Requires hd/2 <= 64; the half is processed in 32-lane
# chunks (a single sub-32 chunk when hd/2 < 32).
# ---------------------------------------------------------------------------
    .globl qwen3_rope
qwen3_rope:
    srli    t6, a3, 1                     # half = hd/2
    li      t5, 32
    bltu    t6, t5, 1f
    li      t6, 32                        # lane count per chunk
1:
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    SETIV   SEL_VC, CF_DTYPE, DT_F32, \R
    SETX    SEL_VC, 1, CF_LEN, (0 + \R), X_T6
    .endr
    # stream descriptors (stride 4, inc): cos vm0/vm1, sin vm2/vm3,
    # x1 vm4/vm5, x2 vm6/vm7; x2 starts half elements into x (hd*2 bytes).
    # Broadcast stride/inc first: cfg.copy copies every field, base included.
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 1, 0, 6          # vm1..vm7 = vm0 stride/inc
    SETX    SEL_VM, 1, CF_BASE, 0, X_A1
    addi    t0, a1, 128
    SETX    SEL_VM, 1, CF_BASE, 1, X_T0
    SETX    SEL_VM, 1, CF_BASE, 2, X_A2
    addi    t0, a2, 128
    SETX    SEL_VM, 1, CF_BASE, 3, X_T0
    SETX    SEL_VM, 1, CF_BASE, 4, X_A0
    addi    t0, a0, 128
    SETX    SEL_VM, 1, CF_BASE, 5, X_T0
    slli    t0, a3, 1
    add     t0, a0, t0
    SETX    SEL_VM, 1, CF_BASE, 6, X_T0
    addi    t0, t0, 128
    SETX    SEL_VM, 1, CF_BASE, 7, X_T0
    # store descriptors vm12..vm15: base = x + c*128, stride 4, no inc
    mv      t0, a0
    .irp R,0,1,2,3
    SETX    SEL_VM, 1, CF_BASE, (12 + \R), X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, (12 + \R)
    SETIV   SEL_VM, CF_VFLAGS, 0, (12 + \R)
    addi    t0, t0, 128
    .endr
    srli    t5, a3, 6                     # 32-lane chunks per half
    li      t1, 2
    beq     t5, t1, .Lrp_half64           # only hd = 128 has two chunks
    # single (possibly short) chunk per half
    VLOAD   0                             # cos
    VLOAD   2                             # sin
    VLOAD   4                             # x1
    VLOAD   6                             # x2
    VMUL    8, 4, 0
    VMUL    9, 6, 2
    VSUB    12, 8, 9                      # out1
    VMUL    8, 6, 0
    VMUL    9, 4, 2
    VADD    13, 8, 9                      # out2
    VSTORE  12                            # vm12 = x
    slli    t0, a3, 1
    add     t0, a0, t0
    SETX    SEL_VM, 1, CF_BASE, 13, X_T0  # vm13 = x + half
    VSTORE  13
    ret
.Lrp_half64:
    # two 32-lane chunks per half (hd = 128)
    VLOAD   0                             # cos c0
    VLOAD   1                             # cos c1
    VLOAD   2                             # sin c0
    VLOAD   3                             # sin c1
    VLOAD   4                             # x1 c0
    VLOAD   5                             # x1 c1
    VLOAD   6                             # x2 c0
    VLOAD   7                             # x2 c1
    VMUL    8, 4, 0
    VMUL    9, 6, 2
    VSUB    12, 8, 9                      # out1 c0
    VMUL    8, 5, 1
    VMUL    9, 7, 3
    VSUB    13, 8, 9                      # out1 c1
    VMUL    8, 6, 0
    VMUL    9, 4, 2
    VADD    14, 8, 9                      # out2 c0
    VMUL    8, 7, 1
    VMUL    9, 5, 3
    VADD    15, 8, 9                      # out2 c1
    VSTORE  12
    VSTORE  13
    VSTORE  14
    VSTORE  15
    ret

# ---------------------------------------------------------------------------
# void qwen3_scores(const float *q, const char *k_base, uint32_t row_stride,
#                float *scores, uint32_t n_keys, uint32_t hd)
# scores[j] = dot(q, K_j) over hd f32; K_j at k_base + j*row_stride.
# ---------------------------------------------------------------------------
    .globl qwen3_scores
qwen3_scores:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 20
    srli    t6, a5, 5                     # chunks = hd / 32
    # q chunk descriptors vm8..vm11 (base = q + c*128, no inc)
    mv      t0, a0
    .irp R,0,1,2,3
    SETX    SEL_VM, 1, CF_BASE, (8 + \R), X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, (8 + \R)
    SETIV   SEL_VM, CF_VFLAGS, 0, (8 + \R)
    addi    t0, t0, 128
    .endr
    # K chunk stream vm12: base per key, auto-inc within the key
    SETIV   SEL_VM, CF_STRIDE, 4, 12
    SETIV   SEL_VM, CF_VFLAGS, 1, 12
    # one-lane score store: vc17/vm17 are fixed except the base per key
    LI_CWV  t5, DT_F32, 1
    SETW    SEL_VC, 1, 17, X_T5
    SETIV   SEL_VM, CF_STRIDE, 4, 17
    SETIV   SEL_VM, CF_VFLAGS, 0, 17
    VLOAD   8
    li      t5, 1
    beq     t6, t5, .Lsc_qdone
    VLOAD   9
    li      t5, 2
    beq     t6, t5, .Lsc_qdone
    VLOAD   10
    li      t5, 3
    beq     t6, t5, .Lsc_qdone
    VLOAD   11
.Lsc_qdone:
    mv      t4, zero                      # key index
.Lsc_key:
    VFILL   16, 0
    mul     t0, t4, a2
    add     t0, a1, t0
    SETX    SEL_VM, 1, CF_BASE, 12, X_T0
    VLOAD   12
    VFMADD  16, 12, 8, 16
    li      t5, 1
    beq     t6, t5, .Lsc_dot
    VLOAD   12
    VFMADD  16, 12, 9, 16
    li      t5, 2
    beq     t6, t5, .Lsc_dot
    VLOAD   12
    VFMADD  16, 12, 10, 16
    li      t5, 3
    beq     t6, t5, .Lsc_dot
    VLOAD   12
    VFMADD  16, 12, 11, 16
.Lsc_dot:
    VREDUCE 0, 17, 16                   # v17[0] = dot (len-1 scratch)
    slli    t1, t4, 2
    add     t1, a3, t1
    SETX    SEL_VM, 1, CF_BASE, 17, X_T1
    VSTORE  17                          # one-lane score store
    addi    t4, t4, 1
    bltu    t4, a4, .Lsc_key
    ret

# ---------------------------------------------------------------------------
# void qwen3_softmax_exp(float *scores, uint32_t n, uint32_t scale_log2e_bits,
#                     uint32_t *rcp_out)
# In place: s[i] = exp2((s[i] - max) * scale_log2e); *rcp_out = rcp(sum).
# n % 32 == 0 (the caller pads the score tail with -inf).
# ---------------------------------------------------------------------------
    .globl qwen3_softmax_exp
qwen3_softmax_exp:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 4
    LI_CWV  t6, DT_F32, 1
    SETW    SEL_VC, 1, 4, X_T6     # vc4: len-1 store of the rcp result
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    SETIV   SEL_VM, CF_STRIDE, 4, 3
    SETIV   SEL_VM, CF_VFLAGS, 1, 3
    li      t0, 0xFF800000                # -inf
    VFILLX  0, X_T0
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0
    srli    t6, a1, 5
.Lsm_max:
    VLOAD   1
    VMAX    0, 0, 1
    addi    t6, t6, -1
    bnez    t6, .Lsm_max
    VREDUCE 1, 0, 0                       # v0 = max f32 bits in every lane
    VFILL   2, 0
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0   # rewind load
    SETX    SEL_VM, 1, CF_BASE, 3, X_A0   # store stream
    srli    t6, a1, 5
.Lsm_exp:
    VLOAD   1
    VSUB    1, 1, 0
    VMULX   1, 1, X_A2
    VEXP2   1, 1
    VADD    2, 2, 1
    REG_MOVE 3, 2, 3, 1                    # vcopy v3, v1
    VSTORE  3
    addi    t6, t6, -1
    bnez    t6, .Lsm_exp
    VREDUCE 0, 4, 2                       # v4 = sum in every lane
    VRCP    4, 4
    SETX    SEL_VM, 1, CF_BASE, 4, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 4
    SETIV   SEL_VM, CF_VFLAGS, 0, 4
    VSTORE  4                             # *rcp_out = rcp(sum)
    FENCE_MEM                   # scores and rcp are read by scalar code
    ret

# ---------------------------------------------------------------------------
# void qwen3_pv(const char *v_base, uint32_t row_stride, const float *probs,
#            float *out, uint32_t n_keys, uint32_t hd, uint32_t rcp_bits)
# out[c] = rcp * sum_j probs[j] * V_j[c], V_j at v_base + j*row_stride.
# ---------------------------------------------------------------------------
    .globl qwen3_pv
qwen3_pv:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 20
    srli    t6, a5, 5                     # chunks
    SETIV   SEL_VM, CF_STRIDE, 4, 12
    SETIV   SEL_VM, CF_VFLAGS, 1, 12
    VFILL   16, 0
    VFILL   17, 0
    VFILL   18, 0
    VFILL   19, 0
    mv      t4, zero
    mv      t3, a2                        # probs cursor
.Lpv_key:
    lw      t0, 0(t3)
    addi    t3, t3, 4
    VINSERT 20, 0, X_T0
    VBROADCAST 20, 20, 0
    mul     t0, t4, a1
    add     t0, a0, t0
    SETX    SEL_VM, 1, CF_BASE, 12, X_T0
    VLOAD   12
    VFMADD  16, 12, 20, 16
    li      t5, 1
    beq     t6, t5, .Lpv_next
    VLOAD   12
    VFMADD  17, 12, 20, 17
    li      t5, 2
    beq     t6, t5, .Lpv_next
    VLOAD   12
    VFMADD  18, 12, 20, 18
    li      t5, 3
    beq     t6, t5, .Lpv_next
    VLOAD   12
    VFMADD  19, 12, 20, 19
.Lpv_next:
    addi    t4, t4, 1
    bltu    t4, a4, .Lpv_key
    # out descriptors vm16..vm19 (one per accumulator, no inc)
    mv      t0, a3
    .irp R,0,1,2,3
    SETX    SEL_VM, 1, CF_BASE, (16 + \R), X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, (16 + \R)
    SETIV   SEL_VM, CF_VFLAGS, 0, (16 + \R)
    addi    t0, t0, 128
    .endr
    VMULX   16, 16, X_A6
    VSTORE  16
    li      t5, 1
    beq     t6, t5, .Lpv_done
    VMULX   17, 17, X_A6
    VSTORE  17
    li      t5, 2
    beq     t6, t5, .Lpv_done
    VMULX   18, 18, X_A6
    VSTORE  18
    li      t5, 3
    beq     t6, t5, .Lpv_done
    VMULX   19, 19, X_A6
    VSTORE  19
.Lpv_done:
    ret

# ---------------------------------------------------------------------------
# void qwen3_silu_mul(float *gate, const float *up, float *out, uint32_t n)
# out = silu(gate) * up, n % 32 == 0.
# ---------------------------------------------------------------------------
    .globl qwen3_silu_mul
qwen3_silu_mul:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 2
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 1, 0, 1          # vm1, vm2 = vm0 stride/inc
    SETX    SEL_VM, 1, CF_BASE, 0, X_A0
    SETX    SEL_VM, 1, CF_BASE, 1, X_A1
    SETX    SEL_VM, 1, CF_BASE, 2, X_A2
    li      t0, 0xBFB8AA3B                 # -log2(e), f32 bits
    li      t1, 0x3F800000                 # 1.0f, f32 bits
    srli    t6, a3, 5
.Lsl_loop:
    VLOAD   0
    VMULX   1, 0, X_T0                     # -x * log2(e)
    VEXP2   1, 1
    VADDX   1, 1, X_T1
    VRCP    1, 1                           # 1 / (1 + exp(-x))
    VMUL    0, 0, 1
    VLOAD   1
    VMUL    0, 0, 1
    REG_MOVE 3, 2, 2, 0                    # vcopy v2, v0
    VSTORE  2
    addi    t6, t6, -1
    bnez    t6, .Lsl_loop
    ret

# ---------------------------------------------------------------------------
# void qwen3_add(float *dst, const float *src, uint32_t n)   dst += src
# ---------------------------------------------------------------------------
    .globl qwen3_add
qwen3_add:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 2
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 1, 0, 1          # vm1, vm2 = vm0 stride/inc
    SETX    SEL_VM, 1, CF_BASE, 0, X_A0
    SETX    SEL_VM, 1, CF_BASE, 1, X_A1
    SETX    SEL_VM, 1, CF_BASE, 2, X_A0
    srli    t6, a2, 5
.Lad_loop:
    VLOAD   0
    VLOAD   1
    VADD    0, 0, 1
    REG_MOVE 3, 2, 2, 0                    # vcopy v2, v0
    VSTORE  2
    addi    t6, t6, -1
    bnez    t6, .Lad_loop
    ret

# ---------------------------------------------------------------------------
# void qwen3_embed(const int8_t *table, const float *scales, uint32_t id,
#               float *out, uint32_t hidden, uint32_t vocab)
# out = dequant(table[id]) with the row's f32 scale. The embedding table is
# q8s32 with one f32 scale per vocabulary row. It is stored as 32x32 tiles in
# [V/32][D/32][32][32] order, each tile in [V][D] order. For token id i,
# V-block i/32 and row i%32 select the token; its D/32 blocks are 1024 bytes
# apart. One 32x32 Tile therefore holds one token's hidden vector (D <= 1024).
# ---------------------------------------------------------------------------
    .globl qwen3_embed
qwen3_embed:
    srli    t4, a4, 5               # dblocks = D / 32
    srli    t0, a2, 5               # vblock = id / 32
    andi    t1, a2, 31              # vinner = id % 32
    slli    t1, t1, 5               # row offset in a tile
    mul     t5, t0, t4              # vblock * dblocks
    slli    t5, t5, 10
    add     t5, a0, t5
    add     t5, t5, t1              # first D-block row for id
    slli    t0, a2, 2
    add     t0, a1, t0
    lw      t2, 0(t0)               # t2 = scales[id]

    li      t0, (DT_I8 << 16) | (32 << 8)
    or      t0, t0, t4
    SETW    SEL_TC, 0, 0, X_T0      # tc0 = i8 dblocks x 32
    li      t0, (DT_F32 << 16) | (32 << 8)
    or      t0, t0, t4
    SETW    SEL_AC, 0, 0, X_T0      # ac0 = f32 dblocks x 32
    li      t0, DT_F32 << 8
    or      t0, t0, t4
    SETW    SEL_VC, 1, 0, X_T0      # vc0 = f32 len dblocks
    VFILLX  0, X_T2                 # duplicate the token scale in v0
    SETIM   SEL_TM, CF_RSTRIDE, 1024, 0
    SETIM   SEL_TM, CF_CSTRIDE, 1, 0
    SETIM   SEL_TM, CF_FLAGS, 0, 0
    SETX    SEL_TM, 0, CF_BASE, 0, X_T5
    SETX    SEL_AM, 0, CF_BASE, 0, X_A3
    SETIM   SEL_AM, CF_RSTRIDE, 128, 0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 0, 0
    TLOAD   0                       # t0 = dblocks x 32, one token
    TDEQUANT 0, 0, 0                # a0 = f32(t0) * token scale
    ASTORE  0                       # contiguous f32 hidden vector
    ret

# ---------------------------------------------------------------------------
# uint32_t qwen3_argmax(const float *x, uint32_t n)   n % 32 == 0
# Greedy index of the maximum with the ISA argmax rules: NaN candidates are
# ignored, numeric ties keep the smaller index, and an all-NaN chunk contributes
# index 0 with a NaN value. Chunk 0 seeds the running best; later chunks fold
# in through vcmplt plus a mask that replaces an all-NaN best with the first
# numeric candidate. The best index reaches a0 through a one-lane vstore +
# fence + lwu.
# ---------------------------------------------------------------------------
    .globl qwen3_argmax
qwen3_argmax:
    addi    sp, sp, -16
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 0, X_T6, 1   # vc0/vc1 f32: best value, chunk
    LI_CWV  t6, DT_U32, 32
    SETW    SEL_VC, 1, 2, X_T6, 1   # vc2/vc3 u32: best index, candidate index
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 1, 4, X_T6      # vc4 f32: candidate value
    LI_CWV  t6, DT_U32, 32
    SETW    SEL_VC, 1, 5, X_T6, 2   # vc5..vc7 u32: compare/NaN masks
    SETX    SEL_VM, 1, CF_BASE, 1, X_A0
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    li      t4, 32                        # next chunk base
    srli    t6, a1, 5
    VLOAD   1
    VREDUCE_ARGMAX 2, 0, 1                # best = chunk 0 (index 0 if all NaN)
    addi    t6, t6, -1
    beqz    t6, .Lam_done
.Lam_loop:
    VLOAD   1
    VREDUCE_ARGMAX 3, 4, 1                # v3 = chunk argmax (u32), v4 = value
    VADDX   3, 3, X_T4                    # global candidate index
    VCMPLT    5, 0, 4                     # numeric: best < candidate
    VCMPUNORD 6, 0, 0                     # best is NaN (all-NaN chunk)
    VCMPUNORD 7, 4, 4                     # candidate is NaN
    VNOT      7, 7                        # candidate is numeric
    VAND      6, 6, 7                     # replace an all-NaN best
    VOR       5, 5, 6                     # greater numeric, or first numeric after NaN
    VSELECT   0, 5, 4, 0
    VSELECT   2, 5, 3, 2
    addi    t4, t4, 32
    addi    t6, t6, -1
    bnez    t6, .Lam_loop
.Lam_done:
    LI_CWV  t6, DT_U32, 1
    SETW    SEL_VC, 1, 5, X_T6      # vc5: len-1 exit store
    SETX    SEL_VM, 1, CF_BASE, 5, X_SP
    SETIV   SEL_VM, CF_STRIDE, 4, 5
    SETIV   SEL_VM, CF_VFLAGS, 0, 5
    VBROADCAST 5, 2, 0
    VSTORE  5
    FENCE_MEM                   # the index store must be visible to the lwu
    lwu     a0, 0(sp)
    addi    sp, sp, 16
    ret
