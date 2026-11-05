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
# uniform symmetric scale into ceil(k/1024) 32x32 i8 quant blocks t2..t4
# (k <= 3072); zero x padding makes the padded tile rows contribute nothing.
# Returns the f32 activation scale bits. The quant blocks stay resident in t2..t4
# for qwen3_matvec_q.
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_quant_act
qwen3_quant_act:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6, 2
    LI_CWV  t6, DT_F32, 1
    SETW    SEL_VC, 30, X_T6     # vc30: len-1 store of the activation scale

    # activation absmax over the padded activation blocks (v0 max, v1 = current 32-lane chunk)
    SETX    SEL_VM, CF_BASE, 1, X_A0
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    VFILL   0, 0
    addi    t6, a1, 1023
    srli    t6, t6, 10          # x31 (t6) = n_blocks = ceil(k/1024)
    slli    t6, t6, 5           # padded kblock count = n_blocks * 32
.Lqa_amax:
    VLOAD   1
    VABS    1, 1
    VMAX    0, 0, 1
    addi    t6, t6, -1
    bnez    t6, .Lqa_amax
    # v2 = absmax * (1/127): the uniform scale reduces into the len-1
    # scratch v30, broadcasts back as aquantx's scale, and leaves
    # the vector domain through a one-lane store.
    VREDUCE 1, 30, 0            # v30[0] = absmax
    li      t1, 0x3C010204      # f32 bits of 1/127
    VMULX   30, 30, X_T1        # v30[0] = scale
    VBROADCAST 2, 30, 0         # v2 = scale in every lane
    addi    sp, sp, -16
    SETX    SEL_VM, CF_BASE, 30, X_SP
    SETIV   SEL_VM, CF_STRIDE, 4, 30
    SETIV   SEL_VM, CF_VFLAGS, 0, 30
    VSTORE  30
    FENCE_MEM                   # the len-1 store must be visible to the lw
    lw      t2, 0(sp)           # x_scale bits reach scalars only via memory
    addi    sp, sp, 16

    # quantize the activation blocks into t2..t4 (32x32 i8 quant blocks)
    LI_CWM  t3, DT_F32, 32, 32
    SETW    SEL_AC, 0, X_T3
    SETX    SEL_AM, CF_BASE, 0, X_A0
    SETIM   SEL_AM, CF_RSTRIDE, 128, 0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 1, 0        # inc along rows: += 32 * 128
    LI_CWM  t3, DT_I8, 32, 32
    SETW    SEL_TC, 2, X_T3, 2
    addi    t3, a1, 1023
    srli    t3, t3, 10          # x28 (t3) = n_blocks
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
    SETX    SEL_TM, CF_BASE, 2, X_T1
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
# scales. quant(x) is the quant blocks t2..t4 left resident by qwen3_quant_act;
# nothing between the two calls may clobber them. Cross-tile accumulation runs
# in an i32 memory buffer (qwen3_accbuf); the last activation block's last kblock
# uses vmma.nt.acc followed by vdequant (vCol v6, vRow v7) to write f32 output
# directly. n % 32 == 0, 32 | k, k <= 3072.
# ---------------------------------------------------------------------------

# Per-activation-block n-block loop. tile_a names the activation block, which must be a
# compile-time register number because vmma.nt encodes tA and its row in
# the instruction word. One expansion per activation block keeps the loop body free of a
# runtime Tile->Tile copy.
.macro Q3_MATVEC_NBLOCK tile_a
    # x29 (t4) = valid kblocks in this activation block:
    #   min(32, k/32 - ablock*32)
    srli    t4, a5, 5
    slli    t5, s5, 5
    sub     t4, t4, t5
    li      t5, 32
    bltu    t4, t5, 1f
    mv      t4, t5
1:
    srli    s3, a4, 5           # n/32 blocks
    li      s2, 0
2:
    SETIV   SEL_VC, CF_DTYPE, DT_I32, 4
    # x5 (t0) = w + (nb*kblocks + ablock*32)*1024; advance one kblock per
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
    SETX    SEL_VM, CF_BASE, 4, X_T2
    beqz    s5, 3f
    VLOAD   4                   # continue accumulation across activation blocks
    j       4f
3:
    VFILL   4, 0
4:
    .irp R,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
    li      t5, \R
    bgeu    t5, t4, 5f
    SETX    SEL_TM, CF_BASE, 1, X_T0
    TLOAD   1
    addi    t6, t4, -1
    bne     t5, t6, 4f
    addi    t6, s4, -1
    bne     s5, t6, 4f
    VLOAD   6
    VMMA_NT_ACC 4, \tile_a, \R, 1
    VDEQUANT 8, 4, 6, 7
    VSTORE  8
    j       5f
4:
    VMMA_NT_ACC 4, \tile_a, \R, 1
5:
    add     t0, t0, t1
    .endr
    addi    t6, s4, -1
    beq     s5, t6, 6f
    VSTORE  4
6:
    addi    s2, s2, 1
    bltu    s2, s3, 2b
.endm

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
    SETW    SEL_VC, 4, X_T0, 1
    LI_CWV  t0, DT_F32, 32
    SETW    SEL_VC, 6, X_T0
    SETW    SEL_VC, 8, X_T0
    SETIM   SEL_VC, CF_DTYPE, DT_F32, 7
    SETIM   SEL_VC, CF_LEN, 1, 7
    mv      t0, s6
    VINSERT 7, 0, 5

    # phase B configs: one [n][k] weight tile in t1 (tile row_stride = 32)
    LI_CWM  t0, DT_I8, 32, 32
    SETW    SEL_TC, 1, X_T0
    SETIM   SEL_TM, CF_RSTRIDE, 32, 1
    SETIM   SEL_TM, CF_CSTRIDE, 1, 1
    SETIM   SEL_TM, CF_FLAGS, 0, 1
    la      t0, qwen3_accbuf
    SETX    SEL_VM, CF_BASE, 4, X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, 4
    SETIV   SEL_VM, CF_VFLAGS, 0, 4
    SETX    SEL_VM, CF_BASE, 6, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 6
    SETIV   SEL_VM, CF_VFLAGS, 1, 6
    SETX    SEL_VM, CF_BASE, 8, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 8
    SETIV   SEL_VM, CF_VFLAGS, 1, 8

    # per activation block: accumulate its valid kblocks (up to 32) directly
    # from the resident Tile, one static tA per activation block.
    addi    s4, a5, 1023
    srli    s4, s4, 10          # s4 = n_blocks = ceil(k/1024)
    li      s5, 0               # activation block index
.Lmv_ablock:
    beqz    s5, .Lmv_ablock0
    li      t5, 1
    beq     s5, t5, .Lmv_ablock1
    Q3_MATVEC_NBLOCK 4
    j       .Lmv_ablock_done
.Lmv_ablock0:
    Q3_MATVEC_NBLOCK 2
    j       .Lmv_ablock_done
.Lmv_ablock1:
    Q3_MATVEC_NBLOCK 3
.Lmv_ablock_done:
    addi    s5, s5, 1
    bltu    s5, s4, .Lmv_ablock

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
# Per-kblock sibling of qwen3_quant_act: x is [rows, k] f32 row-major (rows are
# tokens, up to 32). Pass 1 folds per-row absmax over the k/32 kblocks
# (areduce.absmax + vmax) and stores xs = absmax * (1/127) as rows f32 scales;
# pass 2 quantizes each rows x 32 kblock with the per-row scales into
# xq, [rows, k] i8 row-major. Rows whose absmax is 0 quantize to 0 (NaN
# quotient rule).
# ---------------------------------------------------------------------------
    .text
    .globl qwen3_quant_rows
qwen3_quant_rows:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6, 1
    SETX    SEL_VC, CF_LEN, 0, X_A4, 1
    LI_CWM  t6, DT_F32, 32, 32
    SETW    SEL_AC, 0, X_T6
    SETX    SEL_AC, CF_ROWS, 0, X_A4
    SETX    SEL_AM, CF_BASE, 0, X_A0
    slli    t0, a1, 2
    SETX    SEL_AM, CF_RSTRIDE, 0, X_T0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 2, 0        # inc along cols: += 32 * 4

    # pass 1: v0 = per-row absmax over all kblocks (v1 = kblock absmax)
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
    SETX    SEL_VM, CF_BASE, 0, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 0, 0
    VSTORE  0                   # xs scales out; v0 stays for pass 2

    # pass 2: quantize kblocks to xq (Tile t0, tm0 inc along cols: += 32 * 1)
    LI_CWM  t6, DT_I8, 32, 32
    SETW    SEL_TC, 0, X_T6
    SETX    SEL_TC, CF_ROWS, 0, X_A4
    SETX    SEL_AM, CF_BASE, 0, X_A0   # rewind x
    SETX    SEL_TM, CF_BASE, 0, X_A2
    SETX    SEL_TM, CF_RSTRIDE, 0, X_A1
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
# the [n/32][k/32][32][32] layout and mma.nt accumulates the k/32 kblocks in
# acc. The last (or only) kblock loads the weight scales into v29 and issues
# mma.nt.acc immediately followed by adequant (vCol v29, vRow v30), so the pair
# can be detected/fused; the f32 result goes to a8 and is stored through am8.
# Full groups of eight nblocks keep accelerators a0..a7 and reuse one xq tile
# across the group. n % 32 == 0, 32 | k, k <= 3072.
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
    SETW    SEL_TC, 0, X_T0, 1
    SETX    SEL_TC, CF_ROWS, 0, X_A7
    SETX    SEL_TM, CF_RSTRIDE, 0, X_A6
    SETIM   SEL_TM, CF_CSTRIDE, 1, 0
    SETIM   SEL_TM, CF_FLAGS, 2, 0      # inc along cols: += 32 * 1
    SETIM   SEL_TM, CF_RSTRIDE, 32, 1
    SETIM   SEL_TM, CF_CSTRIDE, 1, 1
    SETIM   SEL_TM, CF_FLAGS, 0, 1
    LI_CWM  t0, DT_I32, 32, 32
    SETW    SEL_AC, 0, X_T0, 7
    SETX    SEL_AC, CF_ROWS, 0, X_A7, 7
    SETX    SEL_AM, CF_BASE, 0, X_A4
    slli    t0, a5, 2
    SETX    SEL_AM, CF_RSTRIDE, 0, X_T0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 0, 0      # every block sets its own base
    LI_CWM  t0, DT_F32, 32, 32
    SETW    SEL_AC, 8, X_T0
    SETX    SEL_AC, CF_ROWS, 8, X_A7
    COPY    SEL_AM, 1, 0, 7
    LI_CWV  s3, DT_F32, 32
    SETW    SEL_VC, 29, X_S3, 1    # vc29 (ws stream), vc30 (xs)
    SETX    SEL_VC, CF_LEN, 30, X_A7
    SETX    SEL_VM, CF_BASE, 29, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 29
    SETIV   SEL_VM, CF_VFLAGS, 1, 29    # ws stream, one block per load
    SETX    SEL_VM, CF_BASE, 30, X_A3
    SETIV   SEL_VM, CF_STRIDE, 4, 30
    SETIV   SEL_VM, CF_VFLAGS, 0, 30
    VLOAD   30                          # xs stays resident in v30

    srli    s3, a5, 5           # n/32 blocks
    srli    s4, a6, 5           # kblocks = k/32
    li      s2, 0               # first nblock of the group
    li      t0, 32
    bne     a7, t0, .Lmm_nblock  # grouping only pays off for a full xq tile
.Lmm_group:
    # Reset the eight i32 accumulators for the next group.
    # Keep the runtime row count (a7) and the 32-column output shape.
    li      t0, (DT_I32 << 16) | (32 << 8)
    or      t0, t0, a7
    SETW    SEL_AC, 0, X_T0, 7

    sub     t0, s3, s2
    li      t1, 8
    bltu    t0, t1, .Lmm_nblock    # fewer than eight blocks remain

    # Full group: load xq once per kblock and keep eight accumulators.
    SETX    SEL_TM, CF_BASE, 0, X_S6
    mul     t0, s2, s4
    slli    t0, t0, 10
    add     t0, s7, t0              # first weight tile of this nblock group
    slli    t3, s4, 10              # next nblock stride within this kblock
    TLOAD   0
    mv      t4, t0
    li      t5, 1
    beq     s4, t5, .Lmm_group_single
    .irp G,0,1,2,3,4,5,6,7
    SETX    SEL_TM, CF_BASE, 1, X_T4
    TLOAD   1
    MMA_NT_ZERO \G, 0, 1
    add     t4, t4, t3
    .endr
    addi    t0, t0, 1024            # next kblock for the group base
    li      s5, 1
    j       .Lmm_group_middle

.Lmm_group_single:
    slli    t0, s2, 7
    add     t0, a4, t0
    .irp G,0,1,2,3,4,5,6,7
    addi    t1, t0, \G*128
    SETX    SEL_TM, CF_BASE, 1, X_T4
    TLOAD   1
    VLOAD   29
    MMA_NT_ZERO \G, 0, 1
    ADEQUANT 8, \G, 29, 30
    SETX    SEL_AM, CF_BASE, 8, X_T1
    ASTORE  8
    add     t4, t4, t3
    .endr
    j       .Lmm_group_epi

.Lmm_group_middle:
    addi    t5, s4, -1
    bgeu    s5, t5, .Lmm_group_last
    TLOAD   0
    mv      t4, t0
    .irp G,0,1,2,3,4,5,6,7
    SETX    SEL_TM, CF_BASE, 1, X_T4
    TLOAD   1
    MMA_NT_ACC \G, 0, 1
    add     t4, t4, t3
    .endr
    addi    t0, t0, 1024
    addi    s5, s5, 1
    j       .Lmm_group_middle

.Lmm_group_last:
    TLOAD   0
    mv      t4, t0
    slli    t0, s2, 7
    add     t0, a4, t0
    .irp G,0,1,2,3,4,5,6,7
    addi    t1, t0, \G*128
    SETX    SEL_TM, CF_BASE, 1, X_T4
    TLOAD   1
    VLOAD   29
    MMA_NT_ACC \G, 0, 1
    ADEQUANT 8, \G, 29, 30
    SETX    SEL_AM, CF_BASE, 8, X_T1
    ASTORE  8
    add     t4, t4, t3
    .endr

.Lmm_group_epi:
    addi    s2, s2, 8
    bltu    s2, s3, .Lmm_group
    j       .Lmm_done

.Lmm_nblock:
    # Tail: fewer than eight output blocks remain.
    SETIM   SEL_AC, CF_DTYPE, DT_I32, 0
    SETX    SEL_TM, CF_BASE, 0, X_S6
    mul     t0, s2, s4
    slli    t0, t0, 10
    add     t0, s7, t0
    li      t1, 1024
    SETX    SEL_TM, CF_BASE, 1, X_T0
    TLOAD   0
    TLOAD   1
    li      t5, 1
    beq     s4, t5, .Lmm_tail_single
    MMA_NT_ZERO 0, 0, 1
    add     t0, t0, t1
    li      s5, 1
.Lmm_tail_middle:
    addi    t5, s4, -1
    bgeu    s5, t5, .Lmm_tail_last
    SETX    SEL_TM, CF_BASE, 1, X_T0
    TLOAD   0
    TLOAD   1
    MMA_NT_ACC 0, 0, 1
    add     t0, t0, t1
    addi    s5, s5, 1
    j       .Lmm_tail_middle
.Lmm_tail_last:
    SETX    SEL_TM, CF_BASE, 1, X_T0
    TLOAD   0
    TLOAD   1
    VLOAD   29
    MMA_NT_ACC 0, 0, 1
    ADEQUANT 8, 0, 29, 30
    j       .Lmm_tail_store
.Lmm_tail_single:
    VLOAD   29
    MMA_NT_ZERO 0, 0, 1
    ADEQUANT 8, 0, 29, 30
.Lmm_tail_store:
    slli    t1, s2, 7
    add     t1, a4, t1
    SETX    SEL_AM, CF_BASE, 8, X_T1
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
    SETW    SEL_VC, 0, X_T6, 4
    SETX    SEL_VM, CF_BASE, 1, X_A0
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
    SETX    SEL_VM, CF_BASE, 1, X_A0   # rewind x
    SETX    SEL_VM, CF_BASE, 3, X_A1
    SETIV   SEL_VM, CF_STRIDE, 4, 3
    SETIV   SEL_VM, CF_VFLAGS, 1, 3
    SETX    SEL_VM, CF_BASE, 4, X_A2
    SETIV   SEL_VM, CF_STRIDE, 4, 4
    SETIV   SEL_VM, CF_VFLAGS, 1, 4
    srli    t6, a3, 5
.Lrn_out:
    VLOAD   1
    VLOAD   3
    VMUL    3, 3, 2
    VMUL    4, 1, 3
    VSTORE  4
    addi    t6, t6, -1
    bnez    t6, .Lrn_out
    ret

# ---------------------------------------------------------------------------
# void qwen3_rmsnorm_rope(float *x, const float *gamma, const float *cos_h,
#                         const float *sin_h, uint32_t hd, uint32_t eps_bits,
#                         uint32_t inv_n_bits)
#
# Fused per-head RMSNorm and rotate-half RoPE.  Keeping the head in v0..v3
# removes the intermediate store/load that separate rmsnorm and rope calls
# would perform.  Arithmetic order matches qwen3_rmsnorm followed by
# qwen3_rope, so the result is bit-identical.  hd is 64 or 128.
# ---------------------------------------------------------------------------
    .globl qwen3_rmsnorm_rope
qwen3_rmsnorm_rope:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6, 21
    srli    t6, a4, 6                   # chunks per half: 1 (hd=64) or 2 (hd=128)

    # x chunks: vm0..vm3; gamma: vm4..vm7; cos/sin: vm8..vm11
    mv      t0, a0
    .irp R,0,1,2,3
      SETX    SEL_VM, CF_BASE, \R, X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, \R
      SETIV   SEL_VM, CF_VFLAGS, 1, \R
      addi    t0, t0, 128
    .endr
    mv      t0, a1
    .irp R,4,5,6,7
      SETX    SEL_VM, CF_BASE, \R, X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, \R
      SETIV   SEL_VM, CF_VFLAGS, 1, \R
      addi    t0, t0, 128
    .endr
    SETX    SEL_VM, CF_BASE, 8, X_A2
    addi    t0, a2, 128
    SETX    SEL_VM, CF_BASE, 9, X_T0
    SETX    SEL_VM, CF_BASE, 10, X_A3
    addi    t0, a3, 128
    SETX    SEL_VM, CF_BASE, 11, X_T0
    .irp R,8,9,10,11
      SETIV   SEL_VM, CF_STRIDE, 4, \R
      SETIV   SEL_VM, CF_VFLAGS, 1, \R
    .endr

    # store descriptors: vm12..vm15 = x, x+32, x+64, x+96 elements
    mv      t0, a0
    .irp R,12,13,14,15
      SETX    SEL_VM, CF_BASE, \R, X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, \R
      SETIV   SEL_VM, CF_VFLAGS, 0, \R
      addi    t0, t0, 128
    .endr

    # sumsq with the same per-lane chunk order as qwen3_rmsnorm
    VFILL   16, 0
    VLOAD   0
    VFMADD  16, 0, 0, 16
    VLOAD   1
    VFMADD  16, 1, 1, 16
    li      t5, 1
    beq     t6, t5, .Lrr_sum_done
    VLOAD   2
    VFMADD  16, 2, 2, 16
    VLOAD   3
    VFMADD  16, 3, 3, 16
.Lrr_sum_done:
    VREDUCE 0, 17, 16
    VMULX   17, 17, X_A6
    VADDX   17, 17, X_A5
    VRSQRT  17, 17

    # x = x * (gamma * inv_rms), same operand order as qwen3_rmsnorm
    VLOAD   4
    VMUL    4, 4, 17
    VMUL    4, 0, 4
    VLOAD   5
    VMUL    5, 5, 17
    VMUL    5, 1, 5
    li      t5, 1
    beq     t6, t5, .Lrr_norm_done
    VLOAD   6
    VMUL    6, 6, 17
    VMUL    6, 2, 6
    VLOAD   7
    VMUL    7, 7, 17
    VMUL    7, 3, 7
.Lrr_norm_done:
    # rotate-half RoPE on the normalized chunks.  hd=64 has one chunk per
    # half (v4 first, v5 second); hd=128 has two (v4/v5 first, v6/v7 second).
    VLOAD   8
    VLOAD   10
    li      t5, 1
    beq     t6, t5, .Lrr_rope_small
    VMUL    18, 4, 8
    VMUL    19, 6, 10
    VSUB    12, 18, 19
    VMUL    18, 6, 8
    VMUL    19, 4, 10
    VADD    14, 18, 19
    VLOAD   9
    VLOAD   11
    VMUL    18, 5, 9
    VMUL    19, 7, 11
    VSUB    13, 18, 19
    VMUL    18, 7, 9
    VMUL    19, 5, 11
    VADD    15, 18, 19
    VSTORE  12
    VSTORE  13
    VSTORE  14
    VSTORE  15
    ret
.Lrr_rope_small:
    VMUL    18, 4, 8
    VMUL    19, 5, 10
    VSUB    12, 18, 19
    VMUL    18, 5, 8
    VMUL    19, 4, 10
    VADD    13, 18, 19
    VSTORE  12
    VSTORE  13
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
    SETX    SEL_VC, CF_LEN, (0 + \R), X_T6
    .endr
    # stream descriptors (stride 4, inc): cos vm0/vm1, sin vm2/vm3,
    # x1 vm4/vm5, x2 vm6/vm7; x2 starts half elements into x (hd*2 bytes).
    # Broadcast stride/inc first: cfg.copy copies every field, base included.
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 0, 6          # vm1..vm7 = vm0 stride/inc
    SETX    SEL_VM, CF_BASE, 0, X_A1
    addi    t0, a1, 128
    SETX    SEL_VM, CF_BASE, 1, X_T0
    SETX    SEL_VM, CF_BASE, 2, X_A2
    addi    t0, a2, 128
    SETX    SEL_VM, CF_BASE, 3, X_T0
    SETX    SEL_VM, CF_BASE, 4, X_A0
    addi    t0, a0, 128
    SETX    SEL_VM, CF_BASE, 5, X_T0
    slli    t0, a3, 1
    add     t0, a0, t0
    SETX    SEL_VM, CF_BASE, 6, X_T0
    addi    t0, t0, 128
    SETX    SEL_VM, CF_BASE, 7, X_T0
    # store descriptors vm12..vm15: base = x + c*128, stride 4, no inc
    mv      t0, a0
    .irp R,0,1,2,3
    SETX    SEL_VM, CF_BASE, (12 + \R), X_T0
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
    SETX    SEL_VM, CF_BASE, 13, X_T0  # vm13 = x + half
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
    SETW    SEL_VC, 0, X_T6, 20
    srli    t6, a5, 5                     # chunks = hd / 32
    # q chunk descriptors vm8..vm11 (base = q + c*128, no inc)
    mv      t0, a0
    .irp R,0,1,2,3
    SETX    SEL_VM, CF_BASE, (8 + \R), X_T0
    SETIV   SEL_VM, CF_STRIDE, 4, (8 + \R)
    SETIV   SEL_VM, CF_VFLAGS, 0, (8 + \R)
    addi    t0, t0, 128
    .endr
    # K chunk stream vm12: base per key, auto-inc within the key
    SETIV   SEL_VM, CF_STRIDE, 4, 12
    SETIV   SEL_VM, CF_VFLAGS, 1, 12
    # one-lane score store: vc17/vm17 are fixed except the base per key
    LI_CWV  t5, DT_F32, 1
    SETW    SEL_VC, 17, X_T5
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
    SETX    SEL_VM, CF_BASE, 12, X_T0
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
    SETX    SEL_VM, CF_BASE, 17, X_T1
    VSTORE  17                          # one-lane score store
    addi    t4, t4, 1
    bltu    t4, a4, .Lsc_key
    ret

# ---------------------------------------------------------------------------
# void qwen3_scores4(const float *q, uint32_t q_stride, const char *k_base,
#                    uint32_t k_stride, float *scores, uint32_t n_keys,
#                    uint32_t hd)
#
# Four consecutive query rows against the same K rows.  The score rows are
# laid out with stride ((n_keys + 31) & ~31) * sizeof(float).  Each query row
# uses the same K chunk order as qwen3_scores, so its score is bit-identical.
# ---------------------------------------------------------------------------
    .globl qwen3_scores4
qwen3_scores4:
    addi    sp, sp, -64
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    mv      s0, a1                     # q row stride (bytes)
    mv      s1, a2                     # K base
    mv      s2, a3                     # K row stride (bytes)
    mv      s3, a4                     # score base
    mv      s4, a5                     # n_keys
    mv      s5, a6                     # head_dim
    srli    s7, s5, 5                  # chunks
    addi    t0, s4, 31
    srli    t0, t0, 5
    slli    s6, t0, 7                  # padded score row stride (bytes)

    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6, 23        # v0..v23: f32 len 32
    SETW    SEL_VC, 24, X_T6, 7        # v24..v31: f32 len 32
    LI_CWV  t5, DT_F32, 1
    SETW    SEL_VC, 4, X_T5, 3         # v4..v7: one-lane score results

.macro Q4_ROW base, vmbase
    mv      t0, \base
    .irp C,0,1,2,3
      SETX    SEL_VM, CF_BASE, (\vmbase + \C), X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, (\vmbase + \C)
      SETIV   SEL_VM, CF_VFLAGS, 0, (\vmbase + \C)
      addi    t0, t0, 128
    .endr
.endm

    Q4_ROW  a0, 8                      # q0 chunks: v8..v11
    add     t1, a0, s0
    Q4_ROW  t1, 12                     # q1 chunks: v12..v15
    slli    t1, s0, 1
    add     t1, a0, t1
    Q4_ROW  t1, 16                     # q2 chunks: v16..v19
    add     t1, t1, s0
    Q4_ROW  t1, 20                     # q3 chunks: v20..v23

    VLOAD   8                          # q row 0, chunk 0
    VLOAD   12                         # q row 1, chunk 0
    VLOAD   16                         # q row 2, chunk 0
    VLOAD   20                         # q row 3, chunk 0
    li      t5, 1
    beq     s7, t5, .Lq4_qdone
    VLOAD   9
    VLOAD   13
    VLOAD   17
    VLOAD   21
    li      t5, 2
    beq     s7, t5, .Lq4_qdone
    VLOAD   10
    VLOAD   14
    VLOAD   18
    VLOAD   22
    li      t5, 3
    beq     s7, t5, .Lq4_qdone
    VLOAD   11
    VLOAD   15
    VLOAD   19
    VLOAD   23
.Lq4_qdone:

    SETIV   SEL_VM, CF_STRIDE, 4, 24
    SETIV   SEL_VM, CF_VFLAGS, 1, 24   # K chunks auto-increment

    mv      t0, s3
    .irp R,4,5,6,7
      SETX    SEL_VM, CF_BASE, \R, X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, \R
      SETIV   SEL_VM, CF_VFLAGS, 1, \R # one score per key, auto-increment
      add     t0, t0, s6
    .endr

    mv      t4, zero
.Lq4_key:
    mul     t0, t4, s2
    add     t0, s1, t0
    SETX    SEL_VM, CF_BASE, 24, X_T0
    VLOAD   24
    VMUL    28, 24, 8
    VMUL    29, 24, 12
    VMUL    30, 24, 16
    VMUL    31, 24, 20
    li      t5, 1
    beq     s7, t5, .Lq4_store
    VLOAD   24
    VFMADD  28, 24, 9, 28
    VFMADD  29, 24, 13, 29
    VFMADD  30, 24, 17, 30
    VFMADD  31, 24, 21, 31
    li      t5, 2
    beq     s7, t5, .Lq4_store
    VLOAD   24
    VFMADD  28, 24, 10, 28
    VFMADD  29, 24, 14, 29
    VFMADD  30, 24, 18, 30
    VFMADD  31, 24, 22, 31
    li      t5, 3
    beq     s7, t5, .Lq4_store
    VLOAD   24
    VFMADD  28, 24, 11, 28
    VFMADD  29, 24, 15, 29
    VFMADD  30, 24, 19, 30
    VFMADD  31, 24, 23, 31
.Lq4_store:
    VREDUCE 0, 4, 28
    VREDUCE 0, 5, 29
    VREDUCE 0, 6, 30
    VREDUCE 0, 7, 31
    VSTORE  4
    VSTORE  5
    VSTORE  6
    VSTORE  7
    addi    t4, t4, 1
    bltu    t4, s4, .Lq4_key

    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    addi    sp, sp, 64
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
    SETW    SEL_VC, 0, X_T6, 4
    LI_CWV  t6, DT_F32, 1
    SETW    SEL_VC, 4, X_T6     # vc4: len-1 store of the rcp result
    SETIV   SEL_VM, CF_STRIDE, 4, 1
    SETIV   SEL_VM, CF_VFLAGS, 1, 1
    SETIV   SEL_VM, CF_STRIDE, 4, 3
    SETIV   SEL_VM, CF_VFLAGS, 1, 3
    li      t0, 0xFF800000                # -inf
    VFILLX  0, X_T0
    SETX    SEL_VM, CF_BASE, 1, X_A0
    srli    t6, a1, 5
.Lsm_max:
    VLOAD   1
    VMAX    0, 0, 1
    addi    t6, t6, -1
    bnez    t6, .Lsm_max
    VREDUCE 1, 0, 0                       # v0 = max f32 bits in every lane
    VFILL   2, 0
    SETX    SEL_VM, CF_BASE, 1, X_A0   # rewind load
    SETX    SEL_VM, CF_BASE, 3, X_A0   # store stream
    srli    t6, a1, 5
.Lsm_exp:
    VLOAD   1
    VSUB    3, 1, 0
    VMULX   3, 3, X_A2
    VEXP2   3, 3
    VADD    2, 2, 3
    VSTORE  3
    addi    t6, t6, -1
    bnez    t6, .Lsm_exp
    VREDUCE 0, 4, 2                       # v4 = sum in every lane
    VRCP    4, 4
    SETX    SEL_VM, CF_BASE, 4, X_A3
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
    SETW    SEL_VC, 0, X_T6, 20
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
    VFILLX  20, X_T0
    mul     t0, t4, a1
    add     t0, a0, t0
    SETX    SEL_VM, CF_BASE, 12, X_T0
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
    SETX    SEL_VM, CF_BASE, (16 + \R), X_T0
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
# void qwen3_pv4(const char *v_base, uint32_t row_stride, const float *probs,
#                float *out, uint32_t n_keys, uint32_t hd,
#                const uint32_t *rcp4, uint32_t out_stride)
#
# Four consecutive query rows.  Rows use the same max n_keys; the caller
# zeroes the softmax tail so the extra terms contribute exactly zero.  The
# per-row arithmetic order is the same as qwen3_pv.
# ---------------------------------------------------------------------------
    .globl qwen3_pv4
qwen3_pv4:
    addi    sp, sp, -80
    sd      s0, 0(sp)
    sd      s1, 8(sp)
    sd      s2, 16(sp)
    sd      s3, 24(sp)
    sd      s4, 32(sp)
    sd      s5, 40(sp)
    sd      s6, 48(sp)
    sd      s7, 56(sp)
    sd      s8, 64(sp)
    sd      s9, 72(sp)
    mv      s0, a0                     # V base
    mv      s1, a1                     # V row stride
    mv      s2, a2                     # probs base
    mv      s3, a3                     # out base
    mv      s4, a4                     # max n_keys
    mv      s5, a5                     # head_dim
    mv      s6, a6                     # rcp4
    mv      s7, a7                     # out row stride
    srli    s9, s5, 5                  # chunks
    addi    t0, s4, 31
    srli    t0, t0, 5
    slli    s8, t0, 7                  # probs row stride (bytes)

    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 8, X_T6, 23        # v8..v31: f32 len 32
    SETIV   SEL_VM, CF_STRIDE, 4, 12
    SETIV   SEL_VM, CF_VFLAGS, 1, 12   # V chunks auto-increment

.macro PV4_OUT_ROW base, vmbase
    mv      t0, \base
    .irp C,0,1,2,3
      SETX    SEL_VM, CF_BASE, (\vmbase + \C), X_T0
      SETIV   SEL_VM, CF_STRIDE, 4, (\vmbase + \C)
      SETIV   SEL_VM, CF_VFLAGS, 0, (\vmbase + \C)
      addi    t0, t0, 128
    .endr
.endm

    PV4_OUT_ROW s3, 16                 # out row 0: vm16..19
    add     t1, s3, s7
    PV4_OUT_ROW t1, 20                 # out row 1: vm20..23
    slli    t1, s7, 1
    add     t1, s3, t1
    PV4_OUT_ROW t1, 24                 # out row 2: vm24..27
    add     t1, t1, s7
    PV4_OUT_ROW t1, 28                 # out row 3: vm28..31

    VFILL   16, 0
    VFILL   17, 0
    VFILL   18, 0
    VFILL   19, 0
    VFILL   20, 0
    VFILL   21, 0
    VFILL   22, 0
    VFILL   23, 0
    VFILL   24, 0
    VFILL   25, 0
    VFILL   26, 0
    VFILL   27, 0
    VFILL   28, 0
    VFILL   29, 0
    VFILL   30, 0
    VFILL   31, 0

    mv      t3, s2                     # row 0 probs cursor
    mv      t4, zero                   # key index
.Lpv4_key:
    lw      t0, 0(t3)
    VFILLX  8, X_T0
    add     t1, t3, s8
    lw      t0, 0(t1)
    VFILLX  9, X_T0
    add     t1, t1, s8
    lw      t0, 0(t1)
    VFILLX  10, X_T0
    add     t1, t1, s8
    lw      t0, 0(t1)
    VFILLX  11, X_T0

    mul     t0, t4, s1
    add     t0, s0, t0
    SETX    SEL_VM, CF_BASE, 12, X_T0
    VLOAD   12
    VFMADD  16, 12, 8, 16
    VFMADD  20, 12, 9, 20
    VFMADD  24, 12, 10, 24
    VFMADD  28, 12, 11, 28
    li      t5, 1
    beq     s9, t5, .Lpv4_next
    VLOAD   12
    VFMADD  17, 12, 8, 17
    VFMADD  21, 12, 9, 21
    VFMADD  25, 12, 10, 25
    VFMADD  29, 12, 11, 29
    li      t5, 2
    beq     s9, t5, .Lpv4_next
    VLOAD   12
    VFMADD  18, 12, 8, 18
    VFMADD  22, 12, 9, 22
    VFMADD  26, 12, 10, 26
    VFMADD  30, 12, 11, 30
    li      t5, 3
    beq     s9, t5, .Lpv4_next
    VLOAD   12
    VFMADD  19, 12, 8, 19
    VFMADD  23, 12, 9, 23
    VFMADD  27, 12, 10, 27
    VFMADD  31, 12, 11, 31
.Lpv4_next:
    addi    t3, t3, 4
    addi    t4, t4, 1
    bltu    t4, s4, .Lpv4_key

    lw      t0, 0(s6)
    VMULX   16, 16, X_T0
    VMULX   17, 17, X_T0
    VMULX   18, 18, X_T0
    VMULX   19, 19, X_T0
    lw      t0, 4(s6)
    VMULX   20, 20, X_T0
    VMULX   21, 21, X_T0
    VMULX   22, 22, X_T0
    VMULX   23, 23, X_T0
    lw      t0, 8(s6)
    VMULX   24, 24, X_T0
    VMULX   25, 25, X_T0
    VMULX   26, 26, X_T0
    VMULX   27, 27, X_T0
    lw      t0, 12(s6)
    VMULX   28, 28, X_T0
    VMULX   29, 29, X_T0
    VMULX   30, 30, X_T0
    VMULX   31, 31, X_T0
    VSTORE  16
    li      t5, 1
    beq     s9, t5, .Lpv4_r0_done
    VSTORE  17
    li      t5, 2
    beq     s9, t5, .Lpv4_r0_done
    VSTORE  18
    li      t5, 3
    beq     s9, t5, .Lpv4_r0_done
    VSTORE  19
.Lpv4_r0_done:
    VSTORE  20
    li      t5, 1
    beq     s9, t5, .Lpv4_r1_done
    VSTORE  21
    li      t5, 2
    beq     s9, t5, .Lpv4_r1_done
    VSTORE  22
    li      t5, 3
    beq     s9, t5, .Lpv4_r1_done
    VSTORE  23
.Lpv4_r1_done:
    VSTORE  24
    li      t5, 1
    beq     s9, t5, .Lpv4_r2_done
    VSTORE  25
    li      t5, 2
    beq     s9, t5, .Lpv4_r2_done
    VSTORE  26
    li      t5, 3
    beq     s9, t5, .Lpv4_r2_done
    VSTORE  27
.Lpv4_r2_done:
    VSTORE  28
    li      t5, 1
    beq     s9, t5, .Lpv4_r3_done
    VSTORE  29
    li      t5, 2
    beq     s9, t5, .Lpv4_r3_done
    VSTORE  30
    li      t5, 3
    beq     s9, t5, .Lpv4_r3_done
    VSTORE  31
.Lpv4_r3_done:

    ld      s0, 0(sp)
    ld      s1, 8(sp)
    ld      s2, 16(sp)
    ld      s3, 24(sp)
    ld      s4, 32(sp)
    ld      s5, 40(sp)
    ld      s6, 48(sp)
    ld      s7, 56(sp)
    ld      s8, 64(sp)
    ld      s9, 72(sp)
    addi    sp, sp, 80
    ret

# ---------------------------------------------------------------------------
# void qwen3_silu_mul(float *gate, const float *up, float *out, uint32_t n)
# out = silu(gate) * up, n % 32 == 0.
# ---------------------------------------------------------------------------
    .globl qwen3_silu_mul
qwen3_silu_mul:
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6, 2
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 0, 1          # vm1, vm2 = vm0 stride/inc
    SETX    SEL_VM, CF_BASE, 0, X_A0
    SETX    SEL_VM, CF_BASE, 1, X_A1
    SETX    SEL_VM, CF_BASE, 2, X_A2
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
    VMUL    2, 0, 1
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
    SETW    SEL_VC, 0, X_T6, 2
    SETIV   SEL_VM, CF_STRIDE, 4, 0
    SETIV   SEL_VM, CF_VFLAGS, 1, 0
    COPY SEL_VM, 1, 0, 1          # vm1, vm2 = vm0 stride/inc
    SETX    SEL_VM, CF_BASE, 0, X_A0
    SETX    SEL_VM, CF_BASE, 1, X_A1
    SETX    SEL_VM, CF_BASE, 2, X_A0
    srli    t6, a2, 5
.Lad_loop:
    VLOAD   0
    VLOAD   1
    VADD    2, 0, 1
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
    srli    t4, a4, 5               # x29 (t4) = D-blocks = D / 32
    srli    t0, a2, 5               # x5 (t0) = V-block = id / 32
    andi    t1, a2, 31              # x6 (t1) = V-inner = id % 32
    slli    t1, t1, 5               # row offset in a tile
    mul     t5, t0, t4              # V-block * D-blocks
    slli    t5, t5, 10
    add     t5, a0, t5
    add     t5, t5, t1              # first D-block row for id
    slli    t0, a2, 2
    add     t0, a1, t0
    lw      t2, 0(t0)               # x7 (t2) = scales[id]

    li      t0, (DT_I8 << 16) | (32 << 8)
    or      t0, t0, t4
    SETW    SEL_TC, 0, X_T0      # tc0 = i8 D-blocks x 32
    li      t0, (DT_F32 << 16) | (32 << 8)
    or      t0, t0, t4
    SETW    SEL_AC, 0, X_T0      # ac0 = f32 D-blocks x 32
    li      t0, DT_F32 << 8
    or      t0, t0, t4
    SETW    SEL_VC, 0, X_T0      # vc0 = f32 len dblocks
    VFILLX  0, X_T2                 # duplicate the token scale in v0
    SETIM   SEL_TM, CF_RSTRIDE, 1024, 0
    SETIM   SEL_TM, CF_CSTRIDE, 1, 0
    SETIM   SEL_TM, CF_FLAGS, 0, 0
    SETX    SEL_TM, CF_BASE, 0, X_T5
    SETX    SEL_AM, CF_BASE, 0, X_A3
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
# ignored, numeric ties keep the smaller index, and an all-NaN block contributes
# index 0 with a NaN value. Each 32x32 f32 Acc tile holds 32 chunks of 32
# logits; areduce.rows.argmax reduces the rows and vreduce.argmax picks the
# best row. The winning block keeps its row-index vector, best row, and base;
# those are written to the stack once at the end and reloaded through scalar
# loads to form the global index.
# ---------------------------------------------------------------------------
    .globl qwen3_argmax
qwen3_argmax:
    addi    sp, sp, -160
    sd      ra, 144(sp)
    LI_CWV  t6, DT_F32, 32
    SETW    SEL_VC, 0, X_T6      # v0: best value
    SETW    SEL_VC, 4, X_T6      # v4: areduce value
    SETW    SEL_VC, 6, X_T6      # v6: block value
    LI_CWV  t6, DT_U32, 32
    SETW    SEL_VC, 3, X_T6      # v3: areduce index
    SETW    SEL_VC, 5, X_T6      # v5: block row
    SETW    SEL_VC, 8, X_T6, 2   # v8..v10: compare/NaN masks
    SETW    SEL_VC, 11, X_T6, 3  # v11..v14: winning index vector, row, base, current base

    LI_CWM  t6, DT_F32, 32, 32
    SETW    SEL_AC, 0, X_T6
    SETIM   SEL_AM, CF_RSTRIDE, 128, 0
    SETIM   SEL_AM, CF_CSTRIDE, 4, 0
    SETIM   SEL_AM, CF_FLAGS, 0, 0

    SETX    SEL_VM, CF_BASE, 12, X_SP
    SETIV   SEL_VM, CF_STRIDE, 4, 12
    SETIV   SEL_VM, CF_VFLAGS, 0, 12
    addi    t5, sp, 4
    SETX    SEL_VM, CF_BASE, 13, X_T5
    SETIV   SEL_VM, CF_STRIDE, 4, 13
    SETIV   SEL_VM, CF_VFLAGS, 0, 13
    addi    t5, sp, 16
    SETX    SEL_VM, CF_BASE, 11, X_T5
    SETIV   SEL_VM, CF_STRIDE, 4, 11
    SETIV   SEL_VM, CF_VFLAGS, 0, 11

    li      t6, 0x7fc00000
    VFILLX  0, X_T6               # all-NaN best
    VFILL   11, 0                 # no winning block yet
    VFILL   12, 0
    VFILL   13, 0

    srli    t6, a1, 10            # full 1024-logit blocks
    andi    t3, a1, 1023
    srli    t3, t3, 5             # tail rows
    li      t4, 0                 # block base in elements
    beqz    t6, .Lam_tail
.Lam_full:
    slli    t5, t4, 2
    add     t5, a0, t5
    SETX    SEL_AM, CF_BASE, 0, X_T5
    jal     ra, .Lam_block
    addi    t4, t4, 1024
    addi    t6, t6, -1
    bnez    t6, .Lam_full

.Lam_tail:
    beqz    t3, .Lam_done
    SETX    SEL_AC, CF_ROWS, 0, X_T3
    SETX    SEL_VC, CF_LEN, 3, X_T3
    SETX    SEL_VC, CF_LEN, 4, X_T3
    slli    t5, t4, 2
    add     t5, a0, t5
    SETX    SEL_AM, CF_BASE, 0, X_T5
    jal     ra, .Lam_tail_block

.Lam_done:
    LI_CWV  t6, DT_U32, 1
    SETW    SEL_VC, 12, X_T6, 1  # exit row/base as one lane each
    VSTORE  12
    VSTORE  13
    VSTORE  11                   # winning block's row-index vector
    FENCE_MEM
    lwu     t0, 0(sp)            # best row
    lwu     t1, 4(sp)            # winning block base
    slli    t2, t0, 2
    add     t2, sp, t2
    lwu     t2, 16(t2)           # column index of the best row
    slli    t3, t0, 5
    add     t1, t1, t3
    add     a0, t1, t2
    ld      ra, 144(sp)
    addi    sp, sp, 160
    ret

# A full block already has rows=32 and vc3/vc4 length 32.
.Lam_block:
    ALOAD   0
    AREDUCE_ARGMAX 3, 4, 0        # v3 = row col index, v4 = row max
    j       .Lam_block_rest

# The tail block reuses the full tournament after restoring vc3's length.
.Lam_tail_block:
    ALOAD   0
    AREDUCE_ARGMAX 3, 4, 0
    LI_CWV  t6, DT_U32, 32
    SETW    SEL_VC, 3, X_T6

.Lam_block_rest:
    VREDUCE_ARGMAX 5, 6, 4        # v5 = best row, v6 = block max
    VFILLX  14, X_T4              # v14 = current block base
    VCMPLT    8, 0, 6
    VCMPUNORD 9, 0, 0
    VCMPUNORD 10, 6, 6
    VNOT      10, 10
    VAND      9, 9, 10
    VOR       8, 8, 9
    VSELECT   0, 8, 6, 0
    VSELECT   11, 8, 3, 11
    VSELECT   12, 8, 5, 12
    VSELECT   13, 8, 14, 13
    ret
