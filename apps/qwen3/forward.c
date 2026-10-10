#include "qwen3.h"

// Qwen3 forward pass orchestration. Architecture dims come from the blob
// header at runtime; weight pointers are cached per layer at init.

// Numeric alignment probes, fired on the last prompt token's forward pass:
// tagged hex dumps mirrored line for line by scripts/qwen3/ref.py (the numpy
// reference), so the host can align guest numerics against it. Flip to 0 for
// silent runs.
#ifndef QWEN3_PROBES
#define QWEN3_PROBES 1
#endif

uint32_t nq_quant_act(const float *x, uint32_t k);
void nq_matvec_q(const int8_t *w, const float *w_scales, float *out, uint32_t n, uint32_t k,
                 uint32_t xs_bits);
void nq_quant_rows(const float *x, uint32_t k, int8_t *xq, float *xs);
void nq_matmul_q(const int8_t *w, const float *w_scales, const int8_t *xq, const float *xs,
                 float *out, uint32_t n, uint32_t k);
void nq_rmsnorm(float *x, const float *gamma, float *out, uint32_t n, uint32_t eps_bits,
                uint32_t inv_n_bits);
void nq_rope(float *x, const float *cos_h, const float *sin_h, uint32_t hd);
void nq_scores(const float *q, const char *k_base, uint32_t row_stride, float *scores,
               uint32_t n_keys, uint32_t hd);
void nq_softmax_exp(float *scores, uint32_t n, uint32_t scale_log2e_bits, uint32_t *rcp_out);
void nq_pv(const char *v_base, uint32_t row_stride, const float *probs, float *out, uint32_t n_keys,
           uint32_t hd, uint32_t rcp_bits);
void nq_silu_mul(float *gate, const float *up, float *out, uint32_t n);
void nq_add(float *dst, const float *src, uint32_t n);
void nq_embed(const int8_t *table, const float *scales, uint32_t id, float *out, uint32_t hidden);
uint32_t nq_argmax(const float *x, uint32_t n);

const void *blob_section(const char name[8]);
const struct q3b_header *blob_header(void);

#define MAX_LAYERS 32

// Per-layer blob section pointers, indexed LS_*.
enum {
  LS_INLN,
  LS_QW,
  LS_QWS,
  LS_KW,
  LS_KWS,
  LS_VW,
  LS_VWS,
  LS_OW,
  LS_OWS,
  LS_QN,
  LS_KN,
  LS_PTLN,
  LS_GW,
  LS_GWS,
  LS_UW,
  LS_UWS,
  LS_DW,
  LS_DWS,
  LS_COUNT
};

static const void *lsec[MAX_LAYERS][LS_COUNT];
static const int8_t *embed;
static const float *embed_s;
static const float *rope_cos;
static const float *rope_sin;
static const float *fin_norm;

static uint32_t N_LAYERS, HIDDEN, INTER, N_Q, N_KV, HD, VOCAB, MAX_POS, N_PROMPT, EPS_BITS;
static uint32_t Q_DIM, KV_DIM;
static uint32_t SCALE_LOG2E_BITS;

static float x_buf[3072];
static float h1[3072];
static float h2[3072];
static float gate_buf[3072];
static float up_buf[3072];
static float attn_buf[3072];
static float q_buf[2048];
static float k_buf[1024];
static float v_buf[1024];
static float scores[512 + 32];
static float logits[151936];
static uint32_t rcp_word[1];

// prefill batch buffers: 32 token rows each, row stride = the vector dim
#define BATCH_ROWS 32
static float pf_x[BATCH_ROWS * 3072];
static float pf_h1[BATCH_ROWS * 3072];
static float pf_h2[BATCH_ROWS * 3072];
static float pf_gate[BATCH_ROWS * 3072];
static float pf_up[BATCH_ROWS * 3072];
static float pf_q[BATCH_ROWS * 2048];
static float pf_attn[BATCH_ROWS * 2048];
static float pf_k[BATCH_ROWS * 1024];
static float pf_v[BATCH_ROWS * 1024];
static int8_t pf_xq[BATCH_ROWS * 3072];
static float pf_xs[BATCH_ROWS];

static char *const kv_region = (char *)KV_REGION_BASE;
static uint32_t kv_stride; // bytes per layer per cache (max_pos * kv_dim * 4)

static char *k_layer(uint32_t layer) { return kv_region + layer * 2 * kv_stride; }

static char *v_layer(uint32_t layer) { return kv_region + (layer * 2 + 1) * kv_stride; }

// ---- numeric alignment probes ---------------------------------------------
// All probe output goes through probe()/probe_head() under the single
// QWEN3_PROBES switch. Fires on the last prompt token's forward pass only,
// the pass ref.py prints.

extern char *nq_debug_qtile; // kernels.S debug hook (alignment only)
extern int32_t nq_accbuf[];  // raw i32 matvec accumulators (kernels.S)
static char qtile_dump[1024];

static void probe(const char *tag, const float *v, int n) {
  print_str(tag);
  serial_putc(' ');
  for (int i = 0; i < n; ++i)
    print_hex32(((const uint32_t *)v)[i]);
  serial_putc('\n');
}

// Head-indexed probe line "<tag><h> w0 ...": ref.py's sc_hN / pb_hN.
static void probe_head(const char *tag, uint32_t h, const uint32_t *words, int n) {
  print_str(tag);
  print_u32(h);
  serial_putc(' ');
  for (int i = 0; i < n; ++i)
    print_hex32(words[i]);
  serial_putc('\n');
}

// attention() probe gate: layer 0, heads 0/1, last prompt token.
static int attn_probing(uint32_t layer, uint32_t h, uint32_t pos) {
  return QWEN3_PROBES && layer == 0 && h < 2 && pos == N_PROMPT - 1;
}

// nq_quant_act with the packed-tile dump hooked around it; active only on
// the probed pass's layer 0. The tag tells the q/k/v and the o-projection
// quantization sites apart.
static uint32_t quant_act(const float *x, uint32_t k, const char *tag, int active) {
  if (active) nq_debug_qtile = qtile_dump;
  uint32_t xs = nq_quant_act(x, k);
  if (active) {
    nq_debug_qtile = 0;
    probe(tag, (const float *)qtile_dump, 16);
  }
  return xs;
}

// PROBE sites all sit in forward(), gated on its `probing` local.
#define PROBE(tag, v, n)                                                                           \
  do {                                                                                             \
    if (probing) probe(tag, v, n);                                                                 \
  } while (0)

static uint32_t inv_n_bits(uint32_t n) {
  // n is a power of two: 1/n = 2^-ctz(n) in f32.
  uint32_t e = 0;
  while (!(n & 1)) {
    n >>= 1;
    ++e;
  }
  return (127 - e) << 23;
}

static void make_name(char name[8], uint32_t layer, const char *suffix) {
  name[0] = 'l';
  name[1] = '0' + layer / 10;
  name[2] = '0' + layer % 10;
  name[3] = '_';
  int i = 4;
  while (*suffix && i < 8)
    name[i++] = *suffix++;
  while (i < 8)
    name[i++] = 0;
}

static void cache_layer_sections(uint32_t layer) {
  static const char *const names[LS_COUNT] = {
      "inln", "qw", "qws",  "kw", "kws", "vw", "vws", "ow", "ows",
      "qn",   "kn", "ptln", "gw", "gws", "uw", "uws", "dw", "dws",
  };
  for (int i = 0; i < LS_COUNT; ++i) {
    char name[8];
    make_name(name, layer, names[i]);
    lsec[layer][i] = blob_section(name);
  }
}

void forward_init(void) {
  const struct q3b_header *h = blob_header();
  N_LAYERS = h->n_layers;
  HIDDEN = h->hidden;
  INTER = h->inter;
  N_Q = h->n_q_heads;
  N_KV = h->n_kv_heads;
  HD = h->head_dim;
  VOCAB = h->vocab;
  MAX_POS = h->max_pos;
  N_PROMPT = h->n_prompt;
  EPS_BITS = h->eps_bits;
  Q_DIM = N_Q * HD;
  KV_DIM = N_KV * HD;
  // the kernels require dims that are multiples of 32 and rope/score chunks
  // that fit 32-lane vectors; inv_n_bits() computes 1/n as 2^-ctz(n), exact
  // only when n is a power of two; logits[] and the kernels.S accumulator
  // buffer are sized 151936. Anything else is rejected up front.
  if (N_LAYERS > MAX_LAYERS || HIDDEN > 3072 || INTER > 3072 || Q_DIM > 2048 || KV_DIM > 1024 ||
      MAX_POS > 512 || VOCAB > 151936 || HIDDEN == 0 || INTER == 0 || N_Q == 0 || N_KV == 0 ||
      VOCAB == 0 || HIDDEN % 32 != 0 || INTER % 32 != 0 || Q_DIM % 32 != 0 || KV_DIM % 32 != 0 ||
      VOCAB % 32 != 0 || N_Q % N_KV != 0 || (HIDDEN & (HIDDEN - 1)) != 0) {
    print_str("unsupported arch dims\n");
    halt(1);
  }
  switch (HD) {
  case 128: SCALE_LOG2E_BITS = 0x3E0293EEu; break;
  case 64: SCALE_LOG2E_BITS = 0x3E38AA3Bu; break;
  case 32: SCALE_LOG2E_BITS = 0x3E8293EEu; break;
  default: print_str("unsupported head_dim\n"); halt(1);
  }
  kv_stride = MAX_POS * KV_DIM * 4;
  if ((uint64_t)2 * N_LAYERS * kv_stride > KV_REGION_SIZE) {
    print_str("kv cache exceeds region\n");
    halt(1);
  }
  embed = blob_section("embed\0\0");
  embed_s = blob_section("embed_s\0");
  rope_cos = blob_section("rope_cos");
  rope_sin = blob_section("rope_sin");
  fin_norm = blob_section("fin_norm");
  for (uint32_t i = 0; i < N_LAYERS; ++i)
    cache_layer_sections(i);
}

static void cache_append(uint32_t layer, uint32_t pos, const float *k, const float *v) {
  float *kd = (float *)(k_layer(layer) + pos * KV_DIM * 4);
  float *vd = (float *)(v_layer(layer) + pos * KV_DIM * 4);
  for (uint32_t i = 0; i < KV_DIM; ++i) {
    kd[i] = k[i];
    vd[i] = v[i];
  }
}

static void attention(uint32_t layer, uint32_t pos, const float *q, float *attn) {
  const uint32_t group = N_Q / N_KV;
  const uint32_t n_keys = pos + 1;
  const uint32_t n_padded = (n_keys + 31) & ~31u;
  for (uint32_t h = 0; h < N_Q; ++h) {
    const uint32_t kvh = h / group;
    const char *kb = k_layer(layer) + kvh * HD * 4;
    const char *vb = v_layer(layer) + kvh * HD * 4;
    nq_scores(q + h * HD, kb, KV_DIM * 4, scores, n_keys, HD);
    // each head's softmax overwrites the whole padded block, so the -inf
    // tail must be re-applied per head
    for (uint32_t i = n_keys; i < n_padded; ++i) {
      ((uint32_t *)scores)[i] = 0xFF800000u;
    }
    if (attn_probing(layer, h, pos)) probe_head("sc_h", h, (const uint32_t *)scores, 6);
    nq_softmax_exp(scores, n_padded, SCALE_LOG2E_BITS, rcp_word);
    // pb_h appends the softmax rcp as a 7th word (ref.py prints it inline)
    if (attn_probing(layer, h, pos)) {
      uint32_t words[7];
      __builtin_memcpy(words, scores, 6 * sizeof(uint32_t));
      words[6] = rcp_word[0];
      probe_head("pb_h", h, words, 7);
    }
    nq_pv(vb, KV_DIM * 4, scores, attn + h * HD, n_keys, HD, rcp_word[0]);
  }
}

uint32_t forward(uint32_t token, uint32_t pos, int want_logits) {
  // probes fire on the last prompt token's logits-producing pass
  const int probing = QWEN3_PROBES && want_logits && pos == N_PROMPT - 1;
  uint32_t xs;
  nq_embed(embed, embed_s, token, x_buf, HIDDEN);
  PROBE("embed", x_buf, 4);
  for (uint32_t L = 0; L < N_LAYERS; ++L) {
    const void **sec = lsec[L];
    nq_rmsnorm(x_buf, sec[LS_INLN], h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
    if (L == 0) PROBE("ln0", h1, 4);
    // q/k/v share one activation quantization; the tiles stay resident
    xs = quant_act(h1, HIDDEN, "qx0", probing && L == 0);
    nq_matvec_q(sec[LS_QW], sec[LS_QWS], q_buf, Q_DIM, HIDDEN, xs);
    if (L == 0) PROBE("ac0", (const float *)nq_accbuf, 8);
    if (L == 0) PROBE("q0", q_buf, 4);
    nq_matvec_q(sec[LS_KW], sec[LS_KWS], k_buf, KV_DIM, HIDDEN, xs);
    nq_matvec_q(sec[LS_VW], sec[LS_VWS], v_buf, KV_DIM, HIDDEN, xs);
    for (uint32_t h = 0; h < N_Q; ++h) {
      nq_rmsnorm(q_buf + h * HD, sec[LS_QN], q_buf + h * HD, HD, EPS_BITS, inv_n_bits(HD));
    }
    if (L == 0) PROBE("q0n", q_buf, 4);
    for (uint32_t h = 0; h < N_KV; ++h) {
      nq_rmsnorm(k_buf + h * HD, sec[LS_KN], k_buf + h * HD, HD, EPS_BITS, inv_n_bits(HD));
    }
    const float *cos_p = rope_cos + pos * (HD / 2);
    const float *sin_p = rope_sin + pos * (HD / 2);
    for (uint32_t h = 0; h < N_Q; ++h) {
      nq_rope(q_buf + h * HD, cos_p, sin_p, HD);
    }
    if (L == 0) PROBE("q0r", q_buf, 4);
    if (L == 0) PROBE("q1r", q_buf + HD, 4);
    for (uint32_t h = 0; h < N_KV; ++h) {
      nq_rope(k_buf + h * HD, cos_p, sin_p, HD);
    }
    cache_append(L, pos, k_buf, v_buf);
    attention(L, pos, q_buf, attn_buf);
    if (L == 0) PROBE("attn0", attn_buf, 4);
    if (L == 0) PROBE("attn1", attn_buf + HD, 4);
    xs = quant_act(attn_buf, Q_DIM, "qxa", probing && L == 0);
    nq_matvec_q(sec[LS_OW], sec[LS_OWS], h2, HIDDEN, Q_DIM, xs);
    if (L == 0) PROBE("aca", (const float *)nq_accbuf, 8);
    if (L == 0) PROBE("o0", h2, 4);
    nq_add(x_buf, h2, HIDDEN);
    if (L == 0) PROBE("xr0", x_buf, 4);
    nq_rmsnorm(x_buf, sec[LS_PTLN], h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
    if (L == 0) PROBE("ln1", h1, 4);
    // gate/up share one activation quantization
    xs = nq_quant_act(h1, HIDDEN);
    nq_matvec_q(sec[LS_GW], sec[LS_GWS], gate_buf, INTER, HIDDEN, xs);
    nq_matvec_q(sec[LS_UW], sec[LS_UWS], up_buf, INTER, HIDDEN, xs);
    if (L == 0) PROBE("g0", gate_buf, 4);
    if (L == 0) PROBE("u0", up_buf, 4);
    nq_silu_mul(gate_buf, up_buf, gate_buf, INTER);
    if (L == 0) PROBE("m0", gate_buf, 4);
    xs = nq_quant_act(gate_buf, INTER);
    nq_matvec_q(sec[LS_DW], sec[LS_DWS], h2, HIDDEN, INTER, xs);
    if (L == 0) PROBE("d0", h2, 4);
    nq_add(x_buf, h2, HIDDEN);
    if (L == 0) PROBE("x_l0", x_buf, 4);
  }
  if (!want_logits) return 0;
  nq_rmsnorm(x_buf, fin_norm, h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
  PROBE("lnf", h1, 4);
  xs = nq_quant_act(h1, HIDDEN);
  nq_matvec_q(embed, embed_s, logits, VOCAB, HIDDEN, xs);
  // numeric alignment hook: ref.py prints its logits top5; the guest dumps
  // the logits at the same fixed ids on the first logits-producing pass
  static int dumped = 0;
  if (QWEN3_PROBES && !dumped) {
    dumped = 1;
    static const uint32_t ids[] = {7, 174, 303, 113, 133};
    uint32_t words[5];
    for (int i = 0; i < 5; ++i)
      words[i] = ((const uint32_t *)logits)[ids[i]];
    probe("logit probe:", (const float *)words, 5);
  }
  return nq_argmax(logits, VOCAB);
}

// Prefill for the first n prompt tokens (positions [0, n)): projections run
// as batched MMA over 32-row token batches; rmsnorm/rope/attention/silu/add
// keep the per-row vector kernels. Rows [bt, 32) of the quantized buffers
// are never consumed. The numerics match the per-token path bit for bit
// (same per-row scales, exact i32 accumulation, same epilogue order).
void forward_prefill(const uint32_t *tokens, uint32_t n) {
  for (uint32_t base = 0; base < n; base += BATCH_ROWS) {
    const uint32_t bt = n - base < BATCH_ROWS ? n - base : BATCH_ROWS;
    for (uint32_t t = 0; t < bt; ++t)
      nq_embed(embed, embed_s, tokens[base + t], pf_x + t * HIDDEN, HIDDEN);
    for (uint32_t L = 0; L < N_LAYERS; ++L) {
      const void **sec = lsec[L];
      for (uint32_t t = 0; t < bt; ++t)
        nq_rmsnorm(pf_x + t * HIDDEN, sec[LS_INLN], pf_h1 + t * HIDDEN, HIDDEN, EPS_BITS,
                   inv_n_bits(HIDDEN));
      nq_quant_rows(pf_h1, HIDDEN, pf_xq, pf_xs);
      nq_matmul_q(sec[LS_QW], sec[LS_QWS], pf_xq, pf_xs, pf_q, Q_DIM, HIDDEN);
      nq_matmul_q(sec[LS_KW], sec[LS_KWS], pf_xq, pf_xs, pf_k, KV_DIM, HIDDEN);
      nq_matmul_q(sec[LS_VW], sec[LS_VWS], pf_xq, pf_xs, pf_v, KV_DIM, HIDDEN);
      for (uint32_t t = 0; t < bt; ++t) {
        for (uint32_t h = 0; h < N_Q; ++h)
          nq_rmsnorm(pf_q + t * Q_DIM + h * HD, sec[LS_QN], pf_q + t * Q_DIM + h * HD, HD, EPS_BITS,
                     inv_n_bits(HD));
        for (uint32_t h = 0; h < N_KV; ++h)
          nq_rmsnorm(pf_k + t * KV_DIM + h * HD, sec[LS_KN], pf_k + t * KV_DIM + h * HD, HD,
                     EPS_BITS, inv_n_bits(HD));
        const float *cos_p = rope_cos + (base + t) * (HD / 2);
        const float *sin_p = rope_sin + (base + t) * (HD / 2);
        for (uint32_t h = 0; h < N_Q; ++h)
          nq_rope(pf_q + t * Q_DIM + h * HD, cos_p, sin_p, HD);
        for (uint32_t h = 0; h < N_KV; ++h)
          nq_rope(pf_k + t * KV_DIM + h * HD, cos_p, sin_p, HD);
        cache_append(L, base + t, pf_k + t * KV_DIM, pf_v + t * KV_DIM);
        attention(L, base + t, pf_q + t * Q_DIM, pf_attn + t * Q_DIM);
      }
      nq_quant_rows(pf_attn, Q_DIM, pf_xq, pf_xs);
      nq_matmul_q(sec[LS_OW], sec[LS_OWS], pf_xq, pf_xs, pf_h2, HIDDEN, Q_DIM);
      for (uint32_t t = 0; t < bt; ++t)
        nq_add(pf_x + t * HIDDEN, pf_h2 + t * HIDDEN, HIDDEN);
      for (uint32_t t = 0; t < bt; ++t)
        nq_rmsnorm(pf_x + t * HIDDEN, sec[LS_PTLN], pf_h1 + t * HIDDEN, HIDDEN, EPS_BITS,
                   inv_n_bits(HIDDEN));
      nq_quant_rows(pf_h1, HIDDEN, pf_xq, pf_xs);
      nq_matmul_q(sec[LS_GW], sec[LS_GWS], pf_xq, pf_xs, pf_gate, INTER, HIDDEN);
      nq_matmul_q(sec[LS_UW], sec[LS_UWS], pf_xq, pf_xs, pf_up, INTER, HIDDEN);
      for (uint32_t t = 0; t < bt; ++t)
        nq_silu_mul(pf_gate + t * INTER, pf_up + t * INTER, pf_gate + t * INTER, INTER);
      nq_quant_rows(pf_gate, INTER, pf_xq, pf_xs);
      nq_matmul_q(sec[LS_DW], sec[LS_DWS], pf_xq, pf_xs, pf_h2, HIDDEN, INTER);
      for (uint32_t t = 0; t < bt; ++t)
        nq_add(pf_x + t * HIDDEN, pf_h2 + t * HIDDEN, HIDDEN);
    }
  }
}
