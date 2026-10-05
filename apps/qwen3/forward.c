#include "qwen3.h"

// Qwen3 forward pass orchestration. Architecture dims come from the blob
// header at runtime; weight pointers are cached per layer at init.
//
// Matvec activation buffers are 3072 f32 each and only ever written within
// [0, k): the matvec kernel reads the zero padding up to its full panels,
// which contributes nothing (see kernels.S).

void nq_matvec(const int8_t *w, const float *w_scales, const float *x,
               float *out, uint32_t n, uint32_t k);
void nq_rmsnorm(float *x, const float *gamma, float *out, uint32_t n,
                uint32_t eps_bits, uint32_t inv_n_bits);
void nq_rope(float *x, const float *cos_h, const float *sin_h, uint32_t hd);
void nq_scores(const float *q, const char *k_base, uint32_t row_stride,
               float *scores, uint32_t n_keys, uint32_t hd);
void nq_softmax_exp(float *scores, uint32_t n, uint32_t scale_log2e_bits,
                    uint32_t *rcp_out);
void nq_pv(const char *v_base, uint32_t row_stride, const float *probs,
           float *out, uint32_t n_keys, uint32_t hd, uint32_t rcp_bits);
void nq_silu_mul(float *gate, const float *up, float *out, uint32_t n);
void nq_add(float *dst, const float *src, uint32_t n);
void nq_embed(const int8_t *table, const float *scales, uint32_t id, float *out,
              uint32_t hidden);
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

static uint32_t N_LAYERS, HIDDEN, INTER, N_Q, N_KV, HD, VOCAB, MAX_POS,
    N_PROMPT, EPS_BITS;
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

static char *const kv_region = (char *)KV_REGION_BASE;
static uint32_t kv_stride; // bytes per layer per cache (max_pos * kv_dim * 4)

static char *k_layer(uint32_t layer) {
  return kv_region + layer * 2 * kv_stride;
}

static char *v_layer(uint32_t layer) {
  return kv_region + (layer * 2 + 1) * kv_stride;
}

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
  if (N_LAYERS > MAX_LAYERS || HIDDEN > 3072 || INTER > 3072 || Q_DIM > 2048 ||
      KV_DIM > 1024 || MAX_POS > 512) {
    print_str("unsupported arch dims\n");
    semihost_exit(1);
  }
  switch (HD) {
  case 128: SCALE_LOG2E_BITS = 0x3E0293EEu; break;
  case 64: SCALE_LOG2E_BITS = 0x3E38AA3Bu; break;
  case 32: SCALE_LOG2E_BITS = 0x3E8293EEu; break;
  case 16: SCALE_LOG2E_BITS = 0x3EB8AA3Bu; break;
  default: print_str("unsupported head_dim\n"); semihost_exit(1);
  }
  kv_stride = MAX_POS * KV_DIM * 4;
  if ((uint64_t)2 * N_LAYERS * kv_stride > KV_REGION_SIZE) {
    print_str("kv cache exceeds region\n");
    semihost_exit(1);
  }
  embed = blob_section("embed\0\0");
  embed_s = blob_section("embed_s\0");
  rope_cos = blob_section("rope_cos");
  rope_sin = blob_section("rope_sin");
  fin_norm = blob_section("fin_norm");
  for (uint32_t i = 0; i < N_LAYERS; ++i)
    cache_layer_sections(i);
}

static void cache_append(uint32_t layer, uint32_t pos) {
  float *kd = (float *)(k_layer(layer) + pos * KV_DIM * 4);
  float *vd = (float *)(v_layer(layer) + pos * KV_DIM * 4);
  for (uint32_t i = 0; i < KV_DIM; ++i) {
    kd[i] = k_buf[i];
    vd[i] = v_buf[i];
  }
}

static void attention(uint32_t layer, uint32_t pos) {
  const uint32_t group = N_Q / N_KV;
  const uint32_t n_keys = pos + 1;
  const uint32_t n_padded = (n_keys + 31) & ~31u;
  for (uint32_t h = 0; h < N_Q; ++h) {
    const uint32_t kvh = h / group;
    const char *kb = k_layer(layer) + kvh * HD * 4;
    const char *vb = v_layer(layer) + kvh * HD * 4;
    nq_scores(q_buf + h * HD, kb, KV_DIM * 4, scores, n_keys, HD);
    // each head's softmax overwrites the whole padded block, so the -inf
    // tail must be re-applied per head
    for (uint32_t i = n_keys; i < n_padded; ++i) {
      ((uint32_t *)scores)[i] = 0xFF800000u;
    }
    if (layer == 0 && pos == N_PROMPT - 1) {
      print_str(h == 0 ? "sc_h0 " : "sc_h1 ");
      for (int i = 0; i < 6; ++i)
        print_hex32(((uint32_t *)scores)[i]);
      semihost_putc('\n');
    }
    nq_softmax_exp(scores, n_padded, SCALE_LOG2E_BITS, rcp_word);
    if (layer == 0 && pos == N_PROMPT - 1) {
      print_str(h == 0 ? "pb_h0 " : "pb_h1 ");
      for (int i = 0; i < 6; ++i)
        print_hex32(((uint32_t *)scores)[i]);
      print_hex32(rcp_word[0]);
      semihost_putc('\n');
    }
    nq_pv(vb, KV_DIM * 4, scores, attn_buf + h * HD, n_keys, HD, rcp_word[0]);
  }
}

uint32_t forward(uint32_t token, uint32_t pos, int want_logits);

extern char *nq_debug_qtile; // kernels.S debug hook (alignment only)
extern int32_t nq_accbuf[];  // raw i32 matvec accumulators (kernels.S)
static char qtile_dump[1024];

// numeric alignment probes (fire on the last prompt token's forward pass)
static void probe(const char *tag, const float *v, int n) {
  print_str(tag);
  semihost_putc(' ');
  for (int i = 0; i < n; ++i)
    print_hex32(((const uint32_t *)v)[i]);
  semihost_putc('\n');
}
#define PROBE(tag, v, n)                                                       \
  do {                                                                         \
    if (want_logits && pos == N_PROMPT - 1) {                                  \
      probe(tag, v, n);                                                        \
    }                                                                          \
  } while (0)

uint32_t forward(uint32_t token, uint32_t pos, int want_logits) {
  nq_embed(embed, embed_s, token, x_buf, HIDDEN);
  PROBE("embed", x_buf, 4);
  for (uint32_t L = 0; L < N_LAYERS; ++L) {
    const void **sec = lsec[L];
    nq_rmsnorm(x_buf, sec[LS_INLN], h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
    if (L == 0) PROBE("ln0", h1, 4);
    if (L == 0 && want_logits && pos == N_PROMPT - 1) {
      nq_debug_qtile = qtile_dump;
    }
    nq_matvec(sec[LS_QW], sec[LS_QWS], h1, q_buf, Q_DIM, HIDDEN);
    if (L == 0 && want_logits && pos == N_PROMPT - 1) {
      nq_debug_qtile = 0;
      probe("qx0", (const float *)qtile_dump, 16);
      probe("ac0", (const float *)nq_accbuf, 8);
    }
    if (L == 0) PROBE("q0", q_buf, 4);
    if (L == 0) PROBE("q0h", q_buf + 16, 4);
    nq_matvec(sec[LS_KW], sec[LS_KWS], h1, k_buf, KV_DIM, HIDDEN);
    nq_matvec(sec[LS_VW], sec[LS_VWS], h1, v_buf, KV_DIM, HIDDEN);
    for (uint32_t h = 0; h < N_Q; ++h) {
      nq_rmsnorm(q_buf + h * HD, sec[LS_QN], q_buf + h * HD, HD, EPS_BITS,
                 inv_n_bits(HD));
    }
    if (L == 0) PROBE("q0n", q_buf, 4);
    for (uint32_t h = 0; h < N_KV; ++h) {
      nq_rmsnorm(k_buf + h * HD, sec[LS_KN], k_buf + h * HD, HD, EPS_BITS,
                 inv_n_bits(HD));
    }
    const float *cos_p = rope_cos + pos * (HD / 2);
    const float *sin_p = rope_sin + pos * (HD / 2);
    if (L == 0) PROBE("cosr", cos_p, 4);
    if (L == 0) PROBE("sinr", sin_p, 4);
    for (uint32_t h = 0; h < N_Q; ++h) {
      nq_rope(q_buf + h * HD, cos_p, sin_p, HD);
    }
    if (L == 0) PROBE("q0r", q_buf, 4);
    if (L == 0) PROBE("q1r", q_buf + 32, 4);
    for (uint32_t h = 0; h < N_KV; ++h) {
      nq_rope(k_buf + h * HD, cos_p, sin_p, HD);
    }
    cache_append(L, pos);
    attention(L, pos);
    if (L == 0) PROBE("attn0", attn_buf, 4);
    if (L == 0) PROBE("attn1", attn_buf + 32, 4);
    if (L == 0 && want_logits && pos == N_PROMPT - 1) {
      nq_debug_qtile = qtile_dump;
    }
    nq_matvec(sec[LS_OW], sec[LS_OWS], attn_buf, h2, HIDDEN, Q_DIM);
    if (L == 0 && want_logits && pos == N_PROMPT - 1) {
      nq_debug_qtile = 0;
      probe("qxa", (const float *)qtile_dump, 16);
      probe("aca", (const float *)nq_accbuf, 8);
    }
    if (L == 0) PROBE("o0", h2, 4);
    nq_add(x_buf, h2, HIDDEN);
    if (L == 0) PROBE("xr0", x_buf, 4);
    nq_rmsnorm(x_buf, sec[LS_PTLN], h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
    if (L == 0) PROBE("ln1", h1, 4);
    nq_matvec(sec[LS_GW], sec[LS_GWS], h1, gate_buf, INTER, HIDDEN);
    nq_matvec(sec[LS_UW], sec[LS_UWS], h1, up_buf, INTER, HIDDEN);
    if (L == 0) PROBE("g0", gate_buf, 4);
    if (L == 0) PROBE("u0", up_buf, 4);
    nq_silu_mul(gate_buf, up_buf, gate_buf, INTER);
    if (L == 0) PROBE("m0", gate_buf, 4);
    nq_matvec(sec[LS_DW], sec[LS_DWS], gate_buf, h2, HIDDEN, INTER);
    if (L == 0) PROBE("d0", h2, 4);
    nq_add(x_buf, h2, HIDDEN);
    if (L == 0) PROBE("x_l0", x_buf, 4);
  }
  if (!want_logits) return 0;
  nq_rmsnorm(x_buf, fin_norm, h1, HIDDEN, EPS_BITS, inv_n_bits(HIDDEN));
  PROBE("lnf", h1, 4);
  nq_matvec(embed, embed_s, h1, logits, VOCAB, HIDDEN);
  static int dumped = 0;
  if (!dumped) { // numeric alignment hook: scripts/qwen3/ref.py top5
    dumped = 1;
    static const uint32_t ids[] = {7, 174, 303, 113, 133};
    print_str("logit probe: ");
    for (int i = 0; i < 5; ++i)
      print_hex32(((uint32_t *)logits)[ids[i]]);
    semihost_putc('\n');
  }
  return nq_argmax(logits, VOCAB);
}
