#include "qwen3.h"

#include "console.h"
#include "dims.h"
#include "kernels.h"
#include "probe.h"
#include "q3b.h"

// Qwen3 forward pass orchestration. Architecture dims come from the blob
// header at runtime; weight pointers are cached per layer at init.
// qwen3_ = Qwen3 NPU kernel, implemented in kernels.S.

#define MAX_LAYERS 32

// Per-layer blob section pointers, indexed by SEC_*.
enum layer_section {
  SEC_INLN,
  SEC_QW,
  SEC_QWS,
  SEC_KW,
  SEC_KWS,
  SEC_VW,
  SEC_VWS,
  SEC_OW,
  SEC_OWS,
  SEC_QN,
  SEC_KN,
  SEC_PTLN,
  SEC_GW,
  SEC_GWS,
  SEC_UW,
  SEC_UWS,
  SEC_DW,
  SEC_DWS,
  SEC_COUNT
};

struct qwen3_model {
  uint32_t n_layers, hidden, inter, n_q_heads, n_kv_heads, head_dim;
  uint32_t vocab, max_pos, n_prompt, eps_bits;
  uint32_t q_dim, kv_dim;
  uint32_t scale_log2e_bits;
  uint32_t kv_stride;
  const int8_t *embed;
  const float *embed_s;
  const float *rope_cos;
  const float *rope_sin;
  const float *fin_norm;
  const void *layer_sections[MAX_LAYERS][SEC_COUNT];
};

static struct qwen3_model model;

static float x_buf[HIDDEN_CAP];
static float h1[HIDDEN_CAP];
static float h2[HIDDEN_CAP];
static float gate_buf[INTER_CAP];
static float up_buf[INTER_CAP];
static float attn_buf[Q_DIM_CAP];
static float q_buf[Q_DIM_CAP];
static float k_buf[KV_DIM_CAP];
static float v_buf[KV_DIM_CAP];
static float scores[POS_CAP + 32];
static float logits[VOCAB_CAP];
static uint32_t rcp_word[1];

// Prefill batch buffers: 32 token rows each, row stride = the vector dim.
#define BATCH_ROWS 32
static float pf_x[BATCH_ROWS * HIDDEN_CAP];
static float pf_h1[BATCH_ROWS * HIDDEN_CAP];
static float pf_h2[BATCH_ROWS * HIDDEN_CAP];
static float pf_gate[BATCH_ROWS * INTER_CAP];
static float pf_up[BATCH_ROWS * INTER_CAP];
static float pf_q[BATCH_ROWS * Q_DIM_CAP];
static float pf_attn[BATCH_ROWS * Q_DIM_CAP];
static float pf_k[BATCH_ROWS * KV_DIM_CAP];
static float pf_v[BATCH_ROWS * KV_DIM_CAP];
static int8_t pf_xq[BATCH_ROWS * HIDDEN_CAP];
static float pf_xs[BATCH_ROWS];

static char *const kv_region = (char *)KV_REGION_BASE;

static char *kv_k_ptr(uint32_t layer) { return kv_region + layer * 2 * model.kv_stride; }

static char *kv_v_ptr(uint32_t layer) { return kv_region + (layer * 2 + 1) * model.kv_stride; }

// qwen3_quant_act with the packed-tile dump hooked around it; active only on
// the probed pass's layer 0. The tag tells the q/k/v and o-projection
// quantization sites apart.
static uint32_t quant_act(const float *x, uint32_t k, const char *tag, int active) {
  return active ? probe_quant_act(x, k, tag) : qwen3_quant_act(x, k);
}

// PROBE sites all sit in qwen3_forward(), gated on its `probing` local.
#define PROBE(name, kind, data, count)                                                             \
  do {                                                                                             \
    if (probing) probe_record(name, -1, kind, (const uint32_t *)(data), count);                    \
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
  static const char *const names[SEC_COUNT] = {
      "inln", "qw", "qws",  "kw", "kws", "vw", "vws", "ow", "ows",
      "qn",   "kn", "ptln", "gw", "gws", "uw", "uws", "dw", "dws",
  };
  for (int i = 0; i < SEC_COUNT; ++i) {
    char name[8];
    make_name(name, layer, names[i]);
    model.layer_sections[layer][i] = q3b_find_section(name);
  }
}

void qwen3_init(uint32_t n_prompt) {
  const struct q3b_header *h = q3b_header();
  model.n_layers = h->n_layers;
  model.hidden = h->hidden;
  model.inter = h->inter;
  model.n_q_heads = h->n_q_heads;
  model.n_kv_heads = h->n_kv_heads;
  model.head_dim = h->head_dim;
  model.vocab = h->vocab;
  model.max_pos = h->max_pos;
  model.n_prompt = n_prompt;
  model.eps_bits = h->eps_bits;
  model.q_dim = model.n_q_heads * model.head_dim;
  model.kv_dim = model.n_kv_heads * model.head_dim;
  // the kernels require dims that are multiples of 32 and rope/score chunks
  // that fit 32-lane vectors; inv_n_bits() computes 1/n as 2^-ctz(n), exact
  // only when n is a power of two; the buffers and the kernels.S accumulator
  // are sized by the dims.h capacities. Anything else is rejected up front.
  if (model.n_layers > MAX_LAYERS || model.hidden > HIDDEN_CAP || model.inter > INTER_CAP ||
      model.q_dim > Q_DIM_CAP || model.kv_dim > KV_DIM_CAP || model.max_pos > POS_CAP ||
      model.vocab > VOCAB_CAP || model.hidden == 0 || model.inter == 0 || model.n_q_heads == 0 ||
      model.n_kv_heads == 0 || model.vocab == 0 || model.hidden % 32 != 0 ||
      model.inter % 32 != 0 || model.q_dim % 32 != 0 || model.kv_dim % 32 != 0 ||
      model.vocab % 32 != 0 || model.n_q_heads % model.n_kv_heads != 0 ||
      (model.hidden & (model.hidden - 1)) != 0) {
    panic("unsupported arch dims");
  }
  switch (model.head_dim) {
  case 128: model.scale_log2e_bits = 0x3E0293EEu; break;
  case 64: model.scale_log2e_bits = 0x3E38AA3Bu; break;
  case 32: model.scale_log2e_bits = 0x3E8293EEu; break;
  default: panic("unsupported head_dim: %u", model.head_dim);
  }
  model.kv_stride = model.max_pos * model.kv_dim * 4;
  if ((uint64_t)2 * model.n_layers * model.kv_stride > KV_REGION_SIZE) {
    panic("kv cache exceeds region");
  }
  model.embed = q3b_find_section("embed\0\0");
  model.embed_s = q3b_find_section("embed_s\0");
  model.rope_cos = q3b_find_section("rope_cos");
  model.rope_sin = q3b_find_section("rope_sin");
  model.fin_norm = q3b_find_section("fin_norm");
  for (uint32_t i = 0; i < model.n_layers; ++i)
    cache_layer_sections(i);
}

static void cache_append(uint32_t layer, uint32_t pos, const float *k, const float *v) {
  float *kd = (float *)(kv_k_ptr(layer) + pos * model.kv_dim * 4);
  float *vd = (float *)(kv_v_ptr(layer) + pos * model.kv_dim * 4);
  for (uint32_t i = 0; i < model.kv_dim; ++i) {
    kd[i] = k[i];
    vd[i] = v[i];
  }
}

static void attention(uint32_t layer, uint32_t pos, const float *q, float *attn) {
  const uint32_t group = model.n_q_heads / model.n_kv_heads;
  const uint32_t n_keys = pos + 1;
  const uint32_t n_padded = (n_keys + 31) & ~31u;
  for (uint32_t h = 0; h < model.n_q_heads; ++h) {
    const uint32_t kvh = h / group;
    const char *kb = kv_k_ptr(layer) + kvh * model.head_dim * 4;
    const char *vb = kv_v_ptr(layer) + kvh * model.head_dim * 4;
    qwen3_scores(q + h * model.head_dim, kb, model.kv_dim * 4, scores, n_keys, model.head_dim);
    // each head's softmax overwrites the whole padded block, so the -inf
    // tail must be re-applied per head
    for (uint32_t i = n_keys; i < n_padded; ++i) {
      ((uint32_t *)scores)[i] = 0xFF800000u;
    }
    if (attention_probe_enabled(layer, h, pos, model.n_prompt))
      probe_record("sc", (int)h, "f32", (const uint32_t *)scores, 6);
    qwen3_softmax_exp(scores, n_padded, model.scale_log2e_bits, rcp_word);
    // pb_h appends the softmax rcp as a 7th word (ref.py prints it inline)
    if (attention_probe_enabled(layer, h, pos, model.n_prompt)) {
      uint32_t words[7];
      __builtin_memcpy(words, scores, 6 * sizeof(uint32_t));
      words[6] = rcp_word[0];
      probe_record("pb", (int)h, "f32", words, 7);
    }
    qwen3_pv(vb, model.kv_dim * 4, scores, attn + h * model.head_dim, n_keys, model.head_dim,
             rcp_word[0]);
  }
}

uint32_t qwen3_forward(uint32_t token, uint32_t pos, int want_logits) {
  // probes fire on the last prompt token's logits-producing pass
  const int probing = QWEN3_PROBES && want_logits && pos == model.n_prompt - 1;
  uint32_t xs;
  qwen3_embed(model.embed, model.embed_s, token, x_buf, model.hidden);
  PROBE("embed", "f32", x_buf, 4);
  for (uint32_t L = 0; L < model.n_layers; ++L) {
    const void **sec = model.layer_sections[L];
    qwen3_rmsnorm(x_buf, sec[SEC_INLN], h1, model.hidden, model.eps_bits, inv_n_bits(model.hidden));
    if (L == 0) PROBE("ln0", "f32", h1, 4);
    // q/k/v share one activation quantization; the tiles stay resident
    xs = quant_act(h1, model.hidden, "qx0", probing && L == 0);
    qwen3_matvec_q(sec[SEC_QW], sec[SEC_QWS], q_buf, model.q_dim, model.hidden, xs);
    if (L == 0) PROBE("ac0", "i32", qwen3_accbuf, 8);
    if (L == 0) PROBE("q0", "f32", q_buf, 4);
    qwen3_matvec_q(sec[SEC_KW], sec[SEC_KWS], k_buf, model.kv_dim, model.hidden, xs);
    qwen3_matvec_q(sec[SEC_VW], sec[SEC_VWS], v_buf, model.kv_dim, model.hidden, xs);
    for (uint32_t h = 0; h < model.n_q_heads; ++h) {
      qwen3_rmsnorm(q_buf + h * model.head_dim, sec[SEC_QN], q_buf + h * model.head_dim,
                    model.head_dim, model.eps_bits, inv_n_bits(model.head_dim));
    }
    if (L == 0) PROBE("q0n", "f32", q_buf, 4);
    for (uint32_t h = 0; h < model.n_kv_heads; ++h) {
      qwen3_rmsnorm(k_buf + h * model.head_dim, sec[SEC_KN], k_buf + h * model.head_dim,
                    model.head_dim, model.eps_bits, inv_n_bits(model.head_dim));
    }
    const float *cos_p = model.rope_cos + pos * (model.head_dim / 2);
    const float *sin_p = model.rope_sin + pos * (model.head_dim / 2);
    for (uint32_t h = 0; h < model.n_q_heads; ++h) {
      qwen3_rope(q_buf + h * model.head_dim, cos_p, sin_p, model.head_dim);
    }
    if (L == 0) PROBE("q0r", "f32", q_buf, 4);
    if (L == 0) PROBE("q1r", "f32", q_buf + model.head_dim, 4);
    for (uint32_t h = 0; h < model.n_kv_heads; ++h) {
      qwen3_rope(k_buf + h * model.head_dim, cos_p, sin_p, model.head_dim);
    }
    cache_append(L, pos, k_buf, v_buf);
    attention(L, pos, q_buf, attn_buf);
    if (L == 0) PROBE("attn0", "f32", attn_buf, 4);
    if (L == 0) PROBE("attn1", "f32", attn_buf + model.head_dim, 4);
    xs = quant_act(attn_buf, model.q_dim, "qxa", probing && L == 0);
    qwen3_matvec_q(sec[SEC_OW], sec[SEC_OWS], h2, model.hidden, model.q_dim, xs);
    if (L == 0) PROBE("aca", "i32", qwen3_accbuf, 8);
    if (L == 0) PROBE("o0", "f32", h2, 4);
    qwen3_add(x_buf, h2, model.hidden);
    if (L == 0) PROBE("xr0", "f32", x_buf, 4);
    qwen3_rmsnorm(x_buf, sec[SEC_PTLN], h1, model.hidden, model.eps_bits, inv_n_bits(model.hidden));
    if (L == 0) PROBE("ln1", "f32", h1, 4);
    // gate/up share one activation quantization
    xs = qwen3_quant_act(h1, model.hidden);
    qwen3_matvec_q(sec[SEC_GW], sec[SEC_GWS], gate_buf, model.inter, model.hidden, xs);
    qwen3_matvec_q(sec[SEC_UW], sec[SEC_UWS], up_buf, model.inter, model.hidden, xs);
    if (L == 0) PROBE("g0", "f32", gate_buf, 4);
    if (L == 0) PROBE("u0", "f32", up_buf, 4);
    qwen3_silu_mul(gate_buf, up_buf, gate_buf, model.inter);
    if (L == 0) PROBE("m0", "f32", gate_buf, 4);
    xs = qwen3_quant_act(gate_buf, model.inter);
    qwen3_matvec_q(sec[SEC_DW], sec[SEC_DWS], h2, model.hidden, model.inter, xs);
    if (L == 0) PROBE("d0", "f32", h2, 4);
    qwen3_add(x_buf, h2, model.hidden);
    if (L == 0) PROBE("x_l0", "f32", x_buf, 4);
  }
  if (!want_logits) return 0;
  qwen3_rmsnorm(x_buf, model.fin_norm, h1, model.hidden, model.eps_bits, inv_n_bits(model.hidden));
  PROBE("lnf", "f32", h1, 4);
  xs = qwen3_quant_act(h1, model.hidden);
  qwen3_matvec_q(model.embed, model.embed_s, logits, model.vocab, model.hidden, xs);
  probe_logits(logits);
  return qwen3_argmax(logits, model.vocab);
}

// Prefill for the first n prompt tokens (positions [0, n)): projections run
// as batched MMA over 32-row token batches; rmsnorm/rope/attention/silu/add
// keep the per-row vector kernels. Rows [bt, 32) of the quantized buffers
// are never consumed. The numerics match the per-token path bit for bit
// (same per-row scales, exact i32 accumulation, same epilogue order).
void qwen3_prefill(const uint32_t *tokens, uint32_t n) {
  for (uint32_t base = 0; base < n; base += BATCH_ROWS) {
    const uint32_t bt = n - base < BATCH_ROWS ? n - base : BATCH_ROWS;
    for (uint32_t t = 0; t < bt; ++t)
      qwen3_embed(model.embed, model.embed_s, tokens[base + t], pf_x + t * model.hidden,
                  model.hidden);
    for (uint32_t L = 0; L < model.n_layers; ++L) {
      const void **sec = model.layer_sections[L];
      for (uint32_t t = 0; t < bt; ++t)
        qwen3_rmsnorm(pf_x + t * model.hidden, sec[SEC_INLN], pf_h1 + t * model.hidden,
                      model.hidden, model.eps_bits, inv_n_bits(model.hidden));
      qwen3_quant_rows(pf_h1, model.hidden, pf_xq, pf_xs);
      qwen3_matmul_q(sec[SEC_QW], sec[SEC_QWS], pf_xq, pf_xs, pf_q, model.q_dim, model.hidden);
      qwen3_matmul_q(sec[SEC_KW], sec[SEC_KWS], pf_xq, pf_xs, pf_k, model.kv_dim, model.hidden);
      qwen3_matmul_q(sec[SEC_VW], sec[SEC_VWS], pf_xq, pf_xs, pf_v, model.kv_dim, model.hidden);
      for (uint32_t t = 0; t < bt; ++t) {
        for (uint32_t h = 0; h < model.n_q_heads; ++h)
          qwen3_rmsnorm(pf_q + t * model.q_dim + h * model.head_dim, sec[SEC_QN],
                        pf_q + t * model.q_dim + h * model.head_dim, model.head_dim, model.eps_bits,
                        inv_n_bits(model.head_dim));
        for (uint32_t h = 0; h < model.n_kv_heads; ++h)
          qwen3_rmsnorm(pf_k + t * model.kv_dim + h * model.head_dim, sec[SEC_KN],
                        pf_k + t * model.kv_dim + h * model.head_dim, model.head_dim,
                        model.eps_bits, inv_n_bits(model.head_dim));
        const float *cos_p = model.rope_cos + (base + t) * (model.head_dim / 2);
        const float *sin_p = model.rope_sin + (base + t) * (model.head_dim / 2);
        for (uint32_t h = 0; h < model.n_q_heads; ++h)
          qwen3_rope(pf_q + t * model.q_dim + h * model.head_dim, cos_p, sin_p, model.head_dim);
        for (uint32_t h = 0; h < model.n_kv_heads; ++h)
          qwen3_rope(pf_k + t * model.kv_dim + h * model.head_dim, cos_p, sin_p, model.head_dim);
        cache_append(L, base + t, pf_k + t * model.kv_dim, pf_v + t * model.kv_dim);
        attention(L, base + t, pf_q + t * model.q_dim, pf_attn + t * model.q_dim);
      }
      qwen3_quant_rows(pf_attn, model.q_dim, pf_xq, pf_xs);
      qwen3_matmul_q(sec[SEC_OW], sec[SEC_OWS], pf_xq, pf_xs, pf_h2, model.hidden, model.q_dim);
      for (uint32_t t = 0; t < bt; ++t)
        qwen3_add(pf_x + t * model.hidden, pf_h2 + t * model.hidden, model.hidden);
      for (uint32_t t = 0; t < bt; ++t)
        qwen3_rmsnorm(pf_x + t * model.hidden, sec[SEC_PTLN], pf_h1 + t * model.hidden,
                      model.hidden, model.eps_bits, inv_n_bits(model.hidden));
      qwen3_quant_rows(pf_h1, model.hidden, pf_xq, pf_xs);
      qwen3_matmul_q(sec[SEC_GW], sec[SEC_GWS], pf_xq, pf_xs, pf_gate, model.inter, model.hidden);
      qwen3_matmul_q(sec[SEC_UW], sec[SEC_UWS], pf_xq, pf_xs, pf_up, model.inter, model.hidden);
      for (uint32_t t = 0; t < bt; ++t)
        qwen3_silu_mul(pf_gate + t * model.inter, pf_up + t * model.inter,
                       pf_gate + t * model.inter, model.inter);
      qwen3_quant_rows(pf_gate, model.inter, pf_xq, pf_xs);
      qwen3_matmul_q(sec[SEC_DW], sec[SEC_DWS], pf_xq, pf_xs, pf_h2, model.hidden, model.inter);
      for (uint32_t t = 0; t < bt; ++t)
        qwen3_add(pf_x + t * model.hidden, pf_h2 + t * model.hidden, model.hidden);
    }
  }
}
