#pragma once

// qwen3_ = Qwen3 NPU kernel. Implementations live in kernels.S.

#include <stdint.h>

uint32_t qwen3_quant_act(const float *x, uint32_t k);
void qwen3_matvec_q(const int8_t *w, const float *w_scales, float *out, uint32_t n, uint32_t k,
                    uint32_t xs_bits);
void qwen3_quant_rows(const float *x, uint32_t k, int8_t *xq, float *xs);
void qwen3_matmul_q(const int8_t *w, const float *w_scales, const int8_t *xq, const float *xs,
                    float *out, uint32_t n, uint32_t k);
void qwen3_rmsnorm(float *x, const float *gamma, float *out, uint32_t n, uint32_t eps_bits,
                   uint32_t inv_n_bits);
void qwen3_rope(float *x, const float *cos_h, const float *sin_h, uint32_t hd);
void qwen3_scores(const float *q, const char *k_base, uint32_t row_stride, float *scores,
                  uint32_t n_keys, uint32_t hd);
void qwen3_softmax_exp(float *scores, uint32_t n, uint32_t scale_log2e_bits, uint32_t *rcp_out);
void qwen3_pv(const char *v_base, uint32_t row_stride, const float *probs, float *out,
              uint32_t n_keys, uint32_t hd, uint32_t rcp_bits);
void qwen3_silu_mul(float *gate, const float *up, float *out, uint32_t n);
void qwen3_add(float *dst, const float *src, uint32_t n);
void qwen3_embed(const int8_t *table, const float *scales, uint32_t id, float *out,
                 uint32_t hidden);
uint32_t qwen3_argmax(const float *x, uint32_t n);

void qwen3_fence_mem(void);

extern char *qwen3_debug_qtile;
extern int32_t qwen3_accbuf[];
