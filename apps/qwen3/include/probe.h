#pragma once

// Numeric alignment probes, compiled in by default for ref.py comparison.

#include <stdint.h>

#ifndef QWEN3_PROBES
#define QWEN3_PROBES 1
#endif

void probe_record(const char *name, int index, const char *kind, const uint32_t *words, int n);
int attention_probe_enabled(uint32_t layer, uint32_t head, uint32_t pos, uint32_t n_prompt);
uint32_t probe_quant_act(const float *x, uint32_t k, const char *tag);
void probe_logits(const float *logits);
#if QWEN3_PROBES
void qwen3_probe_last_prompt(uint32_t token, uint32_t pos);
#endif
