#pragma once

// Qwen3 inference app API and guest memory map.

#include <stdint.h>

// Guest memory map (sim_run -m 1024):
//   0x80000000  this image (code + rodata + data + bss, a few MB)
//   0x81000000  Q3B1 model blob (598 MB for Qwen3-0.6B)
//   0xB0000000  KV cache region (256 MB budget)
//   0xC0000000  stack top

#define KV_REGION_BASE 0xB0000000ull
#define KV_REGION_SIZE (256ull << 20)
#define STACK_TOP 0xC0000000ull

void qwen3_init(void);
void qwen3_prefill(const uint32_t *tokens, uint32_t n);
uint32_t qwen3_forward(uint32_t token, uint32_t pos, int want_logits);
