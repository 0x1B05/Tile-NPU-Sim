#pragma once

// Qwen3 inference app API and guest memory map.

#include <stdint.h>

// Guest memory map (default 1 GiB):
//   0x80000000  this image (code + rodata + data + bss, a few MB)
//   0x80B00000  test snapshot region (4 MiB)
//   0x80F00000  Q3P runtime prompt image
//   0x81000000  Q3B model blob (598 MB for Qwen3-0.6B)
//   0xB0000000  KV cache region (256 MB budget)
//   0xC0000000  stack top

#define SNAPSHOT_BASE 0x80B00000ull
#define SNAPSHOT_SIZE (4ull << 20)
#define KV_REGION_BASE 0xB0000000ull
#define KV_REGION_SIZE (256ull << 20)
#define STACK_TOP 0xC0000000ull

void qwen3_init(uint32_t n_prompt);
uint32_t qwen3_prefill(const uint32_t *tokens, uint32_t n);
uint32_t qwen3_forward(uint32_t token, uint32_t pos, int want_logits);

#if QWEN3_SNAPSHOT
void qwen3_snapshot_last_prompt(uint32_t token, uint32_t pos);
#endif
