#pragma once

// Q3B blob layout and lookup interface.

#include <stdint.h>

#define BLOB_BASE 0x81000000ull
#define Q3B_MAGIC 0x00423351u // "Q3B\0" little-endian

struct q3b_header {
  uint32_t magic; // "Q3B"
  uint32_t n_layers;
  uint32_t hidden;
  uint32_t inter;
  uint32_t n_q_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t vocab;
  uint32_t max_pos;
  uint32_t eps_bits; // f32 rmsnorm eps
  uint32_t n_sections;
  uint32_t reserved; // must be zero
};

struct q3b_section {
  char name[8];
  uint32_t offset; // from blob base
  uint32_t size;
};

const struct q3b_header *q3b_header(void);
const void *q3b_find_section(const char name[8]);
