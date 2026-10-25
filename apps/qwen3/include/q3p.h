#pragma once

// Q3P1 runtime prompt image and guest address.

#include <stdint.h>

#define PROMPT_BASE 0x80F00000ull
#define Q3P_MAGIC 0x31503351u // "Q3P1" little-endian

struct q3p_header {
  uint32_t magic; // "Q3P1"
  uint32_t n_prompt;
  uint32_t reserved[2]; // must be zero
  uint32_t tokens[];
};

const struct q3p_header *q3p_header(void);
