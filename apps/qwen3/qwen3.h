#pragma once

// Qwen3 inference app for the Tile-NPU simulator: shared layout constants,
// the Q3B1 blob format view, and semihosting I/O (monitor conventions, not
// ISA behavior).
//
// Guest memory map (sim_run --run -m 2048):
//   0x80000000  this image (code + rodata + data + bss, a few MB)
//   0x81000000  Q3B1 model blob (598 MB for Qwen3-0.6B)
//   0xB0000000  KV cache region (256 MB budget)
//   0xC0000000  stack top

#include <stdint.h>

#define BLOB_BASE 0x81000000ull
#define KV_REGION_BASE 0xB0000000ull
#define KV_REGION_SIZE (256ull << 20)
#define STACK_TOP 0xC0000000ull

struct q3b_header {
  uint32_t magic; // "Q3B1"
  uint32_t n_layers;
  uint32_t hidden;
  uint32_t inter;
  uint32_t n_q_heads;
  uint32_t n_kv_heads;
  uint32_t head_dim;
  uint32_t vocab;
  uint32_t max_pos;
  uint32_t n_prompt;
  uint32_t eps_bits; // f32 rmsnorm eps
  uint32_t n_sections;
};

struct q3b_section {
  char name[8];
  uint32_t offset; // from blob base
  uint32_t size;
};

// Semihosting ecalls (see src/monitor/runner.h): a7 = 1 prints a0's low
// byte, a7 = 93 exits with a0 as the process exit code.
static inline void semihost_putc(int c) {
  register long a0 asm("a0") = c;
  register long a7 asm("a7") = 1;
  asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
}

static inline void semihost_exit(long code) {
  register long a0 asm("a0") = code;
  register long a7 asm("a7") = 93;
  asm volatile("ecall" : "+r"(a0) : "r"(a7) : "memory");
  __builtin_unreachable();
}

static inline void print_str(const char *s) {
  while (*s)
    semihost_putc(*s++);
}

static inline void print_u32(uint32_t v) {
  char buf[10];
  int n = 0;
  if (v == 0) {
    semihost_putc('0');
    return;
  }
  while (v) {
    buf[n++] = '0' + v % 10;
    v /= 10;
  }
  while (n)
    semihost_putc(buf[--n]);
}

static inline void print_hex32(uint32_t v) {
  for (int shift = 28; shift >= 0; shift -= 4) {
    semihost_putc("0123456789abcdef"[(v >> shift) & 0xf]);
  }
  semihost_putc(' ');
}
