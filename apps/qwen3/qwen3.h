#pragma once

// Qwen3 inference app for the Tile-NPU simulator: shared layout constants,
// the Q3B1 blob format view, and console I/O (platform conventions, not
// ISA behavior).
//
// Guest memory map (sim_run -m 2048):
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

// Console I/O (platform conventions, not ISA behavior): storing a byte at
// the serial THR prints it on the host (see include/tile_npu/serial.h);
// ebreak stops the simulator with a0 as the process exit code.
#define SERIAL_THR ((volatile unsigned char *)0x10000000ull)

static inline void serial_putc(int c) { *SERIAL_THR = (unsigned char)c; }

static inline void halt(long code) {
  register long a0 asm("a0") = code;
  asm volatile("ebreak" : : "r"(a0) : "memory");
  __builtin_unreachable();
}

static inline void print_str(const char *s) {
  while (*s)
    serial_putc(*s++);
}

static inline void print_u32(uint32_t v) {
  char buf[10];
  int n = 0;
  if (v == 0) {
    serial_putc('0');
    return;
  }
  while (v) {
    buf[n++] = '0' + v % 10;
    v /= 10;
  }
  while (n)
    serial_putc(buf[--n]);
}

static inline void print_hex32(uint32_t v) {
  for (int shift = 28; shift >= 0; shift -= 4) {
    serial_putc("0123456789abcdef"[(v >> shift) & 0xf]);
  }
  serial_putc(' ');
}
