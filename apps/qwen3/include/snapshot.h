#pragma once

// Fixed guest-RAM snapshot region used by the tiny Qwen3 regression.

#include "qwen3.h"

#include <stdint.h>

#define SNAPSHOT_MAGIC 0x00504E53u // "SNP\0" little-endian
#define SNAPSHOT_VERSION 1u

enum {
  SNAPSHOT_F32 = 0,
  SNAPSHOT_I8 = 1,
  SNAPSHOT_I32 = 2,
};

struct snapshot_header {
  uint32_t magic;
  uint32_t version;
  uint32_t used;
  uint32_t records;
};

struct snapshot_record {
  char name[8];
  uint32_t kind;
  uint32_t count;
  uint32_t bytes;
  uint32_t reserved[3];
};

void snapshot_reset(void);
void snapshot_add(const char name[8], uint32_t kind, const void *data, uint32_t count);
