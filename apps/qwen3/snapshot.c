#include "snapshot.h"

#include "console.h"

#include <stddef.h>
#include <stdint.h>

static struct snapshot_header *header(void) { return (struct snapshot_header *)SNAPSHOT_BASE; }

static uint8_t *bytes(void) { return (uint8_t *)SNAPSHOT_BASE; }

static uint32_t kind_size(uint32_t kind) {
  switch (kind) {
  case SNAPSHOT_F32:
  case SNAPSHOT_I32: return 4;
  case SNAPSHOT_I8: return 1;
  default: panic("bad snapshot kind: %u", kind);
  }
}

static void copy_name(char dst[8], const char name[8]) {
  for (size_t i = 0; i < 8; ++i) {
    dst[i] = name[i];
    if (name[i] == '\0') {
      for (++i; i < 8; ++i) dst[i] = '\0';
      break;
    }
  }
}

void snapshot_reset(void) {
  struct snapshot_header *h = header();
  h->magic = SNAPSHOT_MAGIC;
  h->version = SNAPSHOT_VERSION;
  h->used = sizeof(*h);
  h->records = 0;
}

void snapshot_add(const char name[8], uint32_t kind, const void *data, uint32_t count) {
  struct snapshot_header *h = header();
  const uint32_t elem_size = kind_size(kind);
  const uint32_t data_bytes = count * elem_size;
  uint32_t used = (h->used + 15u) & ~15u;
  if (used + sizeof(struct snapshot_record) + data_bytes > SNAPSHOT_SIZE) {
    panic("snapshot region full: %u bytes", used + sizeof(struct snapshot_record) + data_bytes);
  }

  struct snapshot_record *record = (struct snapshot_record *)(bytes() + used);
  copy_name(record->name, name);
  record->kind = kind;
  record->count = count;
  record->bytes = data_bytes;
  for (size_t i = 0; i < 3; ++i) record->reserved[i] = 0;

  uint8_t *dst = bytes() + used + sizeof(*record);
  const uint8_t *src = (const uint8_t *)data;
  for (uint32_t i = 0; i < data_bytes; ++i) dst[i] = src[i];

  h->used = used + sizeof(*record) + data_bytes;
  ++h->records;
}
