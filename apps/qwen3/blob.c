#include "qwen3.h"

// Q3B1 blob directory access. Sections are looked up by name at startup;
// the pointers stay valid for the whole run.

static const struct q3b_header *const blob =
    (const struct q3b_header *)BLOB_BASE;

static const struct q3b_section *sections =
    (const struct q3b_section *)(BLOB_BASE + sizeof(struct q3b_header));

const void *blob_section(const char name[8]) {
  for (uint32_t i = 0; i < blob->n_sections; ++i) {
    uint64_t want, have;
    __builtin_memcpy(&want, name, 8);
    __builtin_memcpy(&have, sections[i].name, 8);
    if (want == have) {
      return (const void *)(BLOB_BASE + sections[i].offset);
    }
  }
  print_str("missing blob section: ");
  for (int i = 0; i < 8 && name[i]; ++i)
    semihost_putc(name[i]);
  semihost_putc('\n');
  semihost_exit(1);
}

const struct q3b_header *blob_header(void) { return blob; }
