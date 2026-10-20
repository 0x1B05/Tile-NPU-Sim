#include "q3b.h"

#include "console.h"

// Q3B1 blob directory access. Sections are looked up by name at startup;
// the pointers stay valid for the whole run.

static const struct q3b_header *const blob = (const struct q3b_header *)BLOB_BASE;

static const struct q3b_section *sections =
    (const struct q3b_section *)(BLOB_BASE + sizeof(struct q3b_header));

const void *q3b_find_section(const char name[8]) {
  for (uint32_t i = 0; i < blob->n_sections; ++i) {
    uint64_t want, have;
    __builtin_memcpy(&want, name, 8);
    __builtin_memcpy(&have, sections[i].name, 8);
    if (want == have) {
      return (const void *)(BLOB_BASE + sections[i].offset);
    }
  }
  char text[9];
  __builtin_memcpy(text, name, 8);
  text[8] = '\0';
  panic("missing blob section: %s", text);
}

const struct q3b_header *q3b_header(void) { return blob; }
