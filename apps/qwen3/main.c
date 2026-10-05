#include "qwen3.h"

void forward_init(void);
uint32_t forward(uint32_t token, uint32_t pos, int want_logits);
const void *blob_section(const char name[8]);
const struct q3b_header *blob_header(void);

#define MAX_NEW_TOKENS 32
#define EOS_IM_END 151645u
#define EOS_ENDOFTEXT 151643u

int main(void) {
  const struct q3b_header *h = blob_header();
  if (h->magic != 0x31423351u) { // "Q3B1" little-endian
    print_str("bad blob magic: ");
    print_hex32(h->magic);
    semihost_putc('\n');
    return 1;
  }
  forward_init();
  const uint32_t *tokens = blob_section("tokens\0\0");
  print_str("prompt tokens: ");
  for (uint32_t i = 0; i < h->n_prompt; ++i) {
    print_u32(tokens[i]);
    semihost_putc(' ');
  }
  semihost_putc('\n');

  uint32_t pos = 0;
  for (; pos + 1 < h->n_prompt; ++pos) {
    forward(tokens[pos], pos, 0);
  }
  uint32_t next = forward(tokens[pos], pos, 1);
  ++pos;

  print_str("generated: ");
  for (uint32_t i = 0; i < MAX_NEW_TOKENS && pos < h->max_pos; ++i) {
    print_u32(next);
    semihost_putc(' ');
    if (next == EOS_IM_END || next == EOS_ENDOFTEXT) break;
    next = forward(next, pos, 1);
    ++pos;
  }
  semihost_putc('\n');
  return 0;
}
