#include "qwen3.h"

void forward_init(void);
void forward_prefill(const uint32_t *tokens, uint32_t n);
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
  if (h->n_prompt == 0 || h->n_prompt > h->max_pos) {
    print_str("prompt length out of range\n");
    return 1;
  }
  for (uint32_t i = 0; i < h->n_prompt; ++i) {
    if (tokens[i] >= h->vocab) {
      print_str("prompt token id out of range\n");
      return 1;
    }
  }
  print_str("prompt tokens: ");
  for (uint32_t i = 0; i < h->n_prompt; ++i) {
    print_u32(tokens[i]);
    semihost_putc(' ');
  }
  semihost_putc('\n');

  if (h->n_prompt > 1) forward_prefill(tokens, h->n_prompt - 1);
  uint32_t pos = h->n_prompt - 1;
  uint32_t next = forward(tokens[pos], pos, 1);
  ++pos;

  print_str("generated: ");
  for (uint32_t i = 0; i < MAX_NEW_TOKENS && pos < h->max_pos; ++i) {
    print_u32(next);
    semihost_putc(' ');
    if (next == EOS_IM_END || next == EOS_ENDOFTEXT) break;
    // skip the forward when the next iteration would not run; its result
    // would never be printed
    if (i + 1 == MAX_NEW_TOKENS || pos + 1 >= h->max_pos) break;
    next = forward(next, pos, 1);
    ++pos;
  }
  semihost_putc('\n');
  return 0;
}
