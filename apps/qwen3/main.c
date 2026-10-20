#include "qwen3.h"

#include "console.h"
#include "q3b.h"

#define MAX_NEW_TOKENS 32
#define EOS_IM_END 151645u
#define EOS_ENDOFTEXT 151643u

int main(void) {
  const struct q3b_header *h = q3b_header();
  if ((h)->magic != Q3B_MAGIC) panic("bad blob magic: %x", (h)->magic);

  qwen3_init();

  const uint32_t *tokens = q3b_find_section("tokens\0\0");
  if (h->n_prompt == 0 || h->n_prompt > h->max_pos) panic("prompt length out of range");
  for (uint32_t i = 0; i < h->n_prompt; ++i) {
    if (tokens[i] >= h->vocab) panic("prompt token id out of range: %u", tokens[i]);
  }
  printf("prompt tokens: ");
  for (uint32_t i = 0; i < h->n_prompt; ++i)
    printf("%u ", tokens[i]);
  printf("\n");

  if (h->n_prompt > 1) qwen3_prefill(tokens, h->n_prompt - 1);
  uint32_t pos = h->n_prompt - 1;
  uint32_t next = qwen3_forward(tokens[pos], pos, 1);
  ++pos;

  printf("generated: ");
  for (uint32_t i = 0; i < MAX_NEW_TOKENS && pos < h->max_pos; ++i) {
    printf("%u ", next);
    if (next == EOS_IM_END || next == EOS_ENDOFTEXT) break;
    // skip the forward when the next iteration would not run; its result
    // would never be printed
    if (i + 1 == MAX_NEW_TOKENS || pos + 1 >= h->max_pos) break;
    next = qwen3_forward(next, pos, 1);
    ++pos;
  }
  printf("\n");
  return 0;
}
