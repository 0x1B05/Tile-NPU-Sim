#include "qwen3.h"

#include "console.h"
#include "q3b.h"
#include "q3p.h"

#define MAX_NEW_TOKENS 32
#define EOS_IM_END 151645u
#define EOS_ENDOFTEXT 151643u

int main(void) {
  const struct q3b_header *h = q3b_header();
  if ((h)->magic != Q3B_MAGIC) panic("bad blob magic: %x", (h)->magic);

  const struct q3p_header *p = q3p_header();
  if (p->magic != Q3P_MAGIC) panic("missing prompt image: %x", p->magic);
  if (p->n_prompt == 0 || p->n_prompt > h->max_pos) panic("prompt length out of range");
  for (uint32_t i = 0; i < p->n_prompt; ++i) {
    if (p->tokens[i] >= h->vocab) panic("prompt token id out of range: %u", p->tokens[i]);
  }

  qwen3_init(p->n_prompt);

  const uint32_t *tokens = p->tokens;
  uint32_t n_prompt = p->n_prompt;
  printf("prompt tokens: ");
  for (uint32_t i = 0; i < n_prompt; ++i)
    printf("%u ", tokens[i]);
  printf("\n");

  uint32_t next = qwen3_prefill(tokens, n_prompt);
  uint32_t pos = n_prompt;

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
