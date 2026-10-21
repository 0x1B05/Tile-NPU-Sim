#include "probe.h"

#include "console.h"
#include "kernels.h"

// Packed activation tile dumped around qwen3_quant_act on probed calls.
static char qtile_dump[1024];

void probe_record(const char *name, int index, const char *kind, const uint32_t *words, int n) {
  printf("PROBE %s", name);
  if (index >= 0) printf(" %d", index);
  printf(" %s %d", kind, n);
  for (int i = 0; i < n; ++i)
    printf(" %08x", words[i]);
  printf("\n");
}

int attention_probe_enabled(uint32_t layer, uint32_t head, uint32_t pos, uint32_t n_prompt) {
  return QWEN3_PROBES && layer == 0 && head < 2 && pos == n_prompt - 1;
}

uint32_t probe_quant_act(const float *x, uint32_t k, const char *tag) {
  qwen3_debug_qtile = qtile_dump;
  uint32_t xs = qwen3_quant_act(x, k);
  qwen3_debug_qtile = 0;
  probe_record(tag, -1, "i8", (const uint32_t *)qtile_dump, 16);
  return xs;
}

void probe_logits(const float *logits) {
  static int dumped = 0;
  if (!QWEN3_PROBES || dumped) return;
  dumped = 1;
  static const uint32_t ids[] = {7, 174, 303, 113, 133};
  uint32_t words[5];
  for (int i = 0; i < 5; ++i)
    words[i] = ((const uint32_t *)logits)[ids[i]];
  probe_record("logits", -1, "f32", words, 5);
}
