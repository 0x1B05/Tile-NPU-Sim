#include "q3p.h"

// Q3P1 prompt image is loaded by the runner at a fixed address before the
// guest starts. The header and token buffer stay valid for the whole run.

static const struct q3p_header *const prompt = (const struct q3p_header *)PROMPT_BASE;

const struct q3p_header *q3p_header(void) { return prompt; }
