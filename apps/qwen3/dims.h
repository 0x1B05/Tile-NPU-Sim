#pragma once

/* Buffer capacities shared by forward.c (array sizes and the arch-dim
   validation) and kernels.S (the accumulator .skip), so a capacity changes
   in exactly one place. Models exceeding a capacity are rejected at load
   time. Pure defines: this header is included from both C and assembly. */
#define HIDDEN_CAP 3072
#define INTER_CAP 3072
#define Q_DIM_CAP 2048
#define KV_DIM_CAP 1024
#define POS_CAP 512
#define VOCAB_CAP 151936
