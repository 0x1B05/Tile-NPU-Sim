#include "npu.inc"

# Positive tadd image: one plain tadd, then one with dest aliasing src1.
# The host driver asserts tile contents after each step; ebreak ends the run.

    .globl _start
_start:
    TADD    0, 1, 2
    TADD    1, 1, 2
    ebreak
