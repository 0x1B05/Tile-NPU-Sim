// Minimal riscv-tests environment for the Tile-NPU simulator: no privileged
// state, no tohost/HTIF. Pass/fail ride the runner convention: ebreak with
// a0 == 0 passes; a failing case reports (TESTNUM << 1) | 1.
// Tests that require trap handling (ma_*) or Zifencei (fence_i) are excluded
// by tests/riscv_tests.passlist.

#ifndef _TILE_NPU_RISCV_TEST_ENV_H
#define _TILE_NPU_RISCV_TEST_ENV_H

#define RVTEST_RV64U                                                    \
  .macro init;                                                          \
  .endm

#define INIT_XREG                                                       \
  li x1, 0;  li x2, 0;  li x3, 0;  li x4, 0;                            \
  li x5, 0;  li x6, 0;  li x7, 0;  li x8, 0;                            \
  li x9, 0;  li x10, 0; li x11, 0; li x12, 0;                           \
  li x13, 0; li x14, 0; li x15, 0; li x16, 0;                           \
  li x17, 0; li x18, 0; li x19, 0; li x20, 0;                           \
  li x21, 0; li x22, 0; li x23, 0; li x24, 0;                           \
  li x25, 0; li x26, 0; li x27, 0; li x28, 0;                           \
  li x29, 0; li x30, 0; li x31, 0;

#define RVTEST_CODE_BEGIN                                               \
        .section .text.crt0;                                            \
        .align 2;                                                       \
        .globl _start;                                                  \
_start:                                                                 \
        INIT_XREG;                                                      \
        li TESTNUM, 0;                                                  \
        init;                                                           \
        .section .text;

#define RVTEST_CODE_END                                                 \
        .align 2;

#define TESTNUM gp

#define RVTEST_PASS                                                     \
        li TESTNUM, 1;                                                  \
        li a0, 0;                                                       \
        ebreak

#define RVTEST_FAIL                                                     \
        sll TESTNUM, TESTNUM, 1;                                        \
        ori TESTNUM, TESTNUM, 1;                                        \
        mv a0, TESTNUM;                                                 \
        ebreak

#define RVTEST_DATA_BEGIN                                               \
        .align 4; .global begin_signature; begin_signature:

#define RVTEST_DATA_END .align 4; .global end_signature; end_signature:

#endif
