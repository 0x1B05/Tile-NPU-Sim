#pragma once

// Minimal serial console and panic helpers for the bare-metal guest.

#include <stdint.h>

void halt(long code) __attribute__((noreturn));
int printf(const char *fmt, ...);

#define panic(fmt, ...)                                                                            \
  do {                                                                                             \
    printf("panic: " fmt "\n", ##__VA_ARGS__);                                                     \
    halt(1);                                                                                       \
  } while (0)
