#include "console.h"

#include <stdarg.h>

// Serial THR is memory-mapped by the simulator.
#define SERIAL_THR ((volatile unsigned char *)0x10000000ull)

static void serial_putc(int c) { *SERIAL_THR = (unsigned char)c; }

void halt(long code) {
  register long a0 asm("a0") = code;
  asm volatile("ebreak" : : "r"(a0) : "memory");
  __builtin_unreachable();
}

static int print_text(const char *s) {
  int n = 0;
  if (s == 0) s = "(null)";
  while (*s) {
    serial_putc(*s++);
    ++n;
  }
  return n;
}

static int print_uint(uint32_t value, uint32_t base, int width, int zero_pad, int upper) {
  const char *digits = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  char buf[32];
  int len = 0;
  int n = 0;
  if (value == 0) {
    buf[len++] = '0';
  } else {
    while (value != 0) {
      buf[len++] = digits[value % base];
      value /= base;
    }
  }
  for (int i = len; i < width; ++i) {
    serial_putc(zero_pad ? '0' : ' ');
    ++n;
  }
  while (len > 0) {
    serial_putc(buf[--len]);
    ++n;
  }
  return n;
}

static int print_int(int32_t value) {
  if (value < 0) {
    serial_putc('-');
    return 1 + print_uint((uint32_t)(-value), 10, 0, 0, 0);
  }
  return print_uint((uint32_t)value, 10, 0, 0, 0);
}

int printf(const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  int n = 0;
  while (*fmt) {
    if (*fmt != '%') {
      serial_putc(*fmt++);
      ++n;
      continue;
    }
    ++fmt;
    int zero_pad = 0;
    int width = 0;
    if (*fmt == '0') {
      zero_pad = 1;
      ++fmt;
    }
    while (*fmt >= '0' && *fmt <= '9') {
      width = width * 10 + (*fmt - '0');
      ++fmt;
    }
    switch (*fmt) {
    case 's': n += print_text(va_arg(args, const char *)); break;
    case 'u': n += print_uint(va_arg(args, uint32_t), 10, width, zero_pad, 0); break;
    case 'd': n += print_int(va_arg(args, int32_t)); break;
    case 'x': n += print_uint(va_arg(args, uint32_t), 16, width, zero_pad, 0); break;
    case 'X': n += print_uint(va_arg(args, uint32_t), 16, width, zero_pad, 1); break;
    case 'c':
      serial_putc(va_arg(args, int));
      ++n;
      break;
    case '%':
      serial_putc('%');
      ++n;
      break;
    default:
      serial_putc('%');
      serial_putc(*fmt);
      n += 2;
      break;
    }
    ++fmt;
  }
  va_end(args);
  return n;
}
