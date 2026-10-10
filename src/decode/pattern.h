#pragma once

#include "encoding.h"
#include "tile_npu/decode.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {

// INSTPAT-style instruction pattern: '0'/'1' bits must match, '?' is a
// wildcard, spaces are ignored (use them to mirror field boundaries).
struct Pattern {
  std::uint32_t key = 0;
  std::uint32_t mask = 0;

  template <std::size_t N> consteval Pattern(const char (&str)[N]) {
    int bit = 31;
    for (char c : str) {
      if (c == '\0') break;
      if (c == ' ') continue;
      if (c == '0') {
        mask |= std::uint32_t{1} << bit;
      } else if (c == '1') {
        key |= std::uint32_t{1} << bit;
        mask |= std::uint32_t{1} << bit;
      } else if (c != '?') {
        throw "pattern characters must be 0, 1, or ?";
      }
      --bit;
    }
    if (bit != -1) {
      throw "pattern must have exactly 32 bits";
    }
  }

  constexpr bool matches(std::uint32_t word) const { return (word & mask) == key; }
};

using DecodeHandler = DecodeResult (*)(std::uint32_t word);

// A decode table is scanned in order and the first matching entry wins, so
// specific instruction patterns come before catch-alls.
struct DecodeEntry {
  Pattern pattern;
  DecodeHandler handler;
};

template <std::size_t N> DecodeResult match(const DecodeEntry (&table)[N], std::uint32_t word) {
  for (const auto &entry : table) {
    if (entry.pattern.matches(word)) {
      return entry.handler(word);
    }
  }
  return DecodeError::illegal;
}

// Format-driven scalar operand extraction: the format alone determines how
// the word becomes a DecodedScalarInstr.
inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvR &f) {
  return {op, {f.dest, f.src1, f.src2}};
}

inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvI &f) {
  return {op, {f.dest, f.src1, 0}, f.imm};
}

inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvS &f) {
  return {op, {0, f.src1, f.src2}, f.imm};
}

inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvB &f) {
  return {op, {0, f.src1, f.src2}, f.imm};
}

inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvU &f) {
  return {op, {f.dest, 0, 0}, f.imm};
}

inline DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvJ &f) {
  return {op, {f.dest, 0, 0}, f.imm};
}

template <unsigned ShamtBits>
DecodedScalarInstr make_instr(ScalarOp op, const encoding::RiscvShift<ShamtBits> &f) {
  return {op, {f.dest, f.src1, 0}, f.shamt};
}

template <typename Format, ScalarOp Op> DecodeResult as(std::uint32_t word) {
  return make_instr(Op, Format::from(word));
}

template <ScalarOp Op> DecodeResult no_operands(std::uint32_t) {
  return DecodedScalarInstr{Op, {0, 0, 0}};
}

template <DecodeError E> DecodeResult fail(std::uint32_t) { return E; }

} // namespace tile_npu
