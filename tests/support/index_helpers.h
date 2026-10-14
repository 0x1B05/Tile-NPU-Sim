#pragma once

#include "tile_npu/instruction.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "framework.h"

// Shared decode-side assertion for the IndexOperand variant: an immediate
// index decodes to ImmediateIndex, an xS form to ScalarRegister.
inline void check_index(const tile_npu::IndexOperand &operand, std::uint8_t expected,
                        bool is_scalar, std::string_view what) {
  const std::string context(what);
  if (is_scalar) {
    const auto *reg = std::get_if<tile_npu::ScalarRegister>(&operand);
    check(reg != nullptr, context + ": index is scalar");
    if (reg != nullptr) {
      check_eq(reg->index, expected, context + ": index");
    }
    return;
  }
  const auto *imm = std::get_if<tile_npu::ImmediateIndex>(&operand);
  check(imm != nullptr, context + ": index is immediate");
  if (imm != nullptr) {
    check_eq(imm->value, expected, context + ": index");
  }
}
