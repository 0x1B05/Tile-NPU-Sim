#pragma once

#include "execute.h"
#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <variant>

namespace tile_npu {

// Resolves an IndexOperand: the immediate value, or the named Scalar
// register's value.
inline std::uint64_t resolve_index(const IndexOperand &index, const State &state) {
  return std::visit(
      Overloaded{[](ImmediateIndex imm) { return static_cast<std::uint64_t>(imm.value); },
                 [&state](ScalarRegister reg) { return state.scalar.read_x(reg.index); }},
      index);
}

// decode_scalar(xS) keeps the low element-width bits of xS, interpreted per
// the destination's bound config dtype.
template <typename Element> Element scalar_operand(const State &state, ScalarRegister reg) {
  return static_cast<Element>(state.scalar.read_x(reg.index));
}

// Resolves src2 into an element accessor by storage offset for fn: a
// register source reads the same offset of the source register, a scalar
// source broadcasts one value to every element.
template <typename Regs, typename Fn>
ExecuteResult with_src2(const NpuOperand &src2, const State &state, const Regs &regs, Fn fn) {
  using Element = typename Regs::value_type::value_type;
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 const auto &reg = regs[rhs.index];
                                 return fn([&reg](std::size_t offset) { return reg[offset]; });
                               },
                               [&](ScalarRegister reg) {
                                 const auto value = scalar_operand<Element>(state, reg);
                                 return fn([value](std::size_t) { return value; });
                               }},
                    src2);
}

} // namespace tile_npu
