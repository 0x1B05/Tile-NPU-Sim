#pragma once

#include "execute.h"
#include "tile_npu/matrix_view.h"
#include "utils/overloaded.h"

#include <array>
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

// Resolves a matrix-domain src2 into an element accessor for the run_* body:
// a register source reads the same coordinate of the source register, a
// scalar source broadcasts the same value to every element.
template <typename Element, std::size_t RegCount, typename Body>
ExecuteResult
with_src2_matrix(const NpuOperand &src2, const State &state,
                 const std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
                 Body body) {
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 auto right = matrix_view(regs[rhs.index]);
                                 return body([&](std::size_t row, std::size_t col) {
                                   return right.at(row, col);
                                 });
                               },
                               [&](ScalarRegister reg) {
                                 const auto value = scalar_operand<Element>(state, reg);
                                 return body([value](std::size_t, std::size_t) { return value; });
                               }},
                    src2);
}

// The vector-domain form: a register source reads by lane, a scalar source
// broadcasts to every lane.
template <typename Element, std::size_t RegCount, typename Body>
ExecuteResult with_src2_vector(const NpuOperand &src2, const State &state,
                               const std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
                               Body body) {
  return std::visit(Overloaded{[&](NpuRegister rhs) {
                                 const auto &rhs_reg = regs[rhs.index];
                                 return body([&](std::size_t lane) { return rhs_reg[lane]; });
                               },
                               [&](ScalarRegister reg) {
                                 const auto value = scalar_operand<Element>(state, reg);
                                 return body([value](std::size_t) { return value; });
                               }},
                    src2);
}

} // namespace tile_npu
