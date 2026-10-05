#pragma once

#include "tile_npu/instruction_core.h"

namespace tile_npu {

// INIT_MOVE data initialization and intra-domain moves: immediate fill (I),
// scalar fill, copy and Tile transpose (R2).
enum class InitOp : std::uint8_t { fill, fillx, copy, transpose };

// Operand reading per op:
//   fill:      reg = D, imm = sign-extended imm16
//   fillx:     reg = D, src = xS
//   copy:      reg = D, src = S (same domain)
//   transpose: reg = tD, src = tS (Tile only)
struct InitInstr {
  InitOp op = InitOp::fill;
  NpuRegister reg{};
  RegisterIndex src = 0;
  std::int32_t imm = 0;
};

// ROW_MOVE between a matrix register and its paired vector register
// (EW = 0: Tile <-> Vec8, EW = 1: Acc <-> Vec32). index is the row number
// (immediate or Scalar register).
enum class RowOp : std::uint8_t { insert, extract };

struct RowInstr {
  RowOp op = RowOp::insert;
  NpuRegister matrix{}; // destination for insert, source for extract
  NpuRegister vector{}; // source for insert, destination for extract
  IndexOperand index = ImmediateIndex{};
};

// LANE_MOVE lane access on a vector register (EW = 0: Vec8, EW = 1: Vec32).
// index is the lane number (immediate or Scalar register).
enum class LaneOp : std::uint8_t { extract, insert, broadcast };

// Operand reading per op:
//   extract:   dst = xD,       src = vector S
//   insert:    dst = vector D, src = xS
//   broadcast: dst = vector D, src = vector S
struct LaneInstr {
  LaneOp op = LaneOp::extract;
  NpuOperand dst{};
  NpuOperand src{};
  IndexOperand index = ImmediateIndex{};
};

} // namespace tile_npu
