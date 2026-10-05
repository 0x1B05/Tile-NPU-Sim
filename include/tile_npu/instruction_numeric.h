#pragma once

#include "tile_npu/instruction_core.h"

namespace tile_npu {

// CVT operations (R2): cvt converts i32 -> f32 within Acc or Vec32 and
// updates the destination dtype on success; widen sign-extends a Tile i8
// source into Acc i32. funct3 >= 010 is reserved.
enum class CvtOp : std::uint8_t { cvt = 0b000, widen = 0b001 };

// Operand reading: dest = D (Acc, or Vec32 for cvt), src = S (same domain as
// dest for cvt; Tile for widen).
struct CvtInstr {
  CvtOp op = CvtOp::cvt;
  NpuRegister dest{};
  NpuRegister src{};
};

// AREDUCE/VREDUCE operations: the funct3 codes are shared between the two
// families; argmax is the only two-destination form (VREDUCE R4 variant) and
// absmax exists only on AREDUCE.
enum class ReduceOp : std::uint8_t {
  sum = 0b000,
  max = 0b001,
  min = 0b010,
  sumsq = 0b011,
  argmax = 0b100,
  absmax = 0b101,
};

// Operand reading:
//   areduce.rows.*: dest = vD (Vec32), src = aS (Acc); dest2 unused
//   vreduce.*:      dest = xD (Scalar), src = vS (Vec32); dest2 unused
//   vreduce.argmax: dest = xIndex (Scalar), dest2 = xValue (Scalar), src = vS
struct ReduceInstr {
  ReduceOp op = ReduceOp::sum;
  NpuOperand dest{};
  NpuOperand dest2{};
  NpuRegister src{};
};

// QNT operations (R4): quant covers tquant.rows.q8s32, vquant.q8s32,
// tdequant.rows.f32 and bdequant.f32 (dom2 picks the main destination
// domain, and with it the direction); quant_ext_scale is the external-scale
// row quantize tquant.rows.scale.q8s32 (Tile destination only). funct3 >=
// 010 is reserved.
enum class QntOp : std::uint8_t { quant = 0b000, quant_ext_scale = 0b001 };

// Operand reading: dest = D and src = S are the main operands (tquant:
// tD/aS; tdequant: aD/tS; vquant: bD/vS; bdequant: vD/bS); scale is the
// vScale Vec32 register (destination for quant, source otherwise); index is
// the scale lane (vector forms only, immediate or Scalar register).
struct QntInstr {
  QntOp op = QntOp::quant;
  NpuRegister dest{};
  NpuRegister src{};
  RegisterIndex scale = 0;
  IndexOperand index = ImmediateIndex{};
};

// MMA_BDOT operations (R4): funct3 bit2 picks matrix multiply (0, mma.*) or
// vector-matrix dot (1, bdot.*); bit1 picks the right-operand orientation
// (nn = 0, nt = 1); bit0 picks zero-fill or accumulate for mma (bdot only
// has the zero form, so funct3 codes 101/111 are reserved).
enum class MmaOp : std::uint8_t {
  nn_zero = 0b000,
  nn_acc = 0b001,
  nt_zero = 0b010,
  nt_acc = 0b011,
  bdot_nn = 0b100,
  bdot_nt = 0b110,
};

// Operand reading: dest = aD (Acc) for mma, vD (Vec32) for bdot; src_a = tA
// (Tile) for mma, bA (Vec8) for bdot; src_b is always the right Tile tB.
struct MmaInstr {
  MmaOp op = MmaOp::nn_zero;
  NpuRegister dest{};
  NpuRegister src_a{};
  NpuRegister src_b{};
};

// SYS operations (Z format): no explicit operands. funct3 codes 100-111 are
// reserved.
enum class SysOp : std::uint8_t {
  fence_mem = 0b000,
  fence_sa = 0b001,
  fence_all = 0b010,
  kernel_end = 0b011,
};

struct SysInstr {
  SysOp op = SysOp::fence_mem;
};

} // namespace tile_npu
