#pragma once

#include "tile_npu/instruction_core.h"

namespace tile_npu {

// M_BLK memory access: block load/store, gather load (R2) and row load/store
// (MR). The bound descriptor (TM/AM/BM/VM) is implied by the accessed
// register.
enum class MblkOp : std::uint8_t { load, store, gather, row_load, row_store };

// Operand reading per op:
//   load/gather/row_load: dest = D (gather adds gather_src = vS, the Vec32
//                         index vector)
//   store/row_store:      src = S
//   row_load/row_store:   index is the row number (immediate or Scalar)
struct MblkInstr {
  MblkOp op = MblkOp::load;
  NpuRegister dest{};
  NpuRegister src{};
  IndexOperand index = ImmediateIndex{};
  RegisterIndex gather_src = 0;
};

} // namespace tile_npu
