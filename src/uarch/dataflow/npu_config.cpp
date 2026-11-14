#include "uarch/dataflow/npu_config.h"

#include "uarch/dataflow/npu_instruction.h"
#include "utils/common.h"

#include <type_traits>
#include <utility>
#include <variant>

namespace tile_npu {
namespace {

template <typename> inline constexpr bool always_false_v = false;

void add_operand(ConfigOperands &operands, ConfigRegType type, RegisterIndex index,
                 const ConfigValue &value) {
  for (std::size_t i = 0; i < operands.count; ++i) {
    const auto &existing = operands.values[i];
    if (existing.type == type && existing.index == index) return;
  }
  if (operands.count >= operands.values.size()) unreachable("too many configuration operands");
  operands.values[operands.count++] = ConfigOperand{.type = type, .index = index, .value = value};
}

void add_tile_config(ConfigOperands &operands, const State &state, RegisterIndex index) {
  if (index >= TILE_REGISTERS) unreachable("invalid Tile config index");
  add_operand(operands, ConfigRegType::tc, index, state.tc[index]);
}

void add_acc_config(ConfigOperands &operands, const State &state, RegisterIndex index) {
  if (index >= ACC_REGISTERS) unreachable("invalid Acc config index");
  add_operand(operands, ConfigRegType::ac, index, state.ac[index]);
}

void add_vec_config(ConfigOperands &operands, const State &state, RegisterIndex index) {
  if (index >= VECTOR_REGISTERS) unreachable("invalid Vec32 config index");
  add_operand(operands, ConfigRegType::vc, index, state.vc[index]);
}

void add_npu_config(ConfigOperands &operands, const State &state, const NpuRegister &reg) {
  switch (reg.domain) {
  case NpuDomain::tile: add_tile_config(operands, state, reg.index); break;
  case NpuDomain::acc: add_acc_config(operands, state, reg.index); break;
  case NpuDomain::vec32: add_vec_config(operands, state, reg.index); break;
  case NpuDomain::reserved: unreachable("reserved dynamic domain"); break;
  }
}

void add_npu_descriptor(ConfigOperands &operands, const State &state, const NpuRegister &reg) {
  switch (reg.domain) {
  case NpuDomain::tile:
    if (reg.index >= TILE_REGISTERS) unreachable("invalid Tile descriptor index");
    add_operand(operands, ConfigRegType::tm, reg.index, state.tm[reg.index]);
    break;
  case NpuDomain::acc:
    if (reg.index >= ACC_REGISTERS) unreachable("invalid Acc descriptor index");
    add_operand(operands, ConfigRegType::am, reg.index, state.am[reg.index]);
    break;
  case NpuDomain::vec32:
    if (reg.index >= VECTOR_REGISTERS) unreachable("invalid Vec32 descriptor index");
    add_operand(operands, ConfigRegType::vm, reg.index, state.vm[reg.index]);
    break;
  case NpuDomain::reserved: unreachable("reserved dynamic domain"); break;
  }
}

} // namespace

ConfigOperands capture_config_operands(const DecodedNpuInstr &instr, const State &state) {
  ConfigOperands operands;

  std::visit(
      [&](const auto &decoded) {
        using T = std::remove_cvref_t<decltype(decoded)>;

        if constexpr (is_scalar_npu_v<T>) {
          // Scalar-side instructions use the live configuration at WB.
        } else if constexpr (std::is_same_v<T, MblkInstr>) {
          add_npu_config(operands, state, decoded.reg);
          add_npu_descriptor(operands, state, decoded.reg);
          if (decoded.op == MblkOp::gather) add_vec_config(operands, state, decoded.gather_src);
        } else if constexpr (std::is_same_v<T, FillInstr>) {
          add_npu_config(operands, state, decoded.dest);
        } else if constexpr (std::is_same_v<T, MoveInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, RowInstr>) {
          add_npu_config(operands, state, decoded.matrix);
          add_npu_config(operands, state, decoded.vector);
        } else if constexpr (std::is_same_v<T, LaneExtractInstr>) {
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, LaneInsertInstr>) {
          add_npu_config(operands, state, decoded.dest);
        } else if constexpr (std::is_same_v<T, LaneBroadcastInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, MmaInstr>) {
          add_npu_config(operands, state, decoded.src_a);
          add_npu_config(operands, state, decoded.src_b);
          add_npu_config(operands, state, decoded.dest);
        } else if constexpr (std::is_same_v<T, VmmaInstr>) {
          add_npu_config(operands, state, decoded.src_a);
          add_npu_config(operands, state, decoded.src_b);
          add_npu_config(operands, state, decoded.dest);
        } else if constexpr (std::is_same_v<T, AdequantInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
          add_vec_config(operands, state, decoded.col_scale);
          add_vec_config(operands, state, decoded.row_scale);
        } else if constexpr (std::is_same_v<T, VdequantInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
          add_vec_config(operands, state, decoded.col_scale);
          add_vec_config(operands, state, decoded.row_scale);
        } else if constexpr (std::is_same_v<T, EbinInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src1);
          if (const auto *reg = std::get_if<NpuRegister>(&decoded.src2)) {
            add_npu_config(operands, state, *reg);
          }
        } else if constexpr (std::is_same_v<T, EunInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, CmpInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src1);
          if (const auto *reg = std::get_if<NpuRegister>(&decoded.src2)) {
            add_npu_config(operands, state, *reg);
          }
        } else if constexpr (std::is_same_v<T, BcastInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src1);
          add_npu_config(operands, state, decoded.vector);
        } else if constexpr (std::is_same_v<T, Er4Instr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.arg);
          add_npu_config(operands, state, decoded.src1);
          add_npu_config(operands, state, decoded.src2);
        } else if constexpr (std::is_same_v<T, MaskInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, AreduceInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, AreduceArgmaxInstr>) {
          add_npu_config(operands, state, decoded.index_dest);
          add_npu_config(operands, state, decoded.value_dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, VreduceInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, VreduceArgmaxInstr>) {
          add_npu_config(operands, state, decoded.index_dest);
          add_npu_config(operands, state, decoded.value_dest);
          add_npu_config(operands, state, decoded.src);
        } else if constexpr (std::is_same_v<T, AquantInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
          add_vec_config(operands, state, decoded.scale);
        } else if constexpr (std::is_same_v<T, TdequantInstr>) {
          add_npu_config(operands, state, decoded.dest);
          add_npu_config(operands, state, decoded.src);
          add_vec_config(operands, state, decoded.scale);
        } else {
          static_assert(always_false_v<T>, "missing configuration capture for NPU instruction");
        }
      },
      instr);

  return operands;
}

void apply_config_operands(State &state, const ConfigOperands &operands) {
  for (std::size_t i = 0; i < operands.count; ++i) {
    const auto &operand = operands.values[i];
    switch (operand.type) {
    case ConfigRegType::tc: state.tc[operand.index] = std::get<TileConfig>(operand.value); break;
    case ConfigRegType::ac: state.ac[operand.index] = std::get<AccConfig>(operand.value); break;
    case ConfigRegType::vc: state.vc[operand.index] = std::get<Vec32Config>(operand.value); break;
    case ConfigRegType::tm:
      state.tm[operand.index] = std::get<MatrixDescriptor>(operand.value);
      break;
    case ConfigRegType::am:
      state.am[operand.index] = std::get<MatrixDescriptor>(operand.value);
      break;
    case ConfigRegType::vm:
      state.vm[operand.index] = std::get<VectorDescriptor>(operand.value);
      break;
    default: unreachable("invalid configuration operand type");
    }
  }
}

} // namespace tile_npu
