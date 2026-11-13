#include "uarch/dataflow/npu_operands.h"

#include "utils/common.h"

#include <variant>

namespace tile_npu {
namespace {

void add_scalar(ScalarOperands &operands, RegisterIndex index, const ScalarState &scalar) {
  if (index == 0) return;
  if (operands.count >= operands.values.size()) unreachable("too many scalar operands");
  operands.values[operands.count++] = ScalarOperand{.index = index, .value = scalar.read_x(index)};
}

void add_index(ScalarOperands &operands, const IndexOperand &index, const ScalarState &scalar) {
  if (const auto *reg = std::get_if<ScalarRegister>(&index)) {
    add_scalar(operands, reg->index, scalar);
  }
}

void add_operand(ScalarOperands &operands, const NpuOperand &operand, const ScalarState &scalar) {
  if (const auto *reg = std::get_if<ScalarRegister>(&operand)) {
    add_scalar(operands, reg->index, scalar);
  }
}

} // namespace

std::optional<RegisterIndex> scalar_dest(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

ScalarOperands capture_scalar_operands(const DecodedNpuInstr &instr, const ScalarState &scalar) {
  ScalarOperands operands;

  std::visit(
      Overloaded{
          [&](const FillInstr &fill) {
            if (const auto *reg = std::get_if<ScalarRegister>(&fill.value)) {
              add_scalar(operands, reg->index, scalar);
            }
          },
          [&](const RowInstr &row) { add_index(operands, row.index, scalar); },
          [&](const LaneExtractInstr &extract) { add_index(operands, extract.index, scalar); },
          [&](const LaneInsertInstr &insert) {
            add_scalar(operands, insert.src.index, scalar);
            add_index(operands, insert.index, scalar);
          },
          [&](const LaneBroadcastInstr &broadcast) {
            add_index(operands, broadcast.index, scalar);
          },
          [&](const EbinInstr &ebin) { add_operand(operands, ebin.src2, scalar); },
          [&](const CmpInstr &cmp) { add_operand(operands, cmp.src2, scalar); },
          [&](const MaskInstr &mask) {
            add_scalar(operands, mask.bound, scalar);
            if (const auto *reg = std::get_if<ScalarRegister>(&mask.fill)) {
              add_scalar(operands, reg->index, scalar);
            }
          },
          [&](const CfgSetxInstr &setx) { add_scalar(operands, setx.src, scalar); },
          [&](const CfgSetwInstr &setw) { add_scalar(operands, setw.src, scalar); },
          [&](const auto &) {},
      },
      instr);

  return operands;
}

void apply_scalar_operands(ScalarState &scalar, const ScalarOperands &operands) {
  for (std::size_t i = 0; i < operands.count; ++i) {
    scalar.write_x(operands.values[i].index, operands.values[i].value);
  }
}

} // namespace tile_npu
