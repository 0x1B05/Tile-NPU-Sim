#include "uarch/dataflow/npu_operands.h"

#include "utils/common.h"

#include <variant>

namespace tile_npu {
namespace {

void add_read(ScalarReads &reads, RegisterIndex index) {
  if (index == 0) return;
  if (reads.count >= reads.values.size()) unreachable("too many scalar reads");
  reads.values[reads.count++] = index;
}

void add_index_read(ScalarReads &reads, const IndexOperand &index) {
  if (const auto *reg = std::get_if<ScalarRegister>(&index)) add_read(reads, reg->index);
}

void add_operand_read(ScalarReads &reads, const NpuOperand &operand) {
  if (const auto *reg = std::get_if<ScalarRegister>(&operand)) add_read(reads, reg->index);
}

} // namespace

std::optional<RegisterIndex> scalar_dest(const DecodedNpuInstr &instr) {
  const auto *extract = std::get_if<LaneExtractInstr>(&instr);
  if (extract == nullptr || extract->dest.index == 0) return std::nullopt;
  return extract->dest.index;
}

ScalarReads scalar_reads(const DecodedNpuInstr &instr) {
  ScalarReads reads;

  std::visit(
      Overloaded{
          [&](const FillInstr &fill) {
            if (const auto *reg = std::get_if<ScalarRegister>(&fill.value)) {
              add_read(reads, reg->index);
            }
          },
          [&](const RowInstr &row) { add_index_read(reads, row.index); },
          [&](const LaneExtractInstr &extract) { add_index_read(reads, extract.index); },
          [&](const LaneInsertInstr &insert) {
            add_read(reads, insert.src.index);
            add_index_read(reads, insert.index);
          },
          [&](const LaneBroadcastInstr &broadcast) { add_index_read(reads, broadcast.index); },
          [&](const EbinInstr &ebin) { add_operand_read(reads, ebin.src2); },
          [&](const CmpInstr &cmp) { add_operand_read(reads, cmp.src2); },
          [&](const MaskInstr &mask) {
            add_read(reads, mask.bound);
            if (const auto *reg = std::get_if<ScalarRegister>(&mask.fill)) {
              add_read(reads, reg->index);
            }
          },
          [&](const CfgSetxInstr &setx) { add_read(reads, setx.src); },
          [&](const CfgSetwInstr &setw) { add_read(reads, setw.src); },
          [&](const auto &) {},
      },
      instr);

  return reads;
}

ScalarOperands capture_scalar_operands(const DecodedNpuInstr &instr, const ScalarState &scalar) {
  const auto reads = scalar_reads(instr);
  ScalarOperands operands;
  operands.count = reads.count;
  for (std::size_t i = 0; i < reads.count; ++i) {
    operands.values[i] =
        ScalarOperand{.index = reads.values[i], .value = scalar.read_x(reads.values[i])};
  }
  return operands;
}

void apply_scalar_operands(ScalarState &scalar, const ScalarOperands &operands) {
  for (std::size_t i = 0; i < operands.count; ++i) {
    scalar.write_x(operands.values[i].index, operands.values[i].value);
  }
}

} // namespace tile_npu
