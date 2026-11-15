#include "framework.h"
#include "uarch/dataflow/npu_config.h"

#include <cstdint>
#include <variant>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

const ConfigOperand *find(const ConfigOperands &operands, ConfigRegType type, RegisterIndex index) {
  for (std::size_t i = 0; i < operands.count; ++i) {
    const auto &operand = operands.values[i];
    if (operand.type == type && operand.index == index) return &operand;
  }
  return nullptr;
}

} // namespace

TEST_CASE(npu_config_captures_mma_operands) {
  State state;
  state.tc[0] = TileConfig{.dtype = DType::i8, .rows = 2, .cols = 3};
  state.tc[1] = TileConfig{.dtype = DType::u8, .rows = 4, .cols = 2};
  state.ac[0] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 2};

  const MmaInstr mma{.dest = {.domain = NpuDomain::acc, .index = 0},
                     .src_a = {.domain = NpuDomain::tile, .index = 0},
                     .src_b = {.domain = NpuDomain::tile, .index = 1}};
  const auto operands = capture_config_operands(DecodedNpuInstr{mma}, state);

  check_eq(operands.count, std::uint8_t{3}, "mma captures three configs");
  const auto *tc0 = find(operands, ConfigRegType::tc, 0);
  const auto *tc1 = find(operands, ConfigRegType::tc, 1);
  const auto *ac0 = find(operands, ConfigRegType::ac, 0);
  check(tc0 != nullptr && tc1 != nullptr && ac0 != nullptr, "mma configs are present");
  check_eq(std::get<TileConfig>(tc0->value).rows, std::uint8_t{2}, "tc0 rows");
  check_eq(std::get<TileConfig>(tc1->value).cols, std::uint8_t{2}, "tc1 cols");
  check_eq(std::get<AccConfig>(ac0->value).cols, std::uint8_t{2}, "ac0 cols");
}

TEST_CASE(npu_config_captures_memory_descriptor) {
  State state;
  state.tc[0] = TileConfig{.dtype = DType::i8, .rows = 2, .cols = 3};
  state.tm[0] = MatrixDescriptor{
      .base_addr = 0x2000, .row_stride_bytes = 3, .col_stride_bytes = 1, .flags = 1};
  state.vc[4] = Vec32Config{.dtype = DType::u32, .len = 2};

  MblkInstr load{.op = MblkOp::load, .reg = {.domain = NpuDomain::tile, .index = 0}};
  auto operands = capture_config_operands(DecodedNpuInstr{load}, state);
  check_eq(operands.count, std::uint8_t{2}, "block load captures config and descriptor");
  const auto *tm0 = find(operands, ConfigRegType::tm, 0);
  check(tm0 != nullptr, "tm0 is captured");
  check_eq(std::get<MatrixDescriptor>(tm0->value).base_addr, std::uint64_t{0x2000},
           "descriptor base");

  load.op = MblkOp::gather;
  load.gather_src = 4;
  operands = capture_config_operands(DecodedNpuInstr{load}, state);
  check_eq(operands.count, std::uint8_t{3}, "gather adds the index vector config");
  check(find(operands, ConfigRegType::vc, 4) != nullptr, "gather index config is captured");
}

TEST_CASE(npu_config_captures_widest_instruction) {
  State state;
  for (std::uint8_t i = 0; i < 4; ++i) {
    state.ac[i] =
        AccConfig{.dtype = DType::i32, .rows = static_cast<std::uint8_t>(i + 1), .cols = 1};
  }

  const Er4Instr er4{.dest = {.domain = NpuDomain::acc, .index = 0},
                     .arg = {.domain = NpuDomain::acc, .index = 1},
                     .src1 = {.domain = NpuDomain::acc, .index = 2},
                     .src2 = {.domain = NpuDomain::acc, .index = 3}};
  const auto operands = capture_config_operands(DecodedNpuInstr{er4}, state);

  check_eq(operands.count, std::uint8_t{4}, "er4 captures four configs");
  for (std::uint8_t i = 0; i < 4; ++i) {
    check(find(operands, ConfigRegType::ac, i) != nullptr, "all er4 configs are captured");
  }
}

TEST_CASE(npu_config_deduplicates_operands) {
  State state;
  state.vc[2] = Vec32Config{.dtype = DType::i32, .len = 4};

  const EbinInstr ebin{.dest = {.domain = NpuDomain::vec32, .index = 2},
                       .src1 = {.domain = NpuDomain::vec32, .index = 2},
                       .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 2}};
  const auto operands = capture_config_operands(DecodedNpuInstr{ebin}, state);
  check_eq(operands.count, std::uint8_t{1}, "duplicate config is captured once");
}

TEST_CASE(npu_config_ignores_scalar_side_instructions) {
  State state;
  check_eq(capture_config_operands(DecodedNpuInstr{CfgSetiInstr{}}, state).count, std::uint8_t{0},
           "cfg.seti needs no snapshot");
  check_eq(capture_config_operands(DecodedNpuInstr{SysInstr{}}, state).count, std::uint8_t{0},
           "sys needs no snapshot");
}

TEST_CASE(npu_config_apply_round_trip) {
  State source;
  source.tc[0] = TileConfig{.dtype = DType::u8, .rows = 3, .cols = 4};
  source.tc[1] = TileConfig{.dtype = DType::i8, .rows = 2, .cols = 3};
  source.ac[0] = AccConfig{.dtype = DType::u32, .rows = 2, .cols = 2};

  const MmaInstr mma{.dest = {.domain = NpuDomain::acc, .index = 0},
                     .src_a = {.domain = NpuDomain::tile, .index = 0},
                     .src_b = {.domain = NpuDomain::tile, .index = 1}};
  const auto operands = capture_config_operands(DecodedNpuInstr{mma}, source);

  State target;
  apply_config_operands(target, operands);
  check_eq(target.tc[0].rows, std::uint8_t{3}, "tc0 rows restored");
  check_eq(target.tc[1].cols, std::uint8_t{3}, "tc1 cols restored");
  check_eq(target.ac[0].dtype, DType::u32, "ac0 dtype restored");
}
