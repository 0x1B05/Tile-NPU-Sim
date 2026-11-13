#include "framework.h"
#include "uarch/dataflow/npu_operands.h"

#include <cstdint>
#include <optional>

using namespace tile_npu;
using namespace tile_npu::test;

TEST_CASE(npu_operands_reports_scalar_destination) {
  LaneExtractInstr extract;
  extract.dest.index = 3;
  check_eq(scalar_dest(DecodedNpuInstr{extract}), std::optional<RegisterIndex>{3},
           "lane extract writes x3");

  extract.dest.index = 0;
  check(!scalar_dest(DecodedNpuInstr{extract}).has_value(), "x0 is not a scalar destination");
  check(!scalar_dest(DecodedNpuInstr{MoveInstr{}}).has_value(), "non-extract has no destination");
}

TEST_CASE(npu_operands_capture_fillx_scalar) {
  ScalarState scalar;
  scalar.write_x(5, 42);

  FillInstr fill;
  fill.dest = {.domain = NpuDomain::tile, .index = 0};
  fill.value = ScalarRegister{.index = 5};

  const auto operands = capture_scalar_operands(fill, scalar);
  check_eq(operands.count, std::uint8_t{1}, "one scalar operand");
  check_eq(operands.values[0].index, RegisterIndex{5}, "fillx source index");
  check_eq(operands.values[0].value, std::uint64_t{42}, "fillx source value");
}

TEST_CASE(npu_operands_capture_mask_operands) {
  ScalarState scalar;
  scalar.write_x(3, 11);
  scalar.write_x(7, 22);

  MaskInstr mask;
  mask.bound = 3;
  mask.fill = ScalarRegister{.index = 7};

  const auto operands = capture_scalar_operands(mask, scalar);
  check_eq(operands.count, std::uint8_t{2}, "mask reads two scalars");
  check_eq(operands.values[0].index, RegisterIndex{3}, "mask bound index");
  check_eq(operands.values[0].value, std::uint64_t{11}, "mask bound value");
  check_eq(operands.values[1].index, RegisterIndex{7}, "mask fill index");
  check_eq(operands.values[1].value, std::uint64_t{22}, "mask fill value");
}

TEST_CASE(npu_operands_apply_reconstructs_scalar_inputs) {
  ScalarOperands operands;
  operands.values[0] = ScalarOperand{.index = 4, .value = 99};
  operands.count = 1;

  ScalarState scalar;
  scalar.write_x(4, 1);
  apply_scalar_operands(scalar, operands);

  check_eq(scalar.read_x(4), std::uint64_t{99}, "captured value overwrites the register");
}
