#include "uarch/dyn_inst.h"

#include "framework.h"

#include <cstddef>
#include <cstdint>

using namespace tile_npu;

TEST_CASE(dyn_inst_schema_defaults) {
  DynInst inst{};
  check_eq(inst.id, std::uint64_t{0}, "default id");
  check_eq(inst.pc, std::uint64_t{0}, "default pc");
  check_eq(inst.op, DynOp::other, "default op");
  check_eq(inst.unit, Unit::vec32, "default unit");
  check_eq(inst.src_count, std::uint8_t{0}, "default source count");
  check_eq(inst.dst_count, std::uint8_t{0}, "default destination count");
  check_eq(inst.src.size(), DynInst::MAX_SRC, "source operand capacity");
  check_eq(inst.dst.size(), DynInst::MAX_DST, "destination operand capacity");
  check_eq(inst.rows, std::uint32_t{0}, "default rows");
  check_eq(inst.cols, std::uint32_t{0}, "default cols");
  check_eq(inst.k, std::uint32_t{0}, "default k");
  check_eq(inst.len, std::uint32_t{0}, "default len");
  check_eq(inst.reduce_src_elements, std::uint32_t{0}, "default reduction source elements");
  check_eq(inst.bytes, std::uint32_t{0}, "default bytes");
  check_eq(inst.macs, std::uint32_t{0}, "default macs");
  check_eq(inst.addr, std::uint64_t{0}, "default address");
  check_eq(inst.row_stride, std::int32_t{0}, "default row stride");
  check_eq(inst.col_stride, std::int32_t{0}, "default col stride");
  check_eq(inst.dtype, std::uint8_t{0}, "default dtype");
  check_eq(inst.elem_bytes, std::uint8_t{0}, "default element bytes");
  check(!inst.gather.has_value(), "default gather is unset");
  check(!inst.is_store, "default is not a store");
  check(!inst.transpose, "default is not transposed");
}

TEST_CASE(dyn_inst_schema_operands) {
  DynInst inst{};
  inst.src[0] = RegisterRef{.domain = NpuDomain::tile, .index = 3};
  inst.src[1] = RegisterRef{.domain = NpuDomain::vec32, .index = 7};
  inst.src_count = 2;
  inst.dst[0] = RegisterRef{.domain = NpuDomain::acc, .index = 1};
  inst.dst_count = 1;

  check_eq(inst.src[0].domain, NpuDomain::tile, "source 0 domain");
  check_eq(inst.src[0].index, std::uint8_t{3}, "source 0 index");
  check_eq(inst.src[1].domain, NpuDomain::vec32, "source 1 domain");
  check_eq(inst.src[1].index, std::uint8_t{7}, "source 1 index");
  check_eq(inst.dst[0].domain, NpuDomain::acc, "destination 0 domain");
  check_eq(inst.dst[0].index, std::uint8_t{1}, "destination 0 index");

  const auto copy = inst;
  check_eq(copy.src_count, std::uint8_t{2}, "copy keeps source count");
  check_eq(copy.dst[0].index, std::uint8_t{1}, "copy keeps destination");
}

TEST_CASE(dyn_inst_schema_unit_counters) {
  check_eq(static_cast<std::size_t>(Unit::vec32), std::size_t{0}, "vec32 unit index");
  check_eq(static_cast<std::size_t>(Unit::reduce), std::size_t{1}, "reduce unit index");
  check_eq(static_cast<std::size_t>(Unit::quant), std::size_t{2}, "quant unit index");
  check_eq(static_cast<std::size_t>(Unit::mma), std::size_t{3}, "mma unit index");
  check_eq(static_cast<std::size_t>(Unit::dma), std::size_t{4}, "dma unit index");
  check_eq(static_cast<std::size_t>(Unit::count), std::size_t{5}, "unit count");
}
