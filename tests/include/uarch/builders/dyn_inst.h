#pragma once

// Small DynInst builders shared by backend, driver and scoreboard tests.
#include "uarch/dataflow/dyn_inst.h"

#include <cstdint>
#include <optional>

namespace tile_npu::test {

inline DynInst make_dyn_inst(std::uint64_t id, DynOp op, Unit unit) {
  DynInst inst{};
  inst.id = id;
  inst.op = op;
  inst.unit = unit;
  return inst;
}

inline void add_src(DynInst &inst, NpuDomain domain, RegisterIndex index) {
  inst.src[inst.src_count++] = RegisterRef{.domain = domain, .index = index};
}

inline void add_dst(DynInst &inst, NpuDomain domain, RegisterIndex index) {
  inst.dst[inst.dst_count++] = RegisterRef{.domain = domain, .index = index};
}

inline DynInst vec_write(std::uint64_t id, RegisterIndex dst) {
  auto inst = make_dyn_inst(id, DynOp::elementwise, Unit::vec32);
  add_dst(inst, NpuDomain::vec32, dst);
  return inst;
}

inline DynInst mma_write(std::uint64_t id, RegisterIndex dst) {
  auto inst = make_dyn_inst(id, DynOp::mma, Unit::mma);
  add_dst(inst, NpuDomain::acc, dst);
  return inst;
}

inline DynInst mma_accumulate(std::uint64_t id, RegisterIndex src, RegisterIndex dst) {
  auto inst = make_dyn_inst(id, DynOp::mma, Unit::mma);
  add_src(inst, NpuDomain::acc, src);
  add_dst(inst, NpuDomain::acc, dst);
  return inst;
}

inline DynInst dma_load(std::uint64_t id, std::optional<RegisterIndex> src = std::nullopt) {
  auto inst = make_dyn_inst(id, DynOp::load, Unit::dma);
  if (src) add_src(inst, NpuDomain::acc, *src);
  return inst;
}

} // namespace tile_npu::test
