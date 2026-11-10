#pragma once

#include "tile_npu/instruction.h"
#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

enum class Unit : std::uint8_t { vec32, reduce, quant, mma, dma, count };

// intentionally coarser than the decoded IR: timing cares about the resource class and shape
enum class DynOp : std::uint8_t {
  mma,
  vmma,
  load,
  store,
  gather,
  fill,
  move,
  row,
  lane,
  elementwise,
  compare,
  broadcast,
  select,
  mask,
  reduce,
  quant,
  other,
};

// Register operands that reach the NPU scoreboard. Scalar operands are
// resolved before dispatch and therefore never appear here.
struct RegisterRef {
  NpuDomain domain = NpuDomain::tile;
  RegisterIndex index = 0;
};

struct GatherInfo {
  std::array<std::uint32_t, VECTOR_LANES> indices{};
  std::uint8_t count = 0;
};

struct DynInst {
  std::uint64_t id = 0;
  std::uint64_t pc = 0;
  DynOp op = DynOp::other;
  Unit unit = Unit::vec32;

  // at most 3 sources (Er4/Mask/accumulate MMA) and 2 destinations (argmax).
  static constexpr std::size_t MAX_SRC = 3;
  static constexpr std::size_t MAX_DST = 2;
  std::array<RegisterRef, MAX_SRC> src{};
  std::uint8_t src_count = 0;
  std::array<RegisterRef, MAX_DST> dst{};
  std::uint8_t dst_count = 0;

  // Shape and work size.
  // MMA uses rows=M, cols=N, k=K; vector ops use len; reductions use reduce_src_elements.
  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::uint32_t k = 0;
  std::uint32_t len = 0;
  std::uint32_t reduce_src_elements = 0;

  // Aggregate work used by units and statistics.
  std::uint32_t bytes = 0;
  std::uint32_t macs = 0;

  // Element type and width.
  std::uint8_t dtype = 0;
  std::uint8_t elem_bytes = 0;

  // Memory descriptor snapshot used by DMA.
  std::uint64_t addr = 0;
  std::int32_t row_stride = 0;
  std::int32_t col_stride = 0;
  bool is_store = false;
  bool transpose = false;

  // Gather indices for data-dependent DMA addresses.
  std::optional<GatherInfo> gather;
};

} // namespace tile_npu
