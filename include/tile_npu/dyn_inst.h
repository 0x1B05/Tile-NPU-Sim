#pragma once

#include "tile_npu/state.h"

#include <array>
#include <cstdint>
#include <optional>

namespace tile_npu {

// The architectural domain of one operand. Scalar is separate from the NPU
// domains because ScalarState lives outside the Tile/Acc/Vec arrays.
enum class Domain : std::uint8_t { scalar, tile, acc, vec32 };

// Execution resource a dynamic instruction belongs to. The functional
// executor ignores this; the uarch engine uses it to dispatch to a component.
enum class Unit : std::uint8_t { scalar, mma, vector, dma, config, sync };

// Broad semantic class. It is intentionally coarser than the decoded IR:
// timing cares about the resource class and shape, not the exact opcode.
enum class DynOp : std::uint8_t {
  scalar,
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
  config,
  fence,
  kernel_end,
  other,
};

enum class FenceKind : std::uint8_t { mem, sa, all };

struct DataRef {
  Domain domain = Domain::scalar;
  std::uint8_t index = 0;
};

// Values of the configuration and descriptor registers that the instruction
// actually uses, captured before execution. The uarch engine must not read
// the live State: the functional executor may already have advanced PC or a
// descriptor base, and cfg writes take effect at dispatch.
struct ConfigSnapshot {
  TileConfig tile_a{};
  TileConfig tile_b{};
  AccConfig acc_d{};
  TileConfig tile_d{};
  Vec32Config vec_d{};
  MatrixDescriptor matrix{};
  VectorDescriptor vector{};
};

// A dynamic instruction instance: one execution of one decoded instruction.
// It is produced once by the shared front end and consumed by both the
// functional executor and the uarch engine, so both see identical effective
// operands and configuration.
struct DynInst {
  std::uint64_t id = 0;
  std::uint64_t pc = 0;
  DynOp op = DynOp::other;
  Unit unit = Unit::scalar;

  std::array<DataRef, 4> src{};
  std::uint8_t src_count = 0;
  // Argmax writes an index destination and a value destination; every other
  // instruction writes at most one register.
  std::array<DataRef, 2> dst{};
  std::uint8_t dst_count = 0;

  std::uint32_t rows = 0;
  std::uint32_t cols = 0;
  std::uint32_t k = 0;
  std::uint32_t len = 0;
  std::uint32_t bytes = 0;
  std::uint32_t macs = 0;
  std::uint64_t addr = 0;
  std::int32_t row_stride = 0;
  std::int32_t col_stride = 0;
  std::uint8_t dtype = 0;
  std::uint8_t elem_bytes = 0;

  ConfigSnapshot config{};

  // Filled after the functional executor resolves a control transfer.
  std::optional<std::uint64_t> redirect;

  bool is_store = false;
  bool transpose = false;
  bool kernel_end = false;
  FenceKind fence = FenceKind::mem;
};

} // namespace tile_npu
