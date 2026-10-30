#pragma once

#ifdef CONFIG_MTRACE

#include "tile_npu/state.h"

#include <array>
#include <cstdint>
#include <functional>
#include <string_view>
#include <utility>

namespace tile_npu {

enum class TraceEndpointKind : std::uint8_t { memory, tile, acc, vec32, immediate };

enum class TraceEventType : std::uint8_t { move, compute };

enum class TraceMemoryKind : std::uint8_t { none, linear, block };

enum class TraceDescriptorKind : std::uint8_t { none, tm, am, vm };

enum class TraceRegion : std::uint8_t { ram, serial, mmio, unknown };

struct TraceEndpoint {
  TraceEndpointKind kind = TraceEndpointKind::immediate;
  std::uint8_t index = 0;
  std::uint32_t bytes = 0;
  std::uint8_t elem = 0;
  std::uint8_t dims = 0;
  std::array<std::uint8_t, 2> shape{};
  std::string_view part;
};

struct TraceMemory {
  TraceMemoryKind kind = TraceMemoryKind::none;
  TraceRegion region = TraceRegion::ram;
  std::uint64_t addr = 0;
  std::uint32_t count = 0;
  std::int32_t stride = 0;
  std::uint8_t elem = 0;
  std::uint8_t rows = 0;
  std::uint8_t cols = 0;
  std::int32_t row_stride = 0;
  std::int32_t col_stride = 0;
  bool col_order = false;
  TraceDescriptorKind desc_kind = TraceDescriptorKind::none;
  std::uint8_t desc_index = 0;
};

struct TraceEvent {
  TraceEventType type = TraceEventType::move;
  std::uint64_t seq = 0;
  std::uint64_t instr = 0;
  std::uint64_t pc = 0;
  std::string_view op;
  std::array<TraceEndpoint, 4> src{};
  std::uint8_t src_count = 0;
  std::array<TraceEndpoint, 2> dst{};
  std::uint8_t dst_count = 0;
  TraceMemory mem{};
  std::uint32_t macs = 0;
};

using MtraceCallback = std::function<void(const TraceEvent &)>;

// Assigns the event sequence and current instruction frame, then forwards the
// event to the callback. A disabled recorder has no callback and emits nothing.
class MtraceRecorder {
public:
  void set_callback(MtraceCallback callback) { callback_ = std::move(callback); }

  bool enabled() const { return static_cast<bool>(callback_); }

  void reset() {
    instr_ = 0;
    seq_ = 0;
    pc_ = 0;
  }

  void begin_instruction(std::uint64_t pc) {
    ++instr_;
    pc_ = pc;
  }

  std::uint64_t instruction() const { return instr_; }

  std::uint64_t pc() const { return pc_; }

  void emit(TraceEvent event) {
    if (!callback_) return;
    event.seq = seq_++;
    event.instr = instr_;
    event.pc = pc_;
    callback_(event);
  }

private:
  MtraceCallback callback_;
  std::uint64_t instr_ = 0;
  std::uint64_t seq_ = 0;
  std::uint64_t pc_ = 0;
};

inline constexpr std::uint8_t trace_elem_bytes(DType dtype) {
  return dtype == DType::i8 || dtype == DType::u8 ? 1 : 4;
}

inline TraceEndpoint trace_matrix_endpoint(TraceEndpointKind kind, std::uint8_t index,
                                           const AccConfig &config, std::string_view part) {
  const auto elem = trace_elem_bytes(config.dtype);
  return {.kind = kind,
          .index = index,
          .bytes = static_cast<std::uint32_t>(config.rows) * config.cols * elem,
          .elem = elem,
          .dims = 2,
          .shape = {config.rows, config.cols},
          .part = part};
}

inline TraceEndpoint trace_matrix_endpoint(TraceEndpointKind kind, std::uint8_t index,
                                           const TileConfig &config, std::string_view part) {
  const auto elem = trace_elem_bytes(config.dtype);
  return {.kind = kind,
          .index = index,
          .bytes = static_cast<std::uint32_t>(config.rows) * config.cols * elem,
          .elem = elem,
          .dims = 2,
          .shape = {config.rows, config.cols},
          .part = part};
}

inline TraceEndpoint trace_vector_endpoint(std::uint8_t index, const Vec32Config &config,
                                           std::string_view part) {
  const auto elem = trace_elem_bytes(config.dtype);
  return {.kind = TraceEndpointKind::vec32,
          .index = index,
          .bytes = static_cast<std::uint32_t>(config.len) * elem,
          .elem = elem,
          .dims = 1,
          .shape = {config.len, 0},
          .part = part};
}

} // namespace tile_npu

#endif // CONFIG_MTRACE
