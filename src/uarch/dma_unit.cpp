#include "uarch/dma_unit.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace tile_npu {
namespace {

std::uint32_t operand_bytes(const DynInst &inst, const DataRef &ref, bool source) {
  if (ref.domain == Domain::scalar) return 0;
  // A gather's index vector is much smaller than the destination matrix.
  if (source && inst.op == DynOp::gather && ref.domain == Domain::vec32) {
    const auto count = inst.gather ? inst.gather->count : inst.len;
    return std::max<std::uint32_t>(4, count * 4);
  }
  return inst.bytes;
}

bool vector_access(const DynInst &inst) {
  return (inst.dst_count != 0 && inst.dst[0].domain == Domain::vec32) ||
         (inst.src_count != 0 && inst.src[0].domain == Domain::vec32);
}

void add(std::vector<DmaSegment> &segments, std::uint64_t addr, std::uint32_t bytes) {
  if (bytes != 0) segments.push_back(DmaSegment{.addr = addr, .bytes = bytes});
}

} // namespace

DmaUnit::DmaUnit(SramModel *sram, std::unique_ptr<MemoryService> memory)
    : sram_(sram != nullptr ? sram : &owned_sram_),
      memory_(memory != nullptr ? std::move(memory) : std::make_unique<SimpleMemoryService>()) {}

void DmaUnit::reset() {
  if (sram_ == &owned_sram_) owned_sram_.reset();
  memory_->reset();
}

std::vector<DmaSegment> DmaUnit::segments(const DynInst &inst) const {
  std::vector<DmaSegment> result;
  const auto elem = std::max<std::uint32_t>(1, inst.elem_bytes);

  if (inst.op == DynOp::gather) {
    const auto count = inst.gather ? inst.gather->count : 0;
    const bool vector_target = inst.dst_count != 0 && inst.dst[0].domain == Domain::vec32;
    for (std::uint8_t i = 0; i < count; ++i) {
      const auto index = inst.gather ? inst.gather->indices[i] : 0;
      if (vector_target) {
        add(result, inst.addr + static_cast<std::uint64_t>(index) * inst.col_stride, elem);
        continue;
      }
      const auto row_base = inst.addr + static_cast<std::uint64_t>(index) * inst.row_stride;
      if (inst.col_stride == static_cast<std::int32_t>(elem)) {
        add(result, row_base, inst.cols * elem);
      } else {
        for (std::uint32_t j = 0; j < inst.cols; ++j) {
          add(result, row_base + static_cast<std::int64_t>(j) * inst.col_stride, elem);
        }
      }
    }
    return result;
  }

  if (vector_access(inst)) {
    const auto len = inst.len != 0 ? inst.len : inst.cols;
    const auto stride = inst.col_stride;
    if (stride == static_cast<std::int32_t>(elem)) {
      add(result, inst.addr, len * elem);
    } else {
      for (std::uint32_t j = 0; j < len; ++j) {
        add(result, inst.addr + static_cast<std::int64_t>(j) * stride, elem);
      }
    }
    return result;
  }

  const auto rows = inst.rows;
  const auto cols = inst.cols;
  const auto row_stride = inst.row_stride;
  const auto col_stride = inst.col_stride;

  if (!inst.transpose) {
    if (col_stride == static_cast<std::int32_t>(elem)) {
      for (std::uint32_t i = 0; i < rows; ++i) {
        add(result, inst.addr + static_cast<std::int64_t>(i) * row_stride, cols * elem);
      }
    } else if (row_stride == static_cast<std::int32_t>(elem)) {
      for (std::uint32_t j = 0; j < cols; ++j) {
        add(result, inst.addr + static_cast<std::int64_t>(j) * col_stride, rows * elem);
      }
    } else {
      for (std::uint32_t i = 0; i < rows; ++i) {
        for (std::uint32_t j = 0; j < cols; ++j) {
          add(result,
              inst.addr + static_cast<std::int64_t>(i) * row_stride +
                  static_cast<std::int64_t>(j) * col_stride,
              elem);
        }
      }
    }
  } else {
    // Transpose swaps the logical coordinates: (i,j) reads addr(j,i).
    if (col_stride == static_cast<std::int32_t>(elem)) {
      for (std::uint32_t j = 0; j < cols; ++j) {
        add(result, inst.addr + static_cast<std::int64_t>(j) * row_stride, rows * elem);
      }
    } else if (row_stride == static_cast<std::int32_t>(elem)) {
      for (std::uint32_t i = 0; i < rows; ++i) {
        add(result, inst.addr + static_cast<std::int64_t>(i) * col_stride, cols * elem);
      }
    } else {
      for (std::uint32_t i = 0; i < rows; ++i) {
        for (std::uint32_t j = 0; j < cols; ++j) {
          add(result,
              inst.addr + static_cast<std::int64_t>(j) * row_stride +
                  static_cast<std::int64_t>(i) * col_stride,
              elem);
        }
      }
    }
  }
  return result;
}

std::uint64_t DmaUnit::read_operands(const DynInst &inst, std::uint64_t at) {
  std::uint64_t end = at;
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    const auto bytes = operand_bytes(inst, inst.src[i], true);
    if (bytes == 0) continue;
    const auto access =
        sram_->access(inst.src[i].domain, inst.src[i].index, 0, bytes, at, at, false);
    end = std::max(end, access.end);
  }
  return end;
}

std::uint64_t DmaUnit::write_operands(const DynInst &inst, std::uint64_t at) {
  std::uint64_t end = at;
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    const auto bytes = operand_bytes(inst, inst.dst[i], false);
    if (bytes == 0) continue;
    const auto access =
        sram_->access(inst.dst[i].domain, inst.dst[i].index, 0, bytes, at, at, true);
    end = std::max(end, access.end);
  }
  return end;
}

std::uint64_t DmaUnit::issue_memory(const std::vector<DmaSegment> &segments, std::uint64_t at,
                                    bool write, std::uint64_t *first_data) {
  std::uint64_t end = at;
  for (const auto &segment : segments) {
    const auto completion = memory_->issue(
        MemoryRequest{.addr = segment.addr, .bytes = segment.bytes, .write = write}, at);
    end = std::max(end, completion.complete);
    if (first_data != nullptr) {
      // The first beat is available after the service latency; SRAM writes
      // can overlap the remainder of the memory burst instead of waiting for
      // the whole transfer to finish.
      const auto ready = completion.complete > completion.transactions
                             ? completion.complete - completion.transactions
                             : completion.complete;
      *first_data = std::min(*first_data, ready);
    }
  }
  return end;
}

std::uint64_t DmaUnit::issue(const DynInst &inst, std::uint64_t now) {
  auto work = segments(inst);
  if (work.empty() && inst.bytes != 0) {
    // Defensive fallback for a descriptor shape that does not decompose.
    work.push_back(DmaSegment{.addr = inst.addr, .bytes = inst.bytes});
  }

  if (inst.is_store) {
    // SRAM is read before the write reaches memory.
    const auto sram_end = read_operands(inst, now);
    const auto memory_end = issue_memory(work, sram_end, true);
    return std::max(sram_end, memory_end);
  }

  // Loads may need to read a gather index vector before the memory requests.
  // Destination registers can start accepting beats as soon as the first
  // memory data returns; the rest of the burst overlaps the SRAM write.
  const auto index_end = read_operands(inst, now);
  auto first_data = std::numeric_limits<std::uint64_t>::max();
  const auto memory_end = issue_memory(work, index_end, false, &first_data);
  if (first_data == std::numeric_limits<std::uint64_t>::max()) first_data = memory_end;
  const auto sram_end = write_operands(inst, std::max(index_end, first_data));
  return std::max(memory_end, sram_end);
}

} // namespace tile_npu
