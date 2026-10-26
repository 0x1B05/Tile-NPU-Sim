#include "execute.h"

#include "config_check.h"
#include "descriptor_view.h"
#include "domain.h"
#include "matrix_view.h"
#include "operand.h"
#include "utils/common.h"
#include "utils/endian.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>

namespace tile_npu {
namespace {

template <typename Element>
void load_element(const AddressSpace &memory, std::uint64_t address, Element &out) {
  std::array<std::byte, sizeof(Element)> bytes{};
  if (!memory.read(address, bytes)) {
    unreachable("AddressSpace::read failed after range prevalidation");
  }
  out = load_le<Element>(bytes);
}

template <typename Element>
void store_element(AddressSpace &memory, std::uint64_t address, Element value) {
  std::array<std::byte, sizeof(Element)> bytes{};
  store_le(value, bytes);

  if (!memory.write(address, bytes)) {
    unreachable("AddressSpace::write failed after range prevalidation");
  }
}

bool transform_legal(const MblkInstr &instr, const MatrixDescriptor &desc) {
  const bool full_tile_block = instr.reg.domain == NpuDomain::tile &&
                               (instr.op == MblkOp::load || instr.op == MblkOp::store);
  return desc.transform == 0 || full_tile_block;
}

// Visits a matrix block in row-major order, swapping coordinates for the
// transpose form. Returns false when an address overflows.
template <typename Fn>
bool for_each_block_element(const MatrixDescView &desc, std::size_t rows, std::size_t cols,
                            bool transpose, Fn &&fn) {
  for (std::size_t row = 0; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      const auto address =
          transpose ? desc.matrix_element_addr(col, row) : desc.matrix_element_addr(row, col);
      if (!address) {
        return false;
      }
      fn(*address, row, col);
    }
  }
  return true;
}

// Visits the contiguous source runs of a block. The transpose form's runs are
// source rows, which become columns of the destination register. Returns false
// when either end of a run overflows.
template <typename Fn>
bool for_each_block_run(const MatrixDescView &desc, std::size_t rows, std::size_t cols,
                        bool transpose, Fn &&fn) {
  if (transpose) {
    for (std::size_t src_row = 0; src_row < cols; ++src_row) {
      const auto start = desc.matrix_element_addr(src_row, 0);
      const auto end = desc.matrix_element_addr(src_row, rows - 1);
      if (!start || !end) {
        return false;
      }
      fn(*start, rows, src_row);
    }
  } else {
    for (std::size_t row = 0; row < rows; ++row) {
      const auto start = desc.matrix_element_addr(row, 0);
      const auto end = desc.matrix_element_addr(row, cols - 1);
      if (!start || !end) {
        return false;
      }
      fn(*start, cols, row);
    }
  }
  return true;
}

// Fast path for a block whose columns are contiguous in memory. Runs are
// probed as a whole; a run rejected by the address space falls back to the
// per-element path so the exact fault address is preserved. Returns nullopt
// for that fallback and config_error for an address overflow.
template <typename Reg>
std::optional<ExecuteResult>
try_contiguous_block(const MblkInstr &instr, ExecContext &ctx, const MatrixDescView &desc, Reg &reg,
                     std::size_t rows, std::size_t cols, bool transpose) {
  using Element = typename Reg::value_type;
  const bool store = is_store(instr.op);
  // The ISA defines stores in destination row-major order; a transposed store
  // would write source-row runs instead, which differs when addresses alias.
  if (transpose && store) {
    return std::nullopt;
  }

  bool contained = true;
  if (!for_each_block_run(desc, rows, cols, transpose,
                          [&](std::uint64_t address, std::size_t length, std::size_t) {
                            if (!ctx.memory.contains(address, length * sizeof(Element))) {
                              contained = false;
                            }
                          })) {
    return config_error();
  }
  if (!contained) {
    return std::nullopt;
  }

  auto view = matrix_view(reg);
  if (transpose) {
    // Transposed stores use the per-element order; only loads reach here.
    std::array<Element, MATRIX_EXTENT> buffer{};
    for (std::size_t src_row = 0; src_row < cols; ++src_row) {
      const auto address = desc.matrix_element_addr(src_row, 0);
      if (!address) {
        unreachable("block run overflowed after range prevalidation");
      }
      if (!ctx.memory.read(*address,
                           std::as_writable_bytes(std::span<Element>(buffer.data(), rows)))) {
        unreachable("AddressSpace::read failed after range prevalidation");
      }
      for (std::size_t row = 0; row < rows; ++row) {
        view.at(row, src_row) = buffer[row];
      }
    }
  } else {
    for (std::size_t row = 0; row < rows; ++row) {
      const auto address = desc.matrix_element_addr(row, 0);
      if (!address) {
        unreachable("block run overflowed after range prevalidation");
      }
      Element *data = &view.at(row, 0);
      if (store) {
        if (!ctx.memory.write(*address, std::as_bytes(std::span<const Element>(data, cols)))) {
          unreachable("AddressSpace::write failed after range prevalidation");
        }
      } else {
        if (!ctx.memory.read(*address, std::as_writable_bytes(std::span<Element>(data, cols)))) {
          unreachable("AddressSpace::read failed after range prevalidation");
        }
      }
    }
  }
  return ExecuteResult{.status = StepStatus::retired};
}

// Every access runs in two phases: all addresses are computed first, then
// the transfer happens.
template <std::size_t Max, typename Next>
std::optional<std::array<std::uint64_t, Max>> collect_addresses(std::size_t count, Next next) {
  std::array<std::uint64_t, Max> addrs{};
  for (std::size_t n = 0; n < count; ++n) {
    const auto addr = next(n);
    if (!addr) {
      return std::nullopt;
    }
    addrs[n] = *addr;
  }
  return addrs;
}

// A load prevalidates the full range, then fills the destination register.
template <typename Reg, typename At>
ExecuteResult load_elements(const AddressSpace &memory, Reg &reg,
                            std::span<const std::uint64_t> addrs, At at) {
  using Element = typename Reg::value_type;
  for (const auto addr : addrs) {
    if (!memory.contains(addr, sizeof(Element))) {
      return access_fault(StepStatus::load_fault, addr);
    }
  }
  for (std::size_t n = 0; n < addrs.size(); ++n) {
    load_element(memory, addrs[n], at(reg, n));
  }
  return {.status = StepStatus::retired};
}

// A store reads the source register in place; the whole range is probed
// before the first write to avoid partial stores.
template <typename Reg, typename At>
ExecuteResult store_elements(AddressSpace &memory, const Reg &reg,
                             std::span<const std::uint64_t> addrs, At at) {
  using Element = typename Reg::value_type;
  for (const auto addr : addrs) {
    if (!memory.contains(addr, sizeof(Element))) {
      return access_fault(StepStatus::store_fault, addr);
    }
  }
  for (std::size_t n = 0; n < addrs.size(); ++n) {
    store_element(memory, addrs[n], at(reg, n));
  }
  return {.status = StepStatus::retired};
}

// The element accessor for a matrix transfer: position n is the logical
// coordinate (n / cols, n % cols), read through the row-major view.
template <typename Config> auto matrix_accessor(const Config &config) {
  const auto cols = std::size_t{config.cols};
  return [cols](auto &storage, std::size_t n) -> decltype(auto) {
    return matrix_view(storage).at(n / cols, n % cols);
  };
}

// The element accessor for a matrix row transfer: position n is column n in
// the selected row.
auto matrix_row_accessor(std::size_t row) {
  return [row](auto &storage, std::size_t n) -> decltype(auto) {
    return matrix_view(storage).at(row, n);
  };
}

// The element accessor for a vector transfer: position n is lane n.
auto vector_accessor() {
  return [](auto &storage, std::size_t n) -> decltype(auto) { return storage[n]; };
}

// The gather index vector must use a valid Vec32 configuration with i32 elements.
bool valid_gather_index(const Vec32Config &config) { return valid_config_of(config, DType::i32); }

std::int32_t gather_index(const Vec32 &reg, std::size_t lane) {
  return std::bit_cast<std::int32_t>(reg[lane]);
}

template <NpuDomain Domain> ExecuteResult block_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::tile || Domain == NpuDomain::acc);
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, instr.reg.index);
  if (!valid_domain_config(config)) {
    return config_error();
  }
  MatrixDescView desc(descriptor);
  if (!desc.is_valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }
  const auto rows = std::size_t{config.rows};
  const auto cols = std::size_t{config.cols};
  const auto next_base = desc.incremented_base(rows, cols);
  if (!next_base) {
    return config_error();
  }

  const bool transpose = desc.descriptor().transform;
  using Element = typename std::remove_reference_t<decltype(reg)>::value_type;
  const auto fault_status = is_store(instr.op) ? StepStatus::store_fault : StepStatus::load_fault;

  // A contiguous run per row is the common case and avoids the per-element
  // address-space calls. Multi-byte elements only take this path on a
  // little-endian host because the raw bytes are copied directly.
  if constexpr (sizeof(Element) == 1 || std::endian::native == std::endian::little) {
    if (desc.descriptor().col_stride_bytes == static_cast<std::int32_t>(sizeof(Element))) {
      const auto fast = try_contiguous_block(instr, ctx, desc, reg, rows, cols, transpose);
      if (fast) {
        if (fast->status == StepStatus::retired) {
          desc.commit_base(*next_base);
        }
        return *fast;
      }
    }
  }

  // Probe every element before changing the register or memory, preserving
  // the atomic transfer contract without materializing the address array.
  // Keep scanning after the first fault so a later address-overflow still
  // wins as a configuration error, matching the original probe order.
  std::optional<std::uint64_t> fault_address;
  const auto probe = [&](std::uint64_t address, std::size_t, std::size_t) {
    if (!fault_address && !ctx.memory.contains(address, sizeof(Element))) {
      fault_address = address;
    }
  };
  if (!for_each_block_element(desc, rows, cols, transpose, probe)) {
    return config_error();
  }
  if (fault_address) {
    return access_fault(fault_status, *fault_address);
  }

  auto view = matrix_view(reg);
  if (is_store(instr.op)) {
    const auto store = [&](std::uint64_t address, std::size_t row, std::size_t col) {
      store_element(ctx.memory, address, view.at(row, col));
    };
    if (!for_each_block_element(desc, rows, cols, transpose, store)) {
      unreachable("block address overflowed after range prevalidation");
    }
  } else {
    const auto load = [&](std::uint64_t address, std::size_t row, std::size_t col) {
      load_element(ctx.memory, address, view.at(row, col));
    };
    if (!for_each_block_element(desc, rows, cols, transpose, load)) {
      unreachable("block address overflowed after range prevalidation");
    }
  }
  desc.commit_base(*next_base);
  return {.status = StepStatus::retired};
}

template <NpuDomain Domain> ExecuteResult row_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::tile || Domain == NpuDomain::acc);
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, instr.reg.index);
  if (!valid_domain_config(config)) {
    return config_error();
  }
  MatrixDescView desc(descriptor);
  if (!desc.is_valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }

  const auto row = resolve_index(instr.index, ctx.state);
  // Row numbers outside the effective shape are rejected.
  if (row >= config.rows) {
    return config_error();
  }
  const auto cols = std::size_t{config.cols};
  const auto next_base = desc.incremented_base(1, cols);
  if (!next_base) {
    return config_error();
  }

  const auto addrs = collect_addresses<MATRIX_EXTENT>(
      cols, [&](std::size_t j) { return desc.matrix_element_addr(0, j); });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), cols);
  const auto at = matrix_row_accessor(row);
  const auto result = is_store(instr.op) ? store_elements(ctx.memory, reg, range, at)
                                         : load_elements(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

// tload.gather
ExecuteResult gather_tile_access(const MblkInstr &instr, ExecContext &ctx) {
  const auto index = instr.reg.index;
  const auto &config = ctx.state.tc[index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  const auto &index_config = ctx.state.vc[instr.gather_src];
  if (!valid_gather_index(index_config)) {
    return config_error();
  }
  MatrixDescView desc(ctx.state.tm[index]);
  if (!desc.is_valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }

  const auto rows = std::size_t{config.rows};
  const auto cols = std::size_t{config.cols};
  const auto indexed_rows = std::min(rows, std::size_t{index_config.len});
  const auto &index_reg = ctx.state.v[instr.gather_src];
  std::array<std::uint64_t, MATRIX_ELEMENTS> addrs{};
  for (std::size_t i = 0; i < indexed_rows; ++i) {
    const auto row_base = desc.gather_row_base(gather_index(index_reg, i));
    if (!row_base) {
      return config_error();
    }
    for (std::size_t j = 0; j < cols; ++j) {
      const auto addr = desc.gather_matrix_addr(*row_base, j);
      if (!addr) {
        return config_error();
      }
      addrs[i * cols + j] = *addr;
    }
  }

  auto &reg = ctx.state.t[index];
  const std::span<const std::uint64_t> range(addrs.data(), indexed_rows * cols);
  const auto at = matrix_accessor(config);
  const auto result = load_elements(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  auto dst = matrix_view(reg);
  for (std::size_t i = indexed_rows; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      dst.at(i, j) = 0;
    }
  }
  return {.status = StepStatus::retired};
}

template <typename Reg, typename At>
ExecuteResult gather_vector_access(const MblkInstr &instr, ExecContext &ctx, VectorDescView &desc,
                                   std::size_t len, Reg &reg, At at) {
  const auto &index_config = ctx.state.vc[instr.gather_src];
  if (!valid_gather_index(index_config)) {
    return config_error();
  }

  // Lanes beyond the index vector's effective length are zero-filled. No
  // descriptor increment is performed for gather.
  const auto indexed = std::min(len, std::size_t{index_config.len});
  const auto &index_reg = ctx.state.v[instr.gather_src];
  const auto addrs = collect_addresses<VECTOR_LANES>(
      indexed, [&](std::size_t j) { return desc.gather_vector_addr(gather_index(index_reg, j)); });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), indexed);
  const auto result = load_elements(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  for (std::size_t j = indexed; j < len; ++j) {
    reg[j] = 0;
  }
  return {.status = StepStatus::retired};
}

template <typename Reg, typename At>
ExecuteResult strided_vector_access(const MblkInstr &instr, ExecContext &ctx, VectorDescView &desc,
                                    std::size_t len, Reg &reg, At at) {
  const auto next_base = desc.incremented_base(len);
  if (!next_base) {
    return config_error();
  }
  const auto addrs = collect_addresses<VECTOR_LANES>(
      len, [&](std::size_t j) { return desc.vector_element_addr(j); });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), len);
  const auto result = is_store(instr.op) ? store_elements(ctx.memory, reg, range, at)
                                         : load_elements(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

template <NpuDomain Domain> ExecuteResult vector_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::vec8 || Domain == NpuDomain::vec32);
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, instr.reg.index);
  if (!valid_domain_config(config)) {
    return config_error();
  }
  VectorDescView desc(descriptor);
  if (!desc.is_valid()) {
    return config_error();
  }
  const auto len = std::size_t{config.len};
  const auto at = vector_accessor();

  if (instr.op == MblkOp::gather) {
    return gather_vector_access(instr, ctx, desc, len, reg, at);
  }
  return strided_vector_access(instr, ctx, desc, len, reg, at);
}

} // namespace

ExecuteResult execute(const MblkInstr &instr, ExecContext &ctx) {
  switch (instr.reg.domain) {
  case NpuDomain::tile:
    switch (instr.op) {
    case MblkOp::load:
    case MblkOp::store: return block_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::row_load:
    case MblkOp::row_store: return row_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::gather: return gather_tile_access(instr, ctx);
    default: unreachable("M_BLK op/domain combination rejected by validation");
    }
  case NpuDomain::acc:
    switch (instr.op) {
    case MblkOp::load:
    case MblkOp::store: return block_access<NpuDomain::acc>(instr, ctx);
    case MblkOp::row_load:
    case MblkOp::row_store: return row_access<NpuDomain::acc>(instr, ctx);
    case MblkOp::gather: // gather has no Acc-domain encoding
    default: unreachable("M_BLK op/domain combination rejected by validation");
    }
  case NpuDomain::vec8: return vector_access<NpuDomain::vec8>(instr, ctx);
  case NpuDomain::vec32: return vector_access<NpuDomain::vec32>(instr, ctx);
  default: unreachable("M_BLK op/domain combination rejected by validation");
  }
}

} // namespace tile_npu
