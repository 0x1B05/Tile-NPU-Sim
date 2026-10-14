#include "execute.h"

#include "config_check.h"
#include "descriptor_view.h"
#include "domain.h"
#include "matrix_view.h"
#include "operand.h"
#include "utils/endian.h"
#include "utils/unreachable.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>

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
  return desc.transform == 0 ||
         (instr.op == MblkOp::load && accessed_reg(instr).domain == NpuDomain::tile);
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
// before the first write, so a faulting store leaves architectural state
// unchanged.
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
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, accessed_reg(instr).index);
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

  // A transpose load swaps the coordinates handed to the descriptor.
  const bool transpose = desc.descriptor().transform != 0;
  const auto addrs = collect_addresses<MATRIX_ELEMENTS>(rows * cols, [&](std::size_t n) {
    return transpose ? desc.matrix_element_addr(n % cols, n / cols)
                     : desc.matrix_element_addr(n / cols, n % cols);
  });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), rows * cols);
  const auto at = matrix_accessor(config);
  const auto result = is_store(instr.op) ? store_elements(ctx.memory, reg, range, at)
                                         : load_elements(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

template <NpuDomain Domain> ExecuteResult row_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::tile || Domain == NpuDomain::acc);
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, accessed_reg(instr).index);
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
  const auto index = accessed_reg(instr).index;
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
  auto [reg, config, descriptor] = bind_register<Domain>(ctx.state, accessed_reg(instr).index);
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
  switch (accessed_reg(instr).domain) {
  case NpuDomain::tile:
    switch (instr.op) {
    case MblkOp::load:
    case MblkOp::store: return block_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::row_load:
    case MblkOp::row_store: return row_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::gather: return gather_tile_access(instr, ctx);
    }
    break;
  case NpuDomain::acc:
    switch (instr.op) {
    case MblkOp::load:
    case MblkOp::store: return block_access<NpuDomain::acc>(instr, ctx);
    case MblkOp::row_load:
    case MblkOp::row_store: return row_access<NpuDomain::acc>(instr, ctx);
    case MblkOp::gather: break; // gather has no Acc-domain encoding
    }
    break;
  case NpuDomain::vec8: return vector_access<NpuDomain::vec8>(instr, ctx);
  case NpuDomain::vec32: return vector_access<NpuDomain::vec32>(instr, ctx);
  }
  unreachable("M_BLK op/domain combination rejected by validation");
}

} // namespace tile_npu
