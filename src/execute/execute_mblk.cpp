#include "execute.h"

#include "config_check.h"
#include "operand.h"
#include "tile_npu/descriptor_view.h"
#include "tile_npu/matrix_view.h"
#include "utils/endian.h"
#include "utils/unreachable.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <tuple>
#include <utility>

namespace tile_npu {
namespace {

template <typename Element>
bool load_element(const AddressSpace &memory, std::uint64_t address, Element &out) {
  std::array<std::byte, sizeof(Element)> bytes{};
  if (!memory.read(address, bytes)) {
    return false;
  }
  out = load_le<Element>(bytes);
  return true;
}

template <typename Element>
bool store_element(AddressSpace &memory, std::uint64_t address, Element value) {
  std::array<std::byte, sizeof(Element)> bytes{};
  store_le(value, bytes);
  return memory.write(address, bytes);
}

bool transform_legal(const MblkInstr &instr, const MatrixDescriptor &desc) {
  return desc.transform == 0 ||
         (instr.op == MblkOp::load && accessed_reg(instr).domain == NpuDomain::tile);
}

// Every access runs in two phases: all addresses are computed first, then
// the transfer happens. Address overflow is a property of the descriptor and
// shape alone, so it reports deterministically as a config error before any
// element is touched.
template <std::size_t Max, typename Next>
std::optional<std::array<std::uint64_t, Max>> fill_addresses(std::size_t count, Next next) {
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

// A load stages a copy of the destination register and commits it only after
// the last element arrived, so a mid-transfer fault leaves architectural
// state unchanged.
template <typename Reg, typename At>
ExecuteResult staged_load(const AddressSpace &memory, Reg &reg,
                          std::span<const std::uint64_t> addrs, At at) {
  auto staging = reg;
  for (std::size_t n = 0; n < addrs.size(); ++n) {
    if (!load_element(memory, addrs[n], at(staging, n))) {
      return access_fault(StepStatus::load_fault, addrs[n]);
    }
  }
  reg = staging;
  return {.status = StepStatus::retired};
}

// A store reads the source register in place; the whole range is probed
// before the first write, so a faulting store leaves architectural state
// unchanged.
template <typename Reg, typename At>
ExecuteResult probed_store(AddressSpace &memory, const Reg &reg,
                           std::span<const std::uint64_t> addrs, At at) {
  using Element = typename Reg::value_type;
  for (const auto addr : addrs) {
    if (!memory.contains(addr, sizeof(Element))) {
      return access_fault(StepStatus::store_fault, addr);
    }
  }
  for (std::size_t n = 0; n < addrs.size(); ++n) {
    if (!store_element(memory, addrs[n], at(reg, n))) {
      unreachable("store failed after the range probe passed");
    }
  }
  return {.status = StepStatus::retired};
}

// The element accessor for a matrix transfer: position n is the logical
// coordinate (n / cols, n % cols), read through the row-major view.
template <typename Config> auto matrix_at(const Config &config) {
  return [&config](auto &storage, std::size_t n) -> decltype(auto) {
    const auto view = matrix_view(storage);
    return view.at(n / config.cols, n % config.cols);
  };
}

// The element accessor for a vector transfer: position n is lane n.
auto vector_at() {
  return [](auto &storage, std::size_t n) -> decltype(auto) { return storage[n]; };
}

// The gather index vector must use a valid Vec32 configuration with i32 elements.
bool valid_gather_index(const Vec32Config &config) {
  return config.dtype == DType::i32 && valid_domain_config(config);
}

std::int32_t gather_index(const Vec32 &reg, std::size_t lane) {
  return std::bit_cast<std::int32_t>(reg[lane]);
}

// Bind the data, read-only compute config, and memory descriptor of one register.
template <NpuDomain Domain> auto register_state(State &state, RegisterIndex index) {
  if constexpr (Domain == NpuDomain::tile) {
    return std::tie(state.t[index], std::as_const(state.tc[index]), state.tm[index]);
  } else if constexpr (Domain == NpuDomain::acc) {
    return std::tie(state.a[index], std::as_const(state.ac[index]), state.am[index]);
  } else if constexpr (Domain == NpuDomain::vec8) {
    return std::tie(state.b[index], std::as_const(state.bc[index]), state.bm[index]);
  } else {
    static_assert(Domain == NpuDomain::vec32);
    return std::tie(state.v[index], std::as_const(state.vc[index]), state.vm[index]);
  }
}

template <NpuDomain Domain> ExecuteResult block_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::tile || Domain == NpuDomain::acc);
  auto [reg, config, descriptor] = register_state<Domain>(ctx.state, accessed_reg(instr).index);
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
  const auto addrs = fill_addresses<MATRIX_ELEMENTS>(rows * cols, [&](std::size_t n) {
    return transpose ? desc.matrix_element_addr(n % cols, n / cols)
                     : desc.matrix_element_addr(n / cols, n % cols);
  });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), rows * cols);
  const auto at = matrix_at(config);
  const auto result = is_store(instr.op) ? probed_store(ctx.memory, reg, range, at)
                                         : staged_load(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

template <NpuDomain Domain> ExecuteResult row_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::tile || Domain == NpuDomain::acc);
  auto [reg, config, descriptor] = register_state<Domain>(ctx.state, accessed_reg(instr).index);
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

  const auto addrs = fill_addresses<MATRIX_EXTENT>(
      cols, [&](std::size_t j) { return desc.matrix_element_addr(0, j); });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), cols);
  // The view spans the whole matrix; only row `row` is transferred.
  const auto at = [row](auto &storage, std::size_t n) -> decltype(auto) {
    const auto view = matrix_view(storage);
    return view.at(row, n);
  };
  const auto result = is_store(instr.op) ? probed_store(ctx.memory, reg, range, at)
                                         : staged_load(ctx.memory, reg, range, at);
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

// tload.gather
ExecuteResult gather_tile(const MblkInstr &instr, ExecContext &ctx) {
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
  const auto result = staged_load(ctx.memory, reg, range, matrix_at(config));
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

template <NpuDomain Domain> ExecuteResult vector_access(const MblkInstr &instr, ExecContext &ctx) {
  static_assert(Domain == NpuDomain::vec8 || Domain == NpuDomain::vec32);
  auto [reg, config, descriptor] = register_state<Domain>(ctx.state, accessed_reg(instr).index);
  if (!valid_domain_config(config)) {
    return config_error();
  }
  VectorDescView desc(descriptor);
  if (!desc.is_valid()) {
    return config_error();
  }
  const auto len = std::size_t{config.len};

  // {b,v}load.gather: lanes beyond the index vector's effective length are
  // zero-filled. No descriptor increment.
  if (instr.op == MblkOp::gather) {
    const auto &index_config = ctx.state.vc[instr.gather_src];
    if (!valid_gather_index(index_config)) {
      return config_error();
    }
    const auto indexed = std::min(len, std::size_t{index_config.len});
    const auto &index_reg = ctx.state.v[instr.gather_src];
    const auto addrs = fill_addresses<VECTOR_LANES>(indexed, [&](std::size_t j) {
      return desc.gather_vector_addr(gather_index(index_reg, j));
    });
    if (!addrs) {
      return config_error();
    }
    const std::span<const std::uint64_t> range(addrs->data(), indexed);
    const auto result = staged_load(ctx.memory, reg, range, vector_at());
    if (result.status != StepStatus::retired) {
      return result;
    }
    for (std::size_t j = indexed; j < len; ++j) {
      reg[j] = 0;
    }
    return {.status = StepStatus::retired};
  }

  const auto next_base = desc.incremented_base(len);
  if (!next_base) {
    return config_error();
  }
  const auto addrs =
      fill_addresses<VECTOR_LANES>(len, [&](std::size_t j) { return desc.vector_element_addr(j); });
  if (!addrs) {
    return config_error();
  }

  const std::span<const std::uint64_t> range(addrs->data(), len);
  const auto result = is_store(instr.op) ? probed_store(ctx.memory, reg, range, vector_at())
                                         : staged_load(ctx.memory, reg, range, vector_at());
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
  return result;
}

} // namespace

ExecuteResult execute_mblk(const MblkInstr &instr, ExecContext &ctx) {
  switch (accessed_reg(instr).domain) {
  case NpuDomain::tile:
    switch (instr.op) {
    case MblkOp::load:
    case MblkOp::store: return block_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::row_load:
    case MblkOp::row_store: return row_access<NpuDomain::tile>(instr, ctx);
    case MblkOp::gather: return gather_tile(instr, ctx);
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
