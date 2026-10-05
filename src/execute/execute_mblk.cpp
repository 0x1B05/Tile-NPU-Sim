#include "execute_mblk.h"

#include "config_check.h"
#include "tile_npu/descriptor_view.h"
#include "tile_npu/endian.h"
#include "tile_npu/matrix_view.h"
#include "tile_npu/unreachable.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tile_npu {
namespace {

ExecuteResult config_error() {
  return {.status = StepStatus::invalid_configuration};
}

ExecuteResult access_fault(StepStatus status, std::uint64_t address) {
  return {.status = status, .fault_address = address};
}

// Loads and stores are raw copies at the configured dtype width, little
// endian; the memory path performs no type conversion.
template <typename Element>
bool load_element(const Memory &memory, std::uint64_t address, Element &out) {
  std::array<std::byte, sizeof(Element)> bytes{};
  if (!memory.read(address, bytes)) {
    return false;
  }
  out = load_le<Element>(bytes);
  return true;
}

template <typename Element>
bool store_element(Memory &memory, std::uint64_t address, Element value) {
  std::array<std::byte, sizeof(Element)> bytes{};
  store_le(value, bytes);
  return memory.write(address, bytes);
}

// The ISA defines the transpose transform only for a full tload; any other
// access with transform != 0 is an undefined combination.
bool transform_legal(const MblkInstr &instr, const MatrixDescriptor &desc) {
  return desc.transform == 0 ||
         (instr.op == MblkOp::load && instr.reg.domain == NpuDomain::tile);
}

// Every access runs in two phases: all addresses are computed first, then
// the transfer happens. Address overflow is a property of the descriptor
// and shape alone, so it reports deterministically as a config error
// instead of surfacing as a mid-access fault at whichever element happens
// to be visited first.
template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
block_access(const MblkInstr &instr, ExecContext &ctx,
             std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
             const std::array<Config, RegCount> &configs,
             std::array<MatrixDescriptor, RegCount> &descs) {
  const auto &config = configs[instr.reg.index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  MatrixDescView desc(descs[instr.reg.index]);
  if (!desc.valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }
  const auto rows = std::size_t{config.rows};
  const auto cols = std::size_t{config.cols};
  if (!desc.incremented_base(rows, cols)) {
    return config_error();
  }

  const bool transpose = desc.descriptor().transform != 0;
  std::array<std::uint64_t, MATRIX_ELEMENTS> addrs{};
  {
    std::size_t n = 0;
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        const auto addr = transpose ? desc.address(j, i) : desc.address(i, j);
        if (!addr) {
          return config_error();
        }
        addrs[n++] = *addr;
      }
    }
  }

  // Loads stage into a copy of the register and stores probe the whole
  // range before writing, so a mid-transfer fault leaves state unchanged.
  const bool load = instr.op == MblkOp::load;
  auto staging = regs[instr.reg.index];
  MatrixView<Element> view(std::span<Element>(staging), rows, cols, 0);
  if (load) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        const auto addr = addrs[n++];
        if (!load_element(ctx.memory, addr, view.at(i, j))) {
          return access_fault(StepStatus::load_fault, addr);
        }
      }
    }
    regs[instr.reg.index] = staging;
  } else {
    for (std::size_t n = 0; n < rows * cols; ++n) {
      if (!ctx.memory.contains(addrs[n], sizeof(Element))) {
        return access_fault(StepStatus::store_fault, addrs[n]);
      }
    }
    std::size_t n = 0;
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        if (!store_element(ctx.memory, addrs[n++], view.at(i, j))) {
          unreachable("store failed after the range probe passed");
        }
      }
    }
  }
  if (!desc.post_increment(rows, cols)) {
    unreachable("increment failed after the pre-validation passed");
  }
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
row_access(const MblkInstr &instr, ExecContext &ctx,
           std::array<std::array<Element, MATRIX_ELEMENTS>, RegCount> &regs,
           const std::array<Config, RegCount> &configs,
           std::array<MatrixDescriptor, RegCount> &descs) {
  const auto &config = configs[instr.reg.index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  MatrixDescView desc(descs[instr.reg.index]);
  if (!desc.valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }
  if (instr.index_is_scalar && instr.index >= SCALAR_REGISTERS) {
    return config_error(); // hand-constructed instruction
  }

  const auto row = instr.index_is_scalar ? ctx.state.scalar.read_x(instr.index)
                                         : std::uint64_t{instr.index};
  // The ISA leaves the row bound open; the simulator follows the row-move
  // convention and rejects row numbers outside the effective shape.
  if (row >= config.rows) {
    return config_error();
  }
  const auto cols = std::size_t{config.cols};
  if (!desc.incremented_base(1, cols)) {
    return config_error();
  }

  std::array<std::uint64_t, MATRIX_EXTENT> addrs{};
  for (std::size_t j = 0; j < cols; ++j) {
    const auto addr = desc.address(0, j);
    if (!addr) {
      return config_error();
    }
    addrs[j] = *addr;
  }

  const bool load = instr.op == MblkOp::row_load;
  auto staging = regs[instr.reg.index];
  MatrixView<Element> view(std::span<Element>(staging), config.rows, cols, 0);
  if (load) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (!load_element(ctx.memory, addrs[j], view.at(row, j))) {
        return access_fault(StepStatus::load_fault, addrs[j]);
      }
    }
    regs[instr.reg.index] = staging;
  } else {
    for (std::size_t j = 0; j < cols; ++j) {
      if (!ctx.memory.contains(addrs[j], sizeof(Element))) {
        return access_fault(StepStatus::store_fault, addrs[j]);
      }
    }
    for (std::size_t j = 0; j < cols; ++j) {
      if (!store_element(ctx.memory, addrs[j], view.at(row, j))) {
        unreachable("store failed after the range probe passed");
      }
    }
  }
  if (!desc.post_increment(1, cols)) {
    unreachable("increment failed after the pre-validation passed");
  }
  return {.status = StepStatus::retired};
}

// tload.gather: row i is indexed by vS[i] (an i32 lane value) times
// col_stride_bytes, then walks row_stride_bytes along the row; rows beyond
// len(vS) are zero-filled. Gather never increments the descriptor, and index
// legality is software's contract.
ExecuteResult gather_tile(const MblkInstr &instr, ExecContext &ctx,
                          std::array<Tile, TILE_REGISTERS> &regs,
                          const std::array<TileConfig, TILE_REGISTERS> &configs,
                          std::array<MatrixDescriptor, TILE_REGISTERS> &descs) {
  const auto &config = configs[instr.reg.index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  if (instr.gather_src >= VECTOR_REGISTERS) {
    return config_error(); // hand-constructed instruction
  }
  const auto &index_config = ctx.state.vc[instr.gather_src];
  if (index_config.len == 0 || index_config.len > VECTOR_LANES) {
    return config_error();
  }
  MatrixDescView desc(descs[instr.reg.index]);
  if (!desc.valid() || !transform_legal(instr, desc.descriptor())) {
    return config_error();
  }

  const auto rows = std::size_t{config.rows};
  const auto cols = std::size_t{config.cols};
  const auto &index_reg = ctx.state.v[instr.gather_src];
  const auto &descriptor = desc.descriptor();
  std::array<std::uint64_t, MATRIX_ELEMENTS> addrs{};
  for (std::size_t i = 0; i < rows; ++i) {
    if (i >= index_config.len) {
      continue; // zero-filled rows have no addresses
    }
    const auto element = std::bit_cast<std::int32_t>(index_reg[i]);
    const auto row_base = detail::add_signed_offset(
        descriptor.base_addr,
        std::int64_t{element} * descriptor.col_stride_bytes);
    if (!row_base) {
      return config_error();
    }
    for (std::size_t j = 0; j < cols; ++j) {
      const auto addr = detail::add_signed_offset(
          *row_base, std::int64_t{descriptor.row_stride_bytes} *
                         static_cast<std::int64_t>(j));
      if (!addr) {
        return config_error();
      }
      addrs[i * cols + j] = *addr;
    }
  }

  auto staging = regs[instr.reg.index];
  MatrixView<std::uint8_t> view(std::span<std::uint8_t>(staging), rows, cols,
                                0);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      if (i >= index_config.len) {
        view.at(i, j) = 0;
        continue;
      }
      const auto addr = addrs[i * cols + j];
      if (!load_element(ctx.memory, addr, view.at(i, j))) {
        return access_fault(StepStatus::load_fault, addr);
      }
    }
  }
  regs[instr.reg.index] = staging;
  return {.status = StepStatus::retired};
}

template <typename Element, typename Config, std::size_t RegCount>
ExecuteResult
vector_access(const MblkInstr &instr, ExecContext &ctx,
              std::array<std::array<Element, VECTOR_LANES>, RegCount> &regs,
              const std::array<Config, RegCount> &configs,
              std::array<VectorDescriptor, RegCount> &descs) {
  const auto &config = configs[instr.reg.index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  VectorDescView desc(descs[instr.reg.index]);
  if (!desc.valid()) {
    return config_error();
  }
  const auto len = std::size_t{config.len};

  // {b,v}load.gather: lane j is indexed by vS[j] times stride_bytes; lanes
  // beyond len(vS) are zero-filled. No descriptor increment.
  if (instr.op == MblkOp::gather) {
    if (instr.gather_src >= VECTOR_REGISTERS) {
      return config_error(); // hand-constructed instruction
    }
    const auto &index_config = ctx.state.vc[instr.gather_src];
    if (index_config.len == 0 || index_config.len > VECTOR_LANES) {
      return config_error();
    }
    const auto &index_reg = ctx.state.v[instr.gather_src];
    std::array<std::uint64_t, VECTOR_LANES> addrs{};
    for (std::size_t j = 0; j < len; ++j) {
      if (j >= index_config.len) {
        continue; // zero-filled lanes have no addresses
      }
      const auto element = std::bit_cast<std::int32_t>(index_reg[j]);
      const auto addr = detail::add_signed_offset(
          desc.descriptor().base_addr,
          std::int64_t{element} * desc.descriptor().stride_bytes);
      if (!addr) {
        return config_error();
      }
      addrs[j] = *addr;
    }
    auto staging = regs[instr.reg.index];
    for (std::size_t j = 0; j < len; ++j) {
      if (j >= index_config.len) {
        staging[j] = Element{0};
        continue;
      }
      if (!load_element(ctx.memory, addrs[j], staging[j])) {
        return access_fault(StepStatus::load_fault, addrs[j]);
      }
    }
    regs[instr.reg.index] = staging;
    return {.status = StepStatus::retired};
  }

  if (!desc.incremented_base(len)) {
    return config_error();
  }
  std::array<std::uint64_t, VECTOR_LANES> addrs{};
  for (std::size_t j = 0; j < len; ++j) {
    const auto addr = desc.address(j);
    if (!addr) {
      return config_error();
    }
    addrs[j] = *addr;
  }

  const bool load = instr.op == MblkOp::load;
  auto staging = regs[instr.reg.index];
  if (load) {
    for (std::size_t j = 0; j < len; ++j) {
      if (!load_element(ctx.memory, addrs[j], staging[j])) {
        return access_fault(StepStatus::load_fault, addrs[j]);
      }
    }
    regs[instr.reg.index] = staging;
  } else {
    for (std::size_t j = 0; j < len; ++j) {
      if (!ctx.memory.contains(addrs[j], sizeof(Element))) {
        return access_fault(StepStatus::store_fault, addrs[j]);
      }
    }
    for (std::size_t j = 0; j < len; ++j) {
      if (!store_element(ctx.memory, addrs[j], staging[j])) {
        unreachable("store failed after the range probe passed");
      }
    }
  }
  if (!desc.post_increment(len)) {
    unreachable("increment failed after the pre-validation passed");
  }
  return {.status = StepStatus::retired};
}

} // namespace

ExecuteResult execute_mblk(const MblkInstr &instr, ExecContext &ctx) {
  State &state = ctx.state;
  // Decode bounds the register numbers and the op/domain combinations;
  // hand-constructed instructions can violate both, so the executor
  // re-checks and treats them as config errors.
  switch (instr.reg.domain) {
  case NpuDomain::tile:
    if (instr.reg.index < TILE_REGISTERS) {
      switch (instr.op) {
      case MblkOp::load:
      case MblkOp::store:
        return block_access(instr, ctx, state.t, state.tc, state.tm);
      case MblkOp::row_load:
      case MblkOp::row_store:
        return row_access(instr, ctx, state.t, state.tc, state.tm);
      case MblkOp::gather:
        return gather_tile(instr, ctx, state.t, state.tc, state.tm);
      }
    }
    break;
  case NpuDomain::acc:
    if (instr.reg.index < ACC_REGISTERS) {
      switch (instr.op) {
      case MblkOp::load:
      case MblkOp::store:
        return block_access(instr, ctx, state.a, state.ac, state.am);
      case MblkOp::row_load:
      case MblkOp::row_store:
        return row_access(instr, ctx, state.a, state.ac, state.am);
      case MblkOp::gather: break; // gather is undefined on the Acc domain
      }
    }
    break;
  case NpuDomain::vec8:
    if (instr.reg.index < VECTOR_REGISTERS && instr.op != MblkOp::row_load &&
        instr.op != MblkOp::row_store) {
      return vector_access(instr, ctx, state.b, state.bc, state.bm);
    }
    break;
  case NpuDomain::vec32:
    if (instr.reg.index < VECTOR_REGISTERS && instr.op != MblkOp::row_load &&
        instr.op != MblkOp::row_store) {
      return vector_access(instr, ctx, state.v, state.vc, state.vm);
    }
    break;
  }
  return config_error();
}

} // namespace tile_npu
