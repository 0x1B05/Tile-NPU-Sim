#include "execute.h"

#include "config_check.h"
#include "descriptor_view.h"
#include "domain.h"
#include "matrix_view.h"
#include "operand.h"
#ifdef CONFIG_MTRACE
#include "trace/mtrace.h"
#endif
#include "utils/common.h"
#include "utils/endian.h"

#include <algorithm>
#include <array>
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

// Visits every element once to probe the address space, then visits it again
// to transfer. Address overflow is a configuration error and wins over a fault
// on an earlier element; otherwise the first fault address is reported.
template <typename Element, typename Visit>
ExecuteResult probe_and_transfer(AddressSpace &memory, Visit &&visit, bool store) {
  bool overflow = false;
  std::optional<std::uint64_t> fault_address;
  visit([&](std::optional<std::uint64_t> address, Element &) {
    if (!address) {
      overflow = true;
    } else if (!fault_address && !memory.contains(*address, sizeof(Element))) {
      fault_address = *address;
    }
  });
  if (overflow) {
    return config_error();
  }
  if (fault_address) {
    return access_fault(store ? StepStatus::store_fault : StepStatus::load_fault, *fault_address);
  }

  visit([&](std::optional<std::uint64_t> address, Element &element) {
    if (!address) {
      unreachable("address overflowed after range prevalidation");
    }
    if (store) {
      store_element(memory, *address, element);
    } else {
      load_element(memory, *address, element);
    }
  });
  return {.status = StepStatus::retired};
}

// Gathers indexed rows and zero-fills the rows beyond the index length. The
// vector form is the special case of one column per row.
template <typename Element, typename RowAddress, typename ColumnAddress, typename ElementAt>
ExecuteResult gather_access(AddressSpace &memory, std::size_t rows, std::size_t cols,
                            std::size_t indexed_rows, RowAddress row_address,
                            ColumnAddress column_address, ElementAt element_at) {
  const auto visit = [&](auto &&fn) {
    for (std::size_t row = 0; row < indexed_rows; ++row) {
      const auto row_base = row_address(row);
      for (std::size_t col = 0; col < cols; ++col) {
        const auto address =
            row_base ? column_address(*row_base, col) : std::optional<std::uint64_t>{};
        fn(address, element_at(row, col));
      }
    }
  };
  const auto result = probe_and_transfer<Element>(memory, visit, false);
  if (result.status != StepStatus::retired) {
    return result;
  }
  for (std::size_t row = indexed_rows; row < rows; ++row) {
    for (std::size_t col = 0; col < cols; ++col) {
      element_at(row, col) = 0;
    }
  }
  return result;
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
  std::array<Element, MATRIX_EXTENT> buffer;
  const auto transfer = [&](std::uint64_t address, std::size_t length, std::size_t run) {
    if (transpose) {
      // Transposed stores use the per-element order; only loads reach here.
      if (!ctx.memory.read(address,
                           std::as_writable_bytes(std::span<Element>(buffer.data(), length)))) {
        unreachable("AddressSpace::read failed after range prevalidation");
      }
      for (std::size_t row = 0; row < length; ++row) {
        view.at(row, run) = buffer[row];
      }
      return;
    }

    Element *data = &view.at(run, 0);
    if (store) {
      if (!ctx.memory.write(address, std::as_bytes(std::span<const Element>(data, length)))) {
        unreachable("AddressSpace::write failed after range prevalidation");
      }
    } else {
      if (!ctx.memory.read(address, std::as_writable_bytes(std::span<Element>(data, length)))) {
        unreachable("AddressSpace::read failed after range prevalidation");
      }
    }
  };
  if (!for_each_block_run(desc, rows, cols, transpose, transfer)) {
    unreachable("block run overflowed after range prevalidation");
  }
  return ExecuteResult{.status = StepStatus::retired};
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
#ifdef CONFIG_MTRACE
  const auto trace_descriptor = desc.descriptor();
  const auto trace_elem = static_cast<std::uint8_t>(sizeof(Element));
#endif

  // A contiguous run per row is the common case and avoids the per-element
  // address-space calls. Multi-byte elements only take this path on a
  // little-endian host because the raw bytes are copied directly.
  if constexpr (sizeof(Element) == 1 || std::endian::native == std::endian::little) {
    if (desc.descriptor().col_stride_bytes == static_cast<std::int32_t>(sizeof(Element))) {
      const auto fast = try_contiguous_block(instr, ctx, desc, reg, rows, cols, transpose);
      if (fast) {
        if (fast->status == StepStatus::retired) {
          desc.commit_base(*next_base);
#ifdef CONFIG_MTRACE
          if (ctx.trace) {
            trace_matrix_move(*ctx.trace, instr, trace_descriptor, rows, cols, trace_elem,
                              transpose);
          }
#endif
        }
        return *fast;
      }
    }
  }

  auto view = matrix_view(reg);
  const auto visit = [&](auto &&fn) {
    for (std::size_t row = 0; row < rows; ++row) {
      for (std::size_t col = 0; col < cols; ++col) {
        const auto address =
            transpose ? desc.matrix_element_addr(col, row) : desc.matrix_element_addr(row, col);
        fn(address, view.at(row, col));
      }
    }
  };
  const auto result = probe_and_transfer<Element>(ctx.memory, visit, is_store(instr.op));
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
#ifdef CONFIG_MTRACE
  if (ctx.trace) {
    trace_matrix_move(*ctx.trace, instr, trace_descriptor, rows, cols, trace_elem, transpose);
  }
#endif
  return result;
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

  using Element = typename std::remove_reference_t<decltype(reg)>::value_type;
#ifdef CONFIG_MTRACE
  const auto trace_descriptor = desc.descriptor();
  const auto trace_elem = static_cast<std::uint8_t>(sizeof(Element));
#endif
  auto view = matrix_view(reg);
  const auto visit = [&](auto &&fn) {
    for (std::size_t col = 0; col < cols; ++col) {
      fn(desc.matrix_element_addr(0, col), view.at(row, col));
    }
  };
  const auto result = probe_and_transfer<Element>(ctx.memory, visit, is_store(instr.op));
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
#ifdef CONFIG_MTRACE
  if (ctx.trace) {
    trace_matrix_row_move(*ctx.trace, instr, trace_descriptor, static_cast<std::uint8_t>(cols),
                          trace_elem);
  }
#endif
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
  if (!valid_config_of(index_config, DType::u32)) {
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
  auto &reg = ctx.state.t[index];
  auto view = matrix_view(reg);
  const auto row_address = [&](std::size_t row) { return desc.gather_row_base(index_reg[row]); };
  const auto column_address = [&](std::uint64_t row_base, std::size_t col) {
    return desc.gather_matrix_addr(row_base, col);
  };
  const auto element = [&](std::size_t row, std::size_t col) -> Tile::value_type & {
    return view.at(row, col);
  };
  const auto result = gather_access<Tile::value_type>(ctx.memory, rows, cols, indexed_rows,
                                                      row_address, column_address, element);
#ifdef CONFIG_MTRACE
  if (result.status == StepStatus::retired && ctx.trace) {
    trace_gather_tile(*ctx.trace, instr, desc.descriptor(), static_cast<std::uint8_t>(indexed_rows),
                      static_cast<std::uint8_t>(cols),
                      static_cast<std::uint8_t>(sizeof(Tile::value_type)), index_reg);
  }
#endif
  return result;
}

ExecuteResult gather_vector_access(const MblkInstr &instr, ExecContext &ctx, VectorDescView &desc,
                                   std::size_t len, Vec32 &reg) {
  const auto &index_config = ctx.state.vc[instr.gather_src];
  if (!valid_config_of(index_config, DType::u32)) {
    return config_error();
  }

  // Lanes beyond the index vector's effective length are zero-filled. No
  // descriptor increment is performed for gather.
  const auto indexed = std::min(len, std::size_t{index_config.len});
  const auto &index_reg = ctx.state.v[instr.gather_src];
  using Element = Vec32::value_type;
  const auto row_address = [&](std::size_t lane) {
    return desc.gather_vector_addr(index_reg[lane]);
  };
  const auto column_address = [](std::uint64_t address, std::size_t) {
    return std::optional<std::uint64_t>{address};
  };
  const auto element = [&](std::size_t lane, std::size_t) -> Element & { return reg[lane]; };
  const auto result =
      gather_access<Element>(ctx.memory, len, 1, indexed, row_address, column_address, element);
#ifdef CONFIG_MTRACE
  if (result.status == StepStatus::retired && ctx.trace) {
    trace_gather_vector(*ctx.trace, instr, desc.descriptor(), static_cast<std::uint8_t>(indexed),
                        static_cast<std::uint8_t>(sizeof(Element)), index_reg);
  }
#endif
  return result;
}

ExecuteResult strided_vector_access(const MblkInstr &instr, ExecContext &ctx, VectorDescView &desc,
                                    std::size_t len, Vec32 &reg) {
  const auto next_base = desc.incremented_base(len);
  if (!next_base) {
    return config_error();
  }

  using Element = Vec32::value_type;
#ifdef CONFIG_MTRACE
  const auto trace_descriptor = desc.descriptor();
  const auto trace_elem = static_cast<std::uint8_t>(sizeof(Element));
#endif
  const auto visit = [&](auto &&fn) {
    for (std::size_t lane = 0; lane < len; ++lane) {
      fn(desc.vector_element_addr(lane), reg[lane]);
    }
  };
  const auto result = probe_and_transfer<Element>(ctx.memory, visit, is_store(instr.op));
  if (result.status != StepStatus::retired) {
    return result;
  }
  desc.commit_base(*next_base);
#ifdef CONFIG_MTRACE
  if (ctx.trace) {
    trace_vector_move(*ctx.trace, instr, trace_descriptor, static_cast<std::uint8_t>(len),
                      trace_elem);
  }
#endif
  return result;
}

ExecuteResult vector_access(const MblkInstr &instr, ExecContext &ctx) {
  const auto index = instr.reg.index;
  auto &reg = ctx.state.v[index];
  const auto &config = ctx.state.vc[index];
  auto &descriptor = ctx.state.vm[index];
  if (!valid_domain_config(config)) {
    return config_error();
  }
  VectorDescView desc(descriptor);
  if (!desc.is_valid()) {
    return config_error();
  }
  const auto len = std::size_t{config.len};

  if (instr.op == MblkOp::gather) {
    return gather_vector_access(instr, ctx, desc, len, reg);
  }
  return strided_vector_access(instr, ctx, desc, len, reg);
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
  case NpuDomain::vec32: return vector_access(instr, ctx);
  default: unreachable("M_BLK op/domain combination rejected by validation");
  }
}

} // namespace tile_npu
