#include "mtrace.h"

#include "tile_npu/instruction.h"
#include "trace/names.h"
#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <variant>

namespace tile_npu {
namespace {

TraceEndpointKind endpoint_kind(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return TraceEndpointKind::tile;
  case NpuDomain::acc: return TraceEndpointKind::acc;
  case NpuDomain::vec32: return TraceEndpointKind::vec32;
  default: return TraceEndpointKind::immediate;
  }
}

TraceDescriptorKind descriptor_kind(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return TraceDescriptorKind::tm;
  case NpuDomain::acc: return TraceDescriptorKind::am;
  case NpuDomain::vec32: return TraceDescriptorKind::vm;
  default: return TraceDescriptorKind::none;
  }
}

TraceEndpoint memory_endpoint(std::uint32_t bytes, std::uint8_t elem) {
  return {.kind = TraceEndpointKind::memory, .bytes = bytes, .elem = elem};
}

TraceEndpoint matrix_endpoint(NpuDomain domain, std::uint8_t index, std::uint8_t rows,
                              std::uint8_t cols, std::uint8_t elem, std::string_view part) {
  return {.kind = endpoint_kind(domain),
          .index = index,
          .bytes = static_cast<std::uint32_t>(rows) * cols * elem,
          .elem = elem,
          .dims = 2,
          .shape = {rows, cols},
          .part = part};
}

TraceEndpoint vector_endpoint(std::uint8_t index, std::uint8_t len, std::uint8_t elem,
                              std::string_view part) {
  return {.kind = TraceEndpointKind::vec32,
          .index = index,
          .bytes = static_cast<std::uint32_t>(len) * elem,
          .elem = elem,
          .dims = 1,
          .shape = {len, 0},
          .part = part};
}

TraceEndpoint domain_endpoint(const State &state, NpuDomain domain, std::uint8_t index,
                              std::string_view part) {
  switch (domain) {
  case NpuDomain::tile:
    return trace_matrix_endpoint(TraceEndpointKind::tile, index, state.tc[index], part);
  case NpuDomain::acc:
    return trace_matrix_endpoint(TraceEndpointKind::acc, index, state.ac[index], part);
  case NpuDomain::vec32: return trace_vector_endpoint(index, state.vc[index], part);
  default: return {.part = part};
  }
}

void add_source(TraceEvent &event, TraceEndpoint endpoint) {
  if (event.src_count < event.src.size()) event.src[event.src_count++] = endpoint;
}

void add_dest(TraceEvent &event, TraceEndpoint endpoint) {
  if (event.dst_count < event.dst.size()) event.dst[event.dst_count++] = endpoint;
}

void emit_move(MtraceRecorder &trace, std::string_view op, TraceEndpoint src, TraceEndpoint dst,
               TraceMemory memory = {}) {
  TraceEvent event;
  event.type = TraceEventType::move;
  event.op = op;
  event.src[0] = src;
  event.src_count = 1;
  event.dst[0] = dst;
  event.dst_count = 1;
  event.mem = memory;
  trace.emit(std::move(event));
}

void emit_write(MtraceRecorder &trace, std::string_view op, TraceEndpoint dst) {
  TraceEvent event;
  event.type = TraceEventType::move;
  event.op = op;
  event.dst[0] = dst;
  event.dst_count = 1;
  trace.emit(std::move(event));
}

void emit_read(MtraceRecorder &trace, std::string_view op, TraceEndpoint src) {
  TraceEvent event;
  event.type = TraceEventType::move;
  event.op = op;
  event.src[0] = src;
  event.src_count = 1;
  trace.emit(std::move(event));
}

void emit_compute(MtraceRecorder &trace, std::string_view op, const TraceEvent &event,
                  std::uint32_t macs = 0) {
  TraceEvent out = event;
  out.type = TraceEventType::compute;
  out.op = op;
  out.macs = macs;
  trace.emit(std::move(out));
}

std::string_view mblk_op_name(const MblkInstr &instr) {
  const bool tile = instr.reg.domain == NpuDomain::tile;
  const bool acc = instr.reg.domain == NpuDomain::acc;
  switch (instr.op) {
  case MblkOp::load: return tile ? "tload" : acc ? "aload" : "vload";
  case MblkOp::store: return tile ? "tstore" : acc ? "astore" : "vstore";
  case MblkOp::gather: return tile ? "tload.gather" : "vload.gather";
  }
  return "mblk";
}

TraceMemory block_memory(const MblkInstr &instr, const MatrixDescriptor &descriptor,
                         std::uint8_t rows, std::uint8_t cols, std::uint8_t elem, bool transpose) {
  return {.kind = TraceMemoryKind::block,
          .region = TraceRegion::ram,
          .addr = descriptor.base_addr,
          .elem = elem,
          .rows = rows,
          .cols = cols,
          .row_stride = descriptor.row_stride_bytes,
          .col_stride = descriptor.col_stride_bytes,
          .col_order = transpose,
          .desc_kind = descriptor_kind(instr.reg.domain),
          .desc_index = instr.reg.index};
}

TraceMemory linear_memory(const MblkInstr &instr, std::uint64_t addr, std::uint32_t count,
                          std::int32_t stride, std::uint8_t elem, TraceDescriptorKind kind) {
  return {.kind = TraceMemoryKind::linear,
          .region = TraceRegion::ram,
          .addr = addr,
          .count = count,
          .stride = stride,
          .elem = elem,
          .desc_kind = kind,
          .desc_index = instr.reg.index};
}

std::uint64_t add_signed(std::uint64_t base, std::int64_t offset) {
  return offset >= 0 ? base + static_cast<std::uint64_t>(offset)
                     : base - static_cast<std::uint64_t>(-offset);
}

} // namespace

void trace_matrix_move(MtraceRecorder &trace, const MblkInstr &instr,
                       const MatrixDescriptor &descriptor, std::uint8_t rows, std::uint8_t cols,
                       std::uint8_t elem, bool transpose) {
  if (!trace.has_callback()) return;
  const auto bytes = static_cast<std::uint32_t>(rows) * cols * elem;
  const auto memory = memory_endpoint(bytes, elem);
  const auto reg = matrix_endpoint(instr.reg.domain, instr.reg.index, rows, cols, elem,
                                   is_store(instr.op) ? "src" : "dst");
  const auto mem = block_memory(instr, descriptor, rows, cols, elem, transpose);
  if (is_store(instr.op)) {
    emit_move(trace, mblk_op_name(instr), reg, memory, mem);
  } else {
    emit_move(trace, mblk_op_name(instr), memory, reg, mem);
  }
}

void trace_vector_move(MtraceRecorder &trace, const MblkInstr &instr,
                       const VectorDescriptor &descriptor, std::uint8_t len, std::uint8_t elem) {
  if (!trace.has_callback()) return;
  const auto bytes = static_cast<std::uint32_t>(len) * elem;
  const auto memory = memory_endpoint(bytes, elem);
  const auto reg = vector_endpoint(instr.reg.index, len, elem, is_store(instr.op) ? "src" : "dst");
  const auto mem = linear_memory(instr, descriptor.base_addr, len, descriptor.stride_bytes, elem,
                                 TraceDescriptorKind::vm);
  if (is_store(instr.op)) {
    emit_move(trace, mblk_op_name(instr), reg, memory, mem);
  } else {
    emit_move(trace, mblk_op_name(instr), memory, reg, mem);
  }
}

void trace_gather_tile(MtraceRecorder &trace, const MblkInstr &instr,
                       const MatrixDescriptor &descriptor, std::uint8_t indexed_rows,
                       std::uint8_t cols, std::uint8_t elem, const Vec32 &index) {
  if (!trace.has_callback()) return;
  const auto bytes = static_cast<std::uint32_t>(cols) * elem;
  for (std::uint8_t row = 0; row < indexed_rows; ++row) {
    const auto addr = add_signed(descriptor.base_addr, static_cast<std::int64_t>(index[row]) *
                                                           descriptor.row_stride_bytes);
    const auto memory = memory_endpoint(bytes, elem);
    const auto reg = matrix_endpoint(instr.reg.domain, instr.reg.index, 1, cols, elem, "dst");
    const auto mem = linear_memory(instr, addr, cols, descriptor.col_stride_bytes, elem,
                                   TraceDescriptorKind::tm);
    emit_move(trace, mblk_op_name(instr), memory, reg, mem);
  }
}

void trace_gather_vector(MtraceRecorder &trace, const MblkInstr &instr,
                         const VectorDescriptor &descriptor, std::uint8_t indexed,
                         std::uint8_t elem, const Vec32 &index) {
  if (!trace.has_callback()) return;
  for (std::uint8_t lane = 0; lane < indexed; ++lane) {
    const auto addr = add_signed(descriptor.base_addr,
                                 static_cast<std::int64_t>(index[lane]) * descriptor.stride_bytes);
    const auto memory = memory_endpoint(elem, elem);
    const auto reg = vector_endpoint(instr.reg.index, 1, elem, "dst");
    const auto mem = linear_memory(instr, addr, 1, 0, elem, TraceDescriptorKind::vm);
    emit_move(trace, mblk_op_name(instr), memory, reg, mem);
  }
}

void trace_npu_dataflow(const DecodedNpuInstr &instr, const State &state, MtraceRecorder &trace) {
  if (!trace.has_callback()) return;

  std::visit(
      Overloaded{
          [&](const FillInstr &i) {
            const bool from_scalar = std::holds_alternative<ScalarRegister>(i.value);
            emit_write(trace, trace::fill_mnemonic(i.dest.domain, from_scalar),
                       domain_endpoint(state, i.dest.domain, i.dest.index, "dst"));
          },
          [&](const MoveInstr &i) {
            emit_move(trace, trace::move_mnemonic(i.dest.domain),
                      domain_endpoint(state, i.src.domain, i.src.index, "src"),
                      domain_endpoint(state, i.dest.domain, i.dest.index, "dst"));
          },
          [&](const RowInstr &i) {
            const auto &mconfig = state.ac[i.matrix.index];
            const auto &vconfig = state.vc[i.vector.index];
            const auto elem = trace_elem_bytes(mconfig.dtype);
            const auto matrix =
                matrix_endpoint(NpuDomain::acc, i.matrix.index, 1, mconfig.cols, elem, "mat");
            const auto vector = vector_endpoint(i.vector.index, vconfig.len, elem, "vec");
            if (i.op == RowOp::insert) {
              emit_move(trace, "ainsert.row", vector, matrix);
            } else {
              emit_move(trace, "aextract.row", matrix, vector);
            }
          },
          [&](const LaneExtractInstr &i) {
            const auto &config = state.vc[i.src.index];
            emit_read(trace, "vextract",
                      vector_endpoint(i.src.index, 1, trace_elem_bytes(config.dtype), "src"));
          },
          [&](const LaneInsertInstr &i) {
            const auto &config = state.vc[i.dest.index];
            emit_write(trace, "vinsert",
                       vector_endpoint(i.dest.index, 1, trace_elem_bytes(config.dtype), "dst"));
          },
          [&](const LaneBroadcastInstr &i) {
            const auto &src_config = state.vc[i.src.index];
            const auto &dst_config = state.vc[i.dest.index];
            const auto elem = trace_elem_bytes(src_config.dtype);
            emit_move(trace, "vbroadcast", vector_endpoint(i.src.index, 1, elem, "src"),
                      vector_endpoint(i.dest.index, dst_config.len, elem, "dst"));
          },
          [&](const MmaInstr &i) {
            const auto &lhs = state.tc[i.src_a.index];
            const auto &rhs = state.tc[i.src_b.index];
            const auto &dst = state.ac[i.dest.index];
            TraceEvent event;
            add_source(event,
                       trace_matrix_endpoint(TraceEndpointKind::tile, i.src_a.index, lhs, "a"));
            add_source(event,
                       trace_matrix_endpoint(TraceEndpointKind::tile, i.src_b.index, rhs, "b"));
            if (i.op == MmaOp::nt_acc) {
              add_source(event, trace_matrix_endpoint(TraceEndpointKind::acc, i.dest.index, dst,
                                                      "acc_in"));
            }
            add_dest(event, trace_matrix_endpoint(TraceEndpointKind::acc, i.dest.index, dst, "d"));
            emit_compute(trace, trace::mma_mnemonic(i.op), event,
                         static_cast<std::uint32_t>(lhs.rows) * rhs.rows * lhs.cols);
          },
          [&](const VmmaInstr &i) {
            const auto &lhs = state.tc[i.src_a.index];
            const auto &rhs = state.tc[i.src_b.index];
            const auto &dst = state.vc[i.dest.index];
            TraceEvent event;
            add_source(event, matrix_endpoint(NpuDomain::tile, i.src_a.index, 1, lhs.cols, 1, "a"));
            add_source(event,
                       trace_matrix_endpoint(TraceEndpointKind::tile, i.src_b.index, rhs, "b"));
            if (i.op == VmmaOp::nt_acc) {
              add_source(event, trace_vector_endpoint(i.dest.index, dst, "acc_in"));
            }
            add_dest(event, trace_vector_endpoint(i.dest.index, dst, "d"));
            emit_compute(trace, trace::vmma_mnemonic(i.op), event,
                         static_cast<std::uint32_t>(rhs.rows) * rhs.cols);
          },
          [&](const EbinInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src1.domain, i.src1.index, "s0"));
            if (const auto *src2 = std::get_if<NpuRegister>(&i.src2)) {
              add_source(event, domain_endpoint(state, src2->domain, src2->index, "s1"));
            }
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            const bool from_scalar = std::holds_alternative<ScalarRegister>(i.src2);
            emit_compute(trace, trace::ebin_mnemonic(i.dest.domain, i.op, from_scalar), event);
          },
          [&](const EunInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::eun_mnemonic(i.dest.domain, i.op), event);
          },
          [&](const CmpInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src1.domain, i.src1.index, "s0"));
            if (const auto *src2 = std::get_if<NpuRegister>(&i.src2)) {
              add_source(event, domain_endpoint(state, src2->domain, src2->index, "s1"));
            }
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            const bool from_scalar = std::holds_alternative<ScalarRegister>(i.src2);
            emit_compute(trace, trace::cmp_mnemonic(i.op, from_scalar), event);
          },
          [&](const BcastInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src1.domain, i.src1.index, "s0"));
            add_source(event,
                       trace_vector_endpoint(i.vector.index, state.vc[i.vector.index], "vec"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::bcast_mnemonic(i.op, i.by_col), event);
          },
          [&](const Er4Instr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.arg.domain, i.arg.index, "arg"));
            add_source(event, domain_endpoint(state, i.src1.domain, i.src1.index, "s0"));
            add_source(event, domain_endpoint(state, i.src2.domain, i.src2.index, "s1"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::er4_mnemonic(i.dest.domain, i.op), event);
          },
          [&](const MaskInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            const bool from_scalar = std::holds_alternative<ScalarRegister>(i.fill);
            emit_compute(trace, trace::mask_mnemonic(i.dest.domain, i.op, from_scalar), event);
          },
          [&](const AreduceInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::areduce_mnemonic(i.op), event);
          },
          [&](const AreduceArgmaxInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.index_dest.domain, i.index_dest.index, "idx"));
            add_dest(event, domain_endpoint(state, i.value_dest.domain, i.value_dest.index, "val"));
            emit_compute(trace, trace::areduce_argmax_mnemonic(), event);
          },
          [&](const VreduceInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::vreduce_mnemonic(i.op), event);
          },
          [&](const VreduceArgmaxInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.index_dest.domain, i.index_dest.index, "idx"));
            add_dest(event, domain_endpoint(state, i.value_dest.domain, i.value_dest.index, "val"));
            emit_compute(trace, trace::vreduce_argmax_mnemonic(), event);
          },
          [&](const CvtInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::cvt_mnemonic(i.dest.domain, i.op), event);
          },
          [&](const AquantInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            add_source(event, trace_vector_endpoint(i.scale, state.vc[i.scale], "scale"));
            emit_compute(trace, trace::aquant_mnemonic(), event);
          },
          [&](const TdequantInstr &i) {
            TraceEvent event;
            add_source(event, domain_endpoint(state, i.src.domain, i.src.index, "s0"));
            add_source(event, trace_vector_endpoint(i.scale, state.vc[i.scale], "scale"));
            add_dest(event, domain_endpoint(state, i.dest.domain, i.dest.index, "d"));
            emit_compute(trace, trace::tdequant_mnemonic(), event);
          },
          [](const auto &) {},
      },
      instr);
}

} // namespace tile_npu
