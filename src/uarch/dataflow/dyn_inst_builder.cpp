#include "uarch/dataflow/dyn_inst_builder.h"

#include "domain.h"
#include "utils/common.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

Domain to_domain(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return Domain::tile;
  case NpuDomain::acc: return Domain::acc;
  case NpuDomain::vec32: return Domain::vec32;
  default: return Domain::scalar;
  }
}

std::uint8_t dtype_bytes(DType dtype) { return dtype == DType::i8 || dtype == DType::u8 ? 1 : 4; }

void add_src(DynInst &inst, Domain domain, std::uint8_t index) {
  if (inst.src_count >= inst.src.size()) return;
  inst.src[inst.src_count++] = DataRef{.domain = domain, .index = index};
}

void add_dst(DynInst &inst, Domain domain, std::uint8_t index) {
  if (inst.dst_count >= inst.dst.size()) return;
  inst.dst[inst.dst_count++] = DataRef{.domain = domain, .index = index};
}

void add_npu_src(DynInst &inst, const NpuRegister &reg) {
  add_src(inst, to_domain(reg.domain), reg.index);
}

void add_npu_dst(DynInst &inst, const NpuRegister &reg) {
  add_dst(inst, to_domain(reg.domain), reg.index);
}

// x0 is hard-wired: a read never waits and a write is discarded.
void add_scalar_src(DynInst &inst, RegisterIndex index) {
  if (index != 0) add_src(inst, Domain::scalar, index);
}

void add_scalar_dst(DynInst &inst, RegisterIndex index) {
  if (index != 0) add_dst(inst, Domain::scalar, index);
}

void add_index_src(DynInst &inst, const IndexOperand &index) {
  if (const auto *reg = std::get_if<ScalarRegister>(&index)) add_scalar_src(inst, reg->index);
}

void add_npu_operand(DynInst &inst, const NpuOperand &operand) {
  if (const auto *reg = std::get_if<NpuRegister>(&operand)) {
    add_npu_src(inst, *reg);
  } else if (const auto *reg = std::get_if<ScalarRegister>(&operand)) {
    add_scalar_src(inst, reg->index);
  }
}

void add_fill_src(DynInst &inst, const FillOperand &fill) {
  if (const auto *reg = std::get_if<ScalarRegister>(&fill)) add_scalar_src(inst, reg->index);
}

void add_fill_src(DynInst &inst, const std::variant<FillImm16, ScalarRegister> &value) {
  if (const auto *reg = std::get_if<ScalarRegister>(&value)) add_scalar_src(inst, reg->index);
}

void capture_gather_indices(DynInst &inst, const State &state, std::uint8_t index,
                            std::uint32_t count) {
  const auto &source = state.v[index];
  const auto &config = state.vc[index];
  GatherInfo gather;
  gather.count = static_cast<std::uint8_t>(std::min<std::uint32_t>(
      {count, static_cast<std::uint32_t>(config.len), static_cast<std::uint32_t>(VECTOR_LANES)}));
  for (std::uint8_t i = 0; i < gather.count; ++i) {
    gather.indices[i] = source[i];
  }
  inst.gather = gather;
}

void fill_mma(DynInst &inst, const MmaInstr &instr, const State &state) {
  const auto &lhs = state.tc[instr.src_a.index];
  const auto &rhs = state.tc[instr.src_b.index];
  const auto &dst = state.ac[instr.dest.index];

  inst.op = DynOp::mma;
  inst.unit = Unit::mma;
  inst.config.tile_a = lhs;
  inst.config.tile_b = rhs;
  inst.config.acc_d = dst;
  inst.rows = lhs.rows;
  inst.cols = rhs.rows;
  inst.k = lhs.cols;
  inst.macs = inst.rows * inst.cols * inst.k;
  inst.dtype = static_cast<std::uint8_t>(lhs.dtype);
  inst.elem_bytes = dtype_bytes(lhs.dtype);

  add_src(inst, Domain::tile, instr.src_a.index);
  add_src(inst, Domain::tile, instr.src_b.index);
  if (instr.op == MmaOp::nt_acc) add_src(inst, Domain::acc, instr.dest.index);
  add_dst(inst, Domain::acc, instr.dest.index);
}

void fill_vmma(DynInst &inst, const VmmaInstr &instr, const State &state) {
  const auto &lhs = state.tc[instr.src_a.index];
  const auto &rhs = state.tc[instr.src_b.index];
  const auto &dst = state.vc[instr.dest.index];

  inst.op = DynOp::vmma;
  // VMMA remains on the MMA queue in the current model. If the RTL routes it
  // through the Reduce/Dot backend instead, this is the single mapping to
  // change to Unit::reduce.
  inst.unit = Unit::mma;
  inst.config.tile_a = lhs;
  inst.config.tile_b = rhs;
  inst.config.vec_d = dst;
  inst.rows = 1;
  inst.cols = rhs.rows;
  inst.k = lhs.cols;
  inst.macs = inst.cols * inst.k;
  inst.dtype = static_cast<std::uint8_t>(lhs.dtype);
  inst.elem_bytes = dtype_bytes(lhs.dtype);

  add_src(inst, Domain::tile, instr.src_a.index);
  add_src(inst, Domain::tile, instr.src_b.index);
  if (instr.op == VmmaOp::nt_acc) add_src(inst, Domain::vec32, instr.dest.index);
  add_dst(inst, Domain::vec32, instr.dest.index);
}

void fill_mblk(DynInst &inst, const MblkInstr &instr, const State &state) {
  inst.unit = Unit::dma;
  inst.is_store = is_store(instr.op);
  switch (instr.op) {
  case MblkOp::load: inst.op = DynOp::load; break;
  case MblkOp::store: inst.op = DynOp::store; break;
  case MblkOp::gather: inst.op = DynOp::gather; break;
  }

  switch (instr.reg.domain) {
  case NpuDomain::tile: {
    const auto &config = state.tc[instr.reg.index];
    const auto &descriptor = state.tm[instr.reg.index];
    inst.config.tile_d = config;
    inst.config.matrix = descriptor;
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.addr = descriptor.base_addr;
    inst.row_stride = descriptor.row_stride_bytes;
    inst.col_stride = descriptor.col_stride_bytes;
    inst.transpose = descriptor.transform != 0;
    if (instr.op == MblkOp::gather) {
      // Rows beyond the index vector are zero-filled, not transferred.
      inst.rows = std::min<std::uint32_t>(config.rows, state.vc[instr.gather_src].len);
      capture_gather_indices(inst, state, instr.gather_src, inst.rows);
      add_src(inst, Domain::vec32, instr.gather_src);
      add_dst(inst, Domain::tile, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::tile, instr.reg.index);
    } else {
      add_dst(inst, Domain::tile, instr.reg.index);
    }
    break;
  }
  case NpuDomain::acc: {
    const auto &config = state.ac[instr.reg.index];
    const auto &descriptor = state.am[instr.reg.index];
    inst.config.acc_d = config;
    inst.config.matrix = descriptor;
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.addr = descriptor.base_addr;
    inst.row_stride = descriptor.row_stride_bytes;
    inst.col_stride = descriptor.col_stride_bytes;
    inst.transpose = descriptor.transform != 0;
    if (instr.op == MblkOp::gather) {
      add_src(inst, Domain::vec32, instr.gather_src);
      add_dst(inst, Domain::acc, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::acc, instr.reg.index);
    } else {
      add_dst(inst, Domain::acc, instr.reg.index);
    }
    break;
  }
  case NpuDomain::vec32: {
    const auto &config = state.vc[instr.reg.index];
    const auto &descriptor = state.vm[instr.reg.index];
    inst.config.vec_d = config;
    inst.config.vector = descriptor;
    inst.len = config.len;
    inst.rows = 1;
    inst.cols = config.len;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.addr = descriptor.base_addr;
    inst.col_stride = descriptor.stride_bytes;
    if (instr.op == MblkOp::gather) {
      // Lanes beyond the index vector are zero-filled, not transferred.
      const auto indexed = std::min<std::uint32_t>(config.len, state.vc[instr.gather_src].len);
      inst.len = indexed;
      inst.rows = indexed;
      inst.cols = 1;
      capture_gather_indices(inst, state, instr.gather_src, indexed);
      add_src(inst, Domain::vec32, instr.gather_src);
      add_dst(inst, Domain::vec32, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::vec32, instr.reg.index);
    } else {
      add_dst(inst, Domain::vec32, instr.reg.index);
    }
    break;
  }
  default: break;
  }

  inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
}

void fill_from_config(DynInst &inst, const State &state, Domain domain, std::uint8_t index) {
  switch (domain) {
  case Domain::tile: {
    const auto &config = state.tc[index];
    inst.config.tile_d = config;
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
    break;
  }
  case Domain::acc: {
    const auto &config = state.ac[index];
    inst.config.acc_d = config;
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
    break;
  }
  case Domain::vec32: {
    const auto &config = state.vc[index];
    inst.config.vec_d = config;
    inst.len = config.len;
    inst.rows = 1;
    inst.cols = config.len;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.len * inst.elem_bytes;
    break;
  }
  case Domain::scalar: break;
  }
}

void fill_row(DynInst &inst, const RowInstr &instr, const State &state) {
  const auto &matrix = state.ac[instr.matrix.index];
  const auto &vector = state.vc[instr.vector.index];
  inst.op = DynOp::row;
  inst.unit = Unit::vec32;
  inst.config.acc_d = matrix;
  inst.config.vec_d = vector;
  inst.rows = 1;
  inst.cols = matrix.cols;
  inst.len = vector.len;
  inst.dtype = static_cast<std::uint8_t>(matrix.dtype);
  inst.elem_bytes = dtype_bytes(matrix.dtype);
  inst.bytes = inst.cols * inst.elem_bytes;
  add_index_src(inst, instr.index);
  if (instr.op == RowOp::insert) {
    add_src(inst, Domain::vec32, instr.vector.index);
    add_dst(inst, Domain::acc, instr.matrix.index);
  } else {
    add_src(inst, Domain::acc, instr.matrix.index);
    add_dst(inst, Domain::vec32, instr.vector.index);
  }
}

void fill_sys(DynInst &inst, const SysInstr &instr) {
  inst.unit = Unit::sync;
  inst.kernel_end = instr.op == SysOp::kernel_end;
  if (inst.kernel_end) {
    inst.op = DynOp::kernel_end;
    inst.fence = FenceKind::all;
    return;
  }
  inst.op = DynOp::fence;
  switch (instr.op) {
  case SysOp::fence_mem: inst.fence = FenceKind::mem; break;
  case SysOp::fence_sa: inst.fence = FenceKind::sa; break;
  case SysOp::fence_all: inst.fence = FenceKind::all; break;
  case SysOp::kernel_end: break;
  }
}

} // namespace

DynInst make_scalar_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedScalarInstr &instr) {
  DynInst out{.id = id, .pc = pc, .op = DynOp::scalar, .unit = Unit::scalar};
  add_scalar_src(out, instr.regs.src1);
  add_scalar_src(out, instr.regs.src2);
  add_scalar_dst(out, instr.regs.dest);
  return out;
}

DynInst make_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedNpuInstr &instr,
                      const State &state) {
  DynInst out{.id = id, .pc = pc};

  std::visit(
      [&](const auto &decoded) {
        using T = std::remove_cvref_t<decltype(decoded)>;
        if constexpr (std::is_same_v<T, MmaInstr>) {
          fill_mma(out, decoded, state);
        } else if constexpr (std::is_same_v<T, VmmaInstr>) {
          fill_vmma(out, decoded, state);
        } else if constexpr (std::is_same_v<T, MblkInstr>) {
          fill_mblk(out, decoded, state);
        } else if constexpr (std::is_same_v<T, RowInstr>) {
          fill_row(out, decoded, state);
        } else if constexpr (std::is_same_v<T, SysInstr>) {
          fill_sys(out, decoded);
        } else if constexpr (std::is_same_v<T, CfgSetiInstr> || std::is_same_v<T, CfgSetxInstr> ||
                             std::is_same_v<T, CfgCopyInstr> || std::is_same_v<T, CfgGetInstr> ||
                             std::is_same_v<T, CfgSetwInstr> || std::is_same_v<T, CfgGetwInstr>) {
          out.op = DynOp::config;
          out.unit = Unit::config;
          // Config registers themselves are not scoreboard registers: every
          // consumer freezes the values it needs at dispatch. Only the scalar
          // operand crosses the register file.
          if constexpr (std::is_same_v<T, CfgSetxInstr> || std::is_same_v<T, CfgSetwInstr>) {
            add_scalar_src(out, decoded.src);
          } else if constexpr (std::is_same_v<T, CfgGetInstr> || std::is_same_v<T, CfgGetwInstr>) {
            add_scalar_dst(out, decoded.dest);
          }
        } else {
          // Vec32 is the only vector ALU in the current ISA. Fill/copy,
          // row/lane moves and broadcasts use its fast path; reductions and
          // quantisation have their own queues because they have distinct
          // execution resources.
          out.unit = Unit::vec32;
          if constexpr (std::is_same_v<T, FillInstr>) {
            out.op = DynOp::fill;
          } else if constexpr (std::is_same_v<T, MoveInstr>) {
            out.op = DynOp::move;
          } else if constexpr (std::is_same_v<T, LaneExtractInstr> ||
                               std::is_same_v<T, LaneInsertInstr> ||
                               std::is_same_v<T, LaneBroadcastInstr>) {
            out.op = DynOp::lane;
          } else if constexpr (std::is_same_v<T, EbinInstr> || std::is_same_v<T, EunInstr>) {
            out.op = DynOp::elementwise;
          } else if constexpr (std::is_same_v<T, CmpInstr>) {
            out.op = DynOp::compare;
          } else if constexpr (std::is_same_v<T, BcastInstr>) {
            out.op = DynOp::broadcast;
          } else if constexpr (std::is_same_v<T, Er4Instr>) {
            out.op = DynOp::select;
          } else if constexpr (std::is_same_v<T, MaskInstr>) {
            out.op = DynOp::mask;
          } else if constexpr (std::is_same_v<T, AreduceInstr> ||
                               std::is_same_v<T, AreduceArgmaxInstr> ||
                               std::is_same_v<T, VreduceInstr> ||
                               std::is_same_v<T, VreduceArgmaxInstr>) {
            out.op = DynOp::reduce;
            out.unit = Unit::reduce;
          } else if constexpr (std::is_same_v<T, AquantInstr> || std::is_same_v<T, TdequantInstr> ||
                               std::is_same_v<T, AdequantInstr> ||
                               std::is_same_v<T, VdequantInstr>) {
            out.op = DynOp::quant;
            out.unit = Unit::quant;
          } else {
            out.op = DynOp::other;
          }

          // Reductions process the source, whose shape is not the destination
          // shape: areduce walks rows*cols, vreduce walks the source length.
          if constexpr (std::is_same_v<T, AreduceInstr> || std::is_same_v<T, AreduceArgmaxInstr>) {
            const auto &src = state.ac[decoded.src.index];
            out.work_elements = static_cast<std::uint32_t>(src.rows) * src.cols;
          } else if constexpr (std::is_same_v<T, VreduceInstr> ||
                               std::is_same_v<T, VreduceArgmaxInstr>) {
            out.work_elements = state.vc[decoded.src.index].len;
          }

          if constexpr (requires { decoded.dest; }) {
            using Dest = std::remove_cvref_t<decltype(decoded.dest)>;
            if constexpr (std::is_same_v<Dest, NpuRegister>) {
              const auto domain = to_domain(decoded.dest.domain);
              add_dst(out, domain, decoded.dest.index);
              fill_from_config(out, state, domain, decoded.dest.index);
            } else if constexpr (std::is_same_v<Dest, ScalarRegister>) {
              add_scalar_dst(out, decoded.dest.index);
            }
          }
          if constexpr (requires { decoded.src; }) {
            using Src = std::remove_cvref_t<decltype(decoded.src)>;
            if constexpr (std::is_same_v<Src, NpuRegister>) {
              add_npu_src(out, decoded.src);
            } else if constexpr (std::is_same_v<Src, ScalarRegister>) {
              add_scalar_src(out, decoded.src.index);
            }
          }
          if constexpr (requires { decoded.src1; }) {
            add_npu_src(out, decoded.src1);
          }
          if constexpr (requires { decoded.src2; }) {
            using Src2 = std::remove_cvref_t<decltype(decoded.src2)>;
            if constexpr (std::is_same_v<Src2, NpuOperand>) {
              add_npu_operand(out, decoded.src2);
            } else if constexpr (std::is_same_v<Src2, NpuRegister>) {
              add_npu_src(out, decoded.src2);
            }
          }
          if constexpr (requires { decoded.vector; }) add_npu_src(out, decoded.vector);
          if constexpr (requires { decoded.arg; }) add_npu_src(out, decoded.arg);
          if constexpr (requires { decoded.index; }) add_index_src(out, decoded.index);
          if constexpr (requires { decoded.fill; }) add_fill_src(out, decoded.fill);
          if constexpr (requires { decoded.value; }) add_fill_src(out, decoded.value);
          if constexpr (requires { decoded.bound; }) add_scalar_src(out, decoded.bound);
          if constexpr (requires { decoded.col_scale; }) {
            add_src(out, Domain::vec32, decoded.col_scale);
          }
          if constexpr (requires { decoded.row_scale; }) {
            add_src(out, Domain::vec32, decoded.row_scale);
          }
          if constexpr (requires { decoded.scale; }) {
            add_src(out, Domain::vec32, decoded.scale);
          }
          if constexpr (requires { decoded.index_dest; }) {
            // Argmax has two architectural destinations; freeze the source
            // shape because there is no single `dest` config to read.
            add_npu_dst(out, decoded.index_dest);
            add_npu_dst(out, decoded.value_dest);
            const auto domain = to_domain(decoded.src.domain);
            fill_from_config(out, state, domain, decoded.src.index);
          }
        }
      },
      instr);

  return out;
}

DynInst make_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodeResult &decoded,
                      const State &state) {
  return std::visit(
      Overloaded{
          [&](const DecodedNpuInstr &npu) { return make_dyn_inst(id, pc, npu, state); },
          [&](const DecodedScalarInstr &scalar) { return make_scalar_dyn_inst(id, pc, scalar); },
          [&](DecodeError) {
            return DynInst{.id = id, .pc = pc, .op = DynOp::other, .unit = Unit::scalar};
          }},
      decoded);
}

} // namespace tile_npu
