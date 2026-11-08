#include "dataflow/dyn_inst_builder.h"

#include "domain.h"

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

void set_dst(DynInst &inst, Domain domain, std::uint8_t index) {
  inst.dst = DataRef{.domain = domain, .index = index};
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
  set_dst(inst, Domain::acc, instr.dest.index);
}

void fill_vmma(DynInst &inst, const VmmaInstr &instr, const State &state) {
  const auto &lhs = state.tc[instr.src_a.index];
  const auto &rhs = state.tc[instr.src_b.index];
  const auto &dst = state.vc[instr.dest.index];

  inst.op = DynOp::vmma;
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
  set_dst(inst, Domain::vec32, instr.dest.index);
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
      const auto indexed = state.vc[instr.gather_src].len;
      inst.rows = indexed;
      add_src(inst, Domain::vec32, instr.gather_src);
      set_dst(inst, Domain::tile, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::tile, instr.reg.index);
    } else {
      set_dst(inst, Domain::tile, instr.reg.index);
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
      set_dst(inst, Domain::acc, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::acc, instr.reg.index);
    } else {
      set_dst(inst, Domain::acc, instr.reg.index);
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
      const auto indexed = state.vc[instr.gather_src].len;
      inst.len = indexed;
      inst.rows = indexed;
      inst.cols = 1;
      add_src(inst, Domain::vec32, instr.gather_src);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, Domain::vec32, instr.reg.index);
    } else {
      set_dst(inst, Domain::vec32, instr.reg.index);
    }
    break;
  }
  default: break;
  }

  if (inst.op == DynOp::gather) {
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
  } else {
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
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

DynInst make_scalar_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedScalarInstr &) {
  return DynInst{.id = id, .pc = pc, .op = DynOp::scalar, .unit = Unit::scalar};
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
        } else if constexpr (std::is_same_v<T, SysInstr>) {
          fill_sys(out, decoded);
        } else if constexpr (std::is_same_v<T, CfgSetiInstr> || std::is_same_v<T, CfgSetxInstr> ||
                             std::is_same_v<T, CfgCopyInstr> || std::is_same_v<T, CfgGetInstr> ||
                             std::is_same_v<T, CfgSetwInstr> || std::is_same_v<T, CfgGetwInstr>) {
          out.op = DynOp::config;
          out.unit = Unit::config;
        } else {
          out.unit = Unit::vector;
          if constexpr (std::is_same_v<T, FillInstr>) {
            out.op = DynOp::fill;
          } else if constexpr (std::is_same_v<T, MoveInstr>) {
            out.op = DynOp::move;
          } else if constexpr (std::is_same_v<T, RowInstr>) {
            out.op = DynOp::row;
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
          } else if constexpr (std::is_same_v<T, AquantInstr> || std::is_same_v<T, TdequantInstr> ||
                               std::is_same_v<T, AdequantInstr> ||
                               std::is_same_v<T, VdequantInstr>) {
            out.op = DynOp::quant;
          } else {
            out.op = DynOp::other;
          }

          if constexpr (requires { decoded.dest; }) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(decoded.dest)>,
                                         NpuRegister>) {
              set_dst(out, to_domain(decoded.dest.domain), decoded.dest.index);
            }
          }
          if constexpr (requires { decoded.src; }) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(decoded.src)>, NpuRegister>) {
              add_src(out, to_domain(decoded.src.domain), decoded.src.index);
            }
          }
          if constexpr (requires { decoded.src1; }) {
            if constexpr (std::is_same_v<std::remove_cvref_t<decltype(decoded.src1)>,
                                         NpuRegister>) {
              add_src(out, to_domain(decoded.src1.domain), decoded.src1.index);
            }
          }
        }
      },
      instr);

  return out;
}

} // namespace tile_npu
