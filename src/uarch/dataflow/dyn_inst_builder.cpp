#include "uarch/dataflow/dyn_inst_builder.h"

#include "uarch/frontend.h"
#include "utils/common.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace tile_npu {
namespace {

std::uint8_t dtype_bytes(DType dtype) { return dtype == DType::i8 || dtype == DType::u8 ? 1 : 4; }

void add_src(DynInst &inst, NpuDomain domain, RegisterIndex index) {
  if (inst.src_count >= inst.src.size()) unreachable("too many dynamic sources");
  inst.src[inst.src_count++] = RegisterRef{.domain = domain, .index = index};
}

void add_dst(DynInst &inst, NpuDomain domain, RegisterIndex index) {
  if (inst.dst_count >= inst.dst.size()) unreachable("too many dynamic destinations");
  inst.dst[inst.dst_count++] = RegisterRef{.domain = domain, .index = index};
}

void add_npu_src(DynInst &inst, const NpuRegister &reg) { add_src(inst, reg.domain, reg.index); }

void add_npu_dst(DynInst &inst, const NpuRegister &reg) { add_dst(inst, reg.domain, reg.index); }

void add_npu_operand(DynInst &inst, const NpuOperand &operand) {
  if (const auto *reg = std::get_if<NpuRegister>(&operand)) add_npu_src(inst, *reg);
}

void capture_gather_indices(DynInst &inst, const State &state, RegisterIndex index,
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

void fill_from_config(DynInst &inst, const State &state, NpuDomain domain, RegisterIndex index) {
  switch (domain) {
  case NpuDomain::tile: {
    const auto &config = state.tc[index];
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
    break;
  }
  case NpuDomain::acc: {
    const auto &config = state.ac[index];
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
    break;
  }
  case NpuDomain::vec32: {
    const auto &config = state.vc[index];
    inst.len = config.len;
    inst.rows = 1;
    inst.cols = config.len;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.bytes = inst.len * inst.elem_bytes;
    break;
  }
  case NpuDomain::reserved: unreachable("reserved dynamic domain"); break;
  }
}

void add_dest_and_shape(DynInst &inst, const State &state, const NpuRegister &dest) {
  add_npu_dst(inst, dest);
  fill_from_config(inst, state, dest.domain, dest.index);
}

void fill(DynInst &inst, const MmaInstr &instr, const State &state) {
  const auto &lhs = state.tc[instr.src_a.index];
  const auto &rhs = state.tc[instr.src_b.index];

  inst.op = DynOp::mma;
  inst.unit = Unit::mma;
  inst.rows = lhs.rows;
  inst.cols = rhs.rows;
  inst.k = lhs.cols;
  inst.macs = inst.rows * inst.cols * inst.k;
  inst.dtype = static_cast<std::uint8_t>(lhs.dtype);
  inst.elem_bytes = dtype_bytes(lhs.dtype);

  add_src(inst, NpuDomain::tile, instr.src_a.index);
  add_src(inst, NpuDomain::tile, instr.src_b.index);
  if (instr.op == MmaOp::nt_acc) add_src(inst, NpuDomain::acc, instr.dest.index);
  add_dst(inst, NpuDomain::acc, instr.dest.index);
}

void fill(DynInst &inst, const VmmaInstr &instr, const State &state) {
  const auto &lhs = state.tc[instr.src_a.index];
  const auto &rhs = state.tc[instr.src_b.index];

  inst.op = DynOp::vmma;
  inst.unit = Unit::mma;
  inst.rows = 1;
  inst.cols = rhs.rows;
  inst.k = lhs.cols;
  inst.macs = inst.cols * inst.k;
  inst.dtype = static_cast<std::uint8_t>(lhs.dtype);
  inst.elem_bytes = dtype_bytes(lhs.dtype);

  add_src(inst, NpuDomain::tile, instr.src_a.index);
  add_src(inst, NpuDomain::tile, instr.src_b.index);
  if (instr.op == VmmaOp::nt_acc) add_src(inst, NpuDomain::vec32, instr.dest.index);
  add_dst(inst, NpuDomain::vec32, instr.dest.index);
}

void fill(DynInst &inst, const MblkInstr &instr, const State &state) {
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
      add_src(inst, NpuDomain::vec32, instr.gather_src);
      add_dst(inst, NpuDomain::tile, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, NpuDomain::tile, instr.reg.index);
    } else {
      add_dst(inst, NpuDomain::tile, instr.reg.index);
    }
    break;
  }
  case NpuDomain::acc: {
    const auto &config = state.ac[instr.reg.index];
    const auto &descriptor = state.am[instr.reg.index];
    inst.rows = config.rows;
    inst.cols = config.cols;
    inst.dtype = static_cast<std::uint8_t>(config.dtype);
    inst.elem_bytes = dtype_bytes(config.dtype);
    inst.addr = descriptor.base_addr;
    inst.row_stride = descriptor.row_stride_bytes;
    inst.col_stride = descriptor.col_stride_bytes;
    inst.transpose = descriptor.transform != 0;
    if (instr.op == MblkOp::gather) {
      inst.rows = std::min<std::uint32_t>(config.rows, state.vc[instr.gather_src].len);
      capture_gather_indices(inst, state, instr.gather_src, inst.rows);
      add_src(inst, NpuDomain::vec32, instr.gather_src);
      add_dst(inst, NpuDomain::acc, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, NpuDomain::acc, instr.reg.index);
    } else {
      add_dst(inst, NpuDomain::acc, instr.reg.index);
    }
    break;
  }
  case NpuDomain::vec32: {
    const auto &config = state.vc[instr.reg.index];
    const auto &descriptor = state.vm[instr.reg.index];
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
      add_src(inst, NpuDomain::vec32, instr.gather_src);
      add_dst(inst, NpuDomain::vec32, instr.reg.index);
    } else if (instr.op == MblkOp::store) {
      add_src(inst, NpuDomain::vec32, instr.reg.index);
    } else {
      add_dst(inst, NpuDomain::vec32, instr.reg.index);
    }
    break;
  }
  case NpuDomain::reserved: unreachable("reserved M_BLK domain"); break;
  }

  inst.bytes = inst.rows * inst.cols * inst.elem_bytes;
}

void fill(DynInst &inst, const RowInstr &instr, const State &state) {
  const auto &matrix = state.ac[instr.matrix.index];
  const auto &vector = state.vc[instr.vector.index];

  inst.op = DynOp::row;
  inst.unit = Unit::vec32;
  inst.rows = 1;
  inst.cols = matrix.cols;
  inst.len = vector.len;
  inst.dtype = static_cast<std::uint8_t>(matrix.dtype);
  inst.elem_bytes = dtype_bytes(matrix.dtype);
  inst.bytes = inst.cols * inst.elem_bytes;

  if (instr.op == RowOp::insert) {
    add_src(inst, NpuDomain::vec32, instr.vector.index);
    add_dst(inst, NpuDomain::acc, instr.matrix.index);
  } else {
    add_src(inst, NpuDomain::acc, instr.matrix.index);
    add_dst(inst, NpuDomain::vec32, instr.vector.index);
  }
}

void fill(DynInst &inst, const FillInstr &instr, const State &state) {
  inst.op = DynOp::fill;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
}

void fill(DynInst &inst, const MoveInstr &instr, const State &state) {
  inst.op = DynOp::move;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const LaneExtractInstr &instr, const State &) {
  inst.op = DynOp::lane;
  inst.unit = Unit::vec32;
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const LaneInsertInstr &instr, const State &state) {
  inst.op = DynOp::lane;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
}

void fill(DynInst &inst, const LaneBroadcastInstr &instr, const State &state) {
  inst.op = DynOp::lane;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const EbinInstr &instr, const State &state) {
  inst.op = DynOp::elementwise;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src1);
  add_npu_operand(inst, instr.src2);
}

void fill(DynInst &inst, const EunInstr &instr, const State &state) {
  inst.op = DynOp::elementwise;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const CmpInstr &instr, const State &state) {
  inst.op = DynOp::compare;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src1);
  add_npu_operand(inst, instr.src2);
}

void fill(DynInst &inst, const BcastInstr &instr, const State &state) {
  inst.op = DynOp::broadcast;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src1);
  add_npu_src(inst, instr.vector);
}

void fill(DynInst &inst, const Er4Instr &instr, const State &state) {
  inst.op = DynOp::select;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.arg);
  add_npu_src(inst, instr.src1);
  add_npu_src(inst, instr.src2);
}

void fill(DynInst &inst, const MaskInstr &instr, const State &state) {
  inst.op = DynOp::mask;
  inst.unit = Unit::vec32;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const AreduceInstr &instr, const State &state) {
  inst.op = DynOp::reduce;
  inst.unit = Unit::reduce;
  inst.reduce_src_elements =
      static_cast<std::uint32_t>(state.ac[instr.src.index].rows) * state.ac[instr.src.index].cols;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const VreduceInstr &instr, const State &state) {
  inst.op = DynOp::reduce;
  inst.unit = Unit::reduce;
  inst.reduce_src_elements = state.vc[instr.src.index].len;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
}

void fill(DynInst &inst, const AreduceArgmaxInstr &instr, const State &state) {
  inst.op = DynOp::reduce;
  inst.unit = Unit::reduce;
  const auto &source = state.ac[instr.src.index];
  inst.reduce_src_elements = static_cast<std::uint32_t>(source.rows) * source.cols;
  add_npu_dst(inst, instr.index_dest);
  add_npu_dst(inst, instr.value_dest);
  add_npu_src(inst, instr.src);
  fill_from_config(inst, state, NpuDomain::acc, instr.src.index);
}

void fill(DynInst &inst, const VreduceArgmaxInstr &instr, const State &state) {
  inst.op = DynOp::reduce;
  inst.unit = Unit::reduce;
  inst.reduce_src_elements = state.vc[instr.src.index].len;
  add_npu_dst(inst, instr.index_dest);
  add_npu_dst(inst, instr.value_dest);
  add_npu_src(inst, instr.src);
  fill_from_config(inst, state, NpuDomain::vec32, instr.src.index);
}

void fill(DynInst &inst, const AquantInstr &instr, const State &state) {
  inst.op = DynOp::quant;
  inst.unit = Unit::quant;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
  add_src(inst, NpuDomain::vec32, instr.scale);
}

void fill(DynInst &inst, const TdequantInstr &instr, const State &state) {
  inst.op = DynOp::quant;
  inst.unit = Unit::quant;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
  add_src(inst, NpuDomain::vec32, instr.scale);
}

void fill(DynInst &inst, const AdequantInstr &instr, const State &state) {
  inst.op = DynOp::quant;
  inst.unit = Unit::quant;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
  add_src(inst, NpuDomain::vec32, instr.col_scale);
  add_src(inst, NpuDomain::vec32, instr.row_scale);
}

void fill(DynInst &inst, const VdequantInstr &instr, const State &state) {
  inst.op = DynOp::quant;
  inst.unit = Unit::quant;
  add_dest_and_shape(inst, state, instr.dest);
  add_npu_src(inst, instr.src);
  add_src(inst, NpuDomain::vec32, instr.col_scale);
  add_src(inst, NpuDomain::vec32, instr.row_scale);
}

} // namespace

DynInst make_npu_dyn_inst(std::uint64_t id, std::uint64_t pc, const DecodedNpuInstr &instr,
                          const State &state) {
  DynInst out{.id = id, .pc = pc};
  std::visit(
      [&](const auto &decoded) {
        using T = std::remove_cvref_t<decltype(decoded)>;
        if constexpr (is_frontend_npu_v<T>) {
          unreachable("frontend NPU instruction has no DynInst");
        } else {
          fill(out, decoded, state);
        }
      },
      instr);
  return out;
}

} // namespace tile_npu
