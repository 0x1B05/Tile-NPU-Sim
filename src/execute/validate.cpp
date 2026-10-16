#include "validate.h"

#include "config.h"
#include "op_info.h"
#include "tile_npu/state.h"
#include "utils/common.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {
namespace {

bool in_range(NpuRegister reg) { return reg.index < domain_register_count(reg.domain); }

bool in_range(ScalarRegister reg) { return reg.index < SCALAR_REGISTERS; }

// E_BIN/E_CMP src2: a same-domain register or a Scalar register.
bool src2_in_range(const NpuOperand &src2, NpuDomain domain) {
  return std::visit(
      Overloaded{[domain](NpuRegister reg) { return reg.domain == domain && in_range(reg); },
                 [](ScalarRegister reg) { return in_range(reg); }},
      src2);
}

// The register-numbered index form names a Scalar register; the immediate
// form carries no structure to check (row/lane bounds are dynamic).
bool index_in_range(const IndexOperand &index) {
  return std::visit(Overloaded{[](ImmediateIndex) { return true; },
                               [](ScalarRegister reg) { return in_range(reg); }},
                    index);
}

// The matrix forms reserve IS/index5: the encoding carries immediate 0.
bool index_imm_zero(const IndexOperand &index) {
  return std::visit(Overloaded{[](ImmediateIndex imm) { return imm.value == 0; },
                               [](ScalarRegister) { return false; }},
                    index);
}

// The register form names a Scalar register; the fill5 form is legal for
// codes 0-8 (codes 5-8 are f32-only, a dynamic dtype check).
bool fill_in_range(const FillOperand &fill) {
  return std::visit(Overloaded{[](Fill5Code code) { return code.code <= FILL5_MAX; },
                               [](ScalarRegister reg) { return in_range(reg); }},
                    fill);
}

template <typename Op, typename Info, std::size_t Size>
bool assigned_op(Op op, const std::array<std::optional<Info>, Size> &table) {
  const auto code = static_cast<std::uint8_t>(op);
  return code < Size && table[code].has_value();
}

bool valid_structure(const EbinInstr &instr) {
  if (!ebin_legal(instr.op, instr.dest.domain)) return false;
  return instr.src1.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.src1) &&
         src2_in_range(instr.src2, instr.dest.domain);
}

bool valid_structure(const EunInstr &instr) {
  if (!eun_legal(instr.op, instr.dest.domain)) return false;
  return instr.src.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.src);
}

bool valid_structure(const CmpInstr &instr) {
  if (!assigned_op(instr.op, CMP_BY_FUNCT3)) return false;
  if (instr.dest.domain != NpuDomain::acc && instr.dest.domain != NpuDomain::vec32) {
    return false;
  }
  return instr.src1.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.src1) &&
         src2_in_range(instr.src2, instr.dest.domain);
}

bool valid_structure(const BcastInstr &instr) {
  if (!is_domains_paired(instr.dest.domain, instr.vector.domain) ||
      !ebin_legal(instr.op, instr.dest.domain)) {
    return false;
  }
  return instr.src1.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.src1) &&
         in_range(instr.vector);
}

bool valid_structure(const Er4Instr &instr) {
  if (!er4_legal(instr.op, instr.dest.domain)) return false;
  return instr.arg.domain == instr.dest.domain && instr.src1.domain == instr.dest.domain &&
         instr.src2.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.arg) &&
         in_range(instr.src1) && in_range(instr.src2);
}

bool valid_structure(const MaskInstr &instr) {
  if (!mask_legal(instr.op, instr.dest.domain)) return false;
  return fill_in_range(instr.fill) && in_range(ScalarRegister{instr.bound}) &&
         instr.src.domain == instr.dest.domain && in_range(instr.dest) && in_range(instr.src);
}

bool valid_structure(const MblkInstr &instr) {
  if (instr.op > MblkOp::row_store) return false;
  const auto &reg = accessed_reg(instr);
  const bool row_form = instr.op == MblkOp::row_load || instr.op == MblkOp::row_store;
  if (row_form && reg.domain != NpuDomain::tile && reg.domain != NpuDomain::acc) {
    return false;
  }
  if (instr.op == MblkOp::gather) {
    if (reg.domain == NpuDomain::acc || instr.gather_src >= VECTOR_REGISTERS) {
      return false;
    }
  }
  return in_range(reg) && index_in_range(instr.index);
}

bool valid_structure(const FillInstr &instr) {
  return in_range(instr.dest) &&
         std::visit(Overloaded{[](FillImm16) { return true; },
                               [](ScalarRegister reg) { return in_range(reg); }},
                    instr.value);
}

bool valid_structure(const MoveInstr &instr) {
  if (instr.dest.domain != instr.src.domain || !in_range(instr.dest) || !in_range(instr.src)) {
    return false;
  }
  switch (instr.op) {
  case MoveOp::copy: return true;
  case MoveOp::transpose: return instr.dest.domain == NpuDomain::tile;
  }
  return false;
}

bool valid_structure(const RowInstr &instr) {
  if (instr.op > RowOp::extract) return false;
  return is_domains_paired(instr.matrix.domain, instr.vector.domain) && in_range(instr.matrix) &&
         in_range(instr.vector) && index_in_range(instr.index);
}

bool valid_structure(const LaneInstr &instr) {
  if (instr.op > LaneOp::broadcast) return false;
  const auto vector_operand = [](const NpuOperand &operand) {
    const auto *reg = std::get_if<NpuRegister>(&operand);
    return reg != nullptr && is_vector_domain(reg->domain) && in_range(*reg);
  };
  switch (instr.op) {
  case LaneOp::extract: {
    const auto *dst = std::get_if<ScalarRegister>(&instr.dst);
    return dst != nullptr && in_range(*dst) && vector_operand(instr.src);
  }
  case LaneOp::insert: {
    const auto *src = std::get_if<ScalarRegister>(&instr.src);
    return src != nullptr && in_range(*src) && vector_operand(instr.dst);
  }
  case LaneOp::broadcast: {
    const auto *dst = std::get_if<NpuRegister>(&instr.dst);
    const auto *src = std::get_if<NpuRegister>(&instr.src);
    return vector_operand(instr.dst) && vector_operand(instr.src) && dst->domain == src->domain;
  }
  }
  return false;
}

bool valid_structure(const CfgInstr &instr) {
  if (instr.op > CfgOp::getw || instr.reg_type > ConfigRegType::vm) {
    return false;
  }
  const bool word_op = instr.op == CfgOp::setw || instr.op == CfgOp::getw;
  if (word_op && !is_compute_config(instr.reg_type)) return false;
  // Only the writes setx/copy/setw carry a register count (N5).
  const bool counted =
      instr.op == CfgOp::setx || instr.op == CfgOp::copy || instr.op == CfgOp::setw;
  if (!counted && instr.count != 1) return false;
  if (instr.op == CfgOp::copy || word_op) {
    if (instr.field != 0) return false;
  } else if (!std::holds_alternative<ConfigField>(
                 config_field_from_code(instr.reg_type, instr.field))) {
    return false;
  }
  const auto count = config_register_count(instr.reg_type);
  const bool span_fits = instr.count >= 1 && std::size_t{instr.dest} + instr.count <= count;
  switch (instr.op) {
  case CfgOp::seti: return instr.dest < count;
  case CfgOp::setx:
  case CfgOp::setw: return span_fits && in_range(ScalarRegister{instr.src});
  case CfgOp::copy: return span_fits && instr.src < count;
  case CfgOp::get:
  case CfgOp::getw: return in_range(ScalarRegister{instr.dest}) && instr.src < count;
  }
  return false;
}

bool valid_structure(const CvtInstr &instr) {
  if (instr.op > CvtOp::widen) return false;
  if (instr.op == CvtOp::widen) {
    return instr.dest.domain == NpuDomain::acc && in_range(instr.dest) &&
           instr.src.domain == NpuDomain::tile && in_range(instr.src);
  }
  const bool cvt_domain =
      instr.dest.domain == NpuDomain::acc || instr.dest.domain == NpuDomain::vec32;
  return cvt_domain && instr.src.domain == instr.dest.domain && in_range(instr.dest) &&
         in_range(instr.src);
}

bool valid_structure(const ReduceInstr &instr) {
  if (instr.op > ReduceOp::absmax) return false;
  return std::visit(Overloaded{// areduce.rows.*: Vec32 destination, Acc source.
                               [&](NpuRegister dst) {
                                 return dst.domain == NpuDomain::vec32 && in_range(dst) &&
                                        instr.src.domain == NpuDomain::acc && in_range(instr.src);
                               },
                               // vreduce.*: Scalar destination, Vec32 source; absmax is
                               // AREDUCE-only.
                               [&](ScalarRegister x_dst) {
                                 if (instr.op == ReduceOp::absmax || !in_range(x_dst) ||
                                     instr.src.domain != NpuDomain::vec32 || !in_range(instr.src)) {
                                   return false;
                                 }
                                 if (instr.op == ReduceOp::argmax) {
                                   const auto *x_value = std::get_if<ScalarRegister>(&instr.dest2);
                                   return x_value != nullptr && in_range(*x_value);
                                 }
                                 return true;
                               }},
                    instr.dest);
}

bool valid_structure(const QntInstr &instr) {
  if (instr.op > QntOp::quant_ext_scale || instr.scale >= VECTOR_REGISTERS) {
    return false;
  }
  if (!qnt_dest_legal(instr.op, instr.dest.domain) ||
      instr.src.domain != qnt_paired_domain(instr.dest.domain)) {
    return false;
  }
  // The matrix forms reserve IS/index5: the encoding carries immediate 0.
  return in_range(instr.dest) && in_range(instr.src) &&
         (is_vector_domain(instr.dest.domain) ? index_in_range(instr.index)
                                              : index_imm_zero(instr.index));
}

bool valid_structure(const MmaInstr &instr) {
  if (!assigned_op(instr.op, MMA_BY_FUNCT3)) return false;
  const auto &info = *MMA_BY_FUNCT3[static_cast<std::uint8_t>(instr.op)];
  return instr.dest.domain == info.dest_domain && in_range(instr.dest) &&
         instr.src_a.domain == info.lhs_domain && in_range(instr.src_a) &&
         instr.src_b.domain == NpuDomain::tile && in_range(instr.src_b);
}

bool valid_structure(const SysInstr &instr) { return instr.op <= SysOp::kernel_end; }

} // namespace

bool valid_structure(const DecodedNpuInstr &instr) {
  return std::visit([](const auto &decoded) { return valid_structure(decoded); }, instr);
}

} // namespace tile_npu
