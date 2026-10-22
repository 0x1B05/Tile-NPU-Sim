#include "validate.h"

#include "config.h"
#include "domain.h"
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
  if (instr.dest.domain != NpuDomain::acc || instr.vector.domain != NpuDomain::vec32 ||
      !ebin_legal(instr.op, NpuDomain::acc)) {
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
  const auto &reg = instr.reg;
  if (!in_range(reg) || !index_in_range(instr.index)) {
    return false;
  }
  switch (instr.op) {
  case MblkOp::load:
  case MblkOp::store: return true;
  case MblkOp::row_load:
  case MblkOp::row_store: return reg.domain == NpuDomain::tile || reg.domain == NpuDomain::acc;
  case MblkOp::gather: return reg.domain != NpuDomain::acc && instr.gather_src < VECTOR_REGISTERS;
  default: return false;
  }
}

bool valid_structure(const FillInstr &instr) {
  return in_range(instr.dest) &&
         std::visit(Overloaded{[](FillImm16) { return true; },
                               [](ScalarRegister reg) { return in_range(reg); }},
                    instr.value);
}

bool valid_structure(const MoveInstr &instr) {
  return instr.dest.domain == instr.src.domain && in_range(instr.dest) && in_range(instr.src);
}

bool valid_structure(const RowInstr &instr) {
  switch (instr.op) {
  case RowOp::insert:
  case RowOp::extract:
    return is_domains_paired(instr.matrix.domain, instr.vector.domain) && in_range(instr.matrix) &&
           in_range(instr.vector) && index_in_range(instr.index);
  default: return false;
  }
}

bool valid_structure(const LaneExtractInstr &instr) {
  return in_range(instr.dest) && is_vector_domain(instr.src.domain) && in_range(instr.src) &&
         index_in_range(instr.index);
}

bool valid_structure(const LaneInsertInstr &instr) {
  return is_vector_domain(instr.dest.domain) && in_range(instr.dest) && in_range(instr.src) &&
         index_in_range(instr.index);
}

bool valid_structure(const LaneBroadcastInstr &instr) {
  return is_vector_domain(instr.dest.domain) && instr.dest.domain == instr.src.domain &&
         in_range(instr.dest) && in_range(instr.src) && index_in_range(instr.index);
}

bool valid_config_type(ConfigRegType reg_type) {
  return static_cast<std::uint8_t>(reg_type) < static_cast<std::uint8_t>(ConfigRegType::count);
}

bool valid_config_index(ConfigRegType reg_type, RegisterIndex index) {
  return valid_config_type(reg_type) && index < config_register_count(reg_type);
}

bool valid_config_range(ConfigRegType reg_type, RegisterIndex dest, std::uint8_t n) {
  return valid_config_type(reg_type) && std::size_t{dest} + n < config_register_count(reg_type);
}

bool valid_structure(const CfgSetiInstr &instr) {
  return valid_config_index(instr.reg_type, instr.dest) &&
         config_field_defined(instr.reg_type, instr.field);
}

bool valid_structure(const CfgSetxInstr &instr) {
  return valid_config_range(instr.reg_type, instr.dest, instr.n) &&
         in_range(ScalarRegister{instr.src}) && config_field_defined(instr.reg_type, instr.field);
}

bool valid_structure(const CfgCopyInstr &instr) {
  return valid_config_range(instr.reg_type, instr.dest, instr.n) &&
         valid_config_index(instr.reg_type, instr.src);
}

bool valid_structure(const CfgGetInstr &instr) {
  return in_range(ScalarRegister{instr.dest}) && valid_config_index(instr.reg_type, instr.src) &&
         config_field_defined(instr.reg_type, instr.field);
}

bool valid_structure(const CfgSetwInstr &instr) {
  return is_compute_config(instr.reg_type) &&
         valid_config_range(instr.reg_type, instr.dest, instr.n) &&
         in_range(ScalarRegister{instr.src});
}

bool valid_structure(const CfgGetwInstr &instr) {
  return is_compute_config(instr.reg_type) && in_range(ScalarRegister{instr.dest}) &&
         valid_config_index(instr.reg_type, instr.src);
}

bool valid_structure(const CvtInstr &instr) {
  if (!in_range(instr.dest) || !in_range(instr.src)) return false;
  switch (instr.op) {
  case CvtOp::cvt:
    return (instr.dest.domain == NpuDomain::acc || instr.dest.domain == NpuDomain::vec32) &&
           instr.src.domain == instr.dest.domain;
  case CvtOp::widen:
    return instr.dest.domain == NpuDomain::acc && instr.src.domain == NpuDomain::tile;
  default: return false;
  }
}

bool valid_structure(const AreduceInstr &instr) {
  return instr.op <= ReduceOp::absmax && instr.dest.domain == NpuDomain::vec32 &&
         in_range(instr.dest) && instr.src.domain == NpuDomain::acc && in_range(instr.src);
}

bool valid_structure(const VreduceInstr &instr) {
  return instr.op <= ReduceOp::sumsq && in_range(instr.dest) &&
         instr.src.domain == NpuDomain::vec32 && in_range(instr.src);
}

bool valid_structure(const VreduceArgmaxInstr &instr) {
  return in_range(instr.index_dest) && in_range(instr.value_dest) &&
         instr.src.domain == NpuDomain::vec32 && in_range(instr.src);
}

template <NpuDomain DestDomain, NpuDomain SrcDomain>
bool valid_matrix_quant(const NpuRegister &dest, const NpuRegister &src, RegisterIndex scale) {
  return dest.domain == DestDomain && src.domain == SrcDomain && scale < VECTOR_REGISTERS &&
         in_range(dest) && in_range(src);
}

template <NpuDomain DestDomain, NpuDomain SrcDomain>
bool valid_vector_quant(const NpuRegister &dest, const NpuRegister &src, RegisterIndex scale,
                        const IndexOperand &index) {
  return valid_matrix_quant<DestDomain, SrcDomain>(dest, src, scale) && index_in_range(index);
}

bool valid_structure(const TquantInstr &instr) {
  return instr.scale_source <= QntScaleSource::provided &&
         valid_matrix_quant<NpuDomain::tile, NpuDomain::acc>(instr.dest, instr.src, instr.scale);
}

bool valid_structure(const TdequantInstr &instr) {
  return valid_matrix_quant<NpuDomain::acc, NpuDomain::tile>(instr.dest, instr.src, instr.scale);
}

bool valid_structure(const VquantInstr &instr) {
  return valid_vector_quant<NpuDomain::vec8, NpuDomain::vec32>(instr.dest, instr.src, instr.scale,
                                                               instr.index);
}

bool valid_structure(const BdequantInstr &instr) {
  return valid_vector_quant<NpuDomain::vec32, NpuDomain::vec8>(instr.dest, instr.src, instr.scale,
                                                               instr.index);
}

bool valid_structure(const MmaInstr &instr) {
  return instr.op <= MmaOp::nn_acc && instr.dest.domain == NpuDomain::acc && in_range(instr.dest) &&
         instr.src_a.domain == NpuDomain::tile && in_range(instr.src_a) &&
         instr.src_b.domain == NpuDomain::tile && in_range(instr.src_b);
}

bool valid_structure(const BdotInstr &instr) {
  return instr.op <= BdotOp::nt_acc && instr.dest.domain == NpuDomain::vec32 &&
         in_range(instr.dest) && instr.src_a.domain == NpuDomain::vec8 && in_range(instr.src_a) &&
         instr.src_b.domain == NpuDomain::tile && in_range(instr.src_b);
}

bool valid_structure(const SysInstr &instr) { return instr.op <= SysOp::kernel_end; }

} // namespace

bool valid_structure(const DecodedNpuInstr &instr) {
  return std::visit([](const auto &decoded) { return valid_structure(decoded); }, instr);
}

} // namespace tile_npu
