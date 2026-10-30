#include "disasm.h"

#include "utils/common.h"
#include "utils/format.h"

#include <array>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>

namespace tile_npu::trace {
namespace {

// RV64 ABI names are used for scalar instructions. NPU scalar operands use
// xN so that an Acc register named a0 cannot be confused with scalar a0.
constexpr std::array<std::string_view, 32> ABI_NAMES = {
    "zero", "ra", "sp", "gp", "tp",  "t0",  "t1", "t2", "s0", "s1", "a0",
    "a1",   "a2", "a3", "a4", "a5",  "a6",  "a7", "s2", "s3", "s4", "s5",
    "s6",   "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
};

std::string abi_reg(RegisterIndex index) {
  return index < ABI_NAMES.size() ? std::string(ABI_NAMES[index]) : "x?";
}

std::string x_reg(RegisterIndex index) { return "x" + std::to_string(index); }

char domain_char(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return 't';
  case NpuDomain::acc: return 'a';
  case NpuDomain::vec32: return 'v';
  default: return '?';
  }
}

std::string npu_reg(const NpuRegister &reg) {
  return std::string(1, domain_char(reg.domain)) + std::to_string(reg.index);
}

std::string index_operand(const IndexOperand &index) {
  return std::visit(Overloaded{
                        [](ImmediateIndex imm) { return std::to_string(imm.value); },
                        [](ScalarRegister reg) { return x_reg(reg.index); },
                    },
                    index);
}

std::string npu_operand(const NpuOperand &operand) {
  return std::visit(Overloaded{
                        [](NpuRegister reg) { return npu_reg(reg); },
                        [](ScalarRegister reg) { return x_reg(reg.index); },
                    },
                    operand);
}

std::string_view mma_suffix(MmaOp op) { return op == MmaOp::nt_zero ? "zero" : "acc"; }

std::string_view vmma_suffix(VmmaOp op) { return op == VmmaOp::nt_zero ? "zero" : "acc"; }

std::string_view ebin_name(EbinOp op) {
  switch (op) {
  case EbinOp::add: return "add";
  case EbinOp::sub: return "sub";
  case EbinOp::mul: return "mul";
  case EbinOp::min: return "min";
  case EbinOp::max: return "max";
  case EbinOp::bit_and: return "and";
  case EbinOp::bit_or: return "or";
  case EbinOp::bit_xor: return "xor";
  case EbinOp::shl: return "shl";
  case EbinOp::shr: return "shr";
  case EbinOp::sra: return "sra";
  }
  return "?";
}

std::string_view eun_name(EunOp op) {
  switch (op) {
  case EunOp::bit_not: return "not";
  case EunOp::abs: return "abs";
  case EunOp::neg: return "neg";
  case EunOp::exp2: return "exp2";
  case EunOp::rcp: return "rcp";
  case EunOp::rsqrt: return "rsqrt";
  }
  return "?";
}

std::string_view cmp_name(CmpOp op) {
  switch (op) {
  case CmpOp::eq: return "eq";
  case CmpOp::ne: return "ne";
  case CmpOp::lt: return "lt";
  case CmpOp::ge: return "ge";
  case CmpOp::le: return "le";
  case CmpOp::unord: return "unord";
  }
  return "?";
}

std::string_view mask_name(MaskOp op) {
  switch (op) {
  case MaskOp::tail: return "tail";
  case MaskOp::tril: return "tril";
  case MaskOp::triu: return "triu";
  }
  return "?";
}

std::string_view reduce_name(ReduceOp op) {
  switch (op) {
  case ReduceOp::sum: return "sum";
  case ReduceOp::max: return "max";
  case ReduceOp::min: return "min";
  case ReduceOp::sumsq: return "sumsq";
  case ReduceOp::absmax: return "absmax";
  }
  return "?";
}

std::string_view config_name(ConfigRegType type) {
  switch (type) {
  case ConfigRegType::tc: return "tc";
  case ConfigRegType::ac: return "ac";
  case ConfigRegType::vc: return "vc";
  case ConfigRegType::tm: return "tm";
  case ConfigRegType::am: return "am";
  case ConfigRegType::vm: return "vm";
  default: return "?";
  }
}

std::string_view field_name(ConfigField field) {
  switch (field) {
  case ConfigField::dtype: return "dtype";
  case ConfigField::rows: return "rows";
  case ConfigField::cols: return "cols";
  case ConfigField::layout: return "layout";
  case ConfigField::arith_mode: return "arith_mode";
  case ConfigField::len: return "len";
  case ConfigField::base_addr: return "base_addr";
  case ConfigField::row_stride_bytes: return "row_stride_bytes";
  case ConfigField::col_stride_bytes: return "col_stride_bytes";
  case ConfigField::transform: return "transform";
  case ConfigField::flags: return "flags";
  case ConfigField::stride_bytes: return "stride_bytes";
  }
  return "?";
}

std::string config_reg(ConfigRegType type, RegisterIndex index) {
  return std::string(config_name(type)) + std::to_string(index);
}

std::string fill_name(Fill5Code code) {
  constexpr std::array<std::string_view, 9> FILL_NAMES = {"0",   "1",    "-1",   "2",   "-2",
                                                          "0.5", "-0.5", "+inf", "-inf"};
  if (code.code < FILL_NAMES.size()) return std::string(FILL_NAMES[code.code]);
  return "fill5(" + std::to_string(code.code) + ")";
}

std::string fill_operand(const FillOperand &fill) {
  return std::visit(Overloaded{
                        [](Fill5Code code) { return fill_name(code); },
                        [](ScalarRegister reg) { return x_reg(reg.index); },
                    },
                    fill);
}

std::string scalar_instr(const DecodedScalarInstr &instr, std::uint64_t pc) {
  std::ostringstream out;
  const auto rd = abi_reg(instr.regs.dest);
  const auto rs1 = abi_reg(instr.regs.src1);
  const auto rs2 = abi_reg(instr.regs.src2);
  const auto imm = static_cast<std::int64_t>(instr.imm);

  switch (instr.op) {
  case ScalarOp::lui:
    out << "lui " << rd << ", " << hex32(static_cast<std::uint32_t>(instr.imm));
    break;
  case ScalarOp::auipc:
    out << "auipc " << rd << ", " << hex32(static_cast<std::uint32_t>(instr.imm));
    break;
  case ScalarOp::jal:
    out << "jal " << rd << ", " << hex64(pc + static_cast<std::uint64_t>(imm));
    break;
  case ScalarOp::jalr: out << "jalr " << rd << ", " << imm << "(" << rs1 << ")"; break;
  case ScalarOp::lb:
  case ScalarOp::lh:
  case ScalarOp::lw:
  case ScalarOp::ld:
  case ScalarOp::lbu:
  case ScalarOp::lhu:
  case ScalarOp::lwu:
    out << (instr.op == ScalarOp::lb    ? "lb"
            : instr.op == ScalarOp::lh  ? "lh"
            : instr.op == ScalarOp::lw  ? "lw"
            : instr.op == ScalarOp::ld  ? "ld"
            : instr.op == ScalarOp::lbu ? "lbu"
            : instr.op == ScalarOp::lhu ? "lhu"
                                        : "lwu")
        << " " << rd << ", " << imm << "(" << rs1 << ")";
    break;
  case ScalarOp::sb:
  case ScalarOp::sh:
  case ScalarOp::sw:
  case ScalarOp::sd:
    out << (instr.op == ScalarOp::sb   ? "sb"
            : instr.op == ScalarOp::sh ? "sh"
            : instr.op == ScalarOp::sw ? "sw"
                                       : "sd")
        << " " << rs2 << ", " << imm << "(" << rs1 << ")";
    break;
  case ScalarOp::beq:
  case ScalarOp::bne:
  case ScalarOp::blt:
  case ScalarOp::bge:
  case ScalarOp::bltu:
  case ScalarOp::bgeu:
    out << (instr.op == ScalarOp::beq    ? "beq"
            : instr.op == ScalarOp::bne  ? "bne"
            : instr.op == ScalarOp::blt  ? "blt"
            : instr.op == ScalarOp::bge  ? "bge"
            : instr.op == ScalarOp::bltu ? "bltu"
                                         : "bgeu")
        << " " << rs1 << ", " << rs2 << ", " << hex64(pc + static_cast<std::uint64_t>(imm));
    break;
  case ScalarOp::addi:
  case ScalarOp::slti:
  case ScalarOp::sltiu:
  case ScalarOp::xori:
  case ScalarOp::ori:
  case ScalarOp::andi:
    out << (instr.op == ScalarOp::addi    ? "addi"
            : instr.op == ScalarOp::slti  ? "slti"
            : instr.op == ScalarOp::sltiu ? "sltiu"
            : instr.op == ScalarOp::xori  ? "xori"
            : instr.op == ScalarOp::ori   ? "ori"
                                          : "andi")
        << " " << rd << ", " << rs1 << ", " << imm;
    break;
  case ScalarOp::slli:
  case ScalarOp::srli:
  case ScalarOp::srai:
  case ScalarOp::slliw:
  case ScalarOp::srliw:
  case ScalarOp::sraiw:
    out << (instr.op == ScalarOp::slli    ? "slli"
            : instr.op == ScalarOp::srli  ? "srli"
            : instr.op == ScalarOp::srai  ? "srai"
            : instr.op == ScalarOp::slliw ? "slliw"
            : instr.op == ScalarOp::srliw ? "srliw"
                                          : "sraiw")
        << " " << rd << ", " << rs1 << ", " << imm;
    break;
  case ScalarOp::addiw: out << "addiw " << rd << ", " << rs1 << ", " << imm; break;
  case ScalarOp::add: out << "add " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sub: out << "sub " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sll: out << "sll " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::slt: out << "slt " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sltu: out << "sltu " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::bit_xor: out << "xor " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::srl: out << "srl " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sra: out << "sra " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::bit_or: out << "or " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::bit_and: out << "and " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::addw: out << "addw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::subw: out << "subw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sllw: out << "sllw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::srlw: out << "srlw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::sraw: out << "sraw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::mul: out << "mul " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::mulh: out << "mulh " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::mulhsu: out << "mulhsu " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::mulhu: out << "mulhu " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::div: out << "div " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::divu: out << "divu " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::rem: out << "rem " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::remu: out << "remu " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::mulw: out << "mulw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::divw: out << "divw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::divuw: out << "divuw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::remw: out << "remw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::remuw: out << "remuw " << rd << ", " << rs1 << ", " << rs2; break;
  case ScalarOp::fence: out << "fence"; break;
  case ScalarOp::ecall: out << "ecall"; break;
  case ScalarOp::ebreak: out << "ebreak"; break;
  }
  return out.str();
}

std::string npu_instr(const DecodedNpuInstr &instr) {
  return std::visit(
      Overloaded{
          [](const MblkInstr &i) {
            std::ostringstream out;
            const auto d = domain_char(i.reg.domain);
            switch (i.op) {
            case MblkOp::load: out << d << "load " << npu_reg(i.reg); break;
            case MblkOp::store: out << d << "store " << npu_reg(i.reg); break;
            case MblkOp::gather:
              out << d << "load.gather " << npu_reg(i.reg) << ", v" << int(i.gather_src);
              break;
            case MblkOp::row_load:
              out << d << "load.row " << npu_reg(i.reg) << "[" << index_operand(i.index) << "]";
              break;
            case MblkOp::row_store:
              out << d << "store.row " << npu_reg(i.reg) << "[" << index_operand(i.index) << "]";
              break;
            }
            return out.str();
          },
          [](const FillInstr &i) {
            std::ostringstream out;
            const bool scalar = std::holds_alternative<ScalarRegister>(i.value);
            out << domain_char(i.dest.domain) << (scalar ? "fillx " : "fill ") << npu_reg(i.dest)
                << ", ";
            if (scalar) {
              out << x_reg(std::get<ScalarRegister>(i.value).index);
            } else {
              out << hex16(std::get<FillImm16>(i.value).bits);
            }
            return out.str();
          },
          [](const MoveInstr &i) {
            return std::string(1, domain_char(i.dest.domain)) + "copy " + npu_reg(i.dest) + ", " +
                   npu_reg(i.src);
          },
          [](const RowInstr &i) {
            std::ostringstream out;
            if (i.op == RowOp::insert) {
              out << "ainsert.row " << npu_reg(i.matrix) << "[" << index_operand(i.index) << "], "
                  << npu_reg(i.vector);
            } else {
              out << "aextract.row " << npu_reg(i.vector) << ", " << npu_reg(i.matrix) << "["
                  << index_operand(i.index) << "]";
            }
            return out.str();
          },
          [](const LaneExtractInstr &i) {
            return "vextract " + x_reg(i.dest.index) + ", " + npu_reg(i.src) + "[" +
                   index_operand(i.index) + "]";
          },
          [](const LaneInsertInstr &i) {
            return "vinsert " + npu_reg(i.dest) + "[" + index_operand(i.index) + "], " +
                   x_reg(i.src.index);
          },
          [](const LaneBroadcastInstr &i) {
            return "vbroadcast " + npu_reg(i.dest) + ", " + npu_reg(i.src) + "[" +
                   index_operand(i.index) + "]";
          },
          [](const MmaInstr &i) {
            return "mma.nt." + std::string(mma_suffix(i.op)) + ".i8.i32 " + npu_reg(i.dest) + ", " +
                   npu_reg(i.src_a) + ", " + npu_reg(i.src_b);
          },
          [](const VmmaInstr &i) {
            return "vmma.nt." + std::string(vmma_suffix(i.op)) + ".i8.i32 " + npu_reg(i.dest) +
                   ", " + npu_reg(i.src_a) + "[" + std::to_string(i.row) + "], " + npu_reg(i.src_b);
          },
          [](const EbinInstr &i) {
            std::ostringstream out;
            const bool scalar = std::holds_alternative<ScalarRegister>(i.src2);
            out << domain_char(i.dest.domain) << ebin_name(i.op) << (scalar ? "x " : " ")
                << npu_reg(i.dest) << ", " << npu_reg(i.src1) << ", " << npu_operand(i.src2);
            return out.str();
          },
          [](const EunInstr &i) {
            return std::string(1, domain_char(i.dest.domain)) + std::string(eun_name(i.op)) + " " +
                   npu_reg(i.dest) + ", " + npu_reg(i.src);
          },
          [](const CmpInstr &i) {
            std::ostringstream out;
            const bool scalar = std::holds_alternative<ScalarRegister>(i.src2);
            out << "vcmp" << (scalar ? "x" : "") << "." << cmp_name(i.op) << " " << npu_reg(i.dest)
                << ", " << npu_reg(i.src1) << ", " << npu_operand(i.src2);
            return out.str();
          },
          [](const BcastInstr &i) {
            std::ostringstream out;
            out << "a" << ebin_name(i.op) << "v." << (i.by_col ? "bycol" : "byrow") << " "
                << npu_reg(i.dest) << ", " << npu_reg(i.src1) << ", " << npu_reg(i.vector);
            return out.str();
          },
          [](const Er4Instr &i) {
            std::ostringstream out;
            out << domain_char(i.dest.domain) << (i.op == Er4Op::select ? "select " : "fmadd ")
                << npu_reg(i.dest) << ", " << npu_reg(i.arg) << ", " << npu_reg(i.src1) << ", "
                << npu_reg(i.src2);
            return out.str();
          },
          [](const MaskInstr &i) {
            std::ostringstream out;
            const bool scalar = std::holds_alternative<ScalarRegister>(i.fill);
            out << domain_char(i.dest.domain) << (scalar ? "maskx." : "mask.") << mask_name(i.op)
                << " " << npu_reg(i.dest) << ", " << npu_reg(i.src) << ", " << x_reg(i.bound)
                << ", " << fill_operand(i.fill);
            return out.str();
          },
          [](const AreduceInstr &i) {
            return "areduce.rows." + std::string(reduce_name(i.op)) + " " + npu_reg(i.dest) + ", " +
                   npu_reg(i.src);
          },
          [](const AreduceArgmaxInstr &i) {
            return "areduce.rows.argmax " + npu_reg(i.index_dest) + ", " + npu_reg(i.value_dest) +
                   ", " + npu_reg(i.src);
          },
          [](const VreduceInstr &i) {
            return "vreduce." + std::string(reduce_name(i.op)) + " " + npu_reg(i.dest) + ", " +
                   npu_reg(i.src);
          },
          [](const VreduceArgmaxInstr &i) {
            return "vreduce.argmax " + npu_reg(i.index_dest) + ", " + npu_reg(i.value_dest) + ", " +
                   npu_reg(i.src);
          },
          [](const CvtInstr &i) {
            if (i.op == CvtOp::widen) {
              return "twiden.i8.i32 " + npu_reg(i.dest) + ", " + npu_reg(i.src);
            }
            return std::string(1, domain_char(i.dest.domain)) + "cvt.i32.f32 " + npu_reg(i.dest) +
                   ", " + npu_reg(i.src);
          },
          [](const AquantInstr &i) {
            std::ostringstream out;
            out << "aquant.rows" << (i.scale_source == QntScaleSource::provided ? ".scale" : "")
                << ".q8s32 " << npu_reg(i.dest) << ", ";
            if (i.scale_source == QntScaleSource::provided) {
              out << npu_reg(i.src) << ", v" << int(i.scale);
            } else {
              out << "v" << int(i.scale) << ", " << npu_reg(i.src);
            }
            return out.str();
          },
          [](const TdequantInstr &i) {
            return "tdequant.rows.f32 " + npu_reg(i.dest) + ", " + npu_reg(i.src) + ", v" +
                   std::to_string(i.scale);
          },
          [](const CfgSetiInstr &i) {
            return "cfg.seti " + config_reg(i.reg_type, i.dest) + ", " +
                   std::string(field_name(i.field)) + ", " + std::to_string(i.imm);
          },
          [](const CfgSetxInstr &i) {
            std::ostringstream out;
            out << "cfg.setx " << config_reg(i.reg_type, i.dest) << ", " << field_name(i.field)
                << ", " << x_reg(i.src);
            if (i.n != 0) out << ", " << int(i.n);
            return out.str();
          },
          [](const CfgCopyInstr &i) {
            std::ostringstream out;
            out << "cfg.copy " << config_reg(i.reg_type, i.dest) << ", "
                << config_reg(i.reg_type, i.src);
            if (i.n != 0) out << ", " << int(i.n);
            return out.str();
          },
          [](const CfgGetInstr &i) {
            return "cfg.get " + x_reg(i.dest) + ", " + config_reg(i.reg_type, i.src) + ", " +
                   std::string(field_name(i.field));
          },
          [](const CfgSetwInstr &i) {
            std::ostringstream out;
            out << "cfg.setw " << config_reg(i.reg_type, i.dest) << ", " << x_reg(i.src);
            if (i.n != 0) out << ", " << int(i.n);
            return out.str();
          },
          [](const CfgGetwInstr &i) {
            return "cfg.getw " + x_reg(i.dest) + ", " + config_reg(i.reg_type, i.src);
          },
          [](const SysInstr &i) {
            switch (i.op) {
            case SysOp::fence_mem: return std::string("fence.mem");
            case SysOp::fence_sa: return std::string("fence.sa");
            case SysOp::fence_all: return std::string("fence.all");
            case SysOp::kernel_end: return std::string("kernel.end");
            }
            return std::string("sys?");
          },
      },
      instr);
}

} // namespace

std::string disassemble(std::uint64_t pc, const DecodeResult &decoded) {
  return std::visit(Overloaded{
                        [&](const DecodedScalarInstr &instr) { return scalar_instr(instr, pc); },
                        [&](const DecodedNpuInstr &instr) { return npu_instr(instr); },
                        [](DecodeError) { return std::string("illegal"); },
                    },
                    decoded);
}

} // namespace tile_npu::trace
