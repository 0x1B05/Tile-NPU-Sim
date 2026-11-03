#pragma once

#include "tile_npu/instruction.h"

#include <string>
#include <string_view>

namespace tile_npu::trace {

// Shared mnemonic components for the ITRACE disassembler and MTRACE events.
constexpr char domain_char(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return 't';
  case NpuDomain::acc: return 'a';
  case NpuDomain::vec32: return 'v';
  default: return '?';
  }
}

constexpr std::string_view mma_suffix(MmaOp op) { return op == MmaOp::nt_zero ? "zero" : "acc"; }

constexpr std::string_view vmma_suffix(VmmaOp op) { return op == VmmaOp::nt_zero ? "zero" : "acc"; }

constexpr std::string_view ebin_name(EbinOp op) {
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

constexpr std::string_view eun_name(EunOp op) {
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

constexpr std::string_view cmp_name(CmpOp op) {
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

constexpr std::string_view mask_name(MaskOp op) {
  switch (op) {
  case MaskOp::tail: return "tail";
  case MaskOp::tril: return "tril";
  case MaskOp::triu: return "triu";
  }
  return "?";
}

constexpr std::string_view reduce_name(ReduceOp op) {
  switch (op) {
  case ReduceOp::sum: return "sum";
  case ReduceOp::max: return "max";
  case ReduceOp::min: return "min";
  case ReduceOp::sumsq: return "sumsq";
  case ReduceOp::absmax: return "absmax";
  }
  return "?";
}

constexpr std::string_view er4_name(Er4Op op) { return op == Er4Op::select ? "select" : "fmadd"; }

inline std::string fill_mnemonic(NpuDomain domain, bool from_scalar) {
  std::string name(1, domain_char(domain));
  name += from_scalar ? "fillx" : "fill";
  return name;
}

inline std::string move_mnemonic(NpuDomain domain) {
  std::string name(1, domain_char(domain));
  name += "copy";
  return name;
}

inline std::string mma_mnemonic(MmaOp op) {
  std::string name = "mma.nt.";
  name += mma_suffix(op);
  return name;
}

inline std::string mma_dequant_mnemonic(MmaOp op) {
  std::string name = "mma.nt.dequant.";
  name += mma_suffix(op);
  return name;
}

inline std::string vmma_mnemonic(VmmaOp op) {
  std::string name = "vmma.nt.";
  name += vmma_suffix(op);
  return name;
}

inline std::string vmma_dequant_mnemonic(VmmaOp op) {
  std::string name = "vmma.nt.dequant.";
  name += vmma_suffix(op);
  return name;
}

inline std::string ebin_mnemonic(NpuDomain domain, EbinOp op, bool from_scalar) {
  std::string name(1, domain_char(domain));
  name += ebin_name(op);
  if (from_scalar) name += 'x';
  return name;
}

inline std::string eun_mnemonic(NpuDomain domain, EunOp op) {
  std::string name(1, domain_char(domain));
  name += eun_name(op);
  return name;
}

inline std::string cmp_mnemonic(CmpOp op, bool from_scalar) {
  std::string name = from_scalar ? "vcmpx." : "vcmp.";
  name += cmp_name(op);
  return name;
}

inline std::string bcast_mnemonic(EbinOp op, bool by_col) {
  std::string name = "a";
  name += ebin_name(op);
  name += "v.";
  name += by_col ? "bycol" : "byrow";
  return name;
}

inline std::string er4_mnemonic(NpuDomain domain, Er4Op op) {
  std::string name(1, domain_char(domain));
  name += er4_name(op);
  return name;
}

inline std::string mask_mnemonic(NpuDomain domain, MaskOp op, bool from_scalar) {
  std::string name(1, domain_char(domain));
  name += from_scalar ? "maskx." : "mask.";
  name += mask_name(op);
  return name;
}

inline std::string areduce_mnemonic(ReduceOp op) {
  std::string name = "areduce.rows.";
  name += reduce_name(op);
  return name;
}

inline std::string areduce_argmax_mnemonic() { return "areduce.rows.argmax"; }

inline std::string vreduce_mnemonic(ReduceOp op) {
  std::string name = "vreduce.";
  name += reduce_name(op);
  return name;
}

inline std::string vreduce_argmax_mnemonic() { return "vreduce.argmax"; }

inline std::string cvt_mnemonic(NpuDomain domain, CvtOp op) {
  if (op == CvtOp::widen) return "twiden.i8.i32";
  std::string name(1, domain_char(domain));
  name += "cvt.i32.f32";
  return name;
}

inline std::string aquant_mnemonic() { return "aquant.rows.q8s32"; }

inline std::string tdequant_mnemonic() { return "tdequant.rows.f32"; }

} // namespace tile_npu::trace
