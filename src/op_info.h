#pragma once

// Encoding-derived operation metadata shared by decode and the structural
// validation at the execute entry: per-domain register counts, op/domain
// legality and the per-family funct tables. The IR itself lives in
// tile_npu/instruction.h.
#include "tile_npu/instruction.h"
#include "tile_npu/state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {

inline constexpr std::size_t domain_register_count(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return TILE_REGISTERS;
  case NpuDomain::acc: return ACC_REGISTERS;
  case NpuDomain::vec8:
  case NpuDomain::vec32: return VECTOR_REGISTERS;
  }
  return 0;
}

inline constexpr bool is_vector_domain(NpuDomain domain) {
  return domain == NpuDomain::vec8 || domain == NpuDomain::vec32;
}

inline constexpr bool is_domains_paired(NpuDomain matrix, NpuDomain vector) {
  return (matrix == NpuDomain::tile && vector == NpuDomain::vec8) ||
         (matrix == NpuDomain::acc && vector == NpuDomain::vec32);
}

// Operation metadata for the E_BIN family:
// tile_allowed is false for mul/div (the tile domain has no 8-bit mul/div
// encodings); integer_allowed is false for div (f32-only); f32_allowed is
// false for the bitwise and shift operations (integer-only).
struct EbinInfo {
  EbinOp op;
  bool tile_allowed;
  bool integer_allowed;
  bool f32_allowed;
};

inline constexpr std::array<std::optional<EbinInfo>, 16> EBIN_BY_FUNCT4 = [] {
  std::array<std::optional<EbinInfo>, 16> table{};
  // clang-format off
  table[0b0000] = EbinInfo{EbinOp::add    , true , true , true};
  table[0b0001] = EbinInfo{EbinOp::sub    , true , true , true};
  table[0b0010] = EbinInfo{EbinOp::mul    , false, true , true};
  table[0b0011] = EbinInfo{EbinOp::div    , false, false, true};
  table[0b0100] = EbinInfo{EbinOp::min    , true , true , true};
  table[0b0101] = EbinInfo{EbinOp::max    , true , true , true};
  table[0b1000] = EbinInfo{EbinOp::bit_and, true , true , false};
  table[0b1001] = EbinInfo{EbinOp::bit_or , true , true , false};
  table[0b1010] = EbinInfo{EbinOp::bit_xor, true , true , false};
  table[0b1011] = EbinInfo{EbinOp::shl    , true , true , false};
  table[0b1100] = EbinInfo{EbinOp::shr    , true , true , false};
  table[0b1101] = EbinInfo{EbinOp::sra    , true , true , false};
  // clang-format on
  return table;
}();

// E_BIN/E_BCAST op legality on a destination domain: an assigned funct4 code,
// no Vec8 form, and no Tile mul/div.
inline constexpr bool ebin_legal(EbinOp op, NpuDomain domain) {
  const auto code = static_cast<std::uint8_t>(op);
  if (code >= EBIN_BY_FUNCT4.size() || !EBIN_BY_FUNCT4[code]) return false;
  return domain != NpuDomain::vec8 &&
         (domain != NpuDomain::tile || EBIN_BY_FUNCT4[code]->tile_allowed);
}

// Vec8 has no E_UNARY form; Tile only has `not`.
inline constexpr bool eun_legal(EunOp op, NpuDomain domain) {
  return op <= EunOp::silu && domain != NpuDomain::vec8 &&
         (domain != NpuDomain::tile || op == EunOp::bit_not);
}

// Condition metadata for the E_CMP family:
// ne/ge are integer-only and le/unord are f32-only; eq/lt accept both.
struct CmpInfo {
  CmpOp op;
  bool integer_allowed;
  bool f32_allowed;
};

// Indexed by funct3 code; unassigned codes (110/111) hold nullopt.
inline constexpr std::array<std::optional<CmpInfo>, 8> CMP_BY_FUNCT3 = [] {
  std::array<std::optional<CmpInfo>, 8> table{};
  // clang-format off
  table[0b000] = CmpInfo{CmpOp::eq   , true , true };
  table[0b001] = CmpInfo{CmpOp::ne   , true , false};
  table[0b010] = CmpInfo{CmpOp::lt   , true , true };
  table[0b011] = CmpInfo{CmpOp::ge   , true , false};
  table[0b100] = CmpInfo{CmpOp::le   , false, true };
  table[0b101] = CmpInfo{CmpOp::unord, false, true };
  // clang-format on
  return table;
}();

// Operand reading per op:
//   select: dest = D, arg = mask M (u8/u32 per domain), src1 = A, src2 = B
//   fmadd:  dest = D, arg = addend C, src1 = A, src2 = B
// Vec8 has no E_R4 form; fmadd excludes Tile.
inline constexpr bool er4_legal(Er4Op op, NpuDomain domain) {
  return op <= Er4Op::fmadd && domain != NpuDomain::vec8 &&
         (op != Er4Op::fmadd || domain != NpuDomain::tile);
}

// tril/triu are matrix-only.
inline constexpr bool mask_legal(MaskOp op, NpuDomain domain) {
  return op <= MaskOp::triu && (op == MaskOp::tail || !is_vector_domain(domain));
}

// The fill5 immediate table: codes 0-4 are the integer values 0/+1/-1/+2/-2,
// codes 5-8 add the f32-only ±0.5/±inf; codes 9-31 are reserved.
inline constexpr std::uint8_t FILL5_INT_MAX = 4;

inline constexpr std::uint8_t FILL5_MAX = 8;

// QNT pairs a quantized 8-bit domain with its f32 counterpart: Tile with Acc
// (the matrix forms) and Vec8 with Vec32 (the vector forms).
inline constexpr bool qnt_domain_is_quantized(NpuDomain domain) {
  return domain == NpuDomain::tile || domain == NpuDomain::vec8;
}

inline constexpr NpuDomain qnt_paired_domain(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return NpuDomain::acc;
  case NpuDomain::acc: return NpuDomain::tile;
  case NpuDomain::vec8: return NpuDomain::vec32;
  case NpuDomain::vec32: return NpuDomain::vec8;
  }
  return NpuDomain::tile;
}

// The destination domains an op can write: quant writes the quantized
// side, dequant the f32 side; the external-scale form is Tile-only.
inline constexpr bool qnt_dest_legal(QntOp op, NpuDomain dest) {
  switch (op) {
  case QntOp::quant: return qnt_domain_is_quantized(dest);
  case QntOp::dequant: return !qnt_domain_is_quantized(dest);
  case QntOp::quant_ext_scale: return dest == NpuDomain::tile;
  }
  return false;
}

// Operation metadata for the MMA_BDOT family:
// each entry carries the operation and its destination/left-source domains.
struct MmaInfo {
  MmaOp op;
  NpuDomain dest_domain;
  NpuDomain lhs_domain;
};

// Indexed by funct3 code; every code is assigned.
inline constexpr std::array<std::optional<MmaInfo>, 8> MMA_BY_FUNCT3 = [] {
  std::array<std::optional<MmaInfo>, 8> table{};
  // clang-format off
  table[0b000] = MmaInfo{MmaOp::nn_zero    , NpuDomain::acc  , NpuDomain::tile};
  table[0b001] = MmaInfo{MmaOp::nn_acc     , NpuDomain::acc  , NpuDomain::tile};
  table[0b010] = MmaInfo{MmaOp::nt_zero    , NpuDomain::acc  , NpuDomain::tile};
  table[0b011] = MmaInfo{MmaOp::nt_acc     , NpuDomain::acc  , NpuDomain::tile};
  table[0b100] = MmaInfo{MmaOp::nn_bdot    , NpuDomain::vec32, NpuDomain::vec8};
  table[0b101] = MmaInfo{MmaOp::nn_bdot_acc, NpuDomain::vec32, NpuDomain::vec8};
  table[0b110] = MmaInfo{MmaOp::nt_bdot    , NpuDomain::vec32, NpuDomain::vec8};
  table[0b111] = MmaInfo{MmaOp::nt_bdot_acc, NpuDomain::vec32, NpuDomain::vec8};
  // clang-format on
  return table;
}();

} // namespace tile_npu
