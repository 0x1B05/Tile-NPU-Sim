#pragma once

// Encoding-derived operation metadata shared by decode and the structural
// validation at the execute entry: op/domain legality and the per-family
// funct tables.
#include "domain.h"
#include "tile_npu/instruction.h"

#include <array>
#include <cstdint>
#include <optional>

namespace tile_npu {

struct EbinInfo {
  bool tile_allowed;
  bool integer_allowed;
  bool f32_allowed;
};

inline constexpr std::array<std::optional<EbinInfo>, 16> EBIN_BY_FUNCT4 = [] {
  std::array<std::optional<EbinInfo>, 16> table{};
  // clang-format off
  //                       tile   int    f32
  table[0b0000] = EbinInfo{true , true , true }; // add
  table[0b0001] = EbinInfo{true , true , true }; // sub
  table[0b0010] = EbinInfo{false, true , true }; // mul
  table[0b0011] = EbinInfo{false, false, true }; // div
  table[0b0100] = EbinInfo{true , true , true }; // min
  table[0b0101] = EbinInfo{true , true , true }; // max
  table[0b1000] = EbinInfo{true , true , false}; // bit_and
  table[0b1001] = EbinInfo{true , true , false}; // bit_or
  table[0b1010] = EbinInfo{true , true , false}; // bit_xor
  table[0b1011] = EbinInfo{true , true , false}; // shl
  table[0b1100] = EbinInfo{true , true , false}; // shr
  table[0b1101] = EbinInfo{true , true , false}; // sra
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

struct CmpInfo {
  bool integer_allowed;
  bool f32_allowed;
};

// Indexed by funct3 code; unassigned codes (110/111) hold nullopt.
inline constexpr std::array<std::optional<CmpInfo>, 8> CMP_BY_FUNCT3 = [] {
  std::array<std::optional<CmpInfo>, 8> table{};
  // clang-format off
  //                     int    f32
  table[0b000] = CmpInfo{true , true }; // eq
  table[0b001] = CmpInfo{true , false}; // ne
  table[0b010] = CmpInfo{true , true }; // lt
  table[0b011] = CmpInfo{true , false}; // ge
  table[0b100] = CmpInfo{false, true }; // le
  table[0b101] = CmpInfo{false, true }; // unord
  // clang-format on
  return table;
}();

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

// Destination and left-source domains for each MMA_BDOT operation.
struct MmaInfo {
  NpuDomain dest_domain;
  NpuDomain lhs_domain;
};

// Indexed by funct3 code; 010/011 (the dropped mma.nt) are reserved.
inline constexpr std::array<std::optional<MmaInfo>, 8> MMA_BY_FUNCT3 = [] {
  std::array<std::optional<MmaInfo>, 8> table{};
  // clang-format off
  //                     dest              lhs
  table[0b000] = MmaInfo{NpuDomain::acc  , NpuDomain::tile}; // nn_zero
  table[0b001] = MmaInfo{NpuDomain::acc  , NpuDomain::tile}; // nn_acc
  table[0b100] = MmaInfo{NpuDomain::vec32, NpuDomain::vec8}; // nn_bdot
  table[0b101] = MmaInfo{NpuDomain::vec32, NpuDomain::vec8}; // nn_bdot_acc
  table[0b110] = MmaInfo{NpuDomain::vec32, NpuDomain::vec8}; // nt_bdot
  table[0b111] = MmaInfo{NpuDomain::vec32, NpuDomain::vec8}; // nt_bdot_acc
  // clang-format on
  return table;
}();

} // namespace tile_npu
