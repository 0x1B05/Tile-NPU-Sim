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
// a defined domain, and no Tile mul.
inline constexpr bool ebin_legal(EbinOp op, NpuDomain domain) {
  const auto code = static_cast<std::uint8_t>(op);
  if (code >= EBIN_BY_FUNCT4.size() || !EBIN_BY_FUNCT4[code]) return false;
  return domain != NpuDomain::reserved &&
         (domain != NpuDomain::tile || EBIN_BY_FUNCT4[code]->tile_allowed);
}

// Tile only has `not`.
inline constexpr bool eun_legal(EunOp op, NpuDomain domain) {
  if (domain == NpuDomain::reserved) return false;
  switch (op) {
  case EunOp::bit_not: return true;
  case EunOp::abs:
  case EunOp::neg:
  case EunOp::exp2:
  case EunOp::rcp:
  case EunOp::rsqrt: return domain == NpuDomain::acc || domain == NpuDomain::vec32;
  default: return false;
  }
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

// fmadd excludes Tile.
inline constexpr bool er4_legal(Er4Op op, NpuDomain domain) {
  if (domain == NpuDomain::reserved) return false;
  switch (op) {
  case Er4Op::select: return true;
  case Er4Op::fmadd: return domain == NpuDomain::acc || domain == NpuDomain::vec32;
  default: return false;
  }
}

// tril/triu are matrix-only.
inline constexpr bool mask_legal(MaskOp op, NpuDomain domain) {
  if (domain == NpuDomain::reserved) return false;
  switch (op) {
  case MaskOp::tail: return true;
  case MaskOp::tril:
  case MaskOp::triu: return !is_vector_domain(domain);
  default: return false;
  }
}

// The fill5 immediate table: codes 0-4 are the integer values 0/+1/-1/+2/-2,
// codes 5-8 add the f32-only ±0.5/±inf; codes 9-31 are reserved.
inline constexpr std::uint8_t FILL5_INT_MAX = 4;

inline constexpr std::uint8_t FILL5_MAX = 8;

} // namespace tile_npu
