#include "decode_npu.h"

#include "domain.h"
#include "encoding.h"
#include "op_info.h"

#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

// The BX bit selects the second source: BX = 0 a same-domain register, BX = 1
// the Scalar register xS. The register form fails on an out-of-range number.
std::optional<NpuOperand> decode_bx_src2(std::uint8_t sel3, std::uint8_t src2, NpuDomain domain) {
  if ((sel3 & 1) != 0) {
    return NpuOperand{ScalarRegister{src2}};
  }
  if (src2 >= domain_register_count(domain)) {
    return std::nullopt;
  }
  return NpuOperand{NpuRegister{domain, src2}};
}

} // namespace

// E_BIN (R3): major4 = 0000, q = 00.
// 31    28 27  26  25  24    21 20       17 16      12 11       7 6        2 1   0
// [major4] [dom2] [BX] [funct4] [reserved4] [   B5   ] [   A5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 11 Vec32; 10 reserved.
// BX: 0 = same-domain B5, 1 = Scalar xS in B5.
// funct4 = {G, op3}, checked by EBIN_BY_FUNCT4; Tile mul/div are reserved.
DecodeResult decode_ebin(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<EbinOp>(f.funct4);
  if (!ebin_legal(op, domain)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  const auto src2 = decode_bx_src2(f.sel3, f.src2, domain);
  if (!src2) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{EbinInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src1 = {.domain = domain, .index = f.src1},
      .src2 = *src2,
  }};
}

// E_UNARY (R2): major4 = 0001, q = 00.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile (only the not operation), 01 Acc, 11 Vec32; 10 reserved.
// funct3: 000 not, 001 abs, 010 neg, 011 exp2.approx, 100 rcp.approx,
//         101 rsqrt.approx; 110/111 reserved.
DecodeResult decode_eun(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<EunOp>(f.funct3);
  const auto count = domain_register_count(domain);
  if (!eun_legal(op, domain) || f.dest >= count || f.src2 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{EunInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src = {.domain = domain, .index = f.src2},
  }};
}

// E_CMP (R4): major4 = 0010, q = 00.
// 31    28 27  26  25  24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [dom2] [BX] [funct3] [reserved5] [   B5   ] [   A5   ] [   M5   ] [ q ]
// dom2: 01 Acc, 11 Vec32; 00/10 reserved. M5 is the u32 mask destination.
// BX: 0 = same-domain B5, 1 = Scalar xS in B5.
// funct3: 000 eq, 001 ne, 010 lt, 011 ge, 100 le, 101 unord; 110/111 reserved.
DecodeResult decode_ecmp(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  if (!CMP_BY_FUNCT3[f.funct3]) {
    return DecodeError::illegal;
  }
  const auto op = static_cast<CmpOp>(f.funct3);
  const auto count = domain_register_count(domain);
  if (f.src1 >= count || f.dest >= count) {
    return DecodeError::illegal;
  }
  const auto src2 = decode_bx_src2(f.sel3, f.src2, domain);
  if (!src2) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CmpInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src1 = {.domain = domain, .index = f.src1},
      .src2 = *src2,
  }};
}

// E_BCAST (R3): major4 = 0101, q = 00.
// 31    28  27  26   25   24    21 20       17 16      12 11       7 6        2 1   0
// [major4] [0] [EW] [DIR] [funct4] [reserved4] [   S5   ] [   A5   ] [   D5   ] [ q ]
// EW: 0 = Tile/Vec8, 1 = Acc/Vec32; A5/D5 are matrices, S5 is a vector.
// DIR: 0 = byrow, 1 = bycol. funct4 = {G, op3}, shared with E_BIN.
DecodeResult decode_ebcast(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  const auto domains = paired_domains(f.sel3);
  const auto op = static_cast<EbinOp>(f.funct4);
  if (!ebin_legal(op, domains.matrix)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domains.matrix);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{BcastInstr{
      .op = op,
      .dest = {.domain = domains.matrix, .index = f.dest},
      .src1 = {.domain = domains.matrix, .index = f.src1},
      .vector = {.domain = domains.vector, .index = f.src2},
      .by_col = (f.sel3 & 1) != 0,
  }};
}

// E_R4 (R4): major4 = 0011, q = 00.
// 31    28 27  26 25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [ M5/C5 ] [   B5   ] [   A5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 11 Vec32; 10 reserved.
// funct3: 000 select (M5/C5 = mask M), 001 fmadd (M5/C5 = addend C);
//         010..111 reserved. fmadd requires Acc/Vec32 with f32 dtype.
DecodeResult decode_er4(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto op = static_cast<Er4Op>(f.funct3);
  if (!er4_legal(op, domain)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.arg >= count || f.src1 >= count || f.src2 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{Er4Instr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .arg = {.domain = domain, .index = f.arg},
      .src1 = {.domain = domain, .index = f.src1},
      .src2 = {.domain = domain, .index = f.src2},
  }};
}

// E_MASK (R4): major4 = 0100, q = 00.
// 31    28 27  26  25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [FS] [funct3] [xBound5 ] [ fill5  ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 Tile, 01 Acc, 10 Vec8, 11 Vec32.
// FS: 0 = fill5 code in [16:12], 1 = Scalar xFill in [16:12].
// funct3: 000 tail, 001 tril, 010 triu; 011..111 reserved. tril/triu are matrix-only.
// xBound5 names Scalar xBounds/xLen for tail, or xDelta for tril/triu.
DecodeResult decode_emask(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const bool fill_is_scalar = (f.sel3 & 1) != 0;
  const auto op = static_cast<MaskOp>(f.funct3);
  if (!mask_legal(op, domain) || (!fill_is_scalar && f.src2 > FILL5_MAX)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  const auto fill =
      fill_is_scalar ? FillOperand{ScalarRegister{f.src2}} : FillOperand{Fill5Code{f.src2}};
  return DecodedNpuInstr{MaskInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src = {.domain = domain, .index = f.src1},
      .bound = f.arg,
      .fill = fill,
  }};
}

} // namespace tile_npu
