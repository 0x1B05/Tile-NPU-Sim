#include "decode_npu.h"

#include "domain.h"
#include "encoding.h"
#include "op_info.h"

#include <cstdint>

namespace tile_npu {

// CVT (R2): major4 = 0011, q = 10.
// 31    28 27  26 25  24    22 21                12 11       7 6        2 1   0
// [major4] [dom2] [0] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// dom2: destination 01 Acc, 11 Vec32; 00/10 reserved.
// funct3: 000 i32 -> f32 (same-domain S5/D5), 001 Tile i8 -> Acc i32.
DecodeResult decode_cvt(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto op = static_cast<CvtOp>(f.funct3);
  const auto domain = sel3_domain(f.sel3);
  const auto src_domain = op == CvtOp::widen ? NpuDomain::tile : domain;
  if (f.dest >= domain_register_count(domain) || f.src2 >= domain_register_count(src_domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CvtInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src = {.domain = src_domain, .index = f.src2},
  }};
}

// AREDUCE (R2): major4 = 0001, q = 10.
// 31    28 27       25 24    22 21                12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// funct3: 000 sum, 001 max, 010 min, 011 sumsq, 100 argmax, 101 absmax;
//         110/111 reserved. S5 is Acc; D5 is Vec32.
DecodeResult decode_areduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  if (f.funct3 > 0b101 || f.src2 >= ACC_REGISTERS) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{AreduceInstr{
      .op = static_cast<ReduceOp>(f.funct3),
      .dest = {.domain = NpuDomain::vec32, .index = f.dest},
      .src = {.domain = NpuDomain::acc, .index = f.src2},
  }};
}

// VREDUCE sum/max/min/sumsq (R2): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21                12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// funct3: 000 sum, 001 max, 010 min, 011 sumsq. S5 is Vec32; D5 is Scalar.
// funct3 = 100 uses the argmax layout below; 101..111 are reserved.
DecodeResult decode_vreduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  return DecodedNpuInstr{VreduceInstr{
      .op = static_cast<ReduceOp>(f.funct3),
      .dest = {.index = f.dest},
      .src = {.domain = NpuDomain::vec32, .index = f.src2},
  }};
}

// VREDUCE argmax (R4): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [ xVal5  ] [   S5   ] [   D5   ] [ q ]
// funct3 = 100. S5 is Vec32; D5/xVal5 are the Scalar xIndex/xValue destinations.
DecodeResult decode_vreduce_argmax(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  return DecodedNpuInstr{VreduceArgmaxInstr{
      .index_dest = {.index = f.dest},
      .value_dest = {.index = f.src2},
      .src = {.domain = NpuDomain::vec32, .index = f.src1},
  }};
}

// QNT (R4): major4 = 0100, q = 10.
// 31    28 27  26  25  24    22 21      17 16      12 11       7 6        2 1   0
// [major4] [dom2] [IS] [funct3] [vScale5 ] [ index5 ] [   S5   ] [   D5   ] [ q ]
// dom2: 00 tquant (Tile), 01 tdequant (Acc), 10 vquant (Vec8), 11 bdequant (Vec32).
// funct3: 000 quant/dequant, 001 external-scale tquant; 010..111 reserved.
// vScale5 is an output for dynamic quant, an input for external-scale quant/dequant.
// Vector forms use IS: 0 = immediate scale lane, 1 = Scalar register in index5.
// Matrix forms require IS = 0 and index5 = 0.
DecodeResult decode_qnt(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const bool index_is_scalar = (f.sel3 & 1) != 0;
  const bool vector_form = is_vector_domain(domain);
  // funct3 = 000 quantizes into Tile/Vec8 destinations and dequantizes into
  // Acc/Vec32 destinations; the source is the paired domain.
  const auto op = f.funct3 == 0b001                 ? QntOp::quant_ext_scale
                  : qnt_domain_is_quantized(domain) ? QntOp::quant
                                                    : QntOp::dequant;
  if (f.funct3 > 0b001 || !qnt_dest_legal(op, domain) ||
      (!vector_form && (index_is_scalar || f.src2 != 0))) {
    return DecodeError::illegal;
  }
  const auto src_domain = qnt_paired_domain(domain);
  if (f.dest >= domain_register_count(domain) || f.src1 >= domain_register_count(src_domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{QntInstr{
      .op = op,
      .dest = {.domain = domain, .index = f.dest},
      .src = {.domain = src_domain, .index = f.src1},
      .scale = f.arg,
      .index = decode_index(f.src2, index_is_scalar),
  }};
}

// MMA_BDOT (R4): major4 = 0000, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [   B5   ] [   A5   ] [   D5   ] [ q ]
// funct3 = {mma/bdot, nn/nt, zero/acc}; nt is bdot-only, so 010/011 are reserved.
DecodeResult decode_mma(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto &info = MMA_BY_FUNCT3[f.funct3];
  if (!info) {
    return DecodeError::illegal;
  }
  const auto op = static_cast<MmaOp>(f.funct3);
  // src2 is always tile
  if (f.dest >= domain_register_count(info->dest_domain) ||
      f.src1 >= domain_register_count(info->lhs_domain) || f.src2 >= TILE_REGISTERS) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{MmaInstr{
      .op = op,
      .dest = {.domain = info->dest_domain, .index = f.dest},
      .src_a = {.domain = info->lhs_domain, .index = f.src1},
      .src_b = {.domain = NpuDomain::tile, .index = f.src2},
  }};
}

} // namespace tile_npu
