#include "decode_npu.h"

#include "domain.h"
#include "encoding.h"

#include <cstdint>

namespace tile_npu {
namespace {

// QNT pairs a quantized 8-bit domain with its f32 counterpart: Tile with Acc
// (the matrix forms) and Vec8 with Vec32 (the vector forms).
constexpr NpuDomain qnt_paired_domain(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return NpuDomain::acc;
  case NpuDomain::acc: return NpuDomain::tile;
  case NpuDomain::vec8: return NpuDomain::vec32;
  case NpuDomain::vec32: return NpuDomain::vec8;
  default: return NpuDomain::tile;
  }
}

} // namespace

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
  if (f.funct3 > 0b101) {
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
// dom2: 00 tquant (Tile), 01 tdequant (Acc), 10 vquant (Vec8), 11 reserved.
// funct3: 000 quant/dequant, 001 external-scale tquant; 010..111 reserved.
// vScale5 is an output for dynamic quant, an input for external-scale quant/dequant.
// Vector forms use IS: 0 = immediate scale lane, 1 = Scalar register in index5.
// Matrix forms require IS = 0 and index5 = 0.
DecodeResult decode_qnt(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain = sel3_domain(f.sel3);
  const auto src_domain = qnt_paired_domain(domain);
  const bool vector_form = is_vector_domain(domain);
  const bool index_is_scalar = (f.sel3 & 1) != 0;

  // Tile has two forms: dynamic (funct3 = 000) and external scale (001).
  // Acc/Vec8/Vec32 each have exactly one form (funct3 = 000).
  const bool valid_funct3 = domain == NpuDomain::tile ? f.funct3 <= 0b001 : f.funct3 == 0b000;

  // Matrix forms reserve IS and index5; vector forms use them as the scale lane.
  const bool valid_index = vector_form || (!index_is_scalar && f.src2 == 0);

  if (!valid_funct3 || !valid_index) {
    return DecodeError::illegal;
  }

  const NpuRegister dest{domain, f.dest};
  const NpuRegister src{src_domain, f.src1};

  // Only vector forms consume the lane index; matrix forms force it to 0 above.
  const auto index = decode_index(f.src2, index_is_scalar);

  switch (domain) {
  case NpuDomain::tile:
    return DecodedNpuInstr{TquantInstr{
        .dest = dest,
        .src = src,
        .scale = f.arg,
        .scale_source = f.funct3 == 0b000 ? QntScaleSource::computed : QntScaleSource::provided,
    }};
  case NpuDomain::acc:
    return DecodedNpuInstr{TdequantInstr{
        .dest = dest,
        .src = src,
        .scale = f.arg,
    }};
  case NpuDomain::vec8:
    return DecodedNpuInstr{VquantInstr{
        .dest = dest,
        .src = src,
        .scale = f.arg,
        .index = index,
    }};
  default: return DecodeError::illegal;
  }
}

// MMA_BDOT (R4): major4 = 0000, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [   B5   ] [   A5   ] [   D5   ] [ q ]
// funct3 = {mma/bdot, nn/nt, zero/acc}; nt is bdot-only, so 010/011 are reserved.
DecodeResult decode_mma_bdot(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto make_mma = [&](MmaOp op) -> DecodeResult {
    return DecodedNpuInstr{MmaInstr{
        .op = op,
        .dest = {.domain = NpuDomain::acc, .index = f.dest},
        .src_a = {.domain = NpuDomain::tile, .index = f.src1},
        .src_b = {.domain = NpuDomain::tile, .index = f.src2},
    }};
  };
  const auto make_bdot = [&](BdotOp op) -> DecodeResult {
    return DecodedNpuInstr{BdotInstr{
        .op = op,
        .dest = {.domain = NpuDomain::vec32, .index = f.dest},
        .src_a = {.domain = NpuDomain::vec8, .index = f.src1},
        .src_b = {.domain = NpuDomain::tile, .index = f.src2},
    }};
  };

  switch (f.funct3) {
  case 0b000: return make_mma(MmaOp::nn_zero);
  case 0b001: return make_mma(MmaOp::nn_acc);
  case 0b100: return make_bdot(BdotOp::nn_zero);
  case 0b101: return make_bdot(BdotOp::nn_acc);
  case 0b110: return make_bdot(BdotOp::nt_zero);
  case 0b111: return make_bdot(BdotOp::nt_acc);
  default: return DecodeError::illegal;
  }
}

} // namespace tile_npu
