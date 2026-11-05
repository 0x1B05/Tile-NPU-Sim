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
  if (!domain_register_valid(domain, f.dest) || !domain_register_valid(src_domain, f.src2)) {
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
// funct3: 000 sum, 001 max, 010 min, 011 sumsq, 101 absmax; 100 uses the
// R4 argmax form below; 110/111 reserved. S5 is Acc; D5 is Vec32.
DecodeResult decode_areduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  if (!is_areduce_op(static_cast<ReduceOp>(f.funct3)) ||
      !domain_register_valid(NpuDomain::acc, f.src2) ||
      !domain_register_valid(NpuDomain::vec32, f.dest)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{AreduceInstr{
      .op = static_cast<ReduceOp>(f.funct3),
      .dest = {.domain = NpuDomain::vec32, .index = f.dest},
      .src = {.domain = NpuDomain::acc, .index = f.src2},
  }};
}

// AREDUCE argmax (R4): major4 = 0001, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [ vVal5  ] [   S5   ] [   D5   ] [ q ]
// funct3 = 100. S5 is the Acc source; D5/vVal5 are the Vec32 vIdx/vVal
// destinations.
DecodeResult decode_areduce_argmax(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  if (f.funct3 != 0b100 || !domain_register_valid(NpuDomain::acc, f.src1) ||
      !domain_register_valid(NpuDomain::vec32, f.dest) ||
      !domain_register_valid(NpuDomain::vec32, f.src2)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{AreduceArgmaxInstr{
      .index_dest = {.domain = NpuDomain::vec32, .index = f.dest},
      .value_dest = {.domain = NpuDomain::vec32, .index = f.src2},
      .src = {.domain = NpuDomain::acc, .index = f.src1},
  }};
}

// VREDUCE sum/max/min/sumsq (R2): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21                12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [    reserved10    ] [   S5   ] [   D5   ] [ q ]
// funct3: 000 sum, 001 max, 010 min, 011 sumsq. S5 is Vec32; D5 is Vec32.
// funct3 = 100 uses the argmax layout below; 101..111 are reserved.
DecodeResult decode_vreduce(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  if (!is_vreduce_op(static_cast<ReduceOp>(f.funct3))) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{VreduceInstr{
      .op = static_cast<ReduceOp>(f.funct3),
      .dest = {.domain = NpuDomain::vec32, .index = f.dest},
      .src = {.domain = NpuDomain::vec32, .index = f.src2},
  }};
}

// VREDUCE argmax (R4): major4 = 0010, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [reserved5] [ vVal5  ] [   S5   ] [   D5   ] [ q ]
// funct3 = 100. S5 is the Vec32 source; D5/vVal5 are the Vec32 vIdx/vVal
// destinations.
DecodeResult decode_vreduce_argmax(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  return DecodedNpuInstr{VreduceArgmaxInstr{
      .index_dest = {.domain = NpuDomain::vec32, .index = f.dest},
      .value_dest = {.domain = NpuDomain::vec32, .index = f.src2},
      .src = {.domain = NpuDomain::vec32, .index = f.src1},
  }};
}

// QNT (R4): major4 = 0100, q = 10.
// [31:28] major4 [27:25] qf3 [24:22] funct3 (FORM/QD)
// [21:17] vCol5 [16:12] vRow5 [11:7] S5 [6:2] D5 [1:0] q
DecodeResult decode_qnt(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  if (f.sel3 != 0) {
    return DecodeError::illegal;
  }
  const auto form = static_cast<std::uint8_t>((f.funct3 >> 1) & 0b011);
  const auto qd = static_cast<std::uint8_t>(f.funct3 & 0b001);
  const auto col = f.arg;
  const auto row = f.src2;

  if (qd == 0) {
    if (form != 0 || col != 0 || !domain_register_valid(NpuDomain::tile, f.dest) ||
        !domain_register_valid(NpuDomain::acc, f.src1) ||
        !domain_register_valid(NpuDomain::vec32, row)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{AquantInstr{
        .dest = {.domain = NpuDomain::tile, .index = f.dest},
        .src = {.domain = NpuDomain::acc, .index = f.src1},
        .scale = row,
    }};
  }

  switch (form) {
  case 0:
    if (col != 0 || !domain_register_valid(NpuDomain::acc, f.dest) ||
        !domain_register_valid(NpuDomain::tile, f.src1) ||
        !domain_register_valid(NpuDomain::vec32, row)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{TdequantInstr{
        .dest = {.domain = NpuDomain::acc, .index = f.dest},
        .src = {.domain = NpuDomain::tile, .index = f.src1},
        .scale = row,
    }};
  case 1:
    if (!domain_register_valid(NpuDomain::acc, f.dest) ||
        !domain_register_valid(NpuDomain::acc, f.src1) ||
        !domain_register_valid(NpuDomain::vec32, col) ||
        !domain_register_valid(NpuDomain::vec32, row)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{AdequantInstr{
        .dest = {.domain = NpuDomain::acc, .index = f.dest},
        .src = {.domain = NpuDomain::acc, .index = f.src1},
        .col_scale = col,
        .row_scale = row,
    }};
  case 2:
    if (!domain_register_valid(NpuDomain::vec32, f.dest) ||
        !domain_register_valid(NpuDomain::vec32, f.src1) ||
        !domain_register_valid(NpuDomain::vec32, col) ||
        !domain_register_valid(NpuDomain::vec32, row)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{VdequantInstr{
        .dest = {.domain = NpuDomain::vec32, .index = f.dest},
        .src = {.domain = NpuDomain::vec32, .index = f.src1},
        .col_scale = col,
        .row_scale = row,
    }};
  default: return DecodeError::illegal;
  }
}

// MMA (R4): major4 = 0000, q = 10.
// 31    28 27       25 24    22 21       17 16      12 11       7 6        2 1   0
// [major4] [reserved3] [funct3] [  row5  ] [   B5   ] [   A5   ] [   D5   ] [ q ]
// funct3 = {000 mma.nt.zero, 001 mma.nt.acc, 010 vmma.nt.zero, 011 vmma.nt.acc};
// 100..111 are reserved. row5 selects the vmma.nt Tile A row; mma requires 0.
DecodeResult decode_mma(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto make_mma = [&](MmaOp op) -> DecodeResult {
    if (f.arg != 0 || !domain_register_valid(NpuDomain::acc, f.dest) ||
        !domain_register_valid(NpuDomain::tile, f.src1) ||
        !domain_register_valid(NpuDomain::tile, f.src2)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{MmaInstr{
        .op = op,
        .dest = {.domain = NpuDomain::acc, .index = f.dest},
        .src_a = {.domain = NpuDomain::tile, .index = f.src1},
        .src_b = {.domain = NpuDomain::tile, .index = f.src2},
    }};
  };
  const auto make_vmma = [&](VmmaOp op) -> DecodeResult {
    if (!domain_register_valid(NpuDomain::vec32, f.dest) ||
        !domain_register_valid(NpuDomain::tile, f.src1) ||
        !domain_register_valid(NpuDomain::tile, f.src2)) {
      return DecodeError::illegal;
    }
    return DecodedNpuInstr{VmmaInstr{
        .op = op,
        .dest = {.domain = NpuDomain::vec32, .index = f.dest},
        .src_a = {.domain = NpuDomain::tile, .index = f.src1},
        .src_b = {.domain = NpuDomain::tile, .index = f.src2},
        .row = f.arg,
    }};
  };

  switch (f.funct3) {
  case 0b000: return make_mma(MmaOp::nt_zero);
  case 0b001: return make_mma(MmaOp::nt_acc);
  case 0b010: return make_vmma(VmmaOp::nt_zero);
  case 0b011: return make_vmma(VmmaOp::nt_acc);
  default: return DecodeError::illegal;
  }
}

} // namespace tile_npu
