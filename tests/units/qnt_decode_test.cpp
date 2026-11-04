#include "tile_npu/decode.h"

#include <array>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/qnt_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::AquantInstr;
using tile_npu::TdequantInstr;

} // namespace

TEST_CASE(qnt_decode) {
  {
    // Anchor word from the QNT_AT layout: sel3 = 000, funct3 = 000 (QD = 0),
    // qf5 = 00000, vScale = 1, S5 = 2, D5 = 0.
    const auto word = 0x40001102u;
    const auto result = tile_npu::decode(word);
    const auto &decoded = expect_npu<AquantInstr>(result, "aquant.rows.q8s32");
    check_eq(decoded.dest.domain, tile_npu::NpuDomain::tile, "aquant destination domain");
    check_eq(decoded.dest.index, 0, "aquant destination");
    check_eq(decoded.src.domain, tile_npu::NpuDomain::acc, "aquant source domain");
    check_eq(decoded.src.index, 2, "aquant source");
    check_eq(decoded.scale, 1, "aquant scale");
  }
  {
    const auto result = tile_npu::decode(make_qnt(1, 0, 6, 7, 8));
    const auto &decoded = expect_npu<TdequantInstr>(result, "tdequant.rows.f32");
    check_eq(decoded.dest.domain, tile_npu::NpuDomain::acc, "tdequant destination domain");
    check_eq(decoded.dest.index, 6, "tdequant destination");
    check_eq(decoded.src.domain, tile_npu::NpuDomain::tile, "tdequant source domain");
    check_eq(decoded.src.index, 7, "tdequant source");
    check_eq(decoded.scale, 8, "tdequant scale");
  }
}

TEST_CASE(qnt_decode_rejects_reserved) {
  constexpr std::array cases{
      // qf5 = 00000 is the only defined quantization format.
      DecodeCase{"aquant qf5 00001", make_qnt(0, 1, 0, 0, 0)},
      DecodeCase{"aquant qf5 11111", make_qnt(0, 31, 0, 0, 0)},
      DecodeCase{"tdequant qf5 00001", make_qnt(1, 1, 0, 0, 0)},
      // funct3 values other than 000/001 are reserved.
      DecodeCase{"aquant funct3 010", make_qnt(2, 0, 0, 0, 0)},
      DecodeCase{"tdequant funct3 111", make_qnt(7, 0, 0, 0, 0)},
      // sel3 is reserved and must be zero.
      DecodeCase{"aquant sel3[0]", make_qnt(0, 0, 0, 0, 0) | (1u << 25)},
      DecodeCase{"aquant sel3[1]", make_qnt(0, 0, 0, 0, 0) | (1u << 26)},
      DecodeCase{"tdequant sel3[2]", make_qnt(1, 0, 0, 0, 0) | (1u << 27)},
      // Acc has 16 registers; a16+ is a reserved encoding.
      DecodeCase{"aquant a16 source", make_qnt(0, 0, 0, 16, 0)},
      DecodeCase{"tdequant a16 destination", make_qnt(1, 0, 16, 0, 0)},
  };
  expect_illegal(cases);
}
