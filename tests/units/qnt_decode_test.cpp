#include "tile_npu/decode.h"

#include <array>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/qnt_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::AquantInstr;
using tile_npu::NpuDomain;
using tile_npu::TdequantInstr;

} // namespace

TEST_CASE(qnt_decode) {
  {
    // Anchor word computed by hand from the encoding figure: the asm operand
    // order is tD, vScale, aS; vScale5 is in [21:17].
    const auto word = 0x40020102u;
    const auto result = tile_npu::decode(word);
    const auto &decoded = expect_npu<AquantInstr>(result, "aquant.rows.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::tile, "aquant destination domain");
    check_eq(decoded.dest.index, 0, "aquant destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "aquant source domain");
    check_eq(decoded.src.index, 2, "aquant source");
    check_eq(decoded.scale, 1, "aquant scale");
    check_eq(decoded.scale_source, tile_npu::QntScaleSource::computed, "aquant source mode");
  }
  {
    const auto result = tile_npu::decode(make_qnt(NpuDomain::tile, 1, 3, 4, 5));
    const auto &decoded = expect_npu<AquantInstr>(result, "aquant.rows.scale.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::tile, "aquant.scale destination domain");
    check_eq(decoded.dest.index, 3, "aquant.scale destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "aquant.scale source domain");
    check_eq(decoded.src.index, 4, "aquant.scale source");
    check_eq(decoded.scale, 5, "aquant.scale scale");
    check_eq(decoded.scale_source, tile_npu::QntScaleSource::provided, "aquant.scale source mode");
  }
  {
    const auto result = tile_npu::decode(make_qnt(NpuDomain::acc, 0, 6, 7, 8));
    const auto &decoded = expect_npu<TdequantInstr>(result, "tdequant.rows.f32");
    check_eq(decoded.dest.domain, NpuDomain::acc, "tdequant destination domain");
    check_eq(decoded.dest.index, 6, "tdequant destination");
    check_eq(decoded.src.domain, NpuDomain::tile, "tdequant source domain");
    check_eq(decoded.src.index, 7, "tdequant source");
    check_eq(decoded.scale, 8, "tdequant scale");
  }
}

TEST_CASE(qnt_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 >= 010 is reserved; the external-scale form is Tile-dst only.
      DecodeCase{"funct3 010", make_qnt(NpuDomain::tile, 2, 0, 0, 0)},
      DecodeCase{"funct3 111", make_qnt(NpuDomain::tile, 7, 0, 0, 0)},
      DecodeCase{"tdequant external scale", make_qnt(NpuDomain::acc, 1, 0, 0, 0)},
      DecodeCase{"dom2 10 reserved", make_qnt(NpuDomain::reserved, 0, 0, 0, 0)},
      DecodeCase{"dom2 11 reserved", make_qnt(NpuDomain::vec32, 0, 0, 0, 0)},
      // The matrix forms reserve IS and index5.
      DecodeCase{"aquant IS", make_qnt(NpuDomain::tile, 0, 0, 0, 0, 0, true)},
      DecodeCase{"tdequant index5", make_qnt(NpuDomain::acc, 0, 0, 0, 0, 1)},
  };
  expect_illegal(cases);
}
