#include "tile_npu/decode.h"

#include <array>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/index_helpers.h"
#include "support/qnt_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::BdequantInstr;
using tile_npu::NpuDomain;
using tile_npu::TdequantInstr;
using tile_npu::TquantInstr;
using tile_npu::VquantInstr;

} // namespace

TEST_CASE(qnt_decode) {
  {
    // Anchor word computed by hand from the encoding figure: the asm operand
    // order is tD, vScale, aS; arg5 carries vScale.
    const auto word = 0x40020102u;
    const auto result = tile_npu::decode(word);
    const auto &decoded = expect_npu<TquantInstr>(result, "tquant.rows.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::tile, "tquant destination domain");
    check_eq(decoded.dest.index, 0, "tquant destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "tquant source domain");
    check_eq(decoded.src.index, 2, "tquant source");
    check_eq(decoded.scale, 1, "tquant scale");
    check_eq(decoded.scale_source, tile_npu::QntScaleSource::computed, "tquant source mode");
  }
  {
    const auto result = tile_npu::decode(make_qnt(NpuDomain::tile, 1, 3, 4, 5));
    const auto &decoded = expect_npu<TquantInstr>(result, "tquant.rows.scale.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::tile, "tquant.scale destination domain");
    check_eq(decoded.dest.index, 3, "tquant.scale destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "tquant.scale source domain");
    check_eq(decoded.src.index, 4, "tquant.scale source");
    check_eq(decoded.scale, 5, "tquant.scale scale");
    check_eq(decoded.scale_source, tile_npu::QntScaleSource::provided, "tquant.scale source mode");
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
  {
    const auto result = tile_npu::decode(make_qnt(NpuDomain::vec8, 0, 9, 11, 10, 3));
    const auto &decoded = expect_npu<VquantInstr>(result, "vquant.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::vec8, "vquant destination domain");
    check_eq(decoded.dest.index, 9, "vquant destination");
    check_eq(decoded.src.domain, NpuDomain::vec32, "vquant source domain");
    check_eq(decoded.src.index, 11, "vquant source");
    check_eq(decoded.scale, 10, "vquant scale");
    check_index(decoded.index, 3, false, "vquant scale lane");
  }
  {
    const auto result = tile_npu::decode(make_qnt(NpuDomain::vec32, 0, 12, 13, 14, 15, true));
    const auto &decoded = expect_npu<BdequantInstr>(result, "bdequant.f32");
    check_eq(decoded.dest.domain, NpuDomain::vec32, "bdequant destination domain");
    check_eq(decoded.dest.index, 12, "bdequant destination");
    check_eq(decoded.src.domain, NpuDomain::vec8, "bdequant source domain");
    check_eq(decoded.src.index, 13, "bdequant source");
    check_eq(decoded.scale, 14, "bdequant scale");
    check_index(decoded.index, 15, true, "bdequant scale lane");
  }
}

TEST_CASE(qnt_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 >= 010 is reserved; the external-scale form is Tile-dst only.
      DecodeCase{"funct3 010", make_qnt(NpuDomain::tile, 2, 0, 0, 0)},
      DecodeCase{"funct3 111", make_qnt(NpuDomain::vec32, 7, 0, 0, 0)},
      DecodeCase{"tdequant external scale", make_qnt(NpuDomain::acc, 1, 0, 0, 0)},
      DecodeCase{"vquant external scale", make_qnt(NpuDomain::vec8, 1, 0, 0, 0)},
      DecodeCase{"bdequant external scale", make_qnt(NpuDomain::vec32, 1, 0, 0, 0)},
      // The matrix forms reserve IS and index5.
      DecodeCase{"tquant IS", make_qnt(NpuDomain::tile, 0, 0, 0, 0, 0, true)},
      DecodeCase{"tdequant index5", make_qnt(NpuDomain::acc, 0, 0, 0, 0, 1)},
      // Register numbers are bounded per domain.
      DecodeCase{"tquant t16", make_qnt(NpuDomain::tile, 0, 16, 0, 0)},
      DecodeCase{"tquant a12", make_qnt(NpuDomain::tile, 0, 0, 12, 0)},
      DecodeCase{"tdequant t16", make_qnt(NpuDomain::acc, 0, 0, 16, 0)},
  };
  expect_illegal(cases);
}
