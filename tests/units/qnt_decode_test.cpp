#include "tile_npu/decode.h"

#include <array>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/qnt_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::AdequantInstr;
using tile_npu::AquantInstr;
using tile_npu::NpuDomain;
using tile_npu::TdequantInstr;
using tile_npu::VdequantInstr;

} // namespace

TEST_CASE(qnt_decode) {
  {
    // Anchor word: qf3 = 000, aquant, vCol = 0, vRow = 1, S5 = 2, D5 = 0.
    const auto word = 0x40001102u;
    const auto result = tile_npu::decode(word);
    const auto &decoded = expect_npu<AquantInstr>(result, "aquant.rows.q8s32");
    check_eq(decoded.dest.domain, NpuDomain::tile, "aquant destination domain");
    check_eq(decoded.dest.index, 0, "aquant destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "aquant source domain");
    check_eq(decoded.src.index, 2, "aquant source");
    check_eq(decoded.scale, 1, "aquant scale");
  }
  {
    const auto result = tile_npu::decode(make_tdequant(6, 7, 8));
    const auto &decoded = expect_npu<TdequantInstr>(result, "tdequant.rows.f32");
    check_eq(decoded.dest.domain, NpuDomain::acc, "tdequant destination domain");
    check_eq(decoded.dest.index, 6, "tdequant destination");
    check_eq(decoded.src.domain, NpuDomain::tile, "tdequant source domain");
    check_eq(decoded.src.index, 7, "tdequant source");
    check_eq(decoded.scale, 8, "tdequant scale");
  }
  {
    const auto result = tile_npu::decode(make_adequant(1, 2, 3, 4));
    const auto &decoded = expect_npu<AdequantInstr>(result, "adequant a1, a2, v3, v4");
    check_eq(decoded.dest.domain, NpuDomain::acc, "adequant destination domain");
    check_eq(decoded.dest.index, 1, "adequant destination");
    check_eq(decoded.src.domain, NpuDomain::acc, "adequant source domain");
    check_eq(decoded.src.index, 2, "adequant source");
    check_eq(decoded.col_scale, 3, "adequant vCol");
    check_eq(decoded.row_scale, 4, "adequant vRow");
  }
  {
    const auto result = tile_npu::decode(make_vdequant(5, 6, 7, 8));
    const auto &decoded = expect_npu<VdequantInstr>(result, "vdequant v5, v6, v7, v8");
    check_eq(decoded.dest.domain, NpuDomain::vec32, "vdequant destination domain");
    check_eq(decoded.dest.index, 5, "vdequant destination");
    check_eq(decoded.src.domain, NpuDomain::vec32, "vdequant source domain");
    check_eq(decoded.src.index, 6, "vdequant source");
    check_eq(decoded.col_scale, 7, "vdequant vCol");
    check_eq(decoded.row_scale, 8, "vdequant vRow");
  }
}

TEST_CASE(qnt_decode_rejects_reserved) {
  constexpr std::array cases{
      // qf3 = 000 is the only defined quantization format.
      DecodeCase{"aquant qf3 001", make_qnt(1, 0b000, 0, 0, 0, 0)},
      // Quantization only defines FORM = 00.
      DecodeCase{"aquant form 01", make_qnt(0, 0b010, 0, 0, 0, 0)},
      DecodeCase{"aquant form 10", make_qnt(0, 0b100, 0, 0, 0, 0)},
      DecodeCase{"aquant form 11", make_qnt(0, 0b110, 0, 0, 0, 0)},
      // Dequantization FORM = 11 is reserved.
      DecodeCase{"dequant form 11", make_qnt(0, 0b111, 0, 0, 0, 0)},
      // The Tile storage path has no column scale.
      DecodeCase{"aquant vCol nonzero", make_qnt(0, 0b000, 0, 0, 1, 0)},
      DecodeCase{"tdequant vCol nonzero", make_qnt(0, 0b001, 0, 0, 1, 0)},
      // Acc has 16 registers; a16+ is a reserved encoding.
      DecodeCase{"aquant a16 source", make_qnt(0, 0b000, 0, 16, 0, 0)},
      DecodeCase{"tdequant a16 destination", make_qnt(0, 0b001, 16, 0, 0, 0)},
  };
  expect_illegal(cases);
}
