#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/index_helpers.h"
#include "support/mblk_helpers.h"

namespace {

using tile_npu::accessed_reg;
using tile_npu::MblkInstr;
using tile_npu::MblkOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  MblkOp op;
  NpuDomain domain;
  std::uint8_t reg;
  std::uint8_t index = 0;
  std::uint8_t gather_src = 0;
  bool index_is_scalar = false;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MblkInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    const auto &reg = accessed_reg(decoded);
    check_eq(reg.domain, test.domain, what + ": domain");
    check_eq(reg.index, test.reg, what + ": reg");
    check_index(decoded.index, test.index, test.index_is_scalar, what);
    check_eq(decoded.gather_src, test.gather_src, what + ": gather_src");
  }
}

constexpr std::uint32_t block(NpuDomain domain, std::uint32_t funct3, std::uint32_t dest,
                              std::uint32_t src2 = 0) {
  return make_mblk(domain, funct3, dest, src2);
}

} // namespace

TEST_CASE(mblk_decode_block_and_gather) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "tload t0",
           .word = 0x00000001u,
           .op = MblkOp::load,
           .domain = NpuDomain::tile,
           .reg = 0},
      Case{.name = "tstore t15",
           .word = block(NpuDomain::tile, 1, 15),
           .op = MblkOp::store,
           .domain = NpuDomain::tile,
           .reg = 15},
      Case{.name = "aload a11",
           .word = block(NpuDomain::acc, 0, 11),
           .op = MblkOp::load,
           .domain = NpuDomain::acc,
           .reg = 11},
      Case{.name = "astore a0",
           .word = block(NpuDomain::acc, 1, 0),
           .op = MblkOp::store,
           .domain = NpuDomain::acc,
           .reg = 0},
      Case{.name = "bload b31",
           .word = block(NpuDomain::vec8, 0, 31),
           .op = MblkOp::load,
           .domain = NpuDomain::vec8,
           .reg = 31},
      Case{.name = "vstore v0",
           .word = block(NpuDomain::vec32, 1, 0),
           .op = MblkOp::store,
           .domain = NpuDomain::vec32,
           .reg = 0},
      Case{.name = "tload.gather t0, v1",
           .word = block(NpuDomain::tile, 2, 0, 1),
           .op = MblkOp::gather,
           .domain = NpuDomain::tile,
           .reg = 0,
           .gather_src = 1},
      Case{.name = "bload.gather b1, v0",
           .word = block(NpuDomain::vec8, 2, 1, 0),
           .op = MblkOp::gather,
           .domain = NpuDomain::vec8,
           .reg = 1,
           .gather_src = 0},
      Case{.name = "vload.gather v2, v3",
           .word = block(NpuDomain::vec32, 2, 2, 3),
           .op = MblkOp::gather,
           .domain = NpuDomain::vec32,
           .reg = 2,
           .gather_src = 3},
  };
  check_cases(cases);
}

TEST_CASE(mblk_decode_row_forms) {
  constexpr std::array cases{
      Case{.name = "tload.row t0[3]",
           .word = make_mrow(false, false, 4, 3, 0),
           .op = MblkOp::row_load,
           .domain = NpuDomain::tile,
           .reg = 0,
           .index = 3},
      Case{.name = "tload.row t1[x5]",
           .word = make_mrow(false, true, 4, 5, 1),
           .op = MblkOp::row_load,
           .domain = NpuDomain::tile,
           .reg = 1,
           .index = 5,
           .index_is_scalar = true},
      Case{.name = "tstore.row t2[0]",
           .word = make_mrow(false, false, 5, 0, 2),
           .op = MblkOp::row_store,
           .domain = NpuDomain::tile,
           .reg = 2,
           .index = 0},
      Case{.name = "aload.row a11[x0]",
           .word = make_mrow(true, true, 4, 0, 11),
           .op = MblkOp::row_load,
           .domain = NpuDomain::acc,
           .reg = 11,
           .index = 0,
           .index_is_scalar = true},
      Case{.name = "astore.row a2[31]",
           .word = make_mrow(true, false, 5, 31, 2),
           .op = MblkOp::row_store,
           .domain = NpuDomain::acc,
           .reg = 2,
           .index = 31},
  };
  check_cases(cases);
}

TEST_CASE(mblk_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 = 011 / 11x are reserved.
      DecodeCase{"funct3 011", block(NpuDomain::tile, 3, 0)},
      DecodeCase{"funct3 110", make_mrow(false, false, 6, 0, 0)},
      DecodeCase{"funct3 111", make_mrow(false, false, 7, 0, 0)},
      // Gather has no Acc form.
      DecodeCase{"aload.gather", block(NpuDomain::acc, 2, 0, 1)},
      // Block forms require S5 = 0 and reserved10 = 0.
      DecodeCase{"block load with S5", 0x00000101u},
      DecodeCase{"block load reserved10", 0x00008001u},
      // R2 forms require sel3 bit 0 = 0.
      DecodeCase{"block load sel3 bit0", 0x02000001u},
      // Register numbers are bounded per domain.
      DecodeCase{"tload t16", block(NpuDomain::tile, 0, 16)},
      DecodeCase{"aload a12", block(NpuDomain::acc, 0, 12)},
      DecodeCase{"tload.row t16[0]", make_mrow(false, false, 4, 0, 16)},
      DecodeCase{"aload.row a12[0]", make_mrow(true, false, 4, 0, 12)},
      // MR reserved fields must be 0.
      DecodeCase{"row load reserved5[21:17]", 0x01020001u},
      DecodeCase{"row load reserved5[11:7]", 0x01000081u},
      DecodeCase{"row load bit27", 0x09000001u},
      // Unassigned data-movement majors are reserved.
      DecodeCase{"major4 0101", 0x50000001u},
  };
  expect_illegal(cases);
}
