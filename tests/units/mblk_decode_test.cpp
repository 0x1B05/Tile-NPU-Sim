#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/mblk_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::MblkInstr;
using tile_npu::MblkOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  MblkOp op;
  NpuDomain domain;
  std::uint8_t reg;
  std::uint8_t gather_src = 0;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MblkInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.reg.domain, test.domain, what + ": domain");
    check_eq(decoded.reg.index, test.reg, what + ": reg");
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
      Case{.name = "tstore t31",
           .word = block(NpuDomain::tile, 1, 31),
           .op = MblkOp::store,
           .domain = NpuDomain::tile,
           .reg = 31},
      Case{.name = "aload a15",
           .word = block(NpuDomain::acc, 0, 15),
           .op = MblkOp::load,
           .domain = NpuDomain::acc,
           .reg = 15},
      Case{.name = "astore a0",
           .word = block(NpuDomain::acc, 1, 0),
           .op = MblkOp::store,
           .domain = NpuDomain::acc,
           .reg = 0},
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
      Case{.name = "vload.gather v2, v3",
           .word = block(NpuDomain::vec32, 2, 2, 3),
           .op = MblkOp::gather,
           .domain = NpuDomain::vec32,
           .reg = 2,
           .gather_src = 3},
  };
  check_cases(cases);
}

TEST_CASE(mblk_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 = 011 / 11x are reserved.
      DecodeCase{"funct3 011", block(NpuDomain::tile, 3, 0)},
      DecodeCase{"funct3 100", block(NpuDomain::tile, 4, 0)},
      DecodeCase{"funct3 101", block(NpuDomain::tile, 5, 0)},
      DecodeCase{"funct3 110", block(NpuDomain::tile, 6, 0)},
      DecodeCase{"funct3 111", block(NpuDomain::tile, 7, 0)},
      // Gather has no Acc form.
      DecodeCase{"aload.gather", block(NpuDomain::acc, 2, 0, 1)},
      // dom2 = 10 is reserved for block, store, and gather forms.
      DecodeCase{"reserved block load", block(NpuDomain::reserved, 0, 0)},
      DecodeCase{"reserved block store", block(NpuDomain::reserved, 1, 0)},
      DecodeCase{"reserved gather", block(NpuDomain::reserved, 2, 0, 0)},
      // Block forms require S5 = 0 and reserved10 = 0.
      DecodeCase{"block load with S5", 0x00000101u},
      DecodeCase{"block load reserved10", 0x00008001u},
      // R2 forms require sel3 bit 0 = 0.
      DecodeCase{"block load sel3 bit0", 0x02000001u},
      // Acc has 16 registers; a16+ is a reserved encoding.
      DecodeCase{"aload a16 out of range", block(NpuDomain::acc, 0, 16)},
      // Unassigned data-movement majors are reserved.
      DecodeCase{"major4 0101", 0x50000001u},
  };
  expect_illegal(cases);
}
