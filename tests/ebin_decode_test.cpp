#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/ebin_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::EbinInstr;
using tile_npu::EbinOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::ScalarRegister;

struct Case {
  const char *name;
  std::uint32_t word;
  EbinOp op;
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t src1;
  bool src2_scalar;
  std::uint8_t src2;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<EbinInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": dest domain");
    check_eq(decoded.src1.domain, test.domain, what + ": src1 domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src1.index, test.src1, what + ": src1");
    if (test.src2_scalar) {
      const auto *scalar = std::get_if<ScalarRegister>(&decoded.src2);
      check(scalar != nullptr, what + ": src2 is scalar");
      check_eq(scalar->index, test.src2, what + ": src2");
    } else {
      const auto *reg = std::get_if<NpuRegister>(&decoded.src2);
      check(reg != nullptr, what + ": src2 is a register");
      check_eq(reg->domain, test.domain, what + ": src2 domain");
      check_eq(reg->index, test.src2, what + ": src2");
    }
  }
}

} // namespace

TEST_CASE(ebin_decode) {
  constexpr std::array cases{
      // Anchor words computed by hand from the encoding figure.
      Case{"tadd t0, t1, t2", 0x00002080u, EbinOp::add, NpuDomain::tile, 0, 1, false, 2},
      Case{"taddx t0, t1, x2", 0x02002080u, EbinOp::add, NpuDomain::tile, 0, 1, true, 2},
      Case{"tsub t15, t14, t13", make_ebin(EbinOp::sub, NpuDomain::tile, false, 15, 14, 13),
           EbinOp::sub, NpuDomain::tile, 15, 14, false, 13},
      Case{"tand t3, t4, t5", make_ebin(EbinOp::bit_and, NpuDomain::tile, false, 3, 4, 5),
           EbinOp::bit_and, NpuDomain::tile, 3, 4, false, 5},
      Case{"tshlx t6, t7, x8", make_ebin(EbinOp::shl, NpuDomain::tile, true, 6, 7, 8), EbinOp::shl,
           NpuDomain::tile, 6, 7, true, 8},
      Case{"amul a0, a1, a2", make_ebin(EbinOp::mul, NpuDomain::acc, false, 0, 1, 2), EbinOp::mul,
           NpuDomain::acc, 0, 1, false, 2},
      Case{"adiv a3, a4, a5", make_ebin(EbinOp::div, NpuDomain::acc, false, 3, 4, 5), EbinOp::div,
           NpuDomain::acc, 3, 4, false, 5},
      Case{"amin a6, a7, a8", make_ebin(EbinOp::min, NpuDomain::acc, false, 6, 7, 8), EbinOp::min,
           NpuDomain::acc, 6, 7, false, 8},
      Case{"amaxx a11, a10, x31", make_ebin(EbinOp::max, NpuDomain::acc, true, 11, 10, 31),
           EbinOp::max, NpuDomain::acc, 11, 10, true, 31},
      Case{"vor v0, v1, v2", make_ebin(EbinOp::bit_or, NpuDomain::vec32, false, 0, 1, 2),
           EbinOp::bit_or, NpuDomain::vec32, 0, 1, false, 2},
      Case{"vxor v31, v30, v29", make_ebin(EbinOp::bit_xor, NpuDomain::vec32, false, 31, 30, 29),
           EbinOp::bit_xor, NpuDomain::vec32, 31, 30, false, 29},
      Case{"vshr v3, v4, v5", make_ebin(EbinOp::shr, NpuDomain::vec32, false, 3, 4, 5), EbinOp::shr,
           NpuDomain::vec32, 3, 4, false, 5},
      Case{"asrax a9, a10, x11", make_ebin(EbinOp::sra, NpuDomain::acc, true, 9, 10, 11),
           EbinOp::sra, NpuDomain::acc, 9, 10, true, 11},
  };
  check_cases(cases);
}

TEST_CASE(ebin_decode_rejects_reserved) {
  const std::array cases{
      DecodeCase{"tile mul has no 8-bit form",
                 make_ebin(EbinOp::mul, NpuDomain::tile, false, 0, 1, 2)},
      DecodeCase{"tile div has no 8-bit form",
                 make_ebin(EbinOp::div, NpuDomain::tile, false, 0, 1, 2)},
      DecodeCase{"tile mulx is reserved as well",
                 make_ebin(EbinOp::mul, NpuDomain::tile, true, 0, 1, 2)},
      // funct4 values 6/7/14/15 are unassigned; the enum has no name for
      // them, so the raw code is cast in explicitly.
      DecodeCase{"funct4 = 0110 unassigned",
                 make_ebin(static_cast<EbinOp>(6), NpuDomain::tile, false, 0, 1, 2)},
      DecodeCase{"funct4 = 0111 unassigned",
                 make_ebin(static_cast<EbinOp>(7), NpuDomain::tile, false, 0, 1, 2)},
      DecodeCase{"funct4 = 1110 unassigned",
                 make_ebin(static_cast<EbinOp>(14), NpuDomain::vec32, false, 0, 1, 2)},
      DecodeCase{"funct4 = 1111 unassigned",
                 make_ebin(static_cast<EbinOp>(15), NpuDomain::vec32, false, 0, 1, 2)},
      DecodeCase{"vec8 has no E_BIN encoding",
                 make_ebin(EbinOp::add, NpuDomain::vec8, false, 0, 1, 2)},
      DecodeCase{"acc dest 12 out of range",
                 make_ebin(EbinOp::add, NpuDomain::acc, false, 12, 1, 2)},
      DecodeCase{"acc src1 12 out of range",
                 make_ebin(EbinOp::add, NpuDomain::acc, false, 0, 12, 2)},
      DecodeCase{"acc src2 12 out of range",
                 make_ebin(EbinOp::add, NpuDomain::acc, false, 0, 1, 12)},
      DecodeCase{"tile dest 16 out of range",
                 make_ebin(EbinOp::add, NpuDomain::tile, false, 16, 1, 2)},
      DecodeCase{"nonzero E_BIN reserved4",
                 make_ebin(EbinOp::add, NpuDomain::tile, false, 0, 1, 2) | (1u << 17)},
  };
  expect_illegal(cases);
}
