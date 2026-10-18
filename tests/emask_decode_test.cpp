#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/emask_helpers.h"

namespace {

using tile_npu::MaskInstr;
using tile_npu::MaskOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  MaskOp op;
  NpuDomain domain;
  std::uint8_t dest;
  std::uint8_t src;
  std::uint8_t bound;
  std::uint8_t fill;
  bool fill_is_scalar;
};

// The fill operand decodes to Fill5Code for FS = 0 and to ScalarRegister for
// FS = 1.
void check_fill(const tile_npu::FillOperand &fill, std::uint8_t expected, bool is_scalar,
                std::string_view what) {
  const std::string context(what);
  if (is_scalar) {
    const auto *reg = std::get_if<tile_npu::ScalarRegister>(&fill);
    check(reg != nullptr, context + ": fill is scalar");
    if (reg != nullptr) {
      check_eq(reg->index, expected, context + ": fill");
    }
    return;
  }
  const auto *code = std::get_if<tile_npu::Fill5Code>(&fill);
  check(code != nullptr, context + ": fill is a fill5 code");
  if (code != nullptr) {
    check_eq(code->code, expected, context + ": fill");
  }
}

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<MaskInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.dest.domain, test.domain, what + ": dest domain");
    check_eq(decoded.dest.index, test.dest, what + ": dest");
    check_eq(decoded.src.domain, test.domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
    check_eq(decoded.bound, test.bound, what + ": bound");
    check_fill(decoded.fill, test.fill, test.fill_is_scalar, what);
  }
}

} // namespace

TEST_CASE(emask_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure: the asm
      // operand order is D, S, xBound, fill; xBound sits in [21:17] and the
      // fill in [16:12].
      Case{.name = "tmask.tail t0, t1, x2, +1",
           .word = 0x40041080u,
           .op = MaskOp::tail,
           .domain = NpuDomain::tile,
           .dest = 0,
           .src = 1,
           .bound = 2,
           .fill = 1,
           .fill_is_scalar = false},
      Case{.name = "amask.tril a3, a4, x5, -1",
           .word = make_emask(NpuDomain::acc, 1, 3, 4, 5, 2),
           .op = MaskOp::tril,
           .domain = NpuDomain::acc,
           .dest = 3,
           .src = 4,
           .bound = 5,
           .fill = 2,
           .fill_is_scalar = false},
      Case{.name = "amaskx.triu a6, a7, x8, x9",
           .word = make_emask(NpuDomain::acc, 2, 6, 7, 8, 9, true),
           .op = MaskOp::triu,
           .domain = NpuDomain::acc,
           .dest = 6,
           .src = 7,
           .bound = 8,
           .fill = 9,
           .fill_is_scalar = true},
      Case{.name = "vmask.tail v10, v11, x12, +inf",
           .word = make_emask(NpuDomain::vec32, 0, 10, 11, 12, 7),
           .op = MaskOp::tail,
           .domain = NpuDomain::vec32,
           .dest = 10,
           .src = 11,
           .bound = 12,
           .fill = 7,
           .fill_is_scalar = false},
      Case{.name = "bmaskx.tail b13, b14, x15, x31",
           .word = make_emask(NpuDomain::vec8, 0, 13, 14, 15, 31, true),
           .op = MaskOp::tail,
           .domain = NpuDomain::vec8,
           .dest = 13,
           .src = 14,
           .bound = 15,
           .fill = 31,
           .fill_is_scalar = true},
  };
  check_cases(cases);
}

TEST_CASE(emask_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 >= 011 is reserved.
      DecodeCase{"funct3 011", make_emask(NpuDomain::acc, 3, 0, 0, 0, 0)},
      DecodeCase{"funct3 111", make_emask(NpuDomain::tile, 7, 0, 0, 0, 0)},
      // tril/triu are matrix-only.
      DecodeCase{"vmask.tril", make_emask(NpuDomain::vec32, 1, 0, 0, 0, 0)},
      DecodeCase{"bmask.triu", make_emask(NpuDomain::vec8, 2, 0, 0, 0, 0)},
      // fill5 codes 9-31 are reserved (FS = 0 only; with FS = 1 [16:12] names
      // xFill and any 5-bit number is a valid Scalar register).
      DecodeCase{"fill5 9", make_emask(NpuDomain::tile, 0, 0, 0, 0, 9)},
      DecodeCase{"fill5 31", make_emask(NpuDomain::acc, 0, 0, 0, 0, 31)},
      // Register numbers are bounded per domain.
      DecodeCase{"tmask.tail t16", make_emask(NpuDomain::tile, 0, 16, 0, 0, 0)},
      DecodeCase{"amask.tail a12 src", make_emask(NpuDomain::acc, 0, 0, 12, 0, 0)},
  };
  expect_illegal(cases);
}
