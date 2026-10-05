#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "cvt_helpers.h"
#include "framework.h"

namespace {

using tile_npu::CvtInstr;
using tile_npu::CvtOp;
using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  CvtOp op;
  NpuDomain dest_domain;
  std::uint8_t dest;
  NpuDomain src_domain;
  std::uint8_t src;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what =
        std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<CvtInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->dest.domain, test.dest_domain, what + ": dest domain");
    check_eq(decoded->dest.index, test.dest, what + ": dest");
    check_eq(decoded->src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded->src.index, test.src, what + ": src");
  }
}

struct ErrorCase {
  const char *name;
  std::uint32_t word;
  DecodeError error;
};

template <std::size_t N>
void check_errors(const std::array<ErrorCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *error = std::get_if<DecodeError>(&result);
    const std::string what =
        std::string(test.name) + " (" + to_hex(test.word) + ")";
    check(error != nullptr, what + " decodes");
    check_eq(*error, test.error, what + ": error kind");
  }
}

} // namespace

TEST_CASE(cvt_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "acvt.i32.f32 a0, a1",
           .word = 0x34000082u,
           .op = CvtOp::cvt,
           .dest_domain = NpuDomain::acc,
           .dest = 0,
           .src_domain = NpuDomain::acc,
           .src = 1},
      Case{.name = "vcvt.i32.f32 v31, v31",
           .word = make_cvt(NpuDomain::vec32, 0, 31, 31),
           .op = CvtOp::cvt,
           .dest_domain = NpuDomain::vec32,
           .dest = 31,
           .src_domain = NpuDomain::vec32,
           .src = 31},
      Case{.name = "twiden.i8.i32 a11, t15",
           .word = make_cvt(NpuDomain::acc, 1, 11, 15),
           .op = CvtOp::widen,
           .dest_domain = NpuDomain::acc,
           .dest = 11,
           .src_domain = NpuDomain::tile,
           .src = 15},
  };
  check_cases(cases);
}

TEST_CASE(cvt_decode_rejects_reserved) {
  constexpr std::array cases{
      // The destination domain is Acc or Vec32 only.
      ErrorCase{"tile destination", make_cvt(NpuDomain::tile, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"vec8 destination", make_cvt(NpuDomain::vec8, 0, 0, 0),
                DecodeError::illegal},
      // funct3 >= 010 is reserved; widen has no Vec32 encoding.
      ErrorCase{"funct3 010", make_cvt(NpuDomain::acc, 2, 0, 0),
                DecodeError::illegal},
      ErrorCase{"funct3 111", make_cvt(NpuDomain::vec32, 7, 0, 0),
                DecodeError::illegal},
      ErrorCase{"vwiden", make_cvt(NpuDomain::vec32, 1, 0, 0),
                DecodeError::illegal},
      // reserved10 and bit 25 must be 0.
      ErrorCase{"reserved10", make_cvt(NpuDomain::acc, 0, 0, 0) | (1u << 12),
                DecodeError::illegal},
      ErrorCase{"sel3 bit0", make_cvt(NpuDomain::acc, 0, 0, 0) | (1u << 25),
                DecodeError::illegal},
      // Register numbers are bounded per domain.
      ErrorCase{"acvt a12", make_cvt(NpuDomain::acc, 0, 12, 0),
                DecodeError::illegal},
      ErrorCase{"twiden t16", make_cvt(NpuDomain::acc, 1, 0, 16),
                DecodeError::illegal},
  };
  check_errors(cases);
}
