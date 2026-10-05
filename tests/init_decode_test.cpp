#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "init_helpers.h"

namespace {

using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
using tile_npu::InitInstr;
using tile_npu::InitOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  InitOp op;
  NpuDomain domain;
  std::uint8_t reg;
  std::uint8_t src = 0;
  std::int32_t imm = 0;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what =
        std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<InitInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->reg.domain, test.domain, what + ": domain");
    check_eq(decoded->reg.index, test.reg, what + ": reg");
    check_eq(decoded->src, test.src, what + ": src");
    check_eq(decoded->imm, test.imm, what + ": imm");
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

TEST_CASE(init_decode_fill) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "tfill t0, 1",
           .word = 0x20000081u,
           .op = InitOp::fill,
           .domain = NpuDomain::tile,
           .reg = 0,
           .imm = 1},
      Case{.name = "afill a11, -1",
           .word = make_init_fill(NpuDomain::acc, 11, 0xffffu),
           .op = InitOp::fill,
           .domain = NpuDomain::acc,
           .reg = 11,
           .imm = -1},
      Case{.name = "bfill b31, 0x7fff",
           .word = make_init_fill(NpuDomain::vec8, 31, 0x7fffu),
           .op = InitOp::fill,
           .domain = NpuDomain::vec8,
           .reg = 31,
           .imm = 0x7fff},
      Case{.name = "vfill v2, -32736",
           .word = make_init_fill(NpuDomain::vec32, 2, 0x8020u),
           .op = InitOp::fill,
           .domain = NpuDomain::vec32,
           .reg = 2,
           .imm = -32736},
  };
  check_cases(cases);
}

TEST_CASE(init_decode_r2_forms) {
  constexpr std::array cases{
      Case{.name = "tfillx t1, x5",
           .word = make_init_r2(NpuDomain::tile, 1, 1, 5),
           .op = InitOp::fillx,
           .domain = NpuDomain::tile,
           .reg = 1,
           .src = 5},
      Case{.name = "afillx a2, x31",
           .word = make_init_r2(NpuDomain::acc, 1, 2, 31),
           .op = InitOp::fillx,
           .domain = NpuDomain::acc,
           .reg = 2,
           .src = 31},
      Case{.name = "bcopy b3, b4",
           .word = make_init_r2(NpuDomain::vec8, 2, 3, 4),
           .op = InitOp::copy,
           .domain = NpuDomain::vec8,
           .reg = 3,
           .src = 4},
      Case{.name = "vcopy v31, v0",
           .word = make_init_r2(NpuDomain::vec32, 2, 31, 0),
           .op = InitOp::copy,
           .domain = NpuDomain::vec32,
           .reg = 31,
           .src = 0},
      Case{.name = "tcopy t15, t1",
           .word = make_init_r2(NpuDomain::tile, 2, 15, 1),
           .op = InitOp::copy,
           .domain = NpuDomain::tile,
           .reg = 15,
           .src = 1},
      Case{.name = "ttranspose t0, t2",
           .word = make_init_r2(NpuDomain::tile, 3, 0, 2),
           .op = InitOp::transpose,
           .domain = NpuDomain::tile,
           .reg = 0,
           .src = 2},
      // The shadowed fill range: imm16 in [0x8000, 0x801F] reads as fillx.
      Case{.name = "fill imm16=0x8005 shadows to fillx",
           .word = make_init_fill(NpuDomain::tile, 9, 0x8005u),
           .op = InitOp::fillx,
           .domain = NpuDomain::tile,
           .reg = 9,
           .src = 5},
  };
  check_cases(cases);
}

TEST_CASE(init_decode_rejects_reserved) {
  constexpr std::array cases{
      // R2 forms require sel3 bit 0 = 0 and reserved10 = 0.
      ErrorCase{"fillx sel3 bit0",
                make_init_r2(NpuDomain::tile, 1, 0, 0) | (1u << 25),
                DecodeError::illegal},
      ErrorCase{"copy reserved10",
                make_init_r2(NpuDomain::tile, 2, 0, 0) | (1u << 12),
                DecodeError::illegal},
      // funct3 = 1xx is reserved.
      ErrorCase{"funct3 100", make_init_r2(NpuDomain::tile, 4, 0, 0),
                DecodeError::illegal},
      ErrorCase{"funct3 111", make_init_r2(NpuDomain::tile, 7, 0, 0),
                DecodeError::illegal},
      // ttranspose is Tile-only.
      ErrorCase{"atranspose", make_init_r2(NpuDomain::acc, 3, 0, 0),
                DecodeError::illegal},
      ErrorCase{"vtranspose", make_init_r2(NpuDomain::vec32, 3, 0, 0),
                DecodeError::illegal},
      // Register numbers are bounded per domain; fillx never bounds xS, but
      // copy/ttranspose bound the source.
      ErrorCase{"tfill t16", make_init_fill(NpuDomain::tile, 16, 0),
                DecodeError::illegal},
      ErrorCase{"afillx a12", make_init_r2(NpuDomain::acc, 1, 12, 0),
                DecodeError::illegal},
      ErrorCase{"tcopy t0, t16", make_init_r2(NpuDomain::tile, 2, 0, 16),
                DecodeError::illegal},
      ErrorCase{"ttranspose t0, t16", make_init_r2(NpuDomain::tile, 3, 0, 16),
                DecodeError::illegal},
      // I-form funct3 != 000 is reserved.
      ErrorCase{"fill funct3 010", 0x21000001u, DecodeError::illegal},
  };
  check_errors(cases);
}
