#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "check.h"
#include "ebcast_helpers.h"

namespace {

using tile_npu::BcastInstr;
using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
using tile_npu::EbinOp;
using tile_npu::NpuDomain;

struct Case {
  const char *name;
  std::uint32_t word;
  EbinOp op;
  NpuDomain matrix_domain;
  NpuDomain vector_domain;
  std::uint8_t dest;
  std::uint8_t src1;
  std::uint8_t vec;
  bool by_col;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what =
        std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<BcastInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->dest.domain, test.matrix_domain, what + ": dest domain");
    check_eq(decoded->dest.index, test.dest, what + ": dest");
    check_eq(decoded->src1.domain, test.matrix_domain, what + ": src1 domain");
    check_eq(decoded->src1.index, test.src1, what + ": src1");
    check_eq(decoded->vector.domain, test.vector_domain, what + ": vec domain");
    check_eq(decoded->vector.index, test.vec, what + ": vec");
    check_eq(decoded->by_col, test.by_col, what + ": by_col");
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

TEST_CASE(ebcast_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{.name = "taddb.byrow t0, t1, b2",
           .word = 0x50002080u,
           .op = EbinOp::add,
           .matrix_domain = NpuDomain::tile,
           .vector_domain = NpuDomain::vec8,
           .dest = 0,
           .src1 = 1,
           .vec = 2,
           .by_col = false},
      Case{.name = "tsubb.bycol t15, t14, b31",
           .word = make_ebcast(false, true, 1, 15, 14, 31),
           .op = EbinOp::sub,
           .matrix_domain = NpuDomain::tile,
           .vector_domain = NpuDomain::vec8,
           .dest = 15,
           .src1 = 14,
           .vec = 31,
           .by_col = true},
      Case{.name = "amulv.byrow a0, a1, v2",
           .word = make_ebcast(true, false, 2, 0, 1, 2),
           .op = EbinOp::mul,
           .matrix_domain = NpuDomain::acc,
           .vector_domain = NpuDomain::vec32,
           .dest = 0,
           .src1 = 1,
           .vec = 2,
           .by_col = false},
      Case{.name = "adivv.bycol a11, a10, v31",
           .word = make_ebcast(true, true, 3, 11, 10, 31),
           .op = EbinOp::div,
           .matrix_domain = NpuDomain::acc,
           .vector_domain = NpuDomain::vec32,
           .dest = 11,
           .src1 = 10,
           .vec = 31,
           .by_col = true},
      Case{.name = "ashlv.byrow a3, a4, v5",
           .word = make_ebcast(true, false, 11, 3, 4, 5),
           .op = EbinOp::shl,
           .matrix_domain = NpuDomain::acc,
           .vector_domain = NpuDomain::vec32,
           .dest = 3,
           .src1 = 4,
           .vec = 5,
           .by_col = false},
      Case{.name = "tsrab.byrow t2, t3, b4",
           .word = make_ebcast(false, false, 13, 2, 3, 4),
           .op = EbinOp::sra,
           .matrix_domain = NpuDomain::tile,
           .vector_domain = NpuDomain::vec8,
           .dest = 2,
           .src1 = 3,
           .vec = 4,
           .by_col = false},
  };
  check_cases(cases);
}

TEST_CASE(ebcast_decode_rejects_reserved) {
  constexpr std::array cases{
      // sel3 bit 1 and reserved4 must be 0.
      ErrorCase{"sel3 bit1", make_ebcast(false, false, 0, 0, 0, 0) | (1u << 26),
                DecodeError::illegal},
      ErrorCase{"reserved4", make_ebcast(false, false, 0, 0, 0, 0) | (1u << 17),
                DecodeError::illegal},
      // Unassigned funct4 codes are reserved (shared E_BIN table).
      ErrorCase{"funct4 0110", make_ebcast(false, false, 6, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"funct4 1111", make_ebcast(true, false, 15, 0, 0, 0),
                DecodeError::illegal},
      // Tile has no mul/div encodings.
      ErrorCase{"tmulb", make_ebcast(false, false, 2, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"tdivb", make_ebcast(false, false, 3, 0, 0, 0),
                DecodeError::illegal},
      // Matrix register numbers are bounded per domain; the vector side
      // takes all 32.
      ErrorCase{"taddb t16", make_ebcast(false, false, 0, 16, 0, 0),
                DecodeError::illegal},
      ErrorCase{"amulv a12", make_ebcast(true, false, 2, 0, 12, 0),
                DecodeError::illegal},
  };
  check_errors(cases);
}
