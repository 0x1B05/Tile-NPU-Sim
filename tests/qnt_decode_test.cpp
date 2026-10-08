#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "index_helpers.h"
#include "qnt_helpers.h"

namespace {

using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;
using tile_npu::NpuDomain;
using tile_npu::QntInstr;
using tile_npu::QntOp;

struct Case {
  const char *name;
  std::uint32_t word;
  QntOp op;
  NpuDomain dest_domain;
  std::uint8_t dest;
  NpuDomain src_domain;
  std::uint8_t src;
  std::uint8_t scale;
  std::uint8_t index;
  bool index_is_scalar;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what =
        std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<QntInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->dest.domain, test.dest_domain, what + ": dest domain");
    check_eq(decoded->dest.index, test.dest, what + ": dest");
    check_eq(decoded->src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded->src.index, test.src, what + ": src");
    check_eq(decoded->scale, test.scale, what + ": scale");
    check_index(decoded->index, test.index, test.index_is_scalar, what);
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

TEST_CASE(qnt_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure: the asm
      // operand order is tD, vScale, aS; arg5 carries vScale.
      Case{.name = "tquant.rows.q8s32 t0, v1, a2",
           .word = 0x40020102u,
           .op = QntOp::quant,
           .dest_domain = NpuDomain::tile,
           .dest = 0,
           .src_domain = NpuDomain::acc,
           .src = 2,
           .scale = 1,
           .index = 0,
           .index_is_scalar = false},
      Case{.name = "tquant.rows.scale.q8s32 t3, a4, v5",
           .word = make_qnt(NpuDomain::tile, 1, 3, 4, 5),
           .op = QntOp::quant_ext_scale,
           .dest_domain = NpuDomain::tile,
           .dest = 3,
           .src_domain = NpuDomain::acc,
           .src = 4,
           .scale = 5,
           .index = 0,
           .index_is_scalar = false},
      Case{.name = "tdequant.rows.f32 a6, t7, v8",
           .word = make_qnt(NpuDomain::acc, 0, 6, 7, 8),
           .op = QntOp::dequant,
           .dest_domain = NpuDomain::acc,
           .dest = 6,
           .src_domain = NpuDomain::tile,
           .src = 7,
           .scale = 8,
           .index = 0,
           .index_is_scalar = false},
      Case{.name = "vquant.q8s32 b9, v10, v11[3]",
           .word = make_qnt(NpuDomain::vec8, 0, 9, 11, 10, 3),
           .op = QntOp::quant,
           .dest_domain = NpuDomain::vec8,
           .dest = 9,
           .src_domain = NpuDomain::vec32,
           .src = 11,
           .scale = 10,
           .index = 3,
           .index_is_scalar = false},
      Case{.name = "bdequant.f32 v12, b13, v14[x15]",
           .word = make_qnt(NpuDomain::vec32, 0, 12, 13, 14, 15, true),
           .op = QntOp::dequant,
           .dest_domain = NpuDomain::vec32,
           .dest = 12,
           .src_domain = NpuDomain::vec8,
           .src = 13,
           .scale = 14,
           .index = 15,
           .index_is_scalar = true},
  };
  check_cases(cases);
}

TEST_CASE(qnt_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 >= 010 is reserved; the external-scale form is Tile-dst only.
      ErrorCase{"funct3 010", make_qnt(NpuDomain::tile, 2, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"funct3 111", make_qnt(NpuDomain::vec32, 7, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"adequant.scale", make_qnt(NpuDomain::acc, 1, 0, 0, 0),
                DecodeError::illegal},
      // The matrix forms reserve IS and index5.
      ErrorCase{"tquant IS", make_qnt(NpuDomain::tile, 0, 0, 0, 0, 0, true),
                DecodeError::illegal},
      ErrorCase{"tdequant index5", make_qnt(NpuDomain::acc, 0, 0, 0, 0, 1),
                DecodeError::illegal},
      // Register numbers are bounded per domain.
      ErrorCase{"tquant t16", make_qnt(NpuDomain::tile, 0, 16, 0, 0),
                DecodeError::illegal},
      ErrorCase{"tquant a12", make_qnt(NpuDomain::tile, 0, 0, 12, 0),
                DecodeError::illegal},
      ErrorCase{"tdequant t16", make_qnt(NpuDomain::acc, 0, 0, 16, 0),
                DecodeError::illegal},
  };
  check_errors(cases);
}
