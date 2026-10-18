#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"
#include "support/decode_helpers.h"
#include "support/index_helpers.h"
#include "support/row_lane_helpers.h"

namespace {

using tile_npu::LaneInstr;
using tile_npu::LaneOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::RowInstr;
using tile_npu::RowOp;
using tile_npu::ScalarRegister;

struct RowCase {
  const char *name;
  std::uint32_t word;
  RowOp op;
  NpuDomain matrix_domain;
  std::uint8_t matrix;
  NpuDomain vector_domain;
  std::uint8_t vector;
  std::uint8_t index;
  bool index_is_scalar = false;
};

template <std::size_t N> void check_row_cases(const std::array<RowCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<RowInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.matrix.domain, test.matrix_domain, what + ": matrix domain");
    check_eq(decoded.matrix.index, test.matrix, what + ": matrix");
    check_eq(decoded.vector.domain, test.vector_domain, what + ": vector domain");
    check_eq(decoded.vector.index, test.vector, what + ": vector");
    check_index(decoded.index, test.index, test.index_is_scalar, what);
  }
}

struct LaneCase {
  const char *name;
  std::uint32_t word;
  LaneOp op;
  bool dst_is_scalar;
  NpuDomain vector_domain;
  std::uint8_t dst;
  std::uint8_t src;
  std::uint8_t index;
  bool index_is_scalar = false;
};

template <std::size_t N> void check_lane_cases(const std::array<LaneCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<LaneInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_index(decoded.index, test.index, test.index_is_scalar, what);
    if (test.dst_is_scalar) {
      const auto *dst = std::get_if<ScalarRegister>(&decoded.dst);
      check(dst != nullptr, what + ": dst is scalar");
      check_eq(dst->index, test.dst, what + ": dst");
      const auto *src = std::get_if<NpuRegister>(&decoded.src);
      check(src != nullptr, what + ": src is vector");
      check_eq(src->domain, test.vector_domain, what + ": vector domain");
      check_eq(src->index, test.src, what + ": src");
    } else {
      const auto *dst = std::get_if<NpuRegister>(&decoded.dst);
      check(dst != nullptr, what + ": dst is vector");
      check_eq(dst->domain, test.vector_domain, what + ": vector domain");
      check_eq(dst->index, test.dst, what + ": dst");
      if (test.op == LaneOp::insert) {
        const auto *src = std::get_if<ScalarRegister>(&decoded.src);
        check(src != nullptr, what + ": src is scalar");
        check_eq(src->index, test.src, what + ": src");
      } else {
        const auto *src = std::get_if<NpuRegister>(&decoded.src);
        check(src != nullptr, what + ": src is vector");
        check_eq(src->domain, test.vector_domain, what + ": vector domain");
        check_eq(src->index, test.src, what + ": src");
      }
    }
  }
}

} // namespace

TEST_CASE(row_move_decode) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      RowCase{.name = "tinsert.row t0[3], b5",
              .word = 0x30003281u,
              .op = RowOp::insert,
              .matrix_domain = NpuDomain::tile,
              .matrix = 0,
              .vector_domain = NpuDomain::vec8,
              .vector = 5,
              .index = 3},
      RowCase{.name = "textract.row b2, t1[x7]",
              .word = make_row_move(false, true, 1, 7, 1, 2),
              .op = RowOp::extract,
              .matrix_domain = NpuDomain::tile,
              .matrix = 1,
              .vector_domain = NpuDomain::vec8,
              .vector = 2,
              .index = 7,
              .index_is_scalar = true},
      RowCase{.name = "ainsert.row a11[0], v0",
              .word = make_row_move(true, false, 0, 0, 0, 11),
              .op = RowOp::insert,
              .matrix_domain = NpuDomain::acc,
              .matrix = 11,
              .vector_domain = NpuDomain::vec32,
              .vector = 0,
              .index = 0},
      RowCase{.name = "aextract.row v31, a2[31]",
              .word = make_row_move(true, false, 1, 31, 2, 31),
              .op = RowOp::extract,
              .matrix_domain = NpuDomain::acc,
              .matrix = 2,
              .vector_domain = NpuDomain::vec32,
              .vector = 31,
              .index = 31},
  };
  check_row_cases(cases);
}

TEST_CASE(lane_move_decode) {
  constexpr std::array cases{
      LaneCase{.name = "vextract x3, v4[2]",
               .word = make_lane_move(true, false, 0, 2, 4, 3),
               .op = LaneOp::extract,
               .dst_is_scalar = true,
               .vector_domain = NpuDomain::vec32,
               .dst = 3,
               .src = 4,
               .index = 2},
      LaneCase{.name = "bextract x0, b9[x1]",
               .word = make_lane_move(false, true, 0, 1, 9, 0),
               .op = LaneOp::extract,
               .dst_is_scalar = true,
               .vector_domain = NpuDomain::vec8,
               .dst = 0,
               .src = 9,
               .index = 1,
               .index_is_scalar = true},
      LaneCase{.name = "vinsert v5[6], x9",
               .word = make_lane_move(true, false, 1, 6, 9, 5),
               .op = LaneOp::insert,
               .dst_is_scalar = false,
               .vector_domain = NpuDomain::vec32,
               .dst = 5,
               .src = 9,
               .index = 6},
      LaneCase{.name = "bbroadcast b1, b2[5]",
               .word = make_lane_move(false, false, 2, 5, 2, 1),
               .op = LaneOp::broadcast,
               .dst_is_scalar = false,
               .vector_domain = NpuDomain::vec8,
               .dst = 1,
               .src = 2,
               .index = 5},
      LaneCase{.name = "vbroadcast v3, v4[x8]",
               .word = make_lane_move(true, true, 2, 8, 4, 3),
               .op = LaneOp::broadcast,
               .dst_is_scalar = false,
               .vector_domain = NpuDomain::vec32,
               .dst = 3,
               .src = 4,
               .index = 8,
               .index_is_scalar = true},
  };
  check_lane_cases(cases);
}

TEST_CASE(row_lane_decode_rejects_reserved) {
  constexpr std::array cases{
      // sel3 bit 27 and reserved5 must be 0.
      DecodeCase{"row sel3 bit27", make_row_move(false, false, 0, 0, 0, 0) | (1u << 27)},
      DecodeCase{"lane reserved5", make_lane_move(true, false, 0, 0, 0, 0) | (1u << 17)},
      // ROW_MOVE has only funct3 000/001; LANE_MOVE stops at 010.
      DecodeCase{"row funct3 010", make_row_move(false, false, 2, 0, 0, 0)},
      DecodeCase{"lane funct3 011", make_lane_move(false, false, 3, 0, 0, 0)},
      // Matrix register numbers are bounded per domain.
      DecodeCase{"tinsert.row t16[0], b0", make_row_move(false, false, 0, 0, 0, 16)},
      DecodeCase{"aextract.row v0, a12[0]", make_row_move(true, false, 1, 0, 12, 0)},
  };
  expect_illegal(cases);
}
