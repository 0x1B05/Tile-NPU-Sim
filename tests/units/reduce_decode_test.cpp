#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

#include "framework.h"
#include "units/decode_helpers.h"
#include "units/reduce_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::AreduceArgmaxInstr;
using tile_npu::AreduceInstr;
using tile_npu::NpuDomain;
using tile_npu::ReduceOp;
using tile_npu::VreduceArgmaxInstr;
using tile_npu::VreduceInstr;

struct ReduceCase {
  const char *name;
  std::uint32_t word;
  ReduceOp op;
  NpuDomain src_domain;
  std::uint8_t src;
};

template <std::size_t N> void check_areduce_cases(const std::array<ReduceCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<AreduceInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

template <std::size_t N> void check_vreduce_cases(const std::array<ReduceCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto &decoded = expect_npu<VreduceInstr>(result, what);
    check_eq(decoded.op, test.op, what + ": op");
    check_eq(decoded.src.domain, test.src_domain, what + ": src domain");
    check_eq(decoded.src.index, test.src, what + ": src");
  }
}

} // namespace

TEST_CASE(reduce_decode) {
  constexpr std::array areduce_cases{
      // Anchor word computed by hand from the encoding figure.
      ReduceCase{.name = "areduce.rows.sum v0, a1",
                 .word = 0x10000082u,
                 .op = ReduceOp::sum,
                 .src_domain = NpuDomain::acc,
                 .src = 1},
      ReduceCase{.name = "areduce.rows.absmax v2, a3",
                 .word = make_areduce(5, 2, 3),
                 .op = ReduceOp::absmax,
                 .src_domain = NpuDomain::acc,
                 .src = 3},
  };
  check_areduce_cases(areduce_cases);

  constexpr std::array vreduce_cases{
      ReduceCase{.name = "vreduce.sum v6, v7",
                 .word = make_vreduce(0, 6, 7),
                 .op = ReduceOp::sum,
                 .src_domain = NpuDomain::vec32,
                 .src = 7},
      ReduceCase{.name = "vreduce.sumsq v8, v9",
                 .word = make_vreduce(3, 8, 9),
                 .op = ReduceOp::sumsq,
                 .src_domain = NpuDomain::vec32,
                 .src = 9},
  };
  check_vreduce_cases(vreduce_cases);

  const auto areduce_argmax = tile_npu::decode(make_areduce_argmax(10, 11, 12));
  const auto &areduce_argmax_instr =
      expect_npu<AreduceArgmaxInstr>(areduce_argmax, "areduce.rows.argmax");
  check_eq(areduce_argmax_instr.index_dest.domain, NpuDomain::vec32,
           "areduce.rows.argmax vIdx domain");
  check_eq(areduce_argmax_instr.index_dest.index, std::uint8_t{10}, "areduce.rows.argmax vIdx");
  check_eq(areduce_argmax_instr.value_dest.domain, NpuDomain::vec32,
           "areduce.rows.argmax vVal domain");
  check_eq(areduce_argmax_instr.value_dest.index, std::uint8_t{11}, "areduce.rows.argmax vVal");
  check_eq(areduce_argmax_instr.src.domain, NpuDomain::acc, "areduce.rows.argmax src domain");
  check_eq(areduce_argmax_instr.src.index, std::uint8_t{12}, "areduce.rows.argmax src");

  const auto argmax = tile_npu::decode(make_vreduce_argmax(10, 11, 12));
  const auto &argmax_instr = expect_npu<VreduceArgmaxInstr>(argmax, "vreduce.argmax");
  check_eq(argmax_instr.index_dest.domain, NpuDomain::vec32, "vreduce.argmax vIdx domain");
  check_eq(argmax_instr.index_dest.index, std::uint8_t{10}, "vreduce.argmax vIdx");
  check_eq(argmax_instr.value_dest.domain, NpuDomain::vec32, "vreduce.argmax vVal domain");
  check_eq(argmax_instr.value_dest.index, std::uint8_t{11}, "vreduce.argmax vVal");
  check_eq(argmax_instr.src.domain, NpuDomain::vec32, "vreduce.argmax src domain");
  check_eq(argmax_instr.src.index, std::uint8_t{12}, "vreduce.argmax src");
}

TEST_CASE(reduce_decode_operands) {
  // Operand roles are represented by the concrete instruction type.
  const auto ar = tile_npu::decode(make_areduce(0, 2, 3));
  const auto &ar_instr = expect_npu<AreduceInstr>(ar, "areduce");
  check_eq(ar_instr.dest.domain, NpuDomain::vec32, "areduce dest domain");
  check_eq(ar_instr.dest.index, std::uint8_t{2}, "areduce dest index");

  const auto vr = tile_npu::decode(make_vreduce(1, 6, 7));
  const auto &vr_instr = expect_npu<VreduceInstr>(vr, "vreduce");
  check_eq(vr_instr.dest.domain, NpuDomain::vec32, "vreduce dest domain");
  check_eq(vr_instr.dest.index, std::uint8_t{6}, "vreduce dest index");

  const auto aax = tile_npu::decode(make_areduce_argmax(10, 11, 12));
  const auto &aax_instr = expect_npu<AreduceArgmaxInstr>(aax, "areduce.rows.argmax");
  check_eq(aax_instr.index_dest.index, std::uint8_t{10}, "areduce.rows.argmax vIdx");
  check_eq(aax_instr.value_dest.index, std::uint8_t{11}, "areduce.rows.argmax vVal");

  const auto ax = tile_npu::decode(make_vreduce_argmax(10, 11, 12));
  const auto &ax_instr = expect_npu<VreduceArgmaxInstr>(ax, "vreduce.argmax");
  check_eq(ax_instr.index_dest.index, std::uint8_t{10}, "vreduce.argmax vIdx");
  check_eq(ax_instr.value_dest.index, std::uint8_t{11}, "vreduce.argmax vVal");
}

TEST_CASE(reduce_decode_rejects_reserved) {
  constexpr std::array cases{
      // funct3 110/111 are reserved; absmax has no VREDUCE encoding.
      DecodeCase{"areduce funct3 110", make_areduce(6, 0, 0)},
      DecodeCase{"areduce funct3 111", make_areduce(7, 0, 0)},
      DecodeCase{"vreduce funct3 101", make_vreduce(5, 0, 0)},
      DecodeCase{"vreduce funct3 111", make_vreduce(7, 0, 0)},
      // reserved3/reserved10 must be 0.
      DecodeCase{"areduce reserved3", make_areduce(0, 0, 0) | (1u << 25)},
      DecodeCase{"areduce reserved10", make_areduce(0, 0, 0) | (1u << 12)},
      DecodeCase{"vreduce reserved10", make_vreduce(0, 0, 0) | (1u << 12)},
      DecodeCase{"areduce.argmax reserved5", make_areduce_argmax(0, 0, 0) | (1u << 17)},
      DecodeCase{"vreduce.argmax reserved5", make_vreduce_argmax(0, 0, 0) | (1u << 17)},
  };
  expect_illegal(cases);
}
