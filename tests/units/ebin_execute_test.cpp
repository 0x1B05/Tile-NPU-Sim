#include "func/core.h"

#include "op_info.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>

#include "units/ebin_helpers.h"
#include "units/execute_helpers.h"

using namespace tile_npu::test;

namespace {
using tile_npu::DType;
using tile_npu::EbinOp;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::FuncCore;
using tile_npu::StepStatus;

} // namespace

TEST_CASE(ebin_acc_mul_wrap32) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::mul, NpuDomain::acc, false, 0, 1,
                                  2)); // amul a0, a1, a2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, DType::i32, 1, 2);
  state.a[1][0] = 0x00010000u;
  state.a[1][1] = static_cast<std::uint32_t>(-2);
  state.a[2][0] = 0x00010000u;
  state.a[2][1] = 3;
  check_eq(sim.step().status, StepStatus::retired, "amul retires");
  // 0x10000 * 0x10000 = 2^32 wraps to 0; -2 * 3 = -6.
  check_eq(hex(state.a[0][0]), hex(0), "amul wraps at 32 bits");
  check_eq(hex(state.a[0][1]), hex(static_cast<std::uint32_t>(-6)), "amul signed lane");
}

TEST_CASE(ebin_acc_min_max_i32_u32) {
  Ram memory(CODE_BASE, 16);
  FuncCore sim(memory);
  write_word(memory, CODE_BASE + 0, make_ebin(EbinOp::min, NpuDomain::acc, false, 0, 1, 2));
  write_word(memory, CODE_BASE + 4, make_ebin(EbinOp::max, NpuDomain::acc, false, 3, 4, 5));
  write_word(memory, CODE_BASE + 8, make_ebin(EbinOp::min, NpuDomain::acc, false, 6, 7, 8));
  write_word(memory, CODE_BASE + 12, make_ebin(EbinOp::max, NpuDomain::acc, false, 9, 10, 11));
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2, 3, 4, 5})
    acc_cfg(state, index, DType::i32, 1, 1);
  for (auto index : {6, 7, 8, 9, 10, 11}) {
    acc_cfg(state, index, DType::u32, 1, 1);
  }
  // Same bit patterns under both dtypes: 0xfffffffb reads as -5 (i32) or
  // 4294967291 (u32).
  state.a[1][0] = state.a[4][0] = state.a[7][0] = state.a[10][0] = 0xfffffffbu;
  state.a[2][0] = state.a[5][0] = state.a[8][0] = state.a[11][0] = 3;
  for (int instr = 0; instr < 4; ++instr) {
    check_eq(sim.step().status, StepStatus::retired, "min/max retires");
  }
  check_eq(hex(state.a[0][0]), hex(0xfffffffbu), "i32 min picks -5");
  check_eq(hex(state.a[3][0]), hex(3), "i32 max picks 3");
  check_eq(hex(state.a[6][0]), hex(3), "u32 min picks 3");
  check_eq(hex(state.a[9][0]), hex(0xfffffffbu), "u32 max picks 4294967291");
}

TEST_CASE(ebin_acc_sra_i32) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sra, NpuDomain::acc, false, 0, 1,
                                  2)); // asra a0, a1, a2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    acc_cfg(state, index, DType::i32, 1, 1);
  state.a[1][0] = static_cast<std::uint32_t>(-1024);
  state.a[2][0] = 33; // 33 & 31 = 1
  check_eq(sim.step().status, StepStatus::retired, "asra retires");
  check_eq(hex(state.a[0][0]), hex(static_cast<std::uint32_t>(-512)),
           "i32 arithmetic right shift masks the amount");
}

TEST_CASE(ebin_vec_sra_top_bit_fill) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::sra, NpuDomain::vec32, false, 0, 1,
                     2)); // vsra v0, v1, v2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::u32, 1);
  state.v[1][0] = 0x80000000u;
  state.v[2][0] = 1;
  check_eq(sim.step().status, StepStatus::retired, "vsra retires");
  // sra fills with the element's top bit regardless of dtype signedness:
  // 0x80000000 sra 1 = 0xc0000000 (shr would give 0x40000000).
  check_eq(hex(state.v[0][0]), hex(0xc0000000u), "u32 sra fills the top bit");
}

TEST_CASE(ebin_acc_scalar_source) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory, make_ebin(EbinOp::sub, NpuDomain::acc, true, 0, 1,
                                  3)); // asubx a0, a1, x3
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 1, 2);
  acc_cfg(state, 1, DType::i32, 1, 2);
  state.a[1][0] = 7;
  state.a[1][1] = static_cast<std::uint32_t>(-3);
  state.scalar.write_x(3, 5);
  check_eq(sim.step().status, StepStatus::retired, "asubx retires");
  // 7 - 5 = 2; -3 - 5 = -8.
  check_eq(hex(state.a[0][0]), hex(2), "asubx lane 0");
  check_eq(hex(state.a[0][1]), hex(static_cast<std::uint32_t>(-8)), "asubx lane 1");
}

TEST_CASE(ebin_vec32_len_and_scalar_source) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, true, 0, 1,
                     3)); // vaddx v0, v1, x3
  auto &state = sim.state();
  vec_cfg(state, 0, DType::i32, 2);
  vec_cfg(state, 1, DType::i32, 2);
  state.v[1][0] = 1;
  state.v[1][1] = 2;
  state.scalar.write_x(3, 0x100000005u); // decode_scalar keeps the low 32 bits
  check_eq(sim.step().status, StepStatus::retired, "vaddx retires");
  check_eq(hex(state.v[0][0]), hex(6), "vaddx lane 0");
  check_eq(hex(state.v[0][1]), hex(7), "vaddx lane 1");
  check_eq(hex(state.v[0][2]), hex(0), "lane past len is untouched");
}

TEST_CASE(ebin_f32_arithmetic) {
  Ram memory(CODE_BASE, 12);
  FuncCore sim(memory);
  write_word(memory, CODE_BASE,
             make_ebin(EbinOp::add, NpuDomain::vec32, false, 0, 1,
                       2)); // vadd v0, v1, v2
  write_word(memory, CODE_BASE + 4,
             make_ebin(EbinOp::mul, NpuDomain::vec32, false, 3, 4,
                       5)); // vmul v3, v4, v5
  write_word(memory, CODE_BASE + 8,
             make_ebin(EbinOp::min, NpuDomain::vec32, false, 6, 7,
                       8)); // vmin v6, v7, v8
  sim.reset(CODE_BASE);
  auto &state = sim.state();
  for (auto index : {0, 1, 2, 3, 4, 5, 6, 7, 8}) {
    vec_cfg(state, index, DType::f32, 1);
  }
  state.v[1][0] = as_u32(1.5f);
  state.v[2][0] = as_u32(0.5f);
  state.v[4][0] = as_u32(3.0f);
  state.v[5][0] = as_u32(2.0f);
  state.v[7][0] = 0x7fc00000u; // qNaN
  state.v[8][0] = as_u32(1.0f);
  check_eq(sim.step().status, StepStatus::retired, "vadd retires");
  check_eq(sim.step().status, StepStatus::retired, "vmul retires");
  check_eq(sim.step().status, StepStatus::retired, "vmin retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(2.0f)), "f32 add");
  check_eq(hex(state.v[3][0]), hex(as_u32(6.0f)), "f32 mul");
  // IEEE minNum: NaN does not propagate, the number wins.
  check_eq(hex(state.v[6][0]), hex(as_u32(1.0f)), "f32 min drops NaN");
}

TEST_CASE(ebin_f32_scalar_source) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  // The scalar form reinterprets the low 32 bits of xS as f32 bits; the
  // register file stores raw bits, so this is the case that pins the
  // decode_scalar convention for floats.
  load_one(sim, memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, true, 0, 1,
                     2)); // vaddx v0, v1, x2
  auto &state = sim.state();
  vec_cfg(state, 0, DType::f32, 2);
  vec_cfg(state, 1, DType::f32, 2);
  state.v[1][0] = as_u32(1.5f);
  state.v[1][1] = as_u32(-2.0f);
  state.scalar.write_x(2, 0xffffffff3f800000u); // f32 1.0 in the low 32 bits
  check_eq(sim.step().status, StepStatus::retired, "vaddx f32 retires");
  check_eq(hex(state.v[0][0]), hex(as_u32(2.5f)), "f32 add 1.5 + 1.0");
  check_eq(hex(state.v[0][1]), hex(as_u32(-1.0f)), "f32 add -2.0 + 1.0");
}

TEST_CASE(ebin_f32_rejects_integer_only_ops) {
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::bit_and, NpuDomain::vec32, false, 0, 1,
                     2)); // vand v0, v1, v2
  auto &state = sim.state();
  for (auto index : {0, 1, 2})
    vec_cfg(state, index, DType::f32, 1);
  state.v[1][0] = as_u32(1.0f);
  check_eq(sim.step().status, StepStatus::invalid_configuration,
           "f32 bitwise op is a config error");
  check_eq(hex(state.v[0][0]), hex(0), "rejected op leaves v0 untouched");
}

TEST_CASE(ebin_rejects_mismatched_configs) {
  // Each case perturbs one constraint of an otherwise legal vadd v0, v1, v2
  // (1-lane i32) and gets a fresh simulator: a rejected step keeps the pc, so
  // a shared simulator would re-execute the first instruction.
  struct Case {
    const char *name;
    void (*perturb)(tile_npu::State &);
  };

  constexpr std::array cases{
      Case{"dtype mismatch", [](tile_npu::State &state) { state.vc[1].dtype = DType::u32; }},
      Case{"shape mismatch",
           [](tile_npu::State &state) {
             state.vc[1].len = 2;
             state.vc[2].len = 2;
           }},
  };
  for (const auto &test : cases) {
    Ram memory(CODE_BASE, 4);
    FuncCore sim(memory);
    load_one(sim, memory,
             make_ebin(EbinOp::add, NpuDomain::vec32, false, 0, 1,
                       2)); // vadd v0, v1, v2
    auto &state = sim.state();
    for (auto index : {0, 1, 2}) {
      vec_cfg(state, index, DType::i32, 1);
    }
    state.v[1][0] = 1;
    state.v[2][0] = 2;
    test.perturb(state);
    check_eq(sim.step().status, StepStatus::invalid_configuration, test.name);
    check_eq(hex(state.scalar.pc), hex(CODE_BASE), "rejected op keeps pc");
    check_eq(hex(state.v[0][0]), hex(0x0u), "rejected op leaves v0 untouched");
  }
}

TEST_CASE(ebin_rejects_empty_shape) {
  // A zero lane count matches across operands, so only the 1..extent bound
  // rejects it; the loops would otherwise run zero iterations and retire.
  Ram memory(CODE_BASE, 4);
  FuncCore sim(memory);
  load_one(sim, memory,
           make_ebin(EbinOp::add, NpuDomain::vec32, false, 0, 1,
                     2)); // vadd v0, v1, v2
  auto &state = sim.state();
  for (auto index : {0, 1, 2}) {
    vec_cfg(state, index, DType::i32, 0);
  }
  check_eq(sim.step().status, StepStatus::invalid_configuration, "zero lane count rejects");
}

TEST_CASE(ebin_legality_matrix) {
  // The ISA's dtype table restated as a sweep: f32 supports only
  // add/sub/mul/min/max; integers support every assigned operation. The op
  // list comes from the shared EBIN_BY_FUNCT4 table so a new op joins the
  // sweep automatically, while the expected legality stays an independent
  // restatement of the manual.
  struct Combo {
    NpuDomain domain;
    DType dtype;
  };

  constexpr std::array combos{
      Combo{NpuDomain::acc, DType::i32},   Combo{NpuDomain::acc, DType::u32},
      Combo{NpuDomain::acc, DType::f32},   Combo{NpuDomain::vec32, DType::i32},
      Combo{NpuDomain::vec32, DType::u32}, Combo{NpuDomain::vec32, DType::f32},
  };
  for (const auto &[domain, dtype] : combos) {
    for (std::size_t funct4 = 0; funct4 < tile_npu::EBIN_BY_FUNCT4.size(); ++funct4) {
      const auto &entry = tile_npu::EBIN_BY_FUNCT4[funct4];
      if (!entry) continue;
      const auto op = static_cast<EbinOp>(funct4);
      const bool f32_legal = op == EbinOp::add || op == EbinOp::sub || op == EbinOp::mul ||
                             op == EbinOp::min || op == EbinOp::max;
      const bool legal = dtype == DType::f32 ? f32_legal : true;
      for (const bool scalar_src2 : {false, true}) {
        Ram memory(CODE_BASE, 4);
        FuncCore sim(memory);
        load_one(sim, memory, make_ebin(op, domain, scalar_src2, 0, 1, 2));
        auto &state = sim.state();
        for (auto index : {0, 1, 2}) {
          if (domain == NpuDomain::acc) {
            acc_cfg(state, index, dtype, 1, 1);
          } else {
            vec_cfg(state, index, dtype, 1);
          }
        }
        const auto value = dtype == DType::f32 ? as_u32(1.0f) : 1u;
        if (domain == NpuDomain::acc) {
          state.a[1][0] = value;
          state.a[2][0] = value;
        } else {
          state.v[1][0] = value;
          state.v[2][0] = value;
        }
        state.scalar.write_x(2, value);
        const std::string what = "legality dom=" + std::to_string(static_cast<unsigned>(domain)) +
                                 " dtype=" + std::to_string(static_cast<unsigned>(dtype)) +
                                 " op=" + std::to_string(static_cast<unsigned>(op)) +
                                 (scalar_src2 ? " bx=1" : " bx=0");
        check_eq(sim.step().status, legal ? StepStatus::retired : StepStatus::invalid_configuration,
                 what);
      }
    }
  }
}
