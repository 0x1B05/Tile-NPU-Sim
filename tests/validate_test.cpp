// Structural validation at the execute entry (src/execute/validate.h):
// hand-constructed IR that no encoding can produce must fail with
// illegal_instruction before dispatch, and valid hand-constructed IR must
// execute normally. decode() guarantees structural legality, so these cases
// build the IR directly.

#include "execute/execute.h"

#include "framework.h"
#include "support/execute_helpers.h"

#include "tile_npu/memory.h"
#include "tile_npu/state.h"

#include <cstdint>

namespace {

using tile_npu::BcastInstr;
using tile_npu::CfgInstr;
using tile_npu::CfgOp;
using tile_npu::CmpInstr;
using tile_npu::CmpOp;
using tile_npu::ConfigRegType;
using tile_npu::CvtInstr;
using tile_npu::CvtOp;
using tile_npu::DecodedNpuInstr;
using tile_npu::DType;
using tile_npu::EbinInstr;
using tile_npu::EbinOp;
using tile_npu::Er4Instr;
using tile_npu::Er4Op;
using tile_npu::EunInstr;
using tile_npu::EunOp;
using tile_npu::ExecContext;
using tile_npu::execute_npu;
using tile_npu::Fill5Code;
using tile_npu::FlatMemory;
using tile_npu::ImmediateIndex;
using tile_npu::InitInstr;
using tile_npu::InitOp;
using tile_npu::LaneInstr;
using tile_npu::LaneOp;
using tile_npu::MaskInstr;
using tile_npu::MaskOp;
using tile_npu::MblkInstr;
using tile_npu::MblkOp;
using tile_npu::MmaInstr;
using tile_npu::MmaOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::QntInstr;
using tile_npu::QntOp;
using tile_npu::ReduceInstr;
using tile_npu::ReduceOp;
using tile_npu::RowInstr;
using tile_npu::RowOp;
using tile_npu::ScalarRegister;
using tile_npu::State;
using tile_npu::StepStatus;
using tile_npu::SysInstr;
using tile_npu::SysOp;

struct Env {
  FlatMemory memory{CODE_BASE, 0x100};
  State state{};
  ExecContext ctx{state, memory};
};

template <typename Instr>
void check_illegal(const Instr &instr, const char *what) {
  Env env;
  const auto result = execute_npu(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::illegal_instruction, what);
}

constexpr NpuRegister T(std::uint8_t i) { return {NpuDomain::tile, i}; }
constexpr NpuRegister A(std::uint8_t i) { return {NpuDomain::acc, i}; }
constexpr NpuRegister B(std::uint8_t i) { return {NpuDomain::vec8, i}; }
constexpr NpuRegister V(std::uint8_t i) { return {NpuDomain::vec32, i}; }

} // namespace

TEST_CASE(validate_rejects_bad_ebin) {
  check_illegal(EbinInstr{static_cast<EbinOp>(14), T(0), T(1), T(2)},
                "unassigned funct4 code");
  check_illegal(EbinInstr{EbinOp::add, B(0), B(1), B(2)}, "vec8 domain");
  check_illegal(EbinInstr{EbinOp::mul, T(0), T(1), T(2)}, "tile mul");
  check_illegal(EbinInstr{EbinOp::add, T(0), A(1), T(2)}, "cross-domain src1");
  check_illegal(EbinInstr{EbinOp::add, T(16), T(1), T(2)}, "dest index 16");
  check_illegal(EbinInstr{EbinOp::add, T(0), T(1), ScalarRegister{32}},
                "scalar src2 index 32");
}

TEST_CASE(validate_rejects_bad_eun) {
  check_illegal(EunInstr{static_cast<EunOp>(7), A(0), A(1)}, "reserved op");
  check_illegal(EunInstr{EunOp::bit_not, B(0), B(1)}, "vec8 domain");
  check_illegal(EunInstr{EunOp::abs, T(0), T(1)}, "tile only has bit_not");
  check_illegal(EunInstr{EunOp::neg, A(0), V(1)}, "cross-domain src");
}

TEST_CASE(validate_rejects_bad_ecmp) {
  check_illegal(CmpInstr{static_cast<CmpOp>(6), A(0), A(1), A(2)},
                "unassigned condition code");
  check_illegal(CmpInstr{CmpOp::eq, T(0), T(1), T(2)}, "tile domain");
  check_illegal(CmpInstr{CmpOp::eq, A(0), A(1), V(2)}, "cross-domain src2");
  check_illegal(CmpInstr{CmpOp::eq, A(0), A(1), ScalarRegister{32}},
                "scalar src2 index 32");
}

TEST_CASE(validate_rejects_bad_ebcast) {
  check_illegal(BcastInstr{EbinOp::add, T(0), T(1), V(0), false},
                "tile paired with vec32");
  check_illegal(BcastInstr{EbinOp::add, V(0), V(1), B(0), false},
                "vector destination");
  check_illegal(BcastInstr{EbinOp::add, A(0), A(1), V(32), false},
                "vector index 32");
  check_illegal(BcastInstr{EbinOp::mul, T(0), T(1), B(0), false}, "tile mul");
}

TEST_CASE(validate_rejects_bad_er4) {
  check_illegal(Er4Instr{static_cast<Er4Op>(2), A(0), A(1), A(2), A(3)},
                "reserved op");
  check_illegal(Er4Instr{Er4Op::fmadd, T(0), T(1), T(2), T(3)},
                "fmadd on tile");
  check_illegal(Er4Instr{Er4Op::select, B(0), B(1), B(2), B(3)}, "vec8 domain");
  check_illegal(Er4Instr{Er4Op::select, A(0), A(1), A(2), V(3)},
                "cross-domain src2");
}

TEST_CASE(validate_rejects_bad_emask) {
  check_illegal(MaskInstr{static_cast<MaskOp>(3), A(0), A(1), 0, Fill5Code{0}},
                "reserved op");
  check_illegal(MaskInstr{MaskOp::tril, B(0), B(1), 0, Fill5Code{0}},
                "tril on a vector domain");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 0, Fill5Code{9}},
                "fill5 code 9");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 0, ScalarRegister{32}},
                "xFill index 32");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 32, Fill5Code{0}},
                "bound index 32");
}

TEST_CASE(validate_rejects_bad_mblk) {
  check_illegal(MblkInstr{MblkOp::gather, A(0), {}, ImmediateIndex{}, 0},
                "gather on acc");
  check_illegal(MblkInstr{MblkOp::row_load, B(0), {}, ImmediateIndex{0}, 0},
                "row form on vec8");
  check_illegal(MblkInstr{MblkOp::gather, T(0), {}, ImmediateIndex{}, 32},
                "gather_src index 32");
  check_illegal(MblkInstr{MblkOp::load, T(16), {}, ImmediateIndex{}, 0},
                "dest index 16");
  check_illegal(
      MblkInstr{static_cast<MblkOp>(5), T(0), {}, ImmediateIndex{}, 0},
      "reserved op");
}

TEST_CASE(validate_rejects_bad_init) {
  check_illegal(InitInstr{InitOp::transpose, A(0), 0, 0}, "transpose on acc");
  check_illegal(InitInstr{InitOp::copy, T(0), 16, 0}, "copy src index 16");
  check_illegal(InitInstr{InitOp::fillx, T(0), 32, 0}, "fillx xS index 32");
  check_illegal(InitInstr{static_cast<InitOp>(4), T(0), 0, 0}, "reserved op");
}

TEST_CASE(validate_rejects_bad_row_move) {
  check_illegal(RowInstr{RowOp::insert, T(0), V(0), ImmediateIndex{0}},
                "tile paired with vec32");
  check_illegal(RowInstr{RowOp::insert, T(16), B(0), ImmediateIndex{0}},
                "matrix index 16");
  check_illegal(RowInstr{static_cast<RowOp>(2), T(0), B(0), ImmediateIndex{0}},
                "reserved op");
}

TEST_CASE(validate_rejects_bad_lane_move) {
  check_illegal(LaneInstr{LaneOp::extract, B(0), B(1), ImmediateIndex{0}},
                "extract with a vector dst");
  check_illegal(LaneInstr{LaneOp::broadcast, B(0), V(1), ImmediateIndex{0}},
                "broadcast with mismatched domains");
  check_illegal(
      LaneInstr{LaneOp::insert, B(0), ScalarRegister{32}, ImmediateIndex{0}},
      "insert xS index 32");
  check_illegal(LaneInstr{LaneOp::extract, T(0), B(1), ImmediateIndex{0}},
                "extract dst variant mismatch");
}

TEST_CASE(validate_rejects_bad_cfg) {
  check_illegal(CfgInstr{CfgOp::seti, ConfigRegType::tc, 0, 0, 7, 0},
                "field code 7");
  check_illegal(
      CfgInstr{CfgOp::seti, static_cast<ConfigRegType>(8), 0, 0, 0, 0},
      "reserved register type");
  check_illegal(CfgInstr{CfgOp::seti, ConfigRegType::tc, 16, 0, 1, 0},
                "seti dest index 16");
  check_illegal(CfgInstr{CfgOp::get, ConfigRegType::tc, 32, 0, 1, 0},
                "get xD index 32");
  check_illegal(CfgInstr{CfgOp::copy, ConfigRegType::tc, 0, 16, 0, 0},
                "copy src index 16");
}

TEST_CASE(validate_rejects_bad_cvt) {
  check_illegal(CvtInstr{CvtOp::widen, A(0), B(0)}, "widen from vec8");
  check_illegal(CvtInstr{CvtOp::cvt, A(0), V(0)}, "cross-domain cvt");
  check_illegal(CvtInstr{CvtOp::cvt, T(0), T(1)}, "cvt on tile");
  check_illegal(CvtInstr{static_cast<CvtOp>(2), A(0), A(1)}, "reserved op");
}

TEST_CASE(validate_rejects_bad_reduce) {
  check_illegal(
      ReduceInstr{static_cast<ReduceOp>(6), ScalarRegister{0}, {}, V(0)},
      "reserved op");
  check_illegal(ReduceInstr{ReduceOp::sum, V(0), {}, V(1)},
                "vector dest with a vec32 src");
  check_illegal(ReduceInstr{ReduceOp::argmax, ScalarRegister{0}, V(0), V(1)},
                "argmax dest2 not a scalar");
  check_illegal(ReduceInstr{ReduceOp::sum, ScalarRegister{32}, {}, V(0)},
                "vreduce xD index 32");
  check_illegal(ReduceInstr{ReduceOp::absmax, ScalarRegister{0}, {}, V(0)},
                "vreduce has no absmax encoding");
}

TEST_CASE(validate_rejects_bad_qnt) {
  check_illegal(QntInstr{QntOp::quant, T(0), A(0), 0, ScalarRegister{0}},
                "matrix form with a scalar index");
  check_illegal(
      QntInstr{QntOp::quant_ext_scale, A(0), T(0), 0, ImmediateIndex{0}},
      "ext_scale with an acc dest");
  check_illegal(QntInstr{QntOp::quant, B(0), B(1), 0, ImmediateIndex{0}},
                "vquant with a vec8 src");
  check_illegal(QntInstr{QntOp::quant, B(0), V(1), 32, ImmediateIndex{0}},
                "scale index 32");
}

TEST_CASE(validate_rejects_bad_mma) {
  check_illegal(MmaInstr{static_cast<MmaOp>(5), V(0), B(0), T(1)},
                "reserved funct3 101");
  check_illegal(MmaInstr{MmaOp::bdot_nn, V(0), T(0), T(1)},
                "bdot with a tile src_a");
  check_illegal(MmaInstr{MmaOp::nn_zero, A(12), T(0), T(1)}, "dest index 12");
  check_illegal(MmaInstr{MmaOp::nn_zero, A(0), T(0), A(1)}, "src_b not a tile");
}

TEST_CASE(validate_rejects_bad_sys) {
  check_illegal(SysInstr{static_cast<SysOp>(4)}, "reserved funct3");
}

TEST_CASE(validate_accepts_legal_hand_built_ebin) {
  Env env;
  for (auto index : {0, 1, 2})
    tile_cfg(env.state, index, DType::i8, 1, 2);
  env.state.t[1][0] = i8_bits(3);
  env.state.t[1][1] = i8_bits(-4);
  env.state.t[2][0] = i8_bits(5);
  env.state.t[2][1] = i8_bits(10);
  const EbinInstr instr{EbinOp::add, T(0), T(1), T(2)}; // tadd t0, t1, t2
  const auto result = execute_npu(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::retired, "hand-built tadd retires");
  check_eq(env.state.t[0][0], i8_bits(8), "lane 0");
  check_eq(env.state.t[0][1], i8_bits(6), "lane 1");
}

TEST_CASE(validate_accepts_legal_hand_built_cfg) {
  Env env;
  const CfgInstr instr{CfgOp::seti, ConfigRegType::tc, 0, 0, 1, 4}; // rows = 4
  const auto result = execute_npu(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::retired, "hand-built cfg.seti retires");
  check_eq(env.state.tc[0].rows, 4, "rows written");
}
