// Structural validation at the execute entry:
// hand-constructed IR that no encoding can produce must fail with
// illegal_instruction before dispatch, and valid hand-constructed IR must
// execute normally. decode() guarantees structural legality, so these cases
// build the IR directly.

#include "execute/execute.h"

#include "framework.h"
#include "support/execute_helpers.h"

#include "tile_npu/platform/ram.h"
#include "tile_npu/state.h"

#include <array>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>

using namespace tile_npu::test;

namespace {

using tile_npu::AreduceInstr;
using tile_npu::BcastInstr;
using tile_npu::CfgCopyInstr;
using tile_npu::CfgGetInstr;
using tile_npu::CfgGetwInstr;
using tile_npu::CfgSetiInstr;
using tile_npu::CfgSetwInstr;
using tile_npu::CfgSetxInstr;
using tile_npu::CmpInstr;
using tile_npu::CmpOp;
using tile_npu::ConfigField;
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
using tile_npu::exec_npu_instr;
using tile_npu::ExecContext;
using tile_npu::Fill5Code;
using tile_npu::FillImm16;
using tile_npu::FillInstr;
using tile_npu::ImmediateIndex;
using tile_npu::LaneBroadcastInstr;
using tile_npu::LaneExtractInstr;
using tile_npu::LaneInsertInstr;
using tile_npu::MaskInstr;
using tile_npu::MaskOp;
using tile_npu::MblkInstr;
using tile_npu::MblkOp;
using tile_npu::MmaInstr;
using tile_npu::MmaOp;
using tile_npu::MoveInstr;
using tile_npu::MoveOp;
using tile_npu::NpuDomain;
using tile_npu::NpuRegister;
using tile_npu::QntInstr;
using tile_npu::QntOp;
using tile_npu::Ram;
using tile_npu::ReduceOp;
using tile_npu::RowInstr;
using tile_npu::RowOp;
using tile_npu::ScalarRegister;
using tile_npu::State;
using tile_npu::StepStatus;
using tile_npu::SysInstr;
using tile_npu::SysOp;
using tile_npu::VreduceArgmaxInstr;
using tile_npu::VreduceInstr;

struct Env {
  Ram memory{CODE_BASE, 0x100};
  State state{};
  ExecContext ctx{state, memory};
};

template <typename Instr> void check_illegal(const Instr &instr, const char *what) {
  Env env;
  const auto result = exec_npu_instr(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::illegal_instruction, what);
}

template <typename Instr> void check_unassigned_ops(Instr instr, std::uint8_t first_unassigned) {
  for (const auto code : {first_unassigned, std::numeric_limits<std::uint8_t>::max()}) {
    instr.op = static_cast<decltype(instr.op)>(code);
    const auto what = "unassigned op " + std::to_string(code);
    check_illegal(instr, what.c_str());
  }
}

constexpr NpuRegister T(std::uint8_t i) { return {NpuDomain::tile, i}; }

constexpr NpuRegister A(std::uint8_t i) { return {NpuDomain::acc, i}; }

constexpr NpuRegister B(std::uint8_t i) { return {NpuDomain::vec8, i}; }

constexpr NpuRegister V(std::uint8_t i) { return {NpuDomain::vec32, i}; }

} // namespace

TEST_CASE(validate_rejects_bad_ebin) {
  check_illegal(EbinInstr{static_cast<EbinOp>(14), T(0), T(1), T(2)}, "unassigned funct4 code");
  check_illegal(EbinInstr{EbinOp::add, B(0), B(1), B(2)}, "vec8 domain");
  check_illegal(EbinInstr{EbinOp::mul, T(0), T(1), T(2)}, "tile mul");
  check_illegal(EbinInstr{EbinOp::add, T(0), A(1), T(2)}, "cross-domain src1");
  check_illegal(EbinInstr{EbinOp::add, T(16), T(1), T(2)}, "dest index 16");
  check_illegal(EbinInstr{EbinOp::add, T(0), T(1), ScalarRegister{32}}, "scalar src2 index 32");
}

TEST_CASE(validate_rejects_bad_eun) {
  check_illegal(EunInstr{static_cast<EunOp>(7), A(0), A(1)}, "reserved op");
  check_illegal(EunInstr{EunOp::bit_not, B(0), B(1)}, "vec8 domain");
  check_illegal(EunInstr{EunOp::abs, T(0), T(1)}, "tile only has bit_not");
  check_illegal(EunInstr{EunOp::neg, A(0), V(1)}, "cross-domain src");
}

TEST_CASE(validate_rejects_bad_ecmp) {
  check_illegal(CmpInstr{static_cast<CmpOp>(6), A(0), A(1), A(2)}, "unassigned condition code");
  check_illegal(CmpInstr{CmpOp::eq, T(0), T(1), T(2)}, "tile domain");
  check_illegal(CmpInstr{CmpOp::eq, A(0), A(1), V(2)}, "cross-domain src2");
  check_illegal(CmpInstr{CmpOp::eq, A(0), A(1), ScalarRegister{32}}, "scalar src2 index 32");
}

TEST_CASE(validate_rejects_bad_ebcast) {
  check_illegal(BcastInstr{EbinOp::add, T(0), T(1), V(0), false}, "tile paired with vec32");
  check_illegal(BcastInstr{EbinOp::add, V(0), V(1), B(0), false}, "vector destination");
  check_illegal(BcastInstr{EbinOp::add, A(0), A(1), V(32), false}, "vector index 32");
  check_illegal(BcastInstr{EbinOp::mul, T(0), T(1), B(0), false}, "tile mul");
}

TEST_CASE(validate_rejects_bad_er4) {
  check_illegal(Er4Instr{static_cast<Er4Op>(2), A(0), A(1), A(2), A(3)}, "reserved op");
  check_illegal(Er4Instr{Er4Op::fmadd, T(0), T(1), T(2), T(3)}, "fmadd on tile");
  check_illegal(Er4Instr{Er4Op::select, B(0), B(1), B(2), B(3)}, "vec8 domain");
  check_illegal(Er4Instr{Er4Op::select, A(0), A(1), A(2), V(3)}, "cross-domain src2");
}

TEST_CASE(validate_rejects_bad_emask) {
  check_illegal(MaskInstr{static_cast<MaskOp>(3), A(0), A(1), 0, Fill5Code{0}}, "reserved op");
  check_illegal(MaskInstr{MaskOp::tril, B(0), B(1), 0, Fill5Code{0}}, "tril on a vector domain");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 0, Fill5Code{9}}, "fill5 code 9");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 0, ScalarRegister{32}}, "xFill index 32");
  check_illegal(MaskInstr{MaskOp::tail, A(0), A(1), 32, Fill5Code{0}}, "bound index 32");
}

TEST_CASE(validate_rejects_bad_mblk) {
  check_illegal(MblkInstr{.op = MblkOp::gather, .reg = A(0)}, "gather on acc");
  check_illegal(MblkInstr{.op = MblkOp::row_load, .reg = B(0)}, "row form on vec8");
  check_illegal(MblkInstr{.op = MblkOp::gather, .reg = T(0), .gather_src = 32},
                "gather_src index 32");
  check_illegal(MblkInstr{.op = MblkOp::load, .reg = T(16)}, "reg index 16");
  check_illegal(MblkInstr{.op = static_cast<MblkOp>(5), .reg = T(0)}, "reserved op");
}

TEST_CASE(validate_rejects_bad_fill) {
  check_illegal(FillInstr{T(16), FillImm16{1}}, "fill dest index 16");
  check_illegal(FillInstr{{static_cast<NpuDomain>(4), 0}, FillImm16{1}}, "fill invalid domain");
  check_illegal(FillInstr{T(0), ScalarRegister{32}}, "fillx xS index 32");
}

TEST_CASE(validate_rejects_bad_move) {
  check_illegal(MoveInstr{MoveOp::transpose, A(0), A(1)}, "transpose on acc");
  check_illegal(MoveInstr{MoveOp::copy, T(0), T(16)}, "copy src index 16");
  check_illegal(MoveInstr{MoveOp::copy, T(16), T(0)}, "copy dest index 16");
  check_illegal(MoveInstr{MoveOp::copy, T(0), A(0)}, "cross-domain copy");
  check_illegal(MoveInstr{MoveOp::transpose, T(0), B(0)}, "cross-domain transpose");
  check_illegal(MoveInstr{static_cast<MoveOp>(2), T(0), T(1)}, "reserved op");
}

TEST_CASE(validate_rejects_bad_row_move) {
  check_illegal(RowInstr{RowOp::insert, T(0), V(0), ImmediateIndex{0}}, "tile paired with vec32");
  check_illegal(RowInstr{RowOp::insert, T(16), B(0), ImmediateIndex{0}}, "matrix index 16");
  check_illegal(RowInstr{static_cast<RowOp>(2), T(0), B(0), ImmediateIndex{0}}, "reserved op");
}

TEST_CASE(validate_rejects_bad_lane_move) {
  check_illegal(
      LaneExtractInstr{.dest = ScalarRegister{32}, .src = B(1), .index = ImmediateIndex{0}},
      "extract xD index 32");
  check_illegal(LaneBroadcastInstr{.dest = B(0), .src = V(1), .index = ImmediateIndex{0}},
                "broadcast with mismatched domains");
  check_illegal(
      LaneInsertInstr{.dest = B(0), .src = ScalarRegister{32}, .index = ImmediateIndex{0}},
      "insert xS index 32");
  check_illegal(
      LaneExtractInstr{.dest = ScalarRegister{0}, .src = T(1), .index = ImmediateIndex{0}},
      "extract source domain");
  check_illegal(
      LaneExtractInstr{.dest = ScalarRegister{0}, .src = B(1), .index = ScalarRegister{32}},
      "extract index xS index 32");
}

TEST_CASE(validate_rejects_bad_cfg) {
  check_illegal(CfgSetiInstr{.reg_type = static_cast<ConfigRegType>(8), .field = ConfigField::rows},
                "reserved register type");
  check_illegal(CfgSetiInstr{.dest = 16, .field = ConfigField::rows}, "seti dest index 16");
  check_illegal(CfgGetInstr{.dest = 32, .field = ConfigField::rows}, "get xD index 32");
  check_illegal(CfgGetInstr{.reg_type = ConfigRegType::ac, .src = 12, .field = ConfigField::dtype},
                "get ac12 out of range");
  check_illegal(CfgCopyInstr{.src = 16}, "copy src index 16");
  check_illegal(CfgSetiInstr{.field = ConfigField::base_addr},
                "tile config cannot use descriptor field");
}

TEST_CASE(validate_rejects_cfg_without_decoded_fields) {
  struct RegisterCase {
    ConfigRegType type;
  };

  constexpr std::array registers{
      RegisterCase{ConfigRegType::tc}, RegisterCase{ConfigRegType::ac},
      RegisterCase{ConfigRegType::bc}, RegisterCase{ConfigRegType::vc},
      RegisterCase{ConfigRegType::tm}, RegisterCase{ConfigRegType::am},
      RegisterCase{ConfigRegType::bm}, RegisterCase{ConfigRegType::vm},
  };
  for (const auto &reg : registers) {
    const std::string what = "cfg seti type " + std::to_string(static_cast<unsigned>(reg.type));
    check_illegal(CfgSetiInstr{.reg_type = reg.type, .field = static_cast<ConfigField>(255)},
                  what.c_str());
    check_illegal(CfgSetxInstr{.reg_type = reg.type, .field = static_cast<ConfigField>(255)},
                  "cfg setx without field");
    check_illegal(CfgGetInstr{.reg_type = reg.type, .field = static_cast<ConfigField>(255)},
                  "cfg get without field");
  }
}

TEST_CASE(validate_rejects_bad_cfg_word) {
  using tile_npu::ConfigRegType;
  check_illegal(CfgSetwInstr{.reg_type = ConfigRegType::tm}, "setw on a descriptor");
  check_illegal(CfgGetwInstr{.reg_type = ConfigRegType::bm}, "getw on a descriptor");
  check_illegal(CfgSetwInstr{.reg_type = ConfigRegType::tc, .dest = 15, .n = 1},
                "setw span past tc15");
  check_illegal(CfgSetwInstr{.reg_type = ConfigRegType::ac, .n = 12}, "setw span past ac11");
}

TEST_CASE(validate_rejects_bad_cvt) {
  check_illegal(CvtInstr{CvtOp::widen, A(0), B(0)}, "widen from vec8");
  check_illegal(CvtInstr{CvtOp::cvt, A(0), V(0)}, "cross-domain cvt");
  check_illegal(CvtInstr{CvtOp::cvt, T(0), T(1)}, "cvt on tile");
  check_illegal(CvtInstr{static_cast<CvtOp>(2), A(0), A(1)}, "reserved op");
}

TEST_CASE(validate_rejects_bad_reduce) {
  check_illegal(AreduceInstr{static_cast<ReduceOp>(6), V(0), A(0)}, "reserved areduce op");
  check_illegal(AreduceInstr{ReduceOp::sum, V(0), V(1)}, "areduce source domain");
  check_illegal(VreduceInstr{static_cast<ReduceOp>(4), ScalarRegister{0}, V(1)},
                "vreduce argmax op");
  check_illegal(VreduceInstr{ReduceOp::sum, ScalarRegister{32}, V(0)}, "vreduce xD index 32");
  check_illegal(VreduceInstr{ReduceOp::absmax, ScalarRegister{0}, V(0)}, "vreduce absmax op");
  check_illegal(VreduceArgmaxInstr{ScalarRegister{0}, ScalarRegister{32}, V(0)},
                "vreduce.argmax xValue index 32");
}

TEST_CASE(validate_rejects_bad_qnt) {
  check_unassigned_ops(QntInstr{QntOp::quant, T(0), A(0), 0, ImmediateIndex{0}}, 3);
  check_illegal(QntInstr{QntOp::quant, T(0), A(0), 0, ScalarRegister{0}},
                "matrix form with a scalar index");
  check_illegal(QntInstr{QntOp::quant_ext_scale, A(0), T(0), 0, ImmediateIndex{0}},
                "ext_scale with an acc dest");
  check_illegal(QntInstr{QntOp::quant, B(0), B(1), 0, ImmediateIndex{0}}, "vquant with a vec8 src");
  check_illegal(QntInstr{QntOp::quant, B(0), V(1), 32, ImmediateIndex{0}}, "scale index 32");
  check_illegal(QntInstr{QntOp::quant, A(0), T(0), 0, ImmediateIndex{0}}, "quant with an acc dest");
  check_illegal(QntInstr{QntOp::dequant, T(0), A(0), 0, ImmediateIndex{0}},
                "dequant with a tile dest");
}

TEST_CASE(validate_rejects_bad_mma) {
  check_illegal(MmaInstr{static_cast<MmaOp>(8), V(0), B(0), T(1)}, "op code past funct3");
  check_illegal(MmaInstr{static_cast<MmaOp>(2), A(0), T(0), T(1)}, "reserved mma.nt code");
  check_illegal(MmaInstr{MmaOp::nn_bdot_acc, A(0), B(0), T(1)}, "bdot.acc with an acc dest");
  check_illegal(MmaInstr{MmaOp::nn_bdot, V(0), T(0), T(1)}, "bdot with a tile src_a");
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
  const auto result = exec_npu_instr(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::retired, "hand-built tadd retires");
  check_eq(env.state.t[0][0], i8_bits(8), "lane 0");
  check_eq(env.state.t[0][1], i8_bits(6), "lane 1");
}

TEST_CASE(validate_accepts_legal_hand_built_cfg) {
  Env env;
  const CfgSetiInstr instr{.field = ConfigField::rows, .imm = 4}; // rows = 4
  const auto result = exec_npu_instr(DecodedNpuInstr{instr}, env.ctx);
  check_eq(result.status, StepStatus::retired, "hand-built cfg.seti retires");
  check_eq(env.state.tc[0].rows, 4, "rows written");
}
