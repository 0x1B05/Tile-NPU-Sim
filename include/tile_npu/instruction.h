#pragma once

#include "tile_npu/state.h"

#include <cstdint>
#include <variant>

namespace tile_npu {

using RegisterIndex = std::uint8_t;

struct ScalarRegs {
  RegisterIndex dest = 0;
  RegisterIndex src1 = 0;
  RegisterIndex src2 = 0;
};

struct ScalarRegister {
  RegisterIndex index = 0;
};

// clang-format off
enum class ScalarOp {
  // RV64I
  lui,   auipc,  jal,    jalr,                            // U/J
  lb,    lh,     lw,     ld,    lbu,   lhu,   lwu,        // load
  sb,    sh,     sw,     sd,                              // store
  beq,   bne,    blt,    bge,   bltu,  bgeu,              // branch
  addi,  slti,   sltiu,  xori,  ori,   andi,              // op-imm
  slli,  srli,   srai,                                    // op-imm shift
  addiw, slliw,  srliw,  sraiw,                           // op-imm-32
  add,   sub,    sll,    slt,   sltu,  bit_xor,           // op
  srl,   sra,    bit_or, bit_and,
  addw,  subw,   sllw,   srlw,  sraw,                     // op-32
  fence, ecall,  ebreak,                                  // misc-mem / system

  // RV64M
  mul,   mulh,   mulhsu, mulhu, div,   divu,  rem,  remu, // M op
  mulw,  divw,   divuw,  remw,  remuw,                    // M op-32
};

enum class NpuDomain : std::uint8_t {
  tile  = 0b00,
  acc   = 0b01,
  vec8  = 0b10,
  vec32 = 0b11,
};
// clang-format on

struct NpuRegister {
  NpuDomain domain = NpuDomain::tile;
  RegisterIndex index = 0;
};

using NpuOperand = std::variant<NpuRegister, ScalarRegister>;

struct ImmediateIndex {
  std::uint8_t value = 0;
};

using IndexOperand = std::variant<ImmediateIndex, ScalarRegister>;

enum class MblkOp : std::uint8_t { load, store, gather, row_load, row_store };

struct MblkInstr {
  MblkOp op = MblkOp::load;
  NpuRegister reg{};                     // Load/gather destination or store source.
  IndexOperand index = ImmediateIndex{}; // Row forms only.
  RegisterIndex gather_src = 0;          // Vec32 index vector.
};

inline constexpr bool is_store(MblkOp op) { return op == MblkOp::store || op == MblkOp::row_store; }

struct FillImm16 {
  std::uint16_t bits = 0;
};

// fill and fillx share a semantic operation despite their different encodings.
struct FillInstr {
  NpuRegister dest{};
  std::variant<FillImm16, ScalarRegister> value = FillImm16{};
};

enum class MoveOp : std::uint8_t { copy, transpose };

struct MoveInstr {
  MoveOp op = MoveOp::copy;
  NpuRegister dest{};
  NpuRegister src{};
};

enum class RowOp : std::uint8_t { insert, extract };

struct RowInstr {
  RowOp op = RowOp::insert;
  NpuRegister matrix{}; // dest for insert, src for extract
  NpuRegister vector{}; // src for insert, dest for extract
  IndexOperand index = ImmediateIndex{};
};

struct LaneExtractInstr {
  ScalarRegister dest{};
  NpuRegister src{};
  IndexOperand index = ImmediateIndex{};
};

struct LaneInsertInstr {
  NpuRegister dest{};
  ScalarRegister src{};
  IndexOperand index = ImmediateIndex{};
};

struct LaneBroadcastInstr {
  NpuRegister dest{};
  NpuRegister src{};
  IndexOperand index = ImmediateIndex{};
};

enum class MmaOp : std::uint8_t {
  nn_zero,
  nn_acc,
};

struct MmaInstr {
  MmaOp op = MmaOp::nn_zero;
  NpuRegister dest{};  // Acc.
  NpuRegister src_a{}; // Tile.
  NpuRegister src_b{}; // Tile.
};

enum class BdotOp : std::uint8_t {
  nn_zero,
  nn_acc,
  nt_zero,
  nt_acc,
};

struct BdotInstr {
  BdotOp op = BdotOp::nn_zero;
  NpuRegister dest{};  // Vec32.
  NpuRegister src_a{}; // Vec8.
  NpuRegister src_b{}; // Tile.
};

enum class EbinOp : std::uint8_t {
  // clang-format off
  add     = 0b0000,
  sub     = 0b0001,
  mul     = 0b0010,
  div     = 0b0011,
  min     = 0b0100,
  max     = 0b0101,
  bit_and = 0b1000,
  bit_or  = 0b1001,
  bit_xor = 0b1010,
  shl     = 0b1011,
  shr     = 0b1100,
  sra     = 0b1101,
  // clang-format on
};

struct EbinInstr {
  EbinOp op = EbinOp::add;
  NpuRegister dest{};
  NpuRegister src1{};
  NpuOperand src2 = NpuRegister{};
};

enum class EunOp : std::uint8_t {
  // clang-format off
  bit_not = 0b000,
  abs     = 0b001,
  neg     = 0b010,
  exp2    = 0b011,
  rcp     = 0b100,
  rsqrt   = 0b101,
  silu    = 0b110,
  // clang-format on
};

struct EunInstr {
  EunOp op = EunOp::bit_not;
  NpuRegister dest{};
  NpuRegister src{};
};

enum class CmpOp : std::uint8_t {
  // clang-format off
  eq    = 0b000,
  ne    = 0b001,
  lt    = 0b010,
  ge    = 0b011,
  le    = 0b100,
  unord = 0b101,
  // clang-format on
};

struct CmpInstr {
  CmpOp op = CmpOp::eq;
  NpuRegister dest{}; // u32 mask.
  NpuRegister src1{};
  NpuOperand src2 = NpuRegister{};
};

struct BcastInstr {
  EbinOp op = EbinOp::add;
  NpuRegister dest{};   // matrix D
  NpuRegister src1{};   // matrix A
  NpuRegister vector{}; // broadcast source S (Vec8/Vec32)
  bool by_col = false;  // true: vector[col]; false: vector[row].
};

enum class Er4Op : std::uint8_t { select = 0b000, fmadd = 0b001 };

struct Er4Instr {
  Er4Op op = Er4Op::select;
  NpuRegister dest{};
  NpuRegister arg{}; // M5/C5: mask for select; addend for fmadd.
  NpuRegister src1{};
  NpuRegister src2{};
};

enum class MaskOp : std::uint8_t { tail = 0b000, tril = 0b001, triu = 0b010 };

// Index into the ISA fill5 value table.
struct Fill5Code {
  std::uint8_t code = 0;
};

using FillOperand = std::variant<Fill5Code, ScalarRegister>;

struct MaskInstr {
  MaskOp op = MaskOp::tail;
  NpuRegister dest{};
  NpuRegister src{};
  RegisterIndex bound = 0; // Scalar: packed rows/cols, length, or signed diagonal offset.
  FillOperand fill = Fill5Code{};
};

enum class ReduceOp : std::uint8_t {
  sum = 0b000,
  max = 0b001,
  min = 0b010,
  sumsq = 0b011,
  argmax = 0b100,
  absmax = 0b101,
};

struct AreduceInstr {
  ReduceOp op = ReduceOp::sum;
  NpuRegister dest{}; // Vec32
  NpuRegister src{};  // Acc
};

struct VreduceInstr {
  ReduceOp op = ReduceOp::sum;
  ScalarRegister dest{};
  NpuRegister src{}; // Vec32
};

struct VreduceArgmaxInstr {
  ScalarRegister index_dest{};
  ScalarRegister value_dest{};
  NpuRegister src{}; // Vec32
};

enum class CvtOp : std::uint8_t { cvt = 0b000, widen = 0b001 };

struct CvtInstr {
  CvtOp op = CvtOp::cvt;
  NpuRegister dest{};
  NpuRegister src{};
};

enum class QntScaleSource : std::uint8_t { computed, provided };

// tquant.rows[.scale].q8s32: per-row quantization with either a computed or
// externally provided Vec32 scale.
struct TquantInstr {
  NpuRegister dest{};      // Tile
  NpuRegister src{};       // Acc
  RegisterIndex scale = 0; // Vec32
  QntScaleSource scale_source = QntScaleSource::computed;
};

// tdequant.rows.f32: per-row scale from a Vec32 register.
struct TdequantInstr {
  NpuRegister dest{}; // Acc
  NpuRegister src{};  // Tile
  RegisterIndex scale = 0;
};

// vquant.q8s32: writes one selected lane of the Vec32 scale register.
struct VquantInstr {
  NpuRegister dest{}; // Vec8
  NpuRegister src{};  // Vec32
  RegisterIndex scale = 0;
  IndexOperand index = ImmediateIndex{};
};

// bdequant.f32: reads one selected lane of the Vec32 scale register.
struct BdequantInstr {
  NpuRegister dest{}; // Vec32
  NpuRegister src{};  // Vec8
  RegisterIndex scale = 0;
  IndexOperand index = ImmediateIndex{};
};

enum class ConfigRegType : std::uint8_t { tc, ac, bc, vc, tm, am, bm, vm, count };

// clang-format off
enum class ConfigField : std::uint8_t {
  dtype, rows, cols, layout, arith_mode, // matrix configs
  len,                                   // vector configs
  base_addr, row_stride_bytes, col_stride_bytes, transform, flags,   // matrix descriptors
  stride_bytes,                                                     // vector descriptors
};
// clang-format on

struct CfgSetiInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  ConfigField field = ConfigField::dtype;
  std::int32_t imm = 0;
};

struct CfgSetxInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  ConfigField field = ConfigField::dtype;
  RegisterIndex src = 0;
  std::uint8_t n = 0;
};

struct CfgCopyInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  std::uint8_t n = 0;
};

struct CfgGetInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  ConfigField field = ConfigField::dtype;
};

struct CfgSetwInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  std::uint8_t n = 0;
};

struct CfgGetwInstr {
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
};

enum class SysOp : std::uint8_t {
  fence_mem = 0b000,
  fence_sa = 0b001,
  fence_all = 0b010,
  kernel_end = 0b011,
};

struct SysInstr {
  SysOp op = SysOp::fence_mem;
};

} // namespace tile_npu
