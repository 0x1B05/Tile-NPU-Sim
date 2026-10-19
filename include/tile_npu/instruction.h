#pragma once

#include "tile_npu/state.h"

#include <cstdint>
#include <optional>
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

enum class LaneOp : std::uint8_t { extract, insert, broadcast };

struct LaneInstr {
  LaneOp op = LaneOp::extract;
  NpuOperand dst{};                      // Scalar for extract; vector otherwise.
  NpuOperand src{};                      // Scalar for insert; vector otherwise.
  IndexOperand index = ImmediateIndex{}; // Destination lane for insert; source lane otherwise.
};

enum class MmaOp : std::uint8_t {
  // clang-format off
  nn_zero     = 0b000,
  nn_acc      = 0b001,
  nn_bdot     = 0b100,
  nn_bdot_acc = 0b101,
  nt_bdot     = 0b110,
  nt_bdot_acc = 0b111,
  // clang-format on
};

struct MmaInstr {
  MmaOp op = MmaOp::nn_zero;
  NpuRegister dest{};  // Acc for mma; Vec32 for bdot.
  NpuRegister src_a{}; // Tile for mma; Vec8 for bdot.
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
  NpuRegister arg{}; // Mask for select; addend for fmadd.
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

// Operand reading:
//   areduce.rows.*: dest = vD (Vec32), src = aS (Acc); dest2 unused
//   vreduce.*:      dest = xD (Scalar), src = vS (Vec32); dest2 unused
//   vreduce.argmax: dest = xIndex (Scalar), dest2 = xValue (Scalar), src = vS
struct ReduceInstr {
  ReduceOp op = ReduceOp::sum;
  NpuOperand dest{};
  NpuOperand dest2{};
  NpuRegister src{};
};

enum class CvtOp : std::uint8_t { cvt = 0b000, widen = 0b001 };

struct CvtInstr {
  CvtOp op = CvtOp::cvt;
  NpuRegister dest{};
  NpuRegister src{};
};

enum class QntOp : std::uint8_t {
  quant,
  dequant,
  quant_ext_scale,
};

struct QntInstr {
  QntOp op = QntOp::quant;
  NpuRegister dest{};
  NpuRegister src{};
  RegisterIndex scale = 0;               // Vec32: written by quant, read otherwise.
  IndexOperand index = ImmediateIndex{}; // Scale lane for vector forms only.
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

enum class CfgOp : std::uint8_t { seti, setx, copy, get, setw, getw };

struct CfgInstr {
  CfgOp op = CfgOp::seti;
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;           // Scalar for get/getw; destination config index otherwise.
  RegisterIndex src = 0;            // Scalar for setx/setw; config for copy/get/getw.
  std::optional<ConfigField> field; // Decoded field for seti/setx/get; absent otherwise.
  std::int32_t imm = 0;             // Sign-extended imm16 for seti.
  std::uint8_t n = 0;               // Zero-based last destination offset N5.
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
