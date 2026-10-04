#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <variant>

namespace tile_npu {

using RegisterIndex = std::uint8_t;

// Scalar operands use the names from the RV64 encoding, while NPU operands
// carry their register domain explicitly. This keeps format decoding separate
// from the semantic operation that consumes the operands.
struct ScalarRegs {
  RegisterIndex dest = 0;
  RegisterIndex src1 = 0;
  RegisterIndex src2 = 0;
};

struct ScalarRegister {
  RegisterIndex index = 0;
};

// Grouped by RV64 opcode space, mirroring the group order of RV64IM_TABLE in
// decode_rv64im.cpp.
// clang-format off
enum class ScalarOp {
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

  // The M extension stays contiguous at the end: execute dispatch ranges
  // over it (is_rv64m in execute.cpp).
  mul,   mulh,   mulhsu, mulhu, div,   divu,  rem,  remu, // M op
  mulw,  divw,   divuw,  remw,  remuw,                    // M op-32
};
// clang-format on

enum class NpuDomain : std::uint8_t {
  tile = 0b00,
  acc = 0b01,
  vec8 = 0b10,
  vec32 = 0b11,
};

struct NpuRegister {
  NpuDomain domain = NpuDomain::tile;
  RegisterIndex index = 0;
};

using NpuOperand = std::variant<NpuRegister, ScalarRegister>;

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

// Operation metadata for the E_BIN family, mirroring the ISA's funct4 table:
// tile_allowed is false for mul/div (the tile domain has no 8-bit mul/div
// encodings); integer_allowed is false for div (f32-only); f32_allowed is
// false for the bitwise and shift operations (integer-only). Decode,
// execution and tests share this table so a new operation is added in one
// place.
struct EbinInfo {
  EbinOp op;
  bool tile_allowed;
  bool integer_allowed;
  bool f32_allowed;
};

// Indexed by funct4 code; unassigned codes hold nullopt.
inline constexpr std::array<std::optional<EbinInfo>, 16> EBIN_BY_FUNCT4 = [] {
  std::array<std::optional<EbinInfo>, 16> table{};
  // clang-format off
  table[0b0000] = EbinInfo{EbinOp::add    , true , true , true};
  table[0b0001] = EbinInfo{EbinOp::sub    , true , true , true};
  table[0b0010] = EbinInfo{EbinOp::mul    , false, true , true};
  table[0b0011] = EbinInfo{EbinOp::div    , false, false, true};
  table[0b0100] = EbinInfo{EbinOp::min    , true , true , true};
  table[0b0101] = EbinInfo{EbinOp::max    , true , true , true};
  table[0b1000] = EbinInfo{EbinOp::bit_and, true , true , false};
  table[0b1001] = EbinInfo{EbinOp::bit_or , true , true , false};
  table[0b1010] = EbinInfo{EbinOp::bit_xor, true , true , false};
  table[0b1011] = EbinInfo{EbinOp::shl    , true , true , false};
  table[0b1100] = EbinInfo{EbinOp::shr    , true , true , false};
  table[0b1101] = EbinInfo{EbinOp::sra    , true , true , false};
  // clang-format on
  return table;
}();

// M_BLK memory access: block load/store, gather load (R2) and row load/store
// (MR). The bound descriptor (TM/AM/BM/VM) is implied by `reg`.
enum class MblkOp : std::uint8_t { load, store, gather, row_load, row_store };

// Operand reading per op:
//   load/store:     reg = D (load) / S (store)
//   gather:         reg = D, gather_src = vS (the Vec32 index vector)
//   row_load/store: reg = D/S; index is the row immediate, or the Scalar
//                   register number when index_is_scalar (IS) is set
struct MblkInstr {
  MblkOp op = MblkOp::load;
  NpuRegister reg{};
  RegisterIndex index = 0;
  RegisterIndex gather_src = 0;
  bool index_is_scalar = false;
};

// INIT_MOVE data initialization and intra-domain moves: immediate fill (I),
// scalar fill, copy and Tile transpose (R2).
enum class InitOp : std::uint8_t { fill, fillx, copy, transpose };

// Operand reading per op:
//   fill:      reg = D, imm = sign-extended imm16
//   fillx:     reg = D, src = xS
//   copy:      reg = D, src = S (same domain)
//   transpose: reg = tD, src = tS (Tile only)
struct InitInstr {
  InitOp op = InitOp::fill;
  NpuRegister reg{};
  RegisterIndex src = 0;
  std::int32_t imm = 0;
};

// ROW_MOVE between a matrix register and its paired vector register
// (EW = 0: Tile <-> Vec8, EW = 1: Acc <-> Vec32). The row number comes from
// index5, or from the Scalar register index5 names when index_is_scalar.
enum class RowOp : std::uint8_t { insert, extract };

struct RowInstr {
  RowOp op = RowOp::insert;
  NpuRegister matrix{}; // destination for insert, source for extract
  NpuRegister vector{}; // source for insert, destination for extract
  RegisterIndex index = 0;
  bool index_is_scalar = false;
};

// LANE_MOVE lane access on a vector register (EW = 0: Vec8, EW = 1: Vec32).
// The lane number comes from index5, or from the Scalar register index5
// names when index_is_scalar.
enum class LaneOp : std::uint8_t { extract, insert, broadcast };

// Operand reading per op:
//   extract:   dst = xD,       src = vector S
//   insert:    dst = vector D, src = xS
//   broadcast: dst = vector D, src = vector S
struct LaneInstr {
  LaneOp op = LaneOp::extract;
  NpuOperand dst{};
  NpuOperand src{};
  RegisterIndex index = 0;
  bool index_is_scalar = false;
};

// E_UNARY single-source elementwise operations (R2): bitwise not, abs, neg
// and the approximate f32 special functions. Vec8 has no E_UNARY; on the
// tile domain only `bit_not` exists. Numeric values match the funct3 codes.
enum class EunOp : std::uint8_t {
  bit_not = 0b000,
  abs = 0b001,
  neg = 0b010,
  exp2 = 0b011,
  rcp = 0b100,
  rsqrt = 0b101,
  silu = 0b110,
};

struct EunInstr {
  EunOp op = EunOp::bit_not;
  NpuRegister dest{};
  NpuRegister src{};
};

// E_CMP comparisons (R4), defined on Acc and Vec32 only: the destination is a
// same-shape u32 mask (true = all ones, false = 0). sel3 = {dom2, BX}; BX
// picks the second source (same-domain register or Scalar xS). Numeric
// values match the funct3 condition codes.
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

// Operand reading: dest = mask M (u32), src1 = A, src2 = B or xS.
struct CmpInstr {
  CmpOp op = CmpOp::eq;
  NpuRegister dest{};
  NpuRegister src1{};
  NpuOperand src2 = NpuRegister{};
};

// Condition metadata for the E_CMP family, mirroring the ISA's funct3 table:
// ne/ge are integer-only and le/unord are f32-only; eq/lt accept both.
// Decode, execution and tests share this table so a new condition is added
// in one place.
struct CmpInfo {
  CmpOp op;
  bool integer_allowed;
  bool f32_allowed;
};

// Indexed by funct3 code; unassigned codes (110/111) hold nullopt.
inline constexpr std::array<std::optional<CmpInfo>, 8> CMP_BY_FUNCT3 = [] {
  std::array<std::optional<CmpInfo>, 8> table{};
  // clang-format off
  table[0b000] = CmpInfo{CmpOp::eq   , true , true };
  table[0b001] = CmpInfo{CmpOp::ne   , true , false};
  table[0b010] = CmpInfo{CmpOp::lt   , true , true };
  table[0b011] = CmpInfo{CmpOp::ge   , true , false};
  table[0b100] = CmpInfo{CmpOp::le   , false, true };
  table[0b101] = CmpInfo{CmpOp::unord, false, true };
  // clang-format on
  return table;
}();

// The architectural configuration register types (TC/AC/BC/VC/TM/AM/BM/VM).
// Decode derives the register type from (major4, sel2) or (sel2, MV), so
// field access is independent of the instruction format that selected it.
enum class ConfigRegType : std::uint8_t { tc, ac, bc, vc, tm, am, bm, vm };

enum class CfgOp : std::uint8_t { seti, setx, copy, get };

// Operand reading per op:
//   seti: dest = C,  field = field3 code, imm = sign-extended imm16
//   setx: dest = C,  field = field3 code, src = xS
//   copy: dest = CD, src = CS
//   get:  dest = xD, src = C, field = field3 code
// `field` stays the raw 3-bit code: register types that lack the field fail
// with CFG_ERROR at execution, while the reserved code 7 never reaches here
// (decode rejects it as illegal).
struct CfgInstr {
  CfgOp op = CfgOp::seti;
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  std::uint8_t field = 0;
  std::int32_t imm = 0;
};

} // namespace tile_npu
