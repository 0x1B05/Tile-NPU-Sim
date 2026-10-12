#pragma once

#include "tile_npu/state.h"

#include <array>
#include <cstddef>
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

// Architectural register count per data domain; decode and the structural
// validation at the execute entry both bound register numbers with it.
inline constexpr std::size_t domain_register_count(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return TILE_REGISTERS;
  case NpuDomain::acc: return ACC_REGISTERS;
  case NpuDomain::vec8:
  case NpuDomain::vec32: return VECTOR_REGISTERS;
  }
  return 0;
}

using NpuOperand = std::variant<NpuRegister, ScalarRegister>;

// An index5-style operand: either an immediate index or the name of the
// Scalar register holding the index (the IS encoding bit selects the form).
struct ImmediateIndex {
  std::uint8_t value = 0;
};

using IndexOperand = std::variant<ImmediateIndex, ScalarRegister>;

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
// (MR). The bound descriptor (TM/AM/BM/VM) is implied by the accessed
// register.
enum class MblkOp : std::uint8_t { load, store, gather, row_load, row_store };

// Operand reading per op:
//   load/gather/row_load: dest = D (gather adds gather_src = vS, the Vec32
//                         index vector)
//   store/row_store:      src = S
//   row_load/row_store:   index is the row number (immediate or Scalar)
struct MblkInstr {
  MblkOp op = MblkOp::load;
  NpuRegister dest{};
  NpuRegister src{};
  IndexOperand index = ImmediateIndex{};
  RegisterIndex gather_src = 0;
};

// The store forms read the transferred register from src; every other form
// writes dest. Decode, the structural validation, the executor and tests
// share this reading.
inline constexpr bool is_store_form(MblkOp op) {
  return op == MblkOp::store || op == MblkOp::row_store;
}

// The register the access transfers.
inline constexpr const NpuRegister &accessed_reg(const MblkInstr &instr) {
  return is_store_form(instr.op) ? instr.src : instr.dest;
}

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
// (EW = 0: Tile <-> Vec8, EW = 1: Acc <-> Vec32). index is the row number
// (immediate or Scalar register).
enum class RowOp : std::uint8_t { insert, extract };

struct RowInstr {
  RowOp op = RowOp::insert;
  NpuRegister matrix{}; // destination for insert, source for extract
  NpuRegister vector{}; // source for insert, destination for extract
  IndexOperand index = ImmediateIndex{};
};

// LANE_MOVE lane access on a vector register (EW = 0: Vec8, EW = 1: Vec32).
// index is the lane number (immediate or Scalar register).
enum class LaneOp : std::uint8_t { extract, insert, broadcast };

// Operand reading per op:
//   extract:   dst = xD,       src = vector S
//   insert:    dst = vector D, src = xS
//   broadcast: dst = vector D, src = vector S
struct LaneInstr {
  LaneOp op = LaneOp::extract;
  NpuOperand dst{};
  NpuOperand src{};
  IndexOperand index = ImmediateIndex{};
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

// E_BCAST vector-broadcast elementwise operations (R3): a matrix binary op
// whose second source is a paired vector register, broadcast along rows
// (byrow: S[i] feeds row i) or columns (bycol: S[j] feeds column j). TA pairs
// Tile with Vec8 and Acc with Vec32. The funct4 operation codes are shared
// with E_BIN (EBIN_BY_FUNCT4).
struct BcastInstr {
  EbinOp op = EbinOp::add;
  NpuRegister dest{};   // matrix D
  NpuRegister src1{};   // matrix A
  NpuRegister vector{}; // broadcast source S (Vec8/Vec32)
  bool by_col = false;
};

// E_R4 four-register operations (R4): select (funct3 = 000) on every
// matrix/vec32 domain, fmadd (funct3 = 001, f32 Acc/Vec32 only). arg carries
// the fourth operand: the mask M for select, the addend C for fmadd.
enum class Er4Op : std::uint8_t { select = 0b000, fmadd = 0b001 };

// Operand reading per op:
//   select: dest = D, arg = mask M (u8/u32 per domain), src1 = A, src2 = B
//   fmadd:  dest = D, arg = addend C, src1 = A, src2 = B
struct Er4Instr {
  Er4Op op = Er4Op::select;
  NpuRegister dest{};
  NpuRegister arg{};
  NpuRegister src1{};
  NpuRegister src2{};
};

// E_MASK operations (R4): inside the effective region, positions failing the
// condition are replaced with a fill value. FS picks the fill source (fill5
// immediate table / Scalar xFill); the bound Scalar register is read per op
// (packed nRows/nCols for matrix tail, xLen for vector tail, signed xDelta
// for tril/triu). tril/triu are matrix-only.
enum class MaskOp : std::uint8_t { tail = 0b000, tril = 0b001, triu = 0b010 };

// An E_MASK fill operand (the FS encoding bit selects the form): either a
// code into the fill5 immediate table or the Scalar register xFill whose low
// element-width bits supply the fill.
struct Fill5Code {
  std::uint8_t code = 0;
};

// The fill5 immediate table: codes 0-4 are the integer values 0/+1/-1/+2/-2,
// codes 5-8 add the f32-only ±0.5/±inf; codes 9-31 are reserved.
inline constexpr std::uint8_t FILL5_INT_MAX = 4;
inline constexpr std::uint8_t FILL5_MAX = 8;

using FillOperand = std::variant<Fill5Code, ScalarRegister>;

// Operand reading: dest = D, src = S, bound = xB (Scalar register), fill per
// FS (see FillOperand).
struct MaskInstr {
  MaskOp op = MaskOp::tail;
  NpuRegister dest{};
  NpuRegister src{};
  RegisterIndex bound = 0;
  FillOperand fill = Fill5Code{};
};

// CVT operations (R2): cvt converts i32 -> f32 within Acc or Vec32 and
// updates the destination dtype on success; widen sign-extends a Tile i8
// source into Acc i32. funct3 >= 010 is reserved.
enum class CvtOp : std::uint8_t { cvt = 0b000, widen = 0b001 };

// Operand reading: dest = D (Acc, or Vec32 for cvt), src = S (same domain as
// dest for cvt; Tile for widen).
struct CvtInstr {
  CvtOp op = CvtOp::cvt;
  NpuRegister dest{};
  NpuRegister src{};
};

// AREDUCE/VREDUCE operations: the funct3 codes are shared between the two
// families; argmax is the only two-destination form (VREDUCE R4 variant) and
// absmax exists only on AREDUCE.
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

// QNT operations (R4): the semantic direction, resolved at decode from
// (funct3, dom2). quant is the dynamic-scale quantization (tquant.rows.q8s32,
// vquant.q8s32), dequant is tdequant.rows.f32 / bdequant.f32, and
// quant_ext_scale is the external-scale row quantization
// tquant.rows.scale.q8s32 (Tile destination only). funct3 >= 010 is
// reserved.
enum class QntOp : std::uint8_t {
  quant,
  dequant,
  quant_ext_scale,
};

// Operand reading: dest = D and src = S are the main operands (tquant:
// tD/aS; tdequant: aD/tS; vquant: bD/vS; bdequant: vD/bS); scale is the
// vScale Vec32 register (written by quant, read otherwise); index is the
// scale lane (vector forms only, immediate or Scalar register).
struct QntInstr {
  QntOp op = QntOp::quant;
  NpuRegister dest{};
  NpuRegister src{};
  RegisterIndex scale = 0;
  IndexOperand index = ImmediateIndex{};
};

// QNT pairs a quantized 8-bit domain with its f32 counterpart: Tile with Acc
// (the matrix forms) and Vec8 with Vec32 (the vector forms). Decode and the
// structural validation share the pairing so it is stated once.
inline constexpr bool qnt_domain_is_quantized(NpuDomain domain) {
  return domain == NpuDomain::tile || domain == NpuDomain::vec8;
}

inline constexpr NpuDomain qnt_paired_domain(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return NpuDomain::acc;
  case NpuDomain::acc: return NpuDomain::tile;
  case NpuDomain::vec8: return NpuDomain::vec32;
  case NpuDomain::vec32: return NpuDomain::vec8;
  }
  return NpuDomain::tile;
}

// The destination domains an op can write: quant writes the quantized
// side, dequant the f32 side; the external-scale form is Tile-only.
inline constexpr bool qnt_dest_legal(QntOp op, NpuDomain dest) {
  switch (op) {
  case QntOp::quant: return qnt_domain_is_quantized(dest);
  case QntOp::dequant: return !qnt_domain_is_quantized(dest);
  case QntOp::quant_ext_scale: return dest == NpuDomain::tile;
  }
  return false;
}

// MMA_BDOT operations (R4): funct3 bit2 picks matrix multiply (0, mma.*) or
// vector-matrix dot (1, bdot.*); bit1 picks the right-operand orientation
// (nn = 0, nt = 1); bit0 picks zero-fill or accumulate for mma (bdot only
// has the zero form, so funct3 codes 101/111 are reserved).
enum class MmaOp : std::uint8_t {
  nn_zero = 0b000,
  nn_acc = 0b001,
  nt_zero = 0b010,
  nt_acc = 0b011,
  bdot_nn = 0b100,
  bdot_nt = 0b110,
};

// Operand reading: dest = aD (Acc) for mma, vD (Vec32) for bdot; src_a = tA
// (Tile) for mma, bA (Vec8) for bdot; src_b is always the right Tile tB.
struct MmaInstr {
  MmaOp op = MmaOp::nn_zero;
  NpuRegister dest{};
  NpuRegister src_a{};
  NpuRegister src_b{};
};

// Operation metadata for the MMA_BDOT family, mirroring the ISA's funct3
// table: each entry carries the operation and its destination/left-source
// domains. Decode and the structural validation share this table so a new
// operation is added in one place.
struct MmaInfo {
  MmaOp op;
  NpuDomain dest_domain;
  NpuDomain lhs_domain;
};

// Indexed by funct3 code; the reserved codes 101/111 hold nullopt.
inline constexpr std::array<std::optional<MmaInfo>, 8> MMA_BY_FUNCT3 = [] {
  std::array<std::optional<MmaInfo>, 8> table{};
  // clang-format off
  table[0b000] = MmaInfo{MmaOp::nn_zero, NpuDomain::acc  , NpuDomain::tile};
  table[0b001] = MmaInfo{MmaOp::nn_acc , NpuDomain::acc  , NpuDomain::tile};
  table[0b010] = MmaInfo{MmaOp::nt_zero, NpuDomain::acc  , NpuDomain::tile};
  table[0b011] = MmaInfo{MmaOp::nt_acc , NpuDomain::acc  , NpuDomain::tile};
  table[0b100] = MmaInfo{MmaOp::bdot_nn, NpuDomain::vec32, NpuDomain::vec8};
  table[0b110] = MmaInfo{MmaOp::bdot_nt, NpuDomain::vec32, NpuDomain::vec8};
  // clang-format on
  return table;
}();

// SYS operations (Z format): no explicit operands. funct3 codes 100-111 are
// reserved.
enum class SysOp : std::uint8_t {
  fence_mem = 0b000,
  fence_sa = 0b001,
  fence_all = 0b010,
  kernel_end = 0b011,
};

struct SysInstr {
  SysOp op = SysOp::fence_mem;
};

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
// `field` is a defined field code for reg_type; copy requires zero.
// Invalid field values fail with CFG_ERROR at execution.
struct CfgInstr {
  CfgOp op = CfgOp::seti;
  ConfigRegType reg_type = ConfigRegType::tc;
  RegisterIndex dest = 0;
  RegisterIndex src = 0;
  std::uint8_t field = 0;
  std::int32_t imm = 0;
};

} // namespace tile_npu
