#include "tile_npu/decode.h"

#include "decode_rv64im.h"
#include "encoding.h"
#include "pattern.h"
#include "tile_npu/config.h"
#include "tile_npu/state.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::size_t domain_register_count(NpuDomain domain) {
  switch (domain) {
  case NpuDomain::tile: return TILE_REGISTERS;
  case NpuDomain::acc: return ACC_REGISTERS;
  case NpuDomain::vec8:
  case NpuDomain::vec32: return VECTOR_REGISTERS;
  }
  return 0;
}

// E_BIN (R3): sel3 = {dom2, BX}; BX=0 takes src2 from the same domain, BX=1
// from scalar xS. Unassigned funct4 values and tile mul/div are reserved;
// both come from the shared EBIN_BY_FUNCT4 table. The dtype-dependent
// restrictions (div is f32-only, shifts/bitwise are integer-only) cannot be
// checked here and fail at execution.
DecodeResult decode_ebin(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel3 >> 1);
  const auto &info = EBIN_BY_FUNCT4[f.funct4];
  if (!info || (domain == NpuDomain::tile && !info->tile_allowed)) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src1 >= count) {
    return DecodeError::illegal;
  }
  const bool scalar_src2 = (f.sel3 & 1) != 0;
  if (!scalar_src2 && f.src2 >= count) {
    return DecodeError::illegal;
  }
  return EbinInstr{info->op,
                   {domain, f.dest},
                   {domain, f.src1},
                   scalar_src2 ? NpuOperand{ScalarRegister{f.src2}}
                               : NpuOperand{NpuRegister{domain, f.src2}}};
}

// E_UNARY (R2): sel3 = {dom2, 0}. dom2 = 10 (Vec8) is reserved, funct3 = 111
// is reserved, and on the tile domain only `not` exists (the first pattern
// pins funct3 = 000 for tile and vec8).
DecodeResult decode_eun(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel3 >> 1);
  if (domain == NpuDomain::vec8 || f.funct3 == 0b111) {
    return DecodeError::illegal;
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || f.src2 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{EunInstr{
      static_cast<EunOp>(f.funct3), {domain, f.dest}, {domain, f.src2}}};
}

// M_BLK block/gather forms (R2): sel3 = {dom2, 0}; funct3 = {form2, LS} with
// LS picking load (0) or store (1). S5 names the gather index vector (a Vec32
// register) and the patterns pin it to 0 for the block forms.
DecodeResult decode_mblk(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel3 >> 1);
  const bool gather = f.funct3 == 0b010;
  // Gather is defined for Tile/Vec8/Vec32; there is no aload.gather.
  if (gather && domain == NpuDomain::acc) {
    return DecodeError::illegal;
  }
  if (f.dest >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  const auto op = gather                ? MblkOp::gather
                  : (f.funct3 & 1) != 0 ? MblkOp::store
                                        : MblkOp::load;
  const auto gather_src = gather ? f.src2 : RegisterIndex{0};
  return DecodedNpuInstr{MblkInstr{op, {domain, f.dest}, 0, gather_src, false}};
}

// M_BLK row forms (MR): sel3 = {TA, 0, IS}; IS=0 takes the row number from
// index5, IS=1 from the Scalar register index5 names.
DecodeResult decode_mblk_row(std::uint32_t word) {
  const auto f = encoding::NpuMR::from(word);
  const auto domain = (f.sel3 & 0b100) != 0 ? NpuDomain::acc : NpuDomain::tile;
  if (f.reg >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  const auto op = (f.funct3 & 1) != 0 ? MblkOp::row_store : MblkOp::row_load;
  return DecodedNpuInstr{
      MblkInstr{op, {domain, f.reg}, f.index, 0, (f.sel3 & 1) != 0}};
}

// INIT_MOVE fill (I format): the sel2/field3 positions carry dom2/funct3;
// the patterns pin funct3 = 000.
DecodeResult decode_init_fill(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel2);
  if (f.dest >= domain_register_count(domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{InitInstr{InitOp::fill, {domain, f.dest}, 0, f.imm}};
}

// INIT_MOVE R2 forms: fillx (S5 = xS), copy (S5 = same-domain source) and
// ttranspose (Tile only), distinguished by funct3.
DecodeResult decode_init_move(std::uint32_t word) {
  const auto f = encoding::NpuR2::from(word);
  const auto domain = static_cast<NpuDomain>(f.sel3 >> 1);
  if (f.funct3 == 0b011 && domain != NpuDomain::tile) {
    return DecodeError::illegal; // ttranspose is Tile-only
  }
  const auto count = domain_register_count(domain);
  if (f.dest >= count || (f.funct3 != 0b001 && f.src2 >= count)) {
    return DecodeError::illegal;
  }
  const auto op = f.funct3 == 0b001   ? InitOp::fillx
                  : f.funct3 == 0b010 ? InitOp::copy
                                      : InitOp::transpose;
  return DecodedNpuInstr{InitInstr{op, {domain, f.dest}, f.src2, 0}};
}

// ROW_MOVE (R4): sel3 = {EW, 0, IS}; EW pairs Tile <-> Vec8 (0) or
// Acc <-> Vec32 (1). The vector side has 32 registers, so only the matrix
// register number needs a bounds check.
DecodeResult decode_row_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto matrix_domain =
      (f.sel3 & 0b100) != 0 ? NpuDomain::acc : NpuDomain::tile;
  const auto vector_domain =
      (f.sel3 & 0b100) != 0 ? NpuDomain::vec32 : NpuDomain::vec8;
  const bool insert = f.funct3 == 0b000;
  const auto matrix_index = insert ? f.dest : f.src1;
  if (matrix_index >= domain_register_count(matrix_domain)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{RowInstr{insert ? RowOp::insert : RowOp::extract,
                                  {matrix_domain, matrix_index},
                                  {vector_domain, insert ? f.src1 : f.dest},
                                  f.src2,
                                  (f.sel3 & 1) != 0}};
}

// LANE_MOVE (R4): sel3 = {EW, 0, IS}; EW picks Vec8 (0) or Vec32 (1). Both
// sides take any 5-bit register number.
DecodeResult decode_lane_move(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto domain =
      (f.sel3 & 0b100) != 0 ? NpuDomain::vec32 : NpuDomain::vec8;
  const auto op = f.funct3 == 0b000   ? LaneOp::extract
                  : f.funct3 == 0b001 ? LaneOp::insert
                                      : LaneOp::broadcast;
  auto dst = op == LaneOp::extract ? NpuOperand{ScalarRegister{f.dest}}
                                   : NpuOperand{NpuRegister{domain, f.dest}};
  auto src = op == LaneOp::insert ? NpuOperand{ScalarRegister{f.src1}}
                                  : NpuOperand{NpuRegister{domain, f.src1}};
  return DecodedNpuInstr{LaneInstr{op, dst, src, f.src2, (f.sel3 & 1) != 0}};
}

// sel2 selects the register family pair, the vector-side bit picks within it:
// 00 -> TC/BC, 01 -> AC/VC, 10 -> TM/BM, 11 -> AM/VM.
constexpr ConfigRegType cfg_reg_type(std::uint8_t sel2, bool vector_side) {
  switch (sel2) {
  case 0: return vector_side ? ConfigRegType::bc : ConfigRegType::tc;
  case 1: return vector_side ? ConfigRegType::vc : ConfigRegType::ac;
  case 2: return vector_side ? ConfigRegType::bm : ConfigRegType::tm;
  default: return vector_side ? ConfigRegType::vm : ConfigRegType::am;
  }
}

DecodeResult decode_cfg_seti(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto reg_type = cfg_reg_type(f.sel2, f.major4 == 0b0110);
  // Out-of-range config register numbers are reserved encodings.
  if (f.dest >= config_register_count(reg_type)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{
      CfgInstr{CfgOp::seti, reg_type, f.dest, 0, f.field3, f.imm}};
}

template <CfgOp Op> DecodeResult decode_cfg_reg(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3 >> 1, (f.sel3 & 1) != 0);
  const auto count = config_register_count(reg_type);
  // Scalar operands (xS in setx, xD in get) always name a valid register;
  // config operands are bounded per register type.
  if (Op != CfgOp::get && f.dest >= count) {
    return DecodeError::illegal;
  }
  if (Op != CfgOp::setx && f.src1 >= count) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CfgInstr{Op, reg_type, f.dest, f.src1,
                                  static_cast<std::uint8_t>(f.arg & 0x7), 0}};
}

// clang-format off

// Pattern field groups mirror the family's format:
//   R3: major4 sel3 funct4 reserved4 src2 src1 dest q
//   I:  major4 sel2 field3 imm16(4x4)  C5   q
//   R4: major4 sel3 funct3 arg5    src2 src1 dest q
//   R2: major4 sel3 funct3 reserved10  S5   D5  q
//   MR: major4 sel3 funct3 reserved5 index5 reserved5 reg5 q
// E_BIN reads sel3 as {dom2, BX}: dom tile=00 acc=01 vec8=10 vec32=11.
constexpr DecodeEntry NPU_TABLE[] = {
    {"0000 00? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // tile
    {"0000 01? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // acc
    {"0000 10? ???? 0000 ????? ????? ????? 00", fail<DecodeError::illegal>},          // vec8 has no E_BIN
    {"0000 11? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // vec32
    {"0000 ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::illegal>},          // reserved4 != 0
    // E_UNARY (R2): sel3 = {dom2, 0}; `not` on any matrix/vec32 domain, the
    // rest of the ops on Acc/Vec32 only (patterns pin funct3 = 000 for
    // dom2 x0, i.e. Tile/Vec8).
    {"0001 ??0 000 0000000000 ????? ????? 00", decode_eun},                           // not
    {"0001 ?10 ??? 0000000000 ????? ????? 00", decode_eun},                           // Acc/Vec32 ops
    {"0001 ??? ??? ????? ????? ????? ????? 00", fail<DecodeError::illegal>},          // reserved E_UNARY encodings
    {"???? ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::not_implemented>},  // other elementwise majors

    // M_BLK (R2): sel3 = {dom2, 0}; S5 is the gather index vector, 0 for the
    // block forms. funct3 = {form2, LS}: 00x block, 010 gather.
    {"0000 ??0 000 0000000000 00000 ????? 01", decode_mblk},                          // block load
    {"0000 ??0 001 0000000000 00000 ????? 01", decode_mblk},                          // block store
    {"0000 ??0 010 0000000000 ????? ????? 01", decode_mblk},                          // gather load
    // M_BLK row forms (MR): sel3 = {TA, 0, IS}, funct3 = 10x.
    {"0000 ?0? 100 00000 ????? 00000 ????? 01", decode_mblk_row},                     // row load
    {"0000 ?0? 101 00000 ????? 00000 ????? 01", decode_mblk_row},                     // row store
    {"0000 ??? ??? ????? ????? ????? ????? 01", fail<DecodeError::illegal>},          // reserved M_BLK encodings
    // INIT_MOVE R2 forms: sel3 = {dom2, 0}, funct3 = fillx/copy/ttranspose.
    // They come before fill: an R2 word (reserved10 = 0) is also a valid
    // fill word with imm16 in [0x8000, 0x801F], and the R2 reading wins —
    // that immediate range is shadowed for fill (see the ISA manual).
    {"0010 ??0 001 0000000000 ????? ????? 01", decode_init_move},                     // fillx
    {"0010 ??0 010 0000000000 ????? ????? 01", decode_init_move},                     // copy
    {"0010 ??0 011 0000000000 ????? ????? 01", decode_init_move},                     // ttranspose
    // INIT_MOVE fill (I format): dom2 + funct3 = 000 + imm16; other funct3
    // values in the I layout are reserved.
    {"0010 ?? 000 ???? ???? ???? ???? ????? 01", decode_init_fill},                   // fill
    {"0010 ??? ??? ????? ????? ????? ????? 01", fail<DecodeError::illegal>},          // reserved INIT_MOVE encodings
    // ROW_MOVE (R4): sel3 = {EW, 0, IS}, funct3 = insert.row/extract.row.
    {"0011 ?0? 000 00000 ????? ????? ????? 01", decode_row_move},                     // insert.row
    {"0011 ?0? 001 00000 ????? ????? ????? 01", decode_row_move},                     // extract.row
    {"0011 ??? ??? ????? ????? ????? ????? 01", fail<DecodeError::illegal>},          // reserved ROW_MOVE encodings
    // LANE_MOVE (R4): sel3 = {EW, 0, IS}, funct3 = extract/insert/broadcast.
    {"0100 ?0? 000 00000 ????? ????? ????? 01", decode_lane_move},                    // extract
    {"0100 ?0? 001 00000 ????? ????? ????? 01", decode_lane_move},                    // insert
    {"0100 ?0? 010 00000 ????? ????? ????? 01", decode_lane_move},                    // broadcast
    {"0100 ??? ??? ????? ????? ????? ????? 01", fail<DecodeError::illegal>},          // reserved LANE_MOVE encodings
    {"???? ??? ???? ???? ????? ????? ????? 01", fail<DecodeError::not_implemented>},  // other data movement majors
    // CFG_MAT_SETI / CFG_VEC_SETI (I format); field3 = 111 is reserved.
    {"0101 ?? 111 ???? ???? ???? ???? ????? 10", fail<DecodeError::illegal>},
    {"0110 ?? 111 ???? ???? ???? ???? ????? 10", fail<DecodeError::illegal>},
    {"0101 ?? ??? ???? ???? ???? ???? ????? 10", decode_cfg_seti},
    {"0110 ?? ??? ???? ???? ???? ???? ????? 10", decode_cfg_seti},

    // CFG_REG (R4): funct3 000/001/010 = setx/copy/get; field5 high bits must
    // be 0, field code 111 is reserved; copy requires field5 = 0; the src2
    // position (reserved5) must be 0. Everything else is reserved.
    {"0111 ??? 000 00111 00000 ????? ????? 10", fail<DecodeError::illegal>},
    {"0111 ??? 010 00111 00000 ????? ????? 10", fail<DecodeError::illegal>},
    {"0111 ??? 000 00??? 00000 ????? ????? 10", decode_cfg_reg<CfgOp::setx>},
    {"0111 ??? 001 00000 00000 ????? ????? 10", decode_cfg_reg<CfgOp::copy>},
    {"0111 ??? 010 00??? 00000 ????? ????? 10", decode_cfg_reg<CfgOp::get>},
    {"0111 ??? ??? ????? ????? ????? ????? 10", fail<DecodeError::illegal>},

    {"???? ??? ???? ???? ????? ????? ????? 10", fail<DecodeError::not_implemented>},  // other cross-domain/control
};

// clang-format on

} // namespace

DecodeResult decode(std::uint32_t word) {
  if (encoding::field<1, 0>(word) == 0b11) {
    return decode_rv64im(word);
  }
  return match(NPU_TABLE, word);
}

} // namespace tile_npu
