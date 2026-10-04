#include "tile_npu/decode.h"

#include "decode_rv64im.h"
#include "encoding.h"
#include "pattern.h"
#include "tile_npu/config.h"
#include "tile_npu/state.h"

#include <cstdint>

namespace tile_npu {
namespace {

// E_BIN add on the tile domain with BX=0. The pattern pins sel3, so only the
// register bounds remain to check here.
DecodeResult decode_tadd(std::uint32_t word) {
  const auto f = encoding::NpuR3::from(word);
  if (f.dest >= TILE_REGISTERS || f.src1 >= TILE_REGISTERS ||
      f.src2 >= TILE_REGISTERS) {
    return DecodeError::illegal;
  }
  return EbinInstr{EbinOp::add,
                   {NpuDomain::tile, f.dest},
                   {NpuDomain::tile, f.src1},
                   NpuRegister{NpuDomain::tile, f.src2}};
}

// Defined E_BIN funct4 values other than add are not implemented yet;
// reserved funct4 values are illegal.
DecodeResult decode_binary_other(std::uint32_t word) {
  switch (static_cast<EbinOp>(encoding::field<24, 21>(word))) {
  case EbinOp::sub:
  case EbinOp::mul:
  case EbinOp::div:
  case EbinOp::min:
  case EbinOp::max:
  case EbinOp::bit_and:
  case EbinOp::bit_or:
  case EbinOp::bit_xor:
  case EbinOp::shl:
  case EbinOp::shr:
  case EbinOp::sra: return DecodeError::not_implemented;
  default: return DecodeError::illegal;
  }
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
// E_BIN reads sel3 as {dom2, BX}: dom tile=00 acc=01 vec8=10 vec32=11.
constexpr DecodeEntry NPU_TABLE[] = {
    {"0000 000 0000 0000 ????? ????? ????? 00", decode_tadd},
    {"0000 0?? 0000 0000 ????? ????? ????? 00", fail<DecodeError::not_implemented>}, // taddx, add @acc
    {"0000 10? ???? 0000 ????? ????? ????? 00", fail<DecodeError::illegal>},          // vec8 has no E_BIN
    {"0000 11? 0000 0000 ????? ????? ????? 00", fail<DecodeError::not_implemented>},  // add @vec32
    {"0000 ??? ???? 0000 ????? ????? ????? 00", decode_binary_other},
    {"0000 ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::illegal>},          // reserved4 != 0
    {"???? ??? ???? ???? ????? ????? ????? 00", fail<DecodeError::not_implemented>},  // other elementwise majors
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

    {"???? ??? ???? ???? ????? ????? ????? 01", fail<DecodeError::not_implemented>},  // data movement
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
