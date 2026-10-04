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
    {"0000 00? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // tile
    {"0000 01? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // acc
    {"0000 10? ???? 0000 ????? ????? ????? 00", fail<DecodeError::illegal>},          // vec8 has no E_BIN
    {"0000 11? ???? 0000 ????? ????? ????? 00", decode_ebin},                         // vec32
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
