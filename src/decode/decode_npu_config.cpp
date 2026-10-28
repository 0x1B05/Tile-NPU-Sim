#include "decode_npu.h"

#include "config.h"
#include "encoding.h"

#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

// cfg.seti: major4 selects the compute or memory group, sel2 selects the
// type within that group. sel2 = 11 is reserved.
std::optional<ConfigRegType> cfg_seti_type(std::uint8_t major4, std::uint8_t sel2) {
  const bool compute = major4 == 0b0101;
  const bool memory = major4 == 0b0110;
  if (!compute && !memory) return std::nullopt;
  switch (sel2) {
  case 0: return compute ? ConfigRegType::tc : ConfigRegType::tm;
  case 1: return compute ? ConfigRegType::ac : ConfigRegType::am;
  case 2: return compute ? ConfigRegType::vc : ConfigRegType::vm;
  default: return std::nullopt;
  }
}

// CFG_REG: sel3 = {sel2, CM}; CM = 0 selects compute configs, CM = 1 selects
// memory descriptors, and sel2 = 11 is reserved.
std::optional<ConfigRegType> cfg_reg_type(std::uint8_t sel3) {
  const auto sel2 = static_cast<std::uint8_t>(sel3 >> 1);
  const bool memory = (sel3 & 1) != 0;
  switch (sel2) {
  case 0: return memory ? ConfigRegType::tm : ConfigRegType::tc;
  case 1: return memory ? ConfigRegType::am : ConfigRegType::ac;
  case 2: return memory ? ConfigRegType::vm : ConfigRegType::vc;
  default: return std::nullopt;
  }
}

} // namespace

// CFG_COMPUTE_SETI / CFG_MEM_SETI (I): major4 = 0101 / 0110, q = 10.
// 31    28 27  26 25    23 22                             7 6        2 1   0
// [major4] [sel2] [field3] [            imm16             ] [   C5   ] [ q ]
// sel2: 00 TC/TM, 01 AC/AM, 10 VC/VM; 11 reserved.
DecodeResult decode_cfg_seti(std::uint32_t word) {
  const auto f = encoding::NpuI::from(word);
  const auto reg_type = cfg_seti_type(f.major4, f.sel2);
  if (!reg_type) return DecodeError::illegal;
  const auto field = decode_config_field(*reg_type, f.field3);
  if (!field) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{
      CfgSetiInstr{.reg_type = *reg_type, .dest = f.dest, .field = *field, .imm = f.imm}};
}

// CFG_REG (R4): major4 = 0111, q = 10.
// 31    28 27  26  25  24    22 21      17 16       12 11       7 6        2 1   0
// [major4] [sel2] [CM] [funct3] [ field5 ] [   N5    ] [   S5   ] [   D5   ] [ q ]
// sel3 = {sel2, CM}: sel2 00/01/10 selects TC/AC/VC (CM = 0) or TM/AM/VM
// (CM = 1); sel2 = 11 is reserved.
// funct3: 000 setx (xS -> C), 001 copy (CS -> CD), 010 get (C -> xD).
// N5 is the last destination offset for the writes (setx/copy target
// D5..D5 + N5); get pins it to zero.
DecodeResult decode_cfg_setx(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3);
  if (!reg_type) return DecodeError::illegal;
  const auto register_count = config_register_count(*reg_type);
  if (f.dest + f.src2 >= register_count) {
    return DecodeError::illegal;
  }
  const auto field =
      decode_config_field(*reg_type, static_cast<std::uint8_t>(encoding::field<2, 0>(f.arg)));
  if (!field) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{CfgSetxInstr{
      .reg_type = *reg_type, .dest = f.dest, .field = *field, .src = f.src1, .n = f.src2}};
}

DecodeResult decode_cfg_copy(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3);
  if (!reg_type) return DecodeError::illegal;
  if (f.dest + f.src2 >= config_register_count(*reg_type)) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{
      CfgCopyInstr{.reg_type = *reg_type, .dest = f.dest, .src = f.src1, .n = f.src2}};
}

DecodeResult decode_cfg_get(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3);
  if (!reg_type) return DecodeError::illegal;
  if (f.src2 != 0) return DecodeError::illegal;
  const auto field =
      decode_config_field(*reg_type, static_cast<std::uint8_t>(encoding::field<2, 0>(f.arg)));
  if (!field) return DecodeError::illegal;
  return DecodedNpuInstr{
      CfgGetInstr{.reg_type = *reg_type, .dest = f.dest, .src = f.src1, .field = *field}};
}

// cfg.setw / cfg.getw (CFG_REG R4, funct3 = 011 / 100): whole-word access
// to the compute configs (CM = 0, sel2 = 00/01/10). setw writes xS (S5) into
// D5..D5 + N5, with N5 = [16:12] as the last destination offset; getw reads
// config S5 into xD (D5). field5 is zero for both.
DecodeResult decode_cfg_setw(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3);
  if (!reg_type || !is_compute_config(*reg_type)) return DecodeError::illegal;
  if (f.dest + f.src2 >= config_register_count(*reg_type)) return DecodeError::illegal;
  return DecodedNpuInstr{
      CfgSetwInstr{.reg_type = *reg_type, .dest = f.dest, .src = f.src1, .n = f.src2}};
}

DecodeResult decode_cfg_getw(std::uint32_t word) {
  const auto f = encoding::NpuR4::from(word);
  const auto reg_type = cfg_reg_type(f.sel3);
  if (!reg_type || !is_compute_config(*reg_type)) return DecodeError::illegal;
  return DecodedNpuInstr{CfgGetwInstr{.reg_type = *reg_type, .dest = f.dest, .src = f.src1}};
}

// SYS (Z): major4 = 1000, q = 10.
// 31    28 27    25 24                                           2 1   0
// [major4] [funct3] [                 reserved23                 ] [ q ]
// funct3: 000 fence.mem, 001 fence.sa, 010 fence.all, 011 kernel.end;
//         100..111 reserved.
DecodeResult decode_sys(std::uint32_t word) {
  const auto funct3 = encoding::field<27, 25>(word);
  if (funct3 > 0b011) {
    return DecodeError::illegal;
  }
  return DecodedNpuInstr{SysInstr{.op = static_cast<SysOp>(funct3)}};
}

} // namespace tile_npu
