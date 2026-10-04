#include "tile_npu/decode.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <variant>

#include "check.h"

namespace {

using tile_npu::CfgInstr;
using tile_npu::CfgOp;
using tile_npu::ConfigRegType;
using tile_npu::DecodedNpuInstr;
using tile_npu::DecodeError;

// I format (cfg.seti): major4 sel2 field3 imm16 C5 q=10.
constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2,
                              std::uint32_t field3, std::int32_t imm,
                              std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

// R4 format (CFG_REG, major4 = 0111): sel2 MV funct3 field5 reserved5 S5 D5 q.
constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t mv,
                                std::uint32_t funct3, std::uint32_t field5,
                                std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (mv << 25) | (funct3 << 22) |
         (field5 << 17) | (s5 << 7) | (d5 << 2) | 0x2u;
}

struct Case {
  const char *name;
  std::uint32_t word;
  CfgOp op;
  ConfigRegType reg_type;
  std::uint8_t dest;
  std::uint8_t src;
  std::uint8_t field;
  std::int32_t imm;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *npu = std::get_if<DecodedNpuInstr>(&result);
    const std::string what =
        std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    check(npu != nullptr, what);
    const auto *decoded = std::get_if<CfgInstr>(npu);
    check(decoded != nullptr, what);
    check_eq(decoded->op, test.op, what + ": op");
    check_eq(decoded->reg_type, test.reg_type, what + ": reg_type");
    check_eq(decoded->dest, test.dest, what + ": dest");
    check_eq(decoded->src, test.src, what + ": src");
    check_eq(decoded->field, test.field, what + ": field");
    check_eq(decoded->imm, test.imm, what + ": imm");
  }
}

struct ErrorCase {
  const char *name;
  std::uint32_t word;
  DecodeError error;
};

template <std::size_t N>
void check_errors(const std::array<ErrorCase, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const auto *error = std::get_if<DecodeError>(&result);
    const std::string what =
        std::string(test.name) + " (" + to_hex(test.word) + ")";
    check(error != nullptr, what + " decodes");
    check_eq(*error, test.error, what + ": error kind");
  }
}

} // namespace

TEST_CASE(cfg_decode_seti) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{"cfg.seti tc0, rows, 32", 0x50801002u, CfgOp::seti,
           ConfigRegType::tc, 0, 0, 1, 32},
      Case{"seti ac cols", cfg_i(0x5, 1, 2, 16, 3), CfgOp::seti,
           ConfigRegType::ac, 3, 0, 2, 16},
      Case{"seti negative imm", cfg_i(0x5, 2, 1, -4, 0), CfgOp::seti,
           ConfigRegType::tm, 0, 0, 1, -4},
      Case{"seti imm min", cfg_i(0x5, 0, 1, -32768, 0), CfgOp::seti,
           ConfigRegType::tc, 0, 0, 1, -32768},
      Case{"seti imm max", cfg_i(0x5, 0, 1, 32767, 0), CfgOp::seti,
           ConfigRegType::tc, 0, 0, 1, 32767},
      Case{"seti vm31 flags (vector side)", cfg_i(0x6, 3, 2, 1, 31),
           CfgOp::seti, ConfigRegType::vm, 31, 0, 2, 1},
      Case{"seti tc15 (last tile)", cfg_i(0x5, 0, 0, 1, 15), CfgOp::seti,
           ConfigRegType::tc, 15, 0, 0, 1},
      Case{"seti ac11 (last acc)", cfg_i(0x5, 1, 0, 1, 11), CfgOp::seti,
           ConfigRegType::ac, 11, 0, 0, 1},
  };
  check_cases(cases);
}

TEST_CASE(cfg_decode_reg) {
  constexpr std::array cases{
      Case{"cfg.setx tm2, row_stride_bytes, x4", cfg_reg(2, 0, 0, 1, 4, 2),
           CfgOp::setx, ConfigRegType::tm, 2, 4, 1, 0},
      Case{"cfg.setx vc0, len, x31", cfg_reg(1, 1, 0, 1, 31, 0), CfgOp::setx,
           ConfigRegType::vc, 0, 31, 1, 0},
      Case{"cfg.copy tc7, tc3", cfg_reg(0, 0, 1, 0, 3, 7), CfgOp::copy,
           ConfigRegType::tc, 7, 3, 0, 0},
      Case{"cfg.copy bm5, bm2 (vector side)", cfg_reg(2, 1, 1, 0, 2, 5),
           CfgOp::copy, ConfigRegType::bm, 5, 2, 0, 0},
      Case{"cfg.get x5, tc0, rows", cfg_reg(0, 0, 2, 1, 0, 5), CfgOp::get,
           ConfigRegType::tc, 5, 0, 1, 0},
      Case{"cfg.get x0, am11, base_addr", cfg_reg(3, 0, 2, 0, 11, 0),
           CfgOp::get, ConfigRegType::am, 0, 11, 0, 0},
  };
  check_cases(cases);
}

TEST_CASE(cfg_decode_rejects_reserved) {
  const std::array cases{
      ErrorCase{"seti field3 = 111 reserved", cfg_i(0x5, 0, 7, 0, 0),
                DecodeError::illegal},
      ErrorCase{"seti tc16 out of range", cfg_i(0x5, 0, 0, 0, 16),
                DecodeError::illegal},
      ErrorCase{"seti ac12 out of range", cfg_i(0x5, 1, 0, 0, 12),
                DecodeError::illegal},
      ErrorCase{"seti am12 out of range", cfg_i(0x5, 3, 0, 0, 12),
                DecodeError::illegal},
      ErrorCase{"setx field code 111 reserved", cfg_reg(0, 0, 0, 7, 4, 0),
                DecodeError::illegal},
      ErrorCase{"get field code 111 reserved", cfg_reg(0, 0, 2, 7, 0, 5),
                DecodeError::illegal},
      ErrorCase{"setx field5 high bits set", cfg_reg(0, 0, 0, 8, 4, 0),
                DecodeError::illegal},
      ErrorCase{"copy field5 nonzero", cfg_reg(0, 0, 1, 1, 3, 7),
                DecodeError::illegal},
      ErrorCase{"reg reserved5 nonzero", cfg_reg(0, 0, 0, 1, 4, 0) | (1u << 12),
                DecodeError::illegal},
      ErrorCase{"funct3 = 011 reserved", cfg_reg(0, 0, 3, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"funct3 = 111 reserved", cfg_reg(0, 0, 7, 0, 0, 0),
                DecodeError::illegal},
      ErrorCase{"setx tm16 out of range", cfg_reg(2, 0, 0, 0, 4, 16),
                DecodeError::illegal},
      ErrorCase{"copy tc16 dest out of range", cfg_reg(0, 0, 1, 0, 0, 16),
                DecodeError::illegal},
      ErrorCase{"copy tc16 src out of range", cfg_reg(0, 0, 1, 0, 16, 0),
                DecodeError::illegal},
      ErrorCase{"get ac12 out of range", cfg_reg(1, 0, 2, 0, 12, 5),
                DecodeError::illegal},
      // Defined but not yet decoded families report not_implemented.
      ErrorCase{"sys family", (0x8u << 28) | 0x2u,
                DecodeError::not_implemented},
      ErrorCase{"reserved major4 1001", (0x9u << 28) | 0x2u,
                DecodeError::not_implemented},
  };
  check_errors(cases);
}
