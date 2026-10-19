#include "tile_npu/decode.h"

#include "config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>

#include "framework.h"
#include "support/decode_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::CfgCopyInstr;
using tile_npu::CfgGetInstr;
using tile_npu::CfgGetwInstr;
using tile_npu::CfgSetiInstr;
using tile_npu::CfgSetwInstr;
using tile_npu::CfgSetxInstr;
using tile_npu::ConfigField;
using tile_npu::ConfigRegType;

// I format (cfg.seti): major4 sel2 field3 imm16 C5 q=10.
constexpr std::uint32_t cfg_i(std::uint32_t major4, std::uint32_t sel2, std::uint32_t field3,
                              std::int32_t imm, std::uint32_t c5) {
  return (major4 << 28) | (sel2 << 26) | (field3 << 23) |
         ((static_cast<std::uint32_t>(imm) & 0xffffu) << 7) | (c5 << 2) | 0x2u;
}

// R4 format (CFG_REG, major4 = 0111): sel2 MV funct3 field5 N5 S5 D5 q; N5
// ([16:12]) is the last destination offset N for writes and zero for reads.
constexpr std::uint32_t cfg_reg(std::uint32_t sel2, std::uint32_t mv, std::uint32_t funct3,
                                std::uint32_t field5, std::uint32_t s5, std::uint32_t d5) {
  return (0x7u << 28) | (sel2 << 26) | (mv << 25) | (funct3 << 22) | (field5 << 17) | (s5 << 7) |
         (d5 << 2) | 0x2u;
}

struct Case {
  const char *name;
  std::uint32_t word;
  enum class Kind { seti, setx, copy, get, setw, getw } kind;
  ConfigRegType reg_type;
  std::uint8_t dest;
  std::uint8_t src;
  std::optional<ConfigField> field;
  std::int32_t imm;
  std::uint8_t n = 0;
};

template <std::size_t N> void check_cases(const std::array<Case, N> &cases) {
  for (const auto &test : cases) {
    const auto result = tile_npu::decode(test.word);
    const std::string what = std::string("decode ") + test.name + " (" + to_hex(test.word) + ")";
    const auto check_common = [&](auto &decoded) {
      check_eq(decoded.reg_type, test.reg_type, what + ": reg_type");
    };
    switch (test.kind) {
    case Case::Kind::seti: {
      const auto &decoded = expect_npu<CfgSetiInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.field, *test.field, what + ": field");
      check_eq(decoded.imm, test.imm, what + ": imm");
      break;
    }
    case Case::Kind::setx: {
      const auto &decoded = expect_npu<CfgSetxInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.src, test.src, what + ": src");
      check_eq(decoded.field, *test.field, what + ": field");
      check_eq(decoded.n, test.n, what + ": N5");
      break;
    }
    case Case::Kind::copy: {
      const auto &decoded = expect_npu<CfgCopyInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.src, test.src, what + ": src");
      check_eq(decoded.n, test.n, what + ": N5");
      break;
    }
    case Case::Kind::get: {
      const auto &decoded = expect_npu<CfgGetInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.src, test.src, what + ": src");
      check_eq(decoded.field, *test.field, what + ": field");
      break;
    }
    case Case::Kind::setw: {
      const auto &decoded = expect_npu<CfgSetwInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.src, test.src, what + ": src");
      check_eq(decoded.n, test.n, what + ": N5");
      break;
    }
    case Case::Kind::getw: {
      const auto &decoded = expect_npu<CfgGetwInstr>(result, what);
      check_common(decoded);
      check_eq(decoded.dest, test.dest, what + ": dest");
      check_eq(decoded.src, test.src, what + ": src");
      break;
    }
    }
  }
}

} // namespace

TEST_CASE(cfg_decode_seti) {
  constexpr std::array cases{
      // Anchor word computed by hand from the encoding figure.
      Case{"cfg.seti tc0, rows, 32", 0x50801002u, Case::Kind::seti, ConfigRegType::tc, 0, 0,
           ConfigField::rows, 32},
      Case{"seti ac cols", cfg_i(0x5, 1, 2, 16, 3), Case::Kind::seti, ConfigRegType::ac, 3, 0,
           ConfigField::cols, 16},
      Case{"seti negative imm", cfg_i(0x5, 2, 1, -4, 0), Case::Kind::seti, ConfigRegType::tm, 0, 0,
           ConfigField::row_stride_bytes, -4},
      Case{"seti imm min", cfg_i(0x5, 0, 1, -32768, 0), Case::Kind::seti, ConfigRegType::tc, 0, 0,
           ConfigField::rows, -32768},
      Case{"seti imm max", cfg_i(0x5, 0, 1, 32767, 0), Case::Kind::seti, ConfigRegType::tc, 0, 0,
           ConfigField::rows, 32767},
      Case{"seti vm31 flags (vector side)", cfg_i(0x6, 3, 2, 1, 31), Case::Kind::seti,
           ConfigRegType::vm, 31, 0, ConfigField::flags, 1},
      Case{"seti tc15 (last tile)", cfg_i(0x5, 0, 0, 1, 15), Case::Kind::seti, ConfigRegType::tc,
           15, 0, ConfigField::dtype, 1},
      Case{"seti ac11 (last acc)", cfg_i(0x5, 1, 0, 1, 11), Case::Kind::seti, ConfigRegType::ac, 11,
           0, ConfigField::dtype, 1},
  };
  check_cases(cases);
}

TEST_CASE(cfg_decode_reg) {
  constexpr std::array cases{
      Case{"cfg.setx tm2, row_stride_bytes, x4", cfg_reg(2, 0, 0, 1, 4, 2), Case::Kind::setx,
           ConfigRegType::tm, 2, 4, ConfigField::row_stride_bytes, 0},
      Case{"cfg.setx vc0, len, x31", cfg_reg(1, 1, 0, 1, 31, 0), Case::Kind::setx,
           ConfigRegType::vc, 0, 31, ConfigField::len, 0},
      Case{"cfg.copy tc7, tc3", cfg_reg(0, 0, 1, 0, 3, 7), Case::Kind::copy, ConfigRegType::tc, 7,
           3, std::nullopt, 0},
      Case{"cfg.copy bm5, bm2 (vector side)", cfg_reg(2, 1, 1, 0, 2, 5), Case::Kind::copy,
           ConfigRegType::bm, 5, 2, std::nullopt, 0},
      Case{"cfg.get x5, tc0, rows", cfg_reg(0, 0, 2, 1, 0, 5), Case::Kind::get, ConfigRegType::tc,
           5, 0, ConfigField::rows, 0},
      Case{"cfg.get x0, am11, base_addr", cfg_reg(3, 0, 2, 0, 11, 0), Case::Kind::get,
           ConfigRegType::am, 0, 11, ConfigField::base_addr, 0},
      // N5 is the last destination offset for the writes.
      Case{"cfg.setx vm0, flags, x6, N=2", cfg_reg(3, 1, 0, 2, 6, 0) | (2u << 12), Case::Kind::setx,
           ConfigRegType::vm, 0, 6, ConfigField::flags, 0, 2},
      Case{"cfg.copy tm1, tm0, N=14", cfg_reg(2, 0, 1, 0, 0, 1) | (14u << 12), Case::Kind::copy,
           ConfigRegType::tm, 1, 0, std::nullopt, 0, 14},
      Case{"cfg.setw tc2, x4, N=2", cfg_reg(0, 0, 3, 0, 4, 2) | (2u << 12), Case::Kind::setw,
           ConfigRegType::tc, 2, 4, std::nullopt, 0, 2},
      Case{"cfg.getw x5, vc0", cfg_reg(1, 1, 4, 0, 0, 5), Case::Kind::getw, ConfigRegType::vc, 5, 0,
           std::nullopt, 0},
  };
  check_cases(cases);
}

TEST_CASE(cfg_decode_field_codes_by_register_type) {
  struct RegisterCase {
    const char *name;
    ConfigRegType type;
    std::uint32_t sel2;
    std::uint32_t mv;
    std::span<const ConfigField> fields;
  };

  constexpr std::array tc_fields{ConfigField::dtype, ConfigField::rows, ConfigField::cols,
                                 ConfigField::layout, ConfigField::arith_mode};
  constexpr std::array ac_fields{ConfigField::dtype, ConfigField::rows, ConfigField::cols,
                                 ConfigField::layout};
  constexpr std::array bc_fields{ConfigField::dtype, ConfigField::len};
  constexpr std::array tm_fields{ConfigField::base_addr, ConfigField::row_stride_bytes,
                                 ConfigField::col_stride_bytes, ConfigField::transform,
                                 ConfigField::flags};
  constexpr std::array bm_fields{ConfigField::base_addr, ConfigField::stride_bytes,
                                 ConfigField::flags};
  const std::array registers{
      RegisterCase{"tc", ConfigRegType::tc, 0, 0, tc_fields},
      RegisterCase{"ac", ConfigRegType::ac, 1, 0, ac_fields},
      RegisterCase{"bc", ConfigRegType::bc, 0, 1, bc_fields},
      RegisterCase{"vc", ConfigRegType::vc, 1, 1, bc_fields},
      RegisterCase{"tm", ConfigRegType::tm, 2, 0, tm_fields},
      RegisterCase{"am", ConfigRegType::am, 3, 0, tm_fields},
      RegisterCase{"bm", ConfigRegType::bm, 2, 1, bm_fields},
      RegisterCase{"vm", ConfigRegType::vm, 3, 1, bm_fields},
  };
  for (const auto &reg : registers) {
    for (std::uint8_t field = 0; field < 8; ++field) {
      if (field < reg.fields.size()) {
        const auto expected = reg.fields[field];
        const std::array cases{
            Case{reg.name, cfg_i(0x5 + reg.mv, reg.sel2, field, 32, 0), Case::Kind::seti, reg.type,
                 0, 0, expected, 32},
            Case{reg.name, cfg_reg(reg.sel2, reg.mv, 0, field, 4, 0), Case::Kind::setx, reg.type, 0,
                 4, expected, 0},
            Case{reg.name, cfg_reg(reg.sel2, reg.mv, 2, field, 0, 5), Case::Kind::get, reg.type, 5,
                 0, expected, 0},
        };
        check_cases(cases);
      } else {
        const std::array cases{
            Case{reg.name, cfg_i(0x5 + reg.mv, reg.sel2, field, 32, 0), Case::Kind::seti, reg.type,
                 0, 0, std::nullopt, 32},
            Case{reg.name, cfg_reg(reg.sel2, reg.mv, 0, field, 4, 0), Case::Kind::setx, reg.type, 0,
                 4, std::nullopt, 0},
            Case{reg.name, cfg_reg(reg.sel2, reg.mv, 2, field, 0, 5), Case::Kind::get, reg.type, 5,
                 0, std::nullopt, 0},
        };
        for (const auto &test : cases) {
          expect_illegal(std::array{DecodeCase{test.name, test.word}});
        }
      }
    }
  }
}

TEST_CASE(cfg_decode_rejects_reserved) {
  const std::array cases{
      DecodeCase{"seti tc16 out of range", cfg_i(0x5, 0, 0, 0, 16)},
      DecodeCase{"seti ac12 out of range", cfg_i(0x5, 1, 0, 0, 12)},
      DecodeCase{"seti am12 out of range", cfg_i(0x5, 3, 0, 0, 12)},
      DecodeCase{"setx field5 high bits set", cfg_reg(0, 0, 0, 8, 4, 0)},
      DecodeCase{"copy field5 nonzero", cfg_reg(0, 0, 1, 1, 3, 7)},
      // N5 is a destination offset for writes and must be 0 for reads.
      DecodeCase{"get N5 nonzero", cfg_reg(0, 0, 2, 0, 0, 5) | (1u << 12)},
      DecodeCase{"setx tc15 + 2", cfg_reg(0, 0, 0, 1, 4, 15) | (1u << 12)},
      DecodeCase{"copy vc31 + 2", cfg_reg(1, 1, 1, 0, 0, 31) | (1u << 12)},
      DecodeCase{"copy am0 + 12", cfg_reg(3, 0, 1, 0, 0, 0) | (11u << 12) | (1u << 2)},
      DecodeCase{"funct3 = 101 reserved", cfg_reg(0, 0, 5, 0, 0, 0)},
      DecodeCase{"funct3 = 111 reserved", cfg_reg(0, 0, 7, 0, 0, 0)},
      // setw/getw: compute configs only, field5 zero, span within the type.
      DecodeCase{"setw on tm", cfg_reg(2, 0, 3, 0, 4, 0)},
      DecodeCase{"getw on vm", cfg_reg(3, 1, 4, 0, 0, 5)},
      DecodeCase{"setw field5 nonzero", cfg_reg(0, 0, 3, 1, 4, 0)},
      DecodeCase{"setw tc15 + 2", cfg_reg(0, 0, 3, 0, 4, 15) | (1u << 12)},
      DecodeCase{"setw ac0 + 12", cfg_reg(1, 0, 3, 0, 4, 0) | (12u << 12)},
      DecodeCase{"getw N5 nonzero", cfg_reg(0, 0, 4, 0, 0, 5) | (1u << 12)},
      DecodeCase{"getw tc16", cfg_reg(0, 0, 4, 0, 16, 5)},
      DecodeCase{"setx tm16 out of range", cfg_reg(2, 0, 0, 0, 4, 16)},
      DecodeCase{"copy tc16 dest out of range", cfg_reg(0, 0, 1, 0, 0, 16)},
      DecodeCase{"copy tc16 src out of range", cfg_reg(0, 0, 1, 0, 16, 0)},
      DecodeCase{"get ac12 out of range", cfg_reg(1, 0, 2, 0, 12, 5)},
      // Reserved major4 encodings report illegal.
      DecodeCase{"reserved major4 1010", (0xau << 28) | 0x2u},
  };
  expect_illegal(cases);
}
