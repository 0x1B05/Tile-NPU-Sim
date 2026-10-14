#include "config.h"

#include <cstdint>
#include <variant>

#include "framework.h"

namespace {

constexpr std::uint64_t code(tile_npu::DType dtype) { return static_cast<std::uint64_t>(dtype); }

} // namespace

TEST_CASE(config_fields) {
  using tile_npu::ConfigAccessStatus;
  using tile_npu::ConfigField;
  using tile_npu::ConfigRegType;
  tile_npu::State state{};

  check_eq(tile_npu::write_config_field(state, ConfigRegType::tc, 0, ConfigField::rows, 7),
           ConfigAccessStatus::success, "write TC rows");
  check_eq(tile_npu::read_config_field(state, ConfigRegType::tc, 0, ConfigField::rows),
           std::uint64_t{7}, "read TC rows");

  check_eq(tile_npu::write_config_field(state, ConfigRegType::tc, 0, ConfigField::dtype,
                                        code(tile_npu::DType::u8)),
           ConfigAccessStatus::success, "write TC dtype");
  check_eq(state.tc[0].dtype, tile_npu::DType::u8, "TC dtype holds the written enum");
  check_eq(tile_npu::read_config_field(state, ConfigRegType::tc, 0, ConfigField::dtype),
           code(tile_npu::DType::u8), "read TC dtype as its numeric code");

  check_eq(tile_npu::write_config_field(state, ConfigRegType::tc, 0, ConfigField::dtype,
                                        code(tile_npu::DType::f32)),
           ConfigAccessStatus::invalid_value, "reject dtype from the wrong domain");
  check_eq(tile_npu::write_config_field(state, ConfigRegType::tc, 0, ConfigField::dtype, 0x100),
           ConfigAccessStatus::invalid_value, "reject out-of-range dtype code");
  check_eq(state.tc[0].dtype, tile_npu::DType::u8, "rejected config write is non-mutating");

  const auto field = tile_npu::config_field_from_code(ConfigRegType::tm, 2);
  const auto *field_value = std::get_if<ConfigField>(&field);
  check(field_value != nullptr, "TM field code 2 maps to a field");
  check_eq(*field_value, ConfigField::col_stride_bytes, "decode TM field code");
  const auto invalid_field = tile_npu::config_field_from_code(ConfigRegType::ac, 4);
  check(std::get_if<ConfigAccessStatus>(&invalid_field) != nullptr,
        "reject unsupported AC field code");

  check_eq(tile_npu::write_config_field(state, ConfigRegType::tm, 0, ConfigField::row_stride_bytes,
                                        UINT64_MAX),
           ConfigAccessStatus::success, "write negative stride");
  check_eq(
      hex(tile_npu::read_config_field(state, ConfigRegType::tm, 0, ConfigField::row_stride_bytes)),
      hex(UINT64_MAX), "preserve signed field bits for cfg.get");
  check_eq(tile_npu::write_config_field(state, ConfigRegType::tm, 0, ConfigField::row_stride_bytes,
                                        0x80000000u),
           ConfigAccessStatus::invalid_value, "reject stride outside i32");

  check_eq(tile_npu::write_config_field(state, ConfigRegType::tc, 0, ConfigField::rows, 0),
           ConfigAccessStatus::invalid_value, "reject zero TC rows");
  check_eq(state.tc[0].rows, std::uint8_t{7}, "rejected shape write is non-mutating");
  check_eq(tile_npu::write_config_field(state, ConfigRegType::vc, 0, ConfigField::len, 0),
           ConfigAccessStatus::invalid_value, "reject zero VC len");
}
