#include "tile_npu/config.h"

#include <cstdint>
#include <variant>

#include "check.h"

TEST_CASE(config_fields) {
  tile_npu::State state{};

  check_eq(tile_npu::write_config_field(state, tile_npu::ConfigBank::tc, 0,
                                        tile_npu::ConfigField::rows,
                                        std::uint64_t{7}),
           tile_npu::ConfigAccessError::none,
           "write TC rows through raw scalar value");
  const auto rows = tile_npu::read_config_field(state, tile_npu::ConfigBank::tc,
                                                0, tile_npu::ConfigField::rows);
  const auto *rows_value = std::get_if<std::uint8_t>(&rows);
  check(rows_value != nullptr, "TC rows reads back as a u8 field");
  check_eq(*rows_value, std::uint8_t{7}, "read TC rows through typed value");

  check_eq(tile_npu::write_config_field(
               state, tile_npu::ConfigBank::tc, 0, tile_npu::ConfigField::dtype,
               tile_npu::ConfigFieldValue{tile_npu::DType::u8}),
           tile_npu::ConfigAccessError::none, "write TC dtype");
  const auto dtype = tile_npu::read_config_field(
      state, tile_npu::ConfigBank::tc, 0, tile_npu::ConfigField::dtype);
  const auto *dtype_value = std::get_if<tile_npu::DType>(&dtype);
  check(dtype_value != nullptr, "TC dtype reads back as a DType field");
  check_eq(*dtype_value, tile_npu::DType::u8, "read TC dtype");

  check_eq(tile_npu::write_config_field(
               state, tile_npu::ConfigBank::tc, 0, tile_npu::ConfigField::dtype,
               tile_npu::ConfigFieldValue{tile_npu::DType::f32}),
           tile_npu::ConfigAccessError::invalid_value,
           "reject dtype from the wrong domain");
  check_eq(state.tc[0].dtype, tile_npu::DType::u8,
           "rejected config write is non-mutating");

  const auto field =
      tile_npu::config_field_from_code(tile_npu::ConfigBank::tm, 2);
  const auto *field_value = std::get_if<tile_npu::ConfigField>(&field);
  check(field_value != nullptr, "TM field code 2 maps to a field");
  check_eq(*field_value, tile_npu::ConfigField::col_stride_bytes,
           "decode TM field code");
  const auto invalid_field =
      tile_npu::config_field_from_code(tile_npu::ConfigBank::ac, 4);
  check(std::get_if<tile_npu::ConfigAccessError>(&invalid_field) != nullptr,
        "reject unsupported AC field code");

  const auto scalar = tile_npu::config_field_as_scalar(
      tile_npu::ConfigFieldValue{std::int32_t{-1}});
  const auto *scalar_value = std::get_if<std::uint64_t>(&scalar);
  check(scalar_value != nullptr, "signed field converts to a scalar");
  check_eq(hex(*scalar_value), hex(UINT64_MAX),
           "preserve signed field bits for cfg.get");
}
