#include "tile_npu/config.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <variant>

namespace {

void check(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

} // namespace

void test_config_fields() {
  tile_npu::State state{};

  state.tc[0].rows = 7;
  const auto rows = tile_npu::read_config_field(state, tile_npu::ConfigBank::tc,
                                                0, tile_npu::ConfigField::rows);
  const auto *rows_value = std::get_if<std::uint8_t>(&rows);
  check(rows_value && *rows_value == 7, "read TC rows through typed value");

  state.tc[0].dtype = tile_npu::DType::u8;
  const auto dtype = tile_npu::read_config_field(
      state, tile_npu::ConfigBank::tc, 0, tile_npu::ConfigField::dtype);
  check(std::get_if<tile_npu::DType>(&dtype) != nullptr &&
            std::get<tile_npu::DType>(dtype) == tile_npu::DType::u8,
        "read TC dtype");

  const auto field =
      tile_npu::config_field_from_code(tile_npu::ConfigBank::tm, 2);
  check(std::get_if<tile_npu::ConfigField>(&field) != nullptr &&
            std::get<tile_npu::ConfigField>(field) ==
                tile_npu::ConfigField::col_stride_bytes,
        "decode TM field code");
  const auto invalid_field =
      tile_npu::config_field_from_code(tile_npu::ConfigBank::ac, 4);
  check(std::get_if<tile_npu::ConfigAccessError>(&invalid_field) != nullptr,
        "reject unsupported AC field code");

  const auto scalar = tile_npu::config_field_as_scalar(
      tile_npu::ConfigFieldValue{std::int32_t{-1}});
  check(std::get_if<std::uint64_t>(&scalar) != nullptr &&
            std::get<std::uint64_t>(scalar) == UINT64_MAX,
        "preserve signed field bits for cfg.get");
}
