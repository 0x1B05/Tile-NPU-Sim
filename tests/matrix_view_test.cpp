#include "tile_npu/matrix_view.h"

#include <cstdint>
#include <span>
#include <type_traits>
#include <utility>

#include "framework.h"

TEST_CASE(matrix_view) {
  tile_npu::Tile tile{};
  tile_npu::MatrixView view{std::span<std::uint8_t, tile_npu::MATRIX_ELEMENTS>(tile)};
  view.at(1, 2) = 42;
  check_eq(tile[34], std::uint8_t{42}, "matrix view maps logical coordinates row-major");
  tile[5 * tile_npu::MATRIX_EXTENT + 7] = 99;
  check_eq(view.at(5, 7), std::uint8_t{99}, "matrix view reads physical storage row-major");
}

TEST_CASE(matrix_view_factory) {
  tile_npu::Tile tile{};
  static_assert(
      std::is_same_v<decltype(tile_npu::matrix_view(tile)), tile_npu::MatrixView<std::uint8_t>>);
  static_assert(std::is_same_v<decltype(tile_npu::matrix_view(std::as_const(tile))),
                               tile_npu::MatrixView<const std::uint8_t>>);

  auto dst = tile_npu::matrix_view(tile);
  dst.at(0, 1) = 7;
  check_eq(tile[1], std::uint8_t{7}, "the factory builds a writable view over the register");
  const auto src = tile_npu::matrix_view(std::as_const(tile));
  check_eq(src.at(1, 2), std::uint8_t{0},
           "the factory builds a read-only view over a const register");
}
