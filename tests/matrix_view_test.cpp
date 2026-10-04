#include "tile_npu/matrix_view.h"

#include <cstdint>

#include "check.h"

TEST_CASE(matrix_view) {
  tile_npu::Tile tile{};
  tile_npu::MatrixView<std::uint8_t> view(std::span<std::uint8_t>(tile), 2, 3,
                                          0);
  check(view.valid(), "canonical layout is supported");
  check(view.contains(1, 2), "coordinate inside shape");
  view.at(1, 2) = 42;
  check_eq(tile[34], std::uint8_t{42},
           "matrix view maps logical coordinates row-major");
  check(view.get(2, 0) == nullptr, "row outside shape has no element");

  tile_npu::MatrixView<std::uint8_t> invalid(std::span<std::uint8_t>(tile), 2,
                                             3, 1);
  check(!invalid.valid(), "matrix view rejects unsupported layout");
  check(invalid.get(0, 0) == nullptr, "invalid view yields no element");
}
