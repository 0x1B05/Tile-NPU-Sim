#include "tile_npu/matrix_view.h"

#include <cstdint>

#include "check.h"

void test_matrix_view() {
  tile_npu::Tile tile{};
  tile_npu::MatrixView<std::uint8_t> view(std::span<std::uint8_t>(tile), 2, 3,
                                          0);
  check(view.valid() && view.contains(1, 2), "valid canonical matrix view");
  view.at(1, 2) = 42;
  check(tile[34] == 42 && view.get(2, 0) == nullptr,
        "matrix view maps logical coordinates");

  tile_npu::MatrixView<std::uint8_t> invalid(std::span<std::uint8_t>(tile), 2,
                                             3, 1);
  check(!invalid.valid() && invalid.get(0, 0) == nullptr,
        "matrix view rejects unsupported layout");
}
