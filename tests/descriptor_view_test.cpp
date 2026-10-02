#include "tile_npu/descriptor_view.h"

#include <cstdint>
#include <limits>

#include "check.h"

void test_descriptor_view() {
  tile_npu::MatrixDescriptor matrix{};
  matrix.base_addr = 0x1000;
  matrix.row_stride_bytes = 64;
  matrix.col_stride_bytes = 2;
  matrix.flags = 1;
  tile_npu::MatrixDescriptorView matrix_view(matrix);
  check(matrix_view.valid() && matrix_view.address(1, 2) == 0x1044,
        "matrix descriptor address calculation");
  check(matrix_view.post_increment(2, 3) && matrix.base_addr == 0x1080,
        "matrix descriptor row increment");

  tile_npu::VectorDescriptor vector{};
  vector.base_addr = 0x2000;
  vector.stride_bytes = -4;
  vector.flags = 1;
  tile_npu::VectorDescriptorView vector_view(vector);
  check(vector_view.valid() && vector_view.address(2) == 0x1ff8,
        "vector descriptor signed stride");
  check(vector_view.post_increment(4) && vector.base_addr == 0x1ff0,
        "vector descriptor increment");

  matrix.flags = 3;
  check(!matrix_view.valid() && !matrix_view.post_increment(1, 1),
        "descriptor view rejects reserved increment mode");

  tile_npu::MatrixDescriptor overflow{};
  overflow.base_addr = std::numeric_limits<std::uint64_t>::max();
  overflow.col_stride_bytes = 1;
  tile_npu::ConstMatrixDescriptorView overflow_view(overflow);
  check(!overflow_view.address(0, 1),
        "descriptor view rejects address overflow");

  tile_npu::VectorDescriptor underflow{};
  underflow.base_addr = 0;
  underflow.stride_bytes = -1;
  tile_npu::ConstVectorDescriptorView underflow_view(underflow);
  check(!underflow_view.address(1),
        "descriptor view rejects address underflow");
}
