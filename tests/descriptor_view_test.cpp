#include "tile_npu/descriptor_view.h"

#include <cstdint>
#include <limits>

#include "framework.h"

TEST_CASE(descriptor_view) {
  tile_npu::MatrixDescriptor matrix{};
  matrix.base_addr = 0x1000;
  matrix.row_stride_bytes = 64;
  matrix.col_stride_bytes = 2;
  matrix.flags = 1;
  tile_npu::MatrixDescView matrix_view(matrix);
  check(matrix_view.valid(), "matrix descriptor with increment mode is valid");
  check_eq(hex(matrix_view.address(1, 2).value_or(0)), hex(0x1044),
           "matrix descriptor address calculation");
  check(matrix_view.post_inc(2, 3), "matrix descriptor row increment");
  check_eq(hex(matrix.base_addr), hex(0x1080), "matrix descriptor increments base address");

  tile_npu::VectorDescriptor vector{};
  vector.base_addr = 0x2000;
  vector.stride_bytes = -4;
  vector.flags = 1;
  tile_npu::VectorDescView vector_view(vector);
  check(vector_view.valid(), "vector descriptor with increment mode is valid");
  check_eq(hex(vector_view.address(2).value_or(0)), hex(0x1ff8), "vector descriptor signed stride");
  check(vector_view.post_inc(4), "vector descriptor increment");
  check_eq(hex(vector.base_addr), hex(0x1ff0), "vector descriptor decrements base address");

  matrix.flags = 3;
  check(!matrix_view.valid(), "descriptor view rejects reserved inc mode");
  check(!matrix_view.post_inc(1, 1),
        "descriptor view rejects increment with reserved inc mode");

  tile_npu::MatrixDescriptor overflow{};
  overflow.base_addr = std::numeric_limits<std::uint64_t>::max();
  overflow.col_stride_bytes = 1;
  tile_npu::ConstMatrixDescView overflow_view(overflow);
  check(!overflow_view.address(0, 1), "descriptor view rejects address overflow");

  tile_npu::VectorDescriptor underflow{};
  underflow.base_addr = 0;
  underflow.stride_bytes = -1;
  tile_npu::ConstVectorDescView underflow_view(underflow);
  check(!underflow_view.address(1), "descriptor view rejects address underflow");
}
