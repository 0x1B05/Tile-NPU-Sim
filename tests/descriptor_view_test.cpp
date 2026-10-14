#include "execute/descriptor_view.h"

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
  check(matrix_view.is_valid(), "matrix descriptor with increment mode is valid");
  check_eq(hex(matrix_view.matrix_element_addr(1, 2).value_or(0)), hex(0x1044),
           "matrix descriptor address calculation");
  const auto matrix_next = matrix_view.incremented_base(2, 3);
  check(matrix_next.has_value(), "matrix descriptor row increment");
  matrix_view.commit_base(*matrix_next);
  check_eq(hex(matrix.base_addr), hex(0x1080), "matrix descriptor increments base address");

  tile_npu::VectorDescriptor vector{};
  vector.base_addr = 0x2000;
  vector.stride_bytes = -4;
  vector.flags = 1;
  tile_npu::VectorDescView vector_view(vector);
  check(vector_view.is_valid(), "vector descriptor with increment mode is valid");
  check_eq(hex(vector_view.vector_element_addr(2).value_or(0)), hex(0x1ff8),
           "vector descriptor signed stride");
  const auto vector_next = vector_view.incremented_base(4);
  check(vector_next.has_value(), "vector descriptor increment");
  vector_view.commit_base(*vector_next);
  check_eq(hex(vector.base_addr), hex(0x1ff0), "vector descriptor decrements base address");

  matrix.flags = 3;
  check(!matrix_view.is_valid(), "descriptor view rejects reserved inc mode");

  tile_npu::MatrixDescriptor overflow{};
  overflow.base_addr = std::numeric_limits<std::uint64_t>::max();
  overflow.col_stride_bytes = 1;
  tile_npu::MatrixDescView overflow_view(overflow);
  check(!overflow_view.matrix_element_addr(0, 1), "descriptor view rejects address overflow");

  tile_npu::VectorDescriptor underflow{};
  underflow.base_addr = 0;
  underflow.stride_bytes = -1;
  tile_npu::VectorDescView underflow_view(underflow);
  check(!underflow_view.vector_element_addr(1), "descriptor view rejects address underflow");
}
