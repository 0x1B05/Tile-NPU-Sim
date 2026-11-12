#include "framework.h"
#include "uarch/components/npu_inst_buffer.h"

#include <cstdint>

using namespace tile_npu::test;

TEST_CASE(npu_inst_buffer_is_bounded_fifo) {
  tile_npu::NpuInstBuffer buffer(2);

  const tile_npu::NpuDispatchRequest first{.pc = 1};
  const tile_npu::NpuDispatchRequest second{.pc = 2};
  const tile_npu::NpuDispatchRequest third{.pc = 3};

  check(buffer.empty(), "new buffer is empty");
  check(buffer.enqueue(first), "first request fits");
  check(buffer.enqueue(second), "second request fits");
  check(buffer.full(), "buffer reaches capacity");
  check(!buffer.enqueue(third), "full buffer rejects a request");
  check_eq(buffer.size(), std::size_t{2}, "buffer size");

  check_eq(buffer.front().pc, std::uint64_t{1}, "fifo head");
  buffer.pop();
  check_eq(buffer.front().pc, std::uint64_t{2}, "fifo advances");
  check(buffer.enqueue(third), "space is reusable");

  buffer.clear();
  check(buffer.empty(), "clear empties the buffer");
}

TEST_CASE(npu_inst_buffer_reports_scalar_writebacks) {
  tile_npu::NpuInstBuffer buffer(4);

  check(!buffer.any_pending(), "empty buffer has no scalar writeback");
  check(buffer.enqueue(tile_npu::NpuDispatchRequest{.pc = 1}), "non-writing request fits");
  check(!buffer.any_pending(), "non-writing request is ignored");

  tile_npu::LaneExtractInstr extract;
  extract.dest.index = 3;
  check(buffer.enqueue(tile_npu::NpuDispatchRequest{.pc = 2, .decoded = extract}),
        "scalar-writing request fits");

  check(buffer.any_pending(), "buffer reports a scalar writeback");
  check(buffer.pending(3), "matching scalar destination");
  check(!buffer.pending(4), "other scalar destination");
  check(!buffer.pending(0), "x0 is never pending");
}
