#include "framework.h"
#include "uarch/components/inst_buffer.h"

#include <cstdint>

using namespace tile_npu::test;

TEST_CASE(inst_buffer_is_bounded_fifo) {
  tile_npu::InstBuffer buffer(2);

  const tile_npu::InstBuffer::Entry first{.pc = 1};
  const tile_npu::InstBuffer::Entry second{.pc = 2};
  const tile_npu::InstBuffer::Entry third{.pc = 3};

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
