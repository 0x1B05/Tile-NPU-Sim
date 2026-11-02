#ifdef CONFIG_TRACE

#include "framework.h"
#include "trace/window.h"

TEST_CASE(trace_window_bounds) {
  using tile_npu::trace::TraceWindow;

  const TraceWindow window{0x80000010, 0x80000020};
  check(!window.contains(0x8000000f), "below start");
  check(window.contains(0x80000010), "start is inclusive");
  check(window.contains(0x8000001f), "last traced PC");
  check(!window.contains(0x80000020), "end is exclusive");

  const TraceWindow unbounded{0x80000010, 0};
  check(!unbounded.contains(0x8000000f), "unbounded below start");
  check(unbounded.contains(0x80000010), "unbounded includes start");
  check(unbounded.contains(0x80001000), "unbounded has no end");
}

#endif // CONFIG_TRACE
