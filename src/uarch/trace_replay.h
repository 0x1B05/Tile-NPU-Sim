#pragma once

#include "uarch/event_engine.h"
#include "uarch/uarch_trace.h"

#include <string>

namespace tile_npu {

// Replays a UarchTrace through the event engine without running the
// functional simulator. It preserves the same dispatch-window and queue
// backpressure as the online model.
class TraceReplay {
public:
  TraceReplay(UarchConfig config, const std::string &path, bool timeline = false);

  bool run(std::string &error);

  const UarchStats &stats() const { return engine_.stats(); }

  const std::vector<TimelineRecord> &timeline() const { return engine_.timeline(); }

private:
  UarchTraceReader reader_;
  EventUarchEngine engine_;
};

} // namespace tile_npu
