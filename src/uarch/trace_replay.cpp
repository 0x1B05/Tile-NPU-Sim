#include "uarch/trace_replay.h"

#include <string>
#include <utility>

namespace tile_npu {

TraceReplay::TraceReplay(UarchConfig config, const std::string &path, bool timeline)
    : reader_(path), engine_(std::move(config)) {
  engine_.set_timeline_enabled(timeline);
  engine_.reset();
}

bool TraceReplay::run(std::string &error) {
  if (!reader_.good()) {
    error = reader_.error();
    return false;
  }

  while (true) {
    DynInst inst;
    UarchTraceType type = UarchTraceType::inst;
    if (!reader_.next(inst, type)) {
      if (!reader_.good()) {
        error = reader_.error();
        return false;
      }
      break;
    }

    while (!engine_.can_dispatch(inst)) {
      engine_.advance();
    }
    engine_.dispatch(inst);
  }

  engine_.finish();
  return true;
}

} // namespace tile_npu
