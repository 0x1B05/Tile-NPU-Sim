#include "uarch/event_backend.h"

#include "utils/common.h"

#include <cstddef>
#include <cstdint>

namespace tile_npu {
namespace {

constexpr std::size_t WINDOW_DEPTH = 32;

} // namespace

EventUarchBackend::EventUarchBackend() {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    queues_[unit] = IssueQueue(queue_depth(static_cast<Unit>(unit)));
  }
}

void EventUarchBackend::reset() {
  stats_ = {};
  events_.clear();
  scoreboard_.reset();
  for (auto &queue : queues_)
    queue.clear();
}

bool EventUarchBackend::can_dispatch(const DynInst &inst) const {
  const auto unit = static_cast<std::size_t>(inst.unit);
  if (unit >= queues_.size()) return false;
  return scoreboard_.size() < WINDOW_DEPTH && !queues_[unit].full();
}

void EventUarchBackend::dispatch(const DynInst &inst) {
  if (!can_dispatch(inst)) unreachable("dispatch into a full uarch backend");

  const auto unit = static_cast<std::size_t>(inst.unit);
  if (!queues_[unit].push(inst.id)) unreachable("dispatch into a full issue queue");
  scoreboard_.dispatch(inst);
}

void EventUarchBackend::retire_scalar() {
  ++stats_.retired;
  ++stats_.scalar_retired;
}

void EventUarchBackend::retire_frontend() { ++stats_.retired; }

void EventUarchBackend::tick() {
  events_.run_until(events_.now() + 1);
  issue_ready();
  stats_.cycles = events_.now();
}

std::uint64_t EventUarchBackend::advance() {
  const auto before = events_.now();
  if (const auto next = events_.next_time()) events_.run_until(*next);
  issue_ready();
  stats_.cycles = events_.now();
  return stats_.cycles - before;
}

void EventUarchBackend::finish() {
  while (!scoreboard_.empty()) {
    if (const auto next = events_.next_time()) events_.run_until(*next);
    const bool issued = issue_ready();
    if (scoreboard_.empty()) break;
    if (!issued && !events_.next_time()) {
      unreachable("uarch backend cannot drain dispatched work");
    }
  }
  stats_.cycles = events_.now();
}

const UarchStats &EventUarchBackend::stats() const { return stats_; }

bool EventUarchBackend::issue_ready() {
  bool issued = false;
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    auto &queue = queues_[unit];
    for (const auto id : queue.entries()) {
      if (!scoreboard_.ready(id)) continue;

      queue.remove(id);
      scoreboard_.mark_issued(id);
      if (executor_ != nullptr) executor_->execute(scoreboard_.inst(id));
      events_.schedule(events_.now() + latency(static_cast<Unit>(unit)),
                       [this, id] { complete(id); });
      issued = true;
      break;
    }
  }
  return issued;
}

void EventUarchBackend::complete(std::uint64_t id) {
  const auto &entries = scoreboard_.entries();
  const auto it = entries.find(id);
  if (it == entries.end()) return;

  const auto unit = it->second.inst.unit;
  if (!scoreboard_.complete(id)) return;
  ++stats_.retired;
  ++stats_.unit_count[static_cast<std::size_t>(unit)];
}

std::size_t EventUarchBackend::queue_depth(Unit unit) {
  switch (unit) {
  case Unit::mma: return 2;
  case Unit::dma: return 16;
  default: return 8;
  }
}

std::uint64_t EventUarchBackend::latency(Unit unit) {
  switch (unit) {
  case Unit::mma: return 10;
  case Unit::dma: return 5;
  case Unit::reduce: return 3;
  case Unit::quant: return 3;
  case Unit::vec32: return 2;
  default: return 1;
  }
}

} // namespace tile_npu
