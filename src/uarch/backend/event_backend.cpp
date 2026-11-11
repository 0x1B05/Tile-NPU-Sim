#include "uarch/backend/event_backend.h"

#include "utils/common.h"

#include <cstddef>
#include <cstdint>
#include <optional>

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
  // UarchStats is owned by the machine and reset by UarchCore::reset().
  events_.clear();
  scoreboard_.reset();
  for (auto &queue : queues_)
    queue.clear();
}

bool EventUarchBackend::dispatch(const DynInst &inst) {
  const auto unit = static_cast<std::size_t>(inst.unit);

  if (scoreboard_.size() >= WINDOW_DEPTH || !queues_[unit].push(inst.id)) return false;

  scoreboard_.dispatch(inst);
  return true;
}

void EventUarchBackend::tick() {
  events_.run_until(events_.now() + 1);
  issue();
  counters().cycles = events_.now();
}

std::uint64_t EventUarchBackend::advance() {
  const auto before = events_.now();
  if (const auto next = events_.next_time()) events_.run_until(*next);
  issue();
  auto &stats = counters();
  stats.cycles = events_.now();
  return stats.cycles - before;
}

void EventUarchBackend::finish() {
  while (!scoreboard_.empty()) {
    if (const auto next = events_.next_time()) events_.run_until(*next);
    issue();
    if (scoreboard_.empty()) break;
    if (!events_.next_time()) {
      unreachable("uarch backend cannot drain dispatched work");
    }
  }
  counters().cycles = events_.now();
}

void EventUarchBackend::issue() {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    auto &queue = queues_[unit];
    std::optional<std::uint64_t> candidate;

    if (static_cast<Unit>(unit) == Unit::dma) {
      // All memory operations share one program-order queue.
      candidate = queue.front();
      if (candidate && !scoreboard_.ready(*candidate)) candidate.reset();
    } else {
      // General queues pick the oldest ready entry.
      for (const auto id : queue.entries()) {
        if (!scoreboard_.ready(id)) continue;
        candidate = id;
        break;
      }
    }

    if (!candidate) continue;

    const auto id = *candidate;
    queue.remove(id);
    scoreboard_.mark_issued(id);
    if (executor_ != nullptr) executor_->execute(scoreboard_.inst(id));
    events_.schedule(events_.now() + latency(static_cast<Unit>(unit)),
                     [this, id] { complete(id); });
  }
}

void EventUarchBackend::complete(std::uint64_t id) {
  const auto &entries = scoreboard_.entries();
  const auto it = entries.find(id);
  if (it == entries.end()) return;

  const auto inst = it->second.inst;
  if (!scoreboard_.complete(id)) return;
  if (executor_ != nullptr) executor_->complete(inst);
  auto &stats = counters();
  ++stats.retired;
  ++stats.unit_count[static_cast<std::size_t>(inst.unit)];
}

UarchStats &EventUarchBackend::counters() {
  if (stats_ == nullptr) unreachable("backend stats are not attached");
  return *stats_;
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
