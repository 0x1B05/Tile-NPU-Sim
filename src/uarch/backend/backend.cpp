#include "uarch/backend/backend.h"

#include "utils/common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace tile_npu {
namespace {

constexpr std::size_t INFLIGHT_CAPACITY = 32;

} // namespace

UarchBackend::UarchBackend(const UarchClock &clock, UarchStats &stats, UarchExecutor &executor)
    : clock_(clock), stats_(stats), executor_(executor) {
  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    queues_[unit] = IssueQueue(queue_depth(static_cast<Unit>(unit)));
  }
}

void UarchBackend::reset() {
  events_.clear();
  scoreboard_.reset();
  for (auto &queue : queues_)
    queue.clear();
  inst_issue_not_before_.clear();
  unit_issue_not_before_.fill(0);
  last_processed_cycle_.reset();
  failure_.reset();
}

std::optional<StepResult> UarchBackend::take_failure() {
  if (!failure_) return std::nullopt;

  auto failure = failure_;
  reset();
  return failure;
}

bool UarchBackend::outstanding(Unit unit) const {
  for (const auto &pair : scoreboard_.entries()) {
    if (pair.second.inst.unit == unit) return true;
  }
  return false;
}

bool UarchBackend::dispatch(const DynInst &inst) {
  if (failure_) return false;

  const auto unit = static_cast<std::size_t>(inst.unit);

  if (scoreboard_.size() >= INFLIGHT_CAPACITY || !queues_[unit].push(inst.id)) return false;

  scoreboard_.dispatch(inst);
  inst_issue_not_before_[inst.id] = clock_.now() + 1;
  return true;
}

CycleResult UarchBackend::process_cur_cycle() {
  CycleResult result;
  if (failure_) return result;

  result.issued = issue();
  if (failure_) return result;

  result.completed = events_.run_due(clock_.now());
  last_processed_cycle_ = clock_.now();
  return result;
}

std::optional<UarchClock::Time> UarchBackend::next_activity_time() const {
  auto next = events_.next_time();

  const bool now_processed = last_processed_cycle_ && *last_processed_cycle_ == clock_.now();
  const auto issue_floor = now_processed ? clock_.now() + 1 : clock_.now();

  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    const auto candidate = next_issuable(unit);
    if (!candidate) continue;

    const auto after = inst_issue_not_before_.find(*candidate);
    if (after == inst_issue_not_before_.end())
      unreachable("queued instruction has no earliest issue time");

    const auto issue_time = std::max({after->second, unit_issue_not_before_[unit], issue_floor});
    next = next ? std::min(*next, issue_time) : issue_time;
  }

  if (next) return next;
  if (!scoreboard_.empty()) unreachable("uarch backend cannot make progress");
  return std::nullopt;
}

std::size_t UarchBackend::issue() {
  std::size_t issued = 0;

  for (std::size_t unit = 0; unit < queues_.size(); ++unit) {
    if (unit_issue_not_before_[unit] > clock_.now()) continue;

    const auto candidate = next_issuable(unit);
    if (!candidate) continue;

    const auto id = *candidate;
    const auto after = inst_issue_not_before_.find(id);
    if (after != inst_issue_not_before_.end() && after->second > clock_.now()) continue;

    queues_[unit].remove(id);
    inst_issue_not_before_.erase(id);
    scoreboard_.mark_issued(id);

    const auto result = executor_.execute(scoreboard_.inst(id));
    if (result.status != StepStatus::retired) {
      reset();
      failure_ = result;
      return issued;
    }

    events_.schedule(clock_.now() + latency(static_cast<Unit>(unit)), [this, id] { complete(id); });
    unit_issue_not_before_[unit] = clock_.now() + 1;
    ++issued;
  }

  return issued;
}

std::optional<std::uint64_t> UarchBackend::next_issuable(std::size_t unit) const {
  const auto &queue = queues_[unit];

  if (static_cast<Unit>(unit) == Unit::dma) {
    // All memory operations share one program-order queue.
    auto candidate = queue.front();
    if (candidate && !scoreboard_.ready(*candidate)) candidate.reset();
    return candidate;
  }

  // General queues pick the oldest ready entry.
  for (const auto id : queue.entries()) {
    if (scoreboard_.ready(id)) return id;
  }
  return std::nullopt;
}

void UarchBackend::complete(std::uint64_t id) {
  const auto &entries = scoreboard_.entries();
  const auto it = entries.find(id);
  if (it == entries.end()) return;

  const auto inst = it->second.inst;
  if (!scoreboard_.complete(id)) return;
  inst_issue_not_before_.erase(id);
  executor_.complete(inst);
  ++stats_.retired;
  ++stats_.unit_count[static_cast<std::size_t>(inst.unit)];
}

std::size_t UarchBackend::queue_depth(Unit unit) {
  switch (unit) {
  case Unit::mma: return 2;
  case Unit::dma: return 16;
  default: return 8;
  }
}

std::uint64_t UarchBackend::latency(Unit unit) {
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
