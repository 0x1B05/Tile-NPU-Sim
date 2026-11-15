#include "framework.h"
#include "uarch/builders/dyn_inst.h"
#include "uarch/components/scoreboard.h"

#include <cstdint>

using namespace tile_npu;
using namespace tile_npu::test;

namespace {

DynInst acc_write(std::uint64_t id, RegisterIndex dst) {
  auto inst = make_dyn_inst(id, DynOp::mma, Unit::mma);
  add_dst(inst, NpuDomain::acc, dst);
  return inst;
}

DynInst acc_read(std::uint64_t id, RegisterIndex src) {
  auto inst = make_dyn_inst(id, DynOp::vmma, Unit::mma);
  add_src(inst, NpuDomain::acc, src);
  return inst;
}

DynInst acc_read2(std::uint64_t id, RegisterIndex src1, RegisterIndex src2) {
  auto inst = make_dyn_inst(id, DynOp::vmma, Unit::mma);
  add_src(inst, NpuDomain::acc, src1);
  add_src(inst, NpuDomain::acc, src2);
  return inst;
}

DynInst acc_rmw(std::uint64_t id, RegisterIndex index) {
  auto inst = make_dyn_inst(id, DynOp::vmma, Unit::mma);
  add_src(inst, NpuDomain::acc, index);
  add_dst(inst, NpuDomain::acc, index);
  return inst;
}

} // namespace

TEST_CASE(scoreboard_dispatches_and_completes) {
  Scoreboard scoreboard;
  check(scoreboard.empty(), "new scoreboard is empty");

  scoreboard.dispatch(acc_write(1, 1));
  check(!scoreboard.empty(), "dispatch adds an entry");
  check_eq(scoreboard.size(), std::size_t{1}, "scoreboard size");
  check_eq(scoreboard.inst(1).id, std::uint64_t{1}, "entry id");
  check(scoreboard.ready(1), "independent instruction is ready");

  scoreboard.mark_issued(1);
  check(scoreboard.complete(1), "complete removes the entry");
  check(scoreboard.empty(), "scoreboard is empty after complete");
  check(!scoreboard.complete(1), "completing an absent id fails");
}

TEST_CASE(scoreboard_waits_for_raw) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_write(1, 1));
  scoreboard.dispatch(acc_read(2, 1));

  check(scoreboard.ready(1), "producer is ready");
  check(!scoreboard.ready(2), "reader waits for the producer");
  check(scoreboard.complete(1), "complete producer");
  check(scoreboard.ready(2), "reader becomes ready after producer completion");
  check(scoreboard.complete(2), "complete reader");
}

TEST_CASE(scoreboard_waits_for_waw) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_write(1, 1));
  scoreboard.dispatch(acc_write(2, 1));

  check(scoreboard.ready(1), "first writer is ready");
  check(!scoreboard.ready(2), "second writer waits for the first");
  check(scoreboard.complete(1), "complete first writer");
  check(scoreboard.ready(2), "second writer becomes ready");
}

TEST_CASE(scoreboard_waits_for_war) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_read(1, 1));
  scoreboard.dispatch(acc_write(2, 1));

  check(scoreboard.ready(1), "reader is ready");
  check(!scoreboard.ready(2), "writer waits for the old reader");
  check(scoreboard.complete(1), "complete old reader");
  check(scoreboard.ready(2), "writer becomes ready after reader completion");
}

TEST_CASE(scoreboard_waits_for_all_sources) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_write(1, 1));
  scoreboard.dispatch(acc_write(2, 2));
  scoreboard.dispatch(acc_read2(3, 1, 2));

  check(!scoreboard.ready(3), "reader waits for both producers");
  check(scoreboard.complete(1), "complete first producer");
  check(!scoreboard.ready(3), "one dependency remains");
  check(scoreboard.complete(2), "complete second producer");
  check(scoreboard.ready(3), "reader becomes ready when both complete");
}

TEST_CASE(scoreboard_handles_read_modify_write) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_write(1, 1));
  scoreboard.dispatch(acc_rmw(2, 1));

  check(!scoreboard.ready(2), "rmw waits for the old producer");
  check(scoreboard.complete(1), "complete old producer");
  check(scoreboard.ready(2), "rmw does not self-block");
}

TEST_CASE(scoreboard_allows_out_of_order_completion) {
  Scoreboard scoreboard;
  scoreboard.dispatch(acc_write(1, 1));
  scoreboard.dispatch(acc_write(2, 2));
  check(scoreboard.ready(1), "first instruction ready");
  check(scoreboard.ready(2), "second instruction ready");

  check(scoreboard.complete(2), "complete younger instruction first");
  check(scoreboard.complete(1), "complete older instruction second");
  check(scoreboard.empty(), "out-of-order completion leaves an empty scoreboard");
}
