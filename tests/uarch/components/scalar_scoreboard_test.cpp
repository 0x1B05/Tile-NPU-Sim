#include "framework.h"
#include "uarch/components/scalar_scoreboard.h"

using namespace tile_npu::test;

TEST_CASE(scalar_scoreboard_tracks_busy_registers) {
  tile_npu::ScalarScoreboard scoreboard;

  check(!scoreboard.busy(1), "new scoreboard is clear");
  scoreboard.mark_busy(1);
  check(scoreboard.busy(1), "marked register is busy");
  check(!scoreboard.busy(2), "other registers stay ready");
  check(!scoreboard.busy(0), "x0 is never busy");

  scoreboard.mark_busy(0);
  check(!scoreboard.busy(0), "marking x0 is ignored");

  scoreboard.mark_ready(1);
  check(!scoreboard.busy(1), "writeback clears the busy bit");

  scoreboard.mark_busy(3);
  scoreboard.mark_busy(31);
  check(scoreboard.busy(3), "low register remains busy");
  check(scoreboard.busy(31), "high register remains busy");

  scoreboard.reset();
  check(!scoreboard.busy(3), "reset clears low register");
  check(!scoreboard.busy(31), "reset clears high register");
}
