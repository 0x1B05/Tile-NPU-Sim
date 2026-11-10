#include "framework.h"
#include "uarch/timing/reservation_table.h"

#include <cstddef>
#include <cstdint>

using namespace tile_npu;

TEST_CASE(reservation_table_reserves_first_gap) {
  ReservationTable table;
  check_eq(table.size(), std::size_t{0}, "new table is empty");
  check_eq(table.earliest_start(0, 0, 1), std::uint64_t{0}, "new table is free");

  check_eq(table.reserve(0, 4), std::uint64_t{0}, "first reservation starts at now");
  check_eq(table.size(), std::size_t{1}, "one interval retained");
  check_eq(table.earliest_start(0, 0, 1), std::uint64_t{4}, "next free cycle follows the interval");

  check_eq(table.reserve(0, 3), std::uint64_t{4}, "next reservation follows the first");
  check_eq(table.earliest_start(0, 0, 1), std::uint64_t{7}, "next free cycle after two intervals");
  check_eq(table.reserve(0, 3), std::uint64_t{7}, "reservations are back-to-back");
  check_eq(table.earliest_start(0, 0, 1), std::uint64_t{10},
           "next free cycle after three intervals");
}

TEST_CASE(reservation_table_earliest_start_does_not_book) {
  ReservationTable table;
  check_eq(table.reserve(0, 4), std::uint64_t{0}, "book the first interval");

  check_eq(table.earliest_start(0, 0, 2), std::uint64_t{4}, "query finds the gap");
  check_eq(table.size(), std::size_t{1}, "query does not mutate the table");
  check_eq(table.reserve_from(0, 0, 2), std::uint64_t{4}, "booking uses the same start");
  check_eq(table.size(), std::size_t{2}, "booking adds an interval");
}

TEST_CASE(reservation_table_fills_gaps_between_future_bookings) {
  ReservationTable table;
  check_eq(table.reserve(0, 2), std::uint64_t{0}, "book [0,2)");
  check_eq(table.reserve_from(0, 10, 2), std::uint64_t{10}, "book [10,12)");

  check_eq(table.earliest_start(0, 2, 3), std::uint64_t{2}, "fit inside the gap");
  check_eq(table.earliest_start(0, 2, 9), std::uint64_t{12}, "skip past the second booking");
  check_eq(table.reserve_from(0, 2, 9), std::uint64_t{12}, "reserve after the gap");
}

TEST_CASE(reservation_table_clamps_past_requests) {
  ReservationTable table;
  check_eq(table.reserve_from(5, 1, 2), std::uint64_t{5}, "past earliest clamps to now");
  check_eq(table.size(), std::size_t{1}, "one interval booked");
  check_eq(table.earliest_start(5, 5, 1), std::uint64_t{7}, "next free cycle after the interval");
}

TEST_CASE(reservation_table_prunes_finished_intervals) {
  ReservationTable table;
  table.reserve(0, 2);
  table.reserve_from(0, 10, 2);
  check_eq(table.size(), std::size_t{2}, "two future intervals");

  check_eq(table.reserve_from(5, 5, 1), std::uint64_t{5}, "reserve after the prune horizon");
  check_eq(table.size(), std::size_t{2}, "finished interval is pruned");
  check_eq(table.earliest_start(5, 5, 1), std::uint64_t{6}, "gap after new booking");
  check_eq(table.earliest_start(5, 6, 5), std::uint64_t{12}, "future booking remains");
}

TEST_CASE(reservation_table_handles_zero_duration) {
  ReservationTable table;
  check_eq(table.reserve(3, 0), std::uint64_t{3}, "zero duration returns now");
  check_eq(table.size(), std::size_t{0}, "zero duration books nothing");
  check_eq(table.earliest_start(3, 3, 1), std::uint64_t{3}, "zero duration leaves the table free");
}

TEST_CASE(reservation_table_clear_removes_bookings) {
  ReservationTable table;
  table.reserve(0, 4);
  table.clear();
  check_eq(table.size(), std::size_t{0}, "clear removes intervals");
  check_eq(table.earliest_start(0, 0, 1), std::uint64_t{0}, "cleared table is free");
}
