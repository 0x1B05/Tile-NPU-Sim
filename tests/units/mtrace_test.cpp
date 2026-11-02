#ifdef CONFIG_MTRACE

#include "tile_npu/simulator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "units/execute_helpers.h"
#include "units/mblk_helpers.h"
#include "units/mma_helpers.h"

using namespace tile_npu::test;

namespace {

using tile_npu::DType;
using tile_npu::NpuDomain;
using tile_npu::Ram;
using tile_npu::Simulator;
using tile_npu::StepStatus;
using tile_npu::TraceEndpointKind;
using tile_npu::TraceEventType;
using tile_npu::TraceMemoryKind;

constexpr std::uint64_t DATA = CODE_BASE + 0x400;

template <std::size_t N>
void poke(Ram &memory, std::uint64_t address, const std::uint8_t (&bytes)[N]) {
  std::array<std::byte, N> raw{};
  for (std::size_t i = 0; i < N; ++i)
    raw[i] = static_cast<std::byte>(bytes[i]);
  check(memory.write(address, raw), "poke memory");
}

} // namespace

TEST_CASE(mtrace_tload_block) {
  Ram memory(CODE_BASE, 0x800);
  Simulator sim(memory);
  std::vector<tile_npu::TraceEvent> events;
  sim.set_mtrace_callback([&](const tile_npu::TraceEvent &event) { events.push_back(event); });

  load_one(sim, memory, make_mblk(NpuDomain::tile, 0, 0)); // tload t0
  auto &state = sim.state();
  tile_cfg(state, 0, DType::i8, 2, 2);
  state.tm[0].base_addr = DATA;
  state.tm[0].row_stride_bytes = 2;
  state.tm[0].col_stride_bytes = 1;
  poke(memory, DATA, {1, 2, 3, 4});

  check_eq(sim.step().status, StepStatus::retired, "tload retires");
  check_eq(events.size(), std::size_t{1}, "one move event");
  const auto &event = events[0];
  check_eq(event.type, TraceEventType::move, "event type");
  check(event.op == "tload", "op name");
  check_eq(event.src_count, std::uint8_t{1}, "source count");
  check_eq(event.dst_count, std::uint8_t{1}, "destination count");

  const auto &src = event.src[0];
  const auto &dst = event.dst[0];
  check_eq(src.kind, TraceEndpointKind::memory, "source is memory");
  check_eq(src.bytes, std::uint32_t{4}, "memory bytes");
  check_eq(dst.kind, TraceEndpointKind::tile, "destination is tile");
  check_eq(dst.index, std::uint8_t{0}, "destination register");
  check_eq(dst.bytes, std::uint32_t{4}, "register bytes");
  check_eq(dst.shape[0], std::uint8_t{2}, "destination rows");
  check_eq(dst.shape[1], std::uint8_t{2}, "destination cols");
  check_eq(event.mem.kind, TraceMemoryKind::block, "block pattern");
  check_eq(event.mem.addr, DATA, "base address");
  check_eq(event.mem.rows, std::uint8_t{2}, "memory rows");
  check_eq(event.mem.cols, std::uint8_t{2}, "memory cols");
  check_eq(event.mem.row_stride, std::int32_t{2}, "row stride");
  check_eq(event.mem.col_stride, std::int32_t{1}, "col stride");
  check_eq(event.mem.desc_index, std::uint8_t{0}, "tm descriptor");
}

TEST_CASE(mtrace_mma_compute) {
  Ram memory(CODE_BASE, 0x800);
  Simulator sim(memory);
  std::vector<tile_npu::TraceEvent> events;
  sim.set_mtrace_callback([&](const tile_npu::TraceEvent &event) { events.push_back(event); });

  load_one(sim, memory, make_mma(0, 0, 1, 2)); // mma.nt.zero a0, t1, t2
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 2, 2);
  tile_cfg(state, 1, DType::i8, 2, 3);
  tile_cfg(state, 2, DType::i8, 2, 3);

  check_eq(sim.step().status, StepStatus::retired, "mma retires");
  check_eq(events.size(), std::size_t{1}, "one compute event");
  const auto &event = events[0];
  check_eq(event.type, TraceEventType::compute, "event type");
  check(event.op == "mma.nt.zero", "op name");
  check_eq(event.src_count, std::uint8_t{2}, "source count");
  check_eq(event.dst_count, std::uint8_t{1}, "destination count");
  check_eq(event.src[0].kind, TraceEndpointKind::tile, "A is tile");
  check_eq(event.src[0].bytes, std::uint32_t{6}, "A bytes");
  check_eq(event.src[1].bytes, std::uint32_t{6}, "B bytes");
  check_eq(event.dst[0].kind, TraceEndpointKind::acc, "D is acc");
  check_eq(event.dst[0].bytes, std::uint32_t{16}, "D bytes");
  check_eq(event.dst[0].shape[0], std::uint8_t{2}, "D rows");
  check_eq(event.dst[0].shape[1], std::uint8_t{2}, "D cols");
  check_eq(event.macs, std::uint32_t{12}, "MAC count");
  check_eq(event.mem.kind, TraceMemoryKind::none, "no memory pattern");
}

TEST_CASE(mtrace_mma_accumulate_reads_destination) {
  Ram memory(CODE_BASE, 0x800);
  Simulator sim(memory);
  std::vector<tile_npu::TraceEvent> events;
  sim.set_mtrace_callback([&](const tile_npu::TraceEvent &event) { events.push_back(event); });

  load_one(sim, memory, make_mma(1, 0, 1, 2)); // mma.nt.acc a0, t1, t2
  auto &state = sim.state();
  acc_cfg(state, 0, DType::i32, 2, 2);
  tile_cfg(state, 1, DType::i8, 2, 3);
  tile_cfg(state, 2, DType::i8, 2, 3);

  check_eq(sim.step().status, StepStatus::retired, "mma.nt.acc retires");
  check_eq(events.size(), std::size_t{1}, "one compute event");
  const auto &event = events[0];
  check_eq(event.src_count, std::uint8_t{3}, "A, B, and old D sources");
  check(event.src[2].part == "acc_in", "old D is the accumulator input");
  check_eq(event.src[2].bytes, std::uint32_t{16}, "old D bytes");
  check_eq(event.dst[0].bytes, std::uint32_t{16}, "new D bytes");
}

#endif // CONFIG_MTRACE
