#include "tile_npu/platform/ram.h"
#include "tile_npu/simulator.h"

#include "framework.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

#include "frontend/common/constants.h"
#include "guest/guest_image.h"
#include "utils/endian.h"

using namespace tile_npu::test;

namespace {
constexpr std::uint64_t IMAGE_BASE = 0x80000000;

void load_image(tile_npu::Ram &memory, std::string_view name) {
  const auto &directory = guest_image_dir();
  check(!directory.empty(), "guest image directory not set (pass --guest-dir)");
  const std::string path = directory + "/" + std::string(name) + ".bin";
  std::ifstream in(path, std::ios::binary);
  check(in.good(), "open guest image " + path);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(in),
                                std::istreambuf_iterator<char>()};
  const auto *data = reinterpret_cast<const std::byte *>(bytes.data());
  check(!bytes.empty() && memory.write(IMAGE_BASE, {data, bytes.size()}),
        "load guest image " + path);
}

// Steps until the guest stops on its own (any non-retired status), so a
// guest fault surfaces immediately. A hung guest fails with the budget,
// the current PC and the instruction word at PC.
tile_npu::StepResult run_until_stop(tile_npu::Simulator &sim, tile_npu::Ram &memory,
                                    int max_steps = 512) {
  tile_npu::StepResult result{};
  for (int steps = 0; steps < max_steps; ++steps) {
    result = sim.step();
    if (result.status != tile_npu::StepStatus::retired) {
      return result;
    }
  }
  const auto pc = sim.state().scalar.pc;
  std::uint32_t word = 0;
  std::array<std::byte, 4> bytes{};
  if (memory.read(pc, bytes)) {
    word = tile_npu::load_le<std::uint32_t>(bytes);
  }
  tile_npu::test::fail("guest image did not stop within " + std::to_string(max_steps) +
                           " steps (pc=" + to_hex(pc) + ", instr=" + to_hex(word) + ")",
                       std::source_location::current());
}

// Runs a self-checking guest image until it stops, then asserts the runner
// conventions: the expected stop status (ebreak; kernel.end for sys.S) and
// a0 == 0.
void check_guest_image(std::string_view name, std::uint64_t memory_size,
                       tile_npu::StepStatus stop = tile_npu::StepStatus::breakpoint) {
  tile_npu::Ram memory{IMAGE_BASE, memory_size};
  tile_npu::Simulator sim(memory);
  load_image(memory, name);
  sim.reset(IMAGE_BASE);
  const auto result = run_until_stop(sim, memory);
  check(result.status == stop, std::string(name) + " image stops (status=" +
                                   std::string(tile_npu::status_name(result.status)) + ")");
  check_eq(hex(sim.state().scalar.read_x(tile_npu::frontend::EXIT_CODE_REG)), hex(0x0u),
           std::string(name) + " image reports success via a0");
}

} // namespace

TEST_CASE(npu_guest_mem) {
  // tests/guest/npu/mem.S self-configures descriptors and configs, then
  // exercises block/transposed/row/gather accesses (u32 gather indices)
  // with store-back checks and a descriptor auto-increment readback. It
  // reaches ebreak with a0 = 0 on success; a failing check reports its stage
  // number in a0.
  check_guest_image("mem", 1024);
}

TEST_CASE(npu_guest_init) {
  // tests/guest/npu/init.S self-configures configs and descriptors, then
  // exercises fill/fillx/copy with store-back checks. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("init", 1024);
}

TEST_CASE(npu_guest_rowlane) {
  // tests/guest/npu/rowlane.S exercises Acc <-> Vec32 row insert/extract and
  // lane extract/insert/broadcast with store-back checks. It reaches ebreak
  // with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("rowlane", 1024);
}

TEST_CASE(npu_guest_eun) {
  // tests/guest/npu/eun.S self-configures configs and descriptors, then
  // exercises the E_UNARY family (not/abs/neg/exp2/rcp/rsqrt) with
  // store-back checks against expected values in .data. It reaches ebreak
  // with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("eun", 1024);
}

TEST_CASE(npu_guest_ecmp) {
  // tests/guest/npu/ecmp.S self-configures configs and descriptors, then
  // exercises register/scalar comparisons on Vec32 (i32 and f32), storing the
  // u32 masks back and checking them against expected arrays. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("ecmp", 1024);
}

TEST_CASE(npu_guest_ebcast) {
  // tests/guest/npu/ebcast.S self-configures configs and descriptors, then
  // exercises an Acc byrow broadcast add (i32) and an Acc bycol broadcast mul
  // (i32) with store-back checks. It reaches ebreak with a0 = 0 on success; a
  // failing check reports its stage in a0.
  check_guest_image("ebcast", 1024);
}

TEST_CASE(npu_guest_er4) {
  // tests/guest/npu/er4.S self-configures configs and descriptors, then
  // exercises a Vec32 select (u32 mask) and a Vec32 fmadd (f32, single
  // rounding) with store-back checks. It reaches ebreak with a0 = 0 on
  // success; a failing check reports its stage in a0.
  check_guest_image("er4", 1024);
}

TEST_CASE(npu_guest_emask) {
  // tests/guest/npu/emask.S self-configures configs and descriptors, then
  // exercises an Acc tril mask (i32, xDelta = 0, fill5 = -1) and a Vec32
  // tail mask (i32, xLen = 3, fill5 = +1) with store-back checks. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("emask", 1024);
}

TEST_CASE(npu_guest_reduce) {
  // tests/guest/npu/reduce.S self-configures configs and descriptors, then
  // exercises areduce.rows.sum/argmax and vreduce.argmax cases (NaN candidates
  // are ignored; all-NaN falls back to index 0 / NaN value) with store-back
  // and in-register checks. It reaches ebreak with a0 = 0 on success; a
  // failing check reports its stage in a0.
  check_guest_image("reduce", 1024);
}

TEST_CASE(npu_guest_qnt) {
  // tests/guest/npu/qnt.S self-configures configs and descriptors, then
  // exercises an aquant with distinct source/scale registers,
  // a tdequant that consumes its output, and a saturation check. Store-back
  // checks reach ebreak with a0 = 0 on success; a failing check reports its
  // stage in a0.
  check_guest_image("qnt", 1024);
}

TEST_CASE(npu_guest_mma) {
  // tests/guest/npu/mma.S self-configures configs and descriptors, loads its
  // operands, then exercises mma.nt.zero and mma.nt.acc on an N x K right Tile
  // (accumulating onto the first product), followed by vmma.nt.zero and
  // vmma.nt.acc reading A from a Tile row, with store-back checks. It reaches
  // ebreak with a0 = 0 on success; a failing check reports its stage in a0.
  check_guest_image("mma", 1024);
}

TEST_CASE(npu_guest_sys) {
  // tests/guest/npu/sys.S interleaves the three fences with a tfill/tstore
  // pair, checks the stored bytes, then terminates with kernel.end; the step
  // stops with StepStatus::kernel_end.
  check_guest_image("sys", 256, tile_npu::StepStatus::kernel_end);
}

TEST_CASE(npu_guest_cfg_self_check) {
  // tests/guest/npu/cfg.S programs config registers with cfg.seti/cfg.setx,
  // reads them back with cfg.get, and reaches ebreak with a0 = 0 on success.
  check_guest_image("cfg", 128);
}
