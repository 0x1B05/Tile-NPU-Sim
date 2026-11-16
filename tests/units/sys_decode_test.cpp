#include "tile_npu/decode.h"

#include <cstdint>
#include <string>
#include <variant>

#include "framework.h"

namespace {

using tile_npu::DecodedNpuInstr;
using tile_npu::SysInstr;
using tile_npu::SysOp;

void check_op(const char *name, std::uint32_t word, SysOp expected) {
  const auto result = tile_npu::decode(word);
  const auto *npu = result ? std::get_if<DecodedNpuInstr>(&*result) : nullptr;
  const std::string what = std::string("decode ") + name;
  check(npu != nullptr, what);
  const auto *decoded = std::get_if<SysInstr>(npu);
  check(decoded != nullptr, what);
  check_eq(decoded->op, expected, what + ": op");
}

void check_illegal(const char *name, std::uint32_t word) {
  const auto result = tile_npu::decode(word);
  const std::string what = std::string(name) + " (" + to_hex(word) + ")";
  check(!result.has_value(), what + " decodes");
}

} // namespace

TEST_CASE(sys_decode) {
  check_op("fence.mem", 0x80000002u, SysOp::fence_mem);
  check_op("fence.sa", 0x82000002u, SysOp::fence_sa);
  check_op("fence.all", 0x84000002u, SysOp::fence_all);
  check_op("kernel.end", 0x86000002u, SysOp::kernel_end);
}

TEST_CASE(sys_decode_rejects_reserved) {
  check_illegal("funct3 100", 0x88000002u);
  check_illegal("funct3 111", 0x8e000002u);
  check_illegal("reserved23 low", 0x80000006u);
  check_illegal("reserved23 high", 0x81000002u);
}
