#ifdef CONFIG_MTRACE

#include "framework.h"

#include "trace/mtrace/mtrace_writer.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace tile_npu;

namespace {

class Cursor {
public:
  explicit Cursor(std::string_view data) : data_(data) {}

  bool empty() const { return pos_ == data_.size(); }

  std::uint8_t u8() {
    check(pos_ < data_.size(), "MTRB payload truncated");
    return static_cast<std::uint8_t>(data_[pos_++]);
  }

  std::uint16_t u16() {
    const auto low = u8();
    const auto high = u8();
    return static_cast<std::uint16_t>(low | (high << 8));
  }

  std::uint32_t u32() {
    std::uint32_t value = 0;
    for (int shift = 0; shift < 32; shift += 8)
      value |= static_cast<std::uint32_t>(u8()) << shift;
    return value;
  }

  std::uint64_t varint() {
    std::uint64_t value = 0;
    int shift = 0;
    while (true) {
      const auto byte = u8();
      value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
      if ((byte & 0x80) == 0) return value;
      shift += 7;
      check(shift <= 63, "MTRB varint too long");
    }
  }

  std::int64_t svarint() {
    const auto value = varint();
    return static_cast<std::int64_t>(value >> 1) ^ -static_cast<std::int64_t>(value & 1);
  }

  std::string_view take(std::size_t size) {
    check(size <= data_.size() - pos_, "MTRB payload truncated");
    const auto value = data_.substr(pos_, size);
    pos_ += size;
    return value;
  }

private:
  std::string_view data_;
  std::size_t pos_ = 0;
};

TraceEndpoint read_endpoint(Cursor &cursor,
                            const std::unordered_map<std::uint64_t, std::string> &parts) {
  TraceEndpoint endpoint;
  const auto kind_index = cursor.u8();
  endpoint.kind = static_cast<TraceEndpointKind>(kind_index & 0x7u);
  endpoint.index = static_cast<std::uint8_t>((kind_index >> 3) & 0x1fu);
  endpoint.bytes = static_cast<std::uint32_t>(cursor.varint());
  endpoint.elem = cursor.u8();
  endpoint.dims = cursor.u8();
  for (std::uint8_t i = 0; i < endpoint.dims; ++i) {
    endpoint.shape[i] = static_cast<std::uint8_t>(cursor.varint());
  }
  const auto part_id = cursor.varint();
  const auto part = parts.find(part_id);
  check(part != parts.end(), "undefined part id");
  endpoint.part = part->second;
  return endpoint;
}

TraceMemory read_memory(Cursor &cursor, std::uint64_t &prev_addr) {
  TraceMemory memory;
  const auto flags = cursor.u8();
  memory.kind = static_cast<TraceMemoryKind>(flags & 0x3u);
  memory.region = static_cast<TraceRegion>((flags >> 2) & 0x3u);
  memory.desc_kind = static_cast<TraceDescriptorKind>((flags >> 4) & 0x3u);
  memory.col_order = (flags & 0x40u) != 0;
  memory.desc_index = static_cast<std::uint8_t>(cursor.u8() & 0x1fu);
  memory.addr = static_cast<std::uint64_t>(static_cast<std::int64_t>(prev_addr) + cursor.svarint());
  prev_addr = memory.addr;
  memory.elem = cursor.u8();
  if (memory.kind == TraceMemoryKind::linear) {
    memory.count = static_cast<std::uint32_t>(cursor.varint());
    memory.stride = static_cast<std::int32_t>(cursor.svarint());
  } else if (memory.kind == TraceMemoryKind::block) {
    memory.rows = cursor.u8();
    memory.cols = cursor.u8();
    memory.row_stride = static_cast<std::int32_t>(cursor.svarint());
    memory.col_stride = static_cast<std::int32_t>(cursor.svarint());
  }
  return memory;
}

TraceEvent read_event(Cursor &cursor, const std::unordered_map<std::uint64_t, std::string> &ops,
                      const std::unordered_map<std::uint64_t, std::string> &parts,
                      std::uint64_t &prev_instr, std::uint64_t &prev_pc, std::uint64_t &prev_addr) {
  TraceEvent event;
  const auto flags = cursor.u8();
  event.type = (flags & 0x1u) != 0 ? TraceEventType::compute : TraceEventType::move;
  event.instr =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(prev_instr) + cursor.svarint());
  event.pc = static_cast<std::uint64_t>(static_cast<std::int64_t>(prev_pc) + cursor.svarint());
  prev_instr = event.instr;
  prev_pc = event.pc;

  const auto op_id = cursor.varint();
  const auto op = ops.find(op_id);
  check(op != ops.end(), "undefined op id");
  event.op = op->second;

  event.src_count = cursor.u8();
  event.dst_count = cursor.u8();
  for (std::uint8_t i = 0; i < event.src_count; ++i) {
    event.src[i] = read_endpoint(cursor, parts);
  }
  for (std::uint8_t i = 0; i < event.dst_count; ++i) {
    event.dst[i] = read_endpoint(cursor, parts);
  }
  if ((flags & 0x2u) != 0) event.mem = read_memory(cursor, prev_addr);
  if ((flags & 0x4u) != 0) event.macs = static_cast<std::uint32_t>(cursor.varint());
  return event;
}

TraceEndpoint tile_endpoint(std::uint8_t index, std::uint32_t bytes, std::string_view part) {
  TraceEndpoint endpoint;
  endpoint.kind = TraceEndpointKind::tile;
  endpoint.index = index;
  endpoint.bytes = bytes;
  endpoint.elem = 1;
  endpoint.dims = 2;
  endpoint.shape = {2, 2};
  endpoint.part = part;
  return endpoint;
}

} // namespace

TEST_CASE(mtrace_writer_roundtrip) {
  std::ostringstream out(std::ios::out | std::ios::binary);
  trace::BinaryMtraceWriter writer(out);
  writer.write_header();

  TraceEvent load;
  load.type = TraceEventType::move;
  load.instr = 5;
  load.pc = 0x80000010;
  load.op = "tload";
  load.src[0] = {.kind = TraceEndpointKind::memory, .bytes = 4, .elem = 1};
  load.src_count = 1;
  load.dst[0] = tile_endpoint(0, 4, "dst");
  load.dst_count = 1;
  load.mem.kind = TraceMemoryKind::block;
  load.mem.region = TraceRegion::ram;
  load.mem.addr = 0x81000000;
  load.mem.elem = 1;
  load.mem.rows = 2;
  load.mem.cols = 2;
  load.mem.row_stride = 2;
  load.mem.col_stride = 1;
  load.mem.desc_kind = TraceDescriptorKind::tm;
  load.mem.desc_index = 0;
  writer(load);

  TraceEvent mma;
  mma.type = TraceEventType::compute;
  mma.instr = 6;
  mma.pc = 0x80000014;
  mma.op = "mma.nt.zero";
  mma.src[0] = tile_endpoint(1, 4, "a");
  mma.src[1] = tile_endpoint(2, 4, "b");
  mma.src_count = 2;
  mma.dst[0].kind = TraceEndpointKind::acc;
  mma.dst[0].index = 0;
  mma.dst[0].bytes = 16;
  mma.dst[0].elem = 4;
  mma.dst[0].dims = 2;
  mma.dst[0].shape = {2, 2};
  mma.dst[0].part = "d";
  mma.dst_count = 1;
  mma.macs = 12;
  writer(mma);

  TraceEvent store;
  store.type = TraceEventType::move;
  store.instr = 7;
  store.pc = 0x80000018;
  store.op = "vstore";
  store.src[0].kind = TraceEndpointKind::vec32;
  store.src[0].index = 3;
  store.src[0].bytes = 8;
  store.src[0].elem = 4;
  store.src[0].dims = 1;
  store.src[0].shape = {2, 0};
  store.src[0].part = "src";
  store.src_count = 1;
  store.dst[0] = {.kind = TraceEndpointKind::memory, .bytes = 8, .elem = 4};
  store.dst_count = 1;
  store.mem.kind = TraceMemoryKind::linear;
  store.mem.region = TraceRegion::ram;
  store.mem.addr = 0x80000000;
  store.mem.elem = 4;
  store.mem.count = 2;
  store.mem.stride = 4;
  store.mem.desc_kind = TraceDescriptorKind::vm;
  store.mem.desc_index = 3;
  writer(store);
  writer.finish();

  const auto data = out.str();
  Cursor cursor(data);
  check(cursor.take(4) == "MTRB", "magic");
  check_eq(cursor.u16(), std::uint16_t{1}, "version");
  check_eq(cursor.u16(), std::uint16_t{1}, "flags");
  check_eq(cursor.u32(), std::uint32_t{16}, "header size");
  check_eq(cursor.u32(), std::uint32_t{0}, "reserved");

  std::unordered_map<std::uint64_t, std::string> ops;
  std::unordered_map<std::uint64_t, std::string> parts{{0, ""}};
  std::vector<TraceEvent> events;
  std::uint64_t prev_instr = 0;
  std::uint64_t prev_pc = 0;
  std::uint64_t prev_addr = 0;
  std::uint64_t end_count = 0;
  bool saw_end = false;
  while (!cursor.empty()) {
    const auto type = cursor.u8();
    Cursor payload(cursor.take(cursor.varint()));
    if (type == 1) {
      const auto id = payload.varint();
      ops[id] = std::string(payload.take(payload.varint()));
    } else if (type == 2) {
      const auto id = payload.varint();
      parts[id] = std::string(payload.take(payload.varint()));
    } else if (type == 3) {
      events.push_back(read_event(payload, ops, parts, prev_instr, prev_pc, prev_addr));
    } else if (type == 4) {
      end_count = payload.varint();
      saw_end = true;
    } else {
      check(false, "unexpected record type");
    }
  }

  check(saw_end, "END record");
  check_eq(end_count, std::uint64_t{3}, "END event count");
  check_eq(ops.size(), std::size_t{3}, "op dictionary size");
  check_eq(parts.size(), std::size_t{6}, "part dictionary size");
  check_eq(events.size(), std::size_t{3}, "event count");

  check_eq(events[0].type, TraceEventType::move, "load type");
  check(events[0].op == "tload", "load op");
  check_eq(events[0].instr, std::uint64_t{5}, "load instruction");
  check_eq(events[0].src_count, std::uint8_t{1}, "load source count");
  check_eq(events[0].dst_count, std::uint8_t{1}, "load destination count");
  check_eq(events[0].dst[0].kind, TraceEndpointKind::tile, "load dst kind");
  check_eq(events[0].dst[0].index, std::uint8_t{0}, "load dst index");
  check_eq(events[0].mem.kind, TraceMemoryKind::block, "load block pattern");
  check_eq(events[0].mem.addr, std::uint64_t{0x81000000}, "load address");
  check_eq(events[0].mem.row_stride, std::int32_t{2}, "load row stride");
  check_eq(events[0].mem.col_stride, std::int32_t{1}, "load col stride");
  check_eq(events[0].mem.desc_index, std::uint8_t{0}, "load descriptor");

  check_eq(events[1].type, TraceEventType::compute, "mma type");
  check(events[1].op == "mma.nt.zero", "mma op");
  check_eq(events[1].instr, std::uint64_t{6}, "mma instruction delta");
  check_eq(events[1].pc, std::uint64_t{0x80000014}, "mma pc delta");
  check_eq(events[1].src_count, std::uint8_t{2}, "mma source count");
  check_eq(events[1].dst[0].kind, TraceEndpointKind::acc, "mma dst kind");
  check_eq(events[1].macs, std::uint32_t{12}, "mma macs");

  check(events[2].op == "vstore", "store op");
  check_eq(events[2].mem.kind, TraceMemoryKind::linear, "store linear pattern");
  check_eq(events[2].mem.addr, std::uint64_t{0x80000000}, "store address delta");
  check_eq(events[2].mem.count, std::uint32_t{2}, "store count");
  check_eq(events[2].mem.stride, std::int32_t{4}, "store stride");
  check_eq(events[2].mem.desc_index, std::uint8_t{3}, "store descriptor");
}

#endif // CONFIG_MTRACE
