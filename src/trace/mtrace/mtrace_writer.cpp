#include "mtrace_writer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string_view>

static_assert(TRACE_FLUSH_RECORDS > 0, "TRACE_FLUSH_RECORDS must be positive");

namespace tile_npu::trace {
namespace {

constexpr std::uint8_t REC_OP_DEF = 1;
constexpr std::uint8_t REC_PART_DEF = 2;
constexpr std::uint8_t REC_EVENT = 3;
constexpr std::uint8_t REC_END = 4;

constexpr std::uint8_t EVENT_COMPUTE = 1u << 0;
constexpr std::uint8_t EVENT_HAS_MEM = 1u << 1;
constexpr std::uint8_t EVENT_HAS_MACS = 1u << 2;

using Buffer = std::vector<std::uint8_t>;

std::uint8_t endpoint_kind_code(TraceEndpointKind kind) {
  switch (kind) {
  case TraceEndpointKind::memory: return 0;
  case TraceEndpointKind::tile: return 1;
  case TraceEndpointKind::acc: return 2;
  case TraceEndpointKind::vec32: return 3;
  case TraceEndpointKind::immediate: return 4;
  }
  return 4;
}

std::uint8_t memory_kind_code(TraceMemoryKind kind) {
  switch (kind) {
  case TraceMemoryKind::none: return 0;
  case TraceMemoryKind::linear: return 1;
  case TraceMemoryKind::block: return 2;
  }
  return 0;
}

std::uint8_t descriptor_kind_code(TraceDescriptorKind kind) {
  switch (kind) {
  case TraceDescriptorKind::none: return 0;
  case TraceDescriptorKind::tm: return 1;
  case TraceDescriptorKind::am: return 2;
  case TraceDescriptorKind::vm: return 3;
  }
  return 0;
}

std::uint8_t region_code(TraceRegion region) {
  switch (region) {
  case TraceRegion::ram: return 0;
  case TraceRegion::serial: return 1;
  case TraceRegion::mmio: return 2;
  case TraceRegion::unknown: return 3;
  }
  return 3;
}

void append_u8(Buffer &out, std::uint8_t value) { out.push_back(value); }

void append_varint(Buffer &out, std::uint64_t value) {
  while (value >= 0x80) {
    out.push_back(static_cast<std::uint8_t>(value) | 0x80u);
    value >>= 7;
  }
  out.push_back(static_cast<std::uint8_t>(value));
}

void append_svarint(Buffer &out, std::int64_t value) {
  const auto bits = static_cast<std::uint64_t>(value);
  const auto sign = value < 0 ? ~std::uint64_t{0} : std::uint64_t{0};
  append_varint(out, (bits << 1) ^ sign);
}

void write_varint(std::ostream &out, std::uint64_t value) {
  while (value >= 0x80) {
    out.put(static_cast<char>(static_cast<std::uint8_t>(value) | 0x80u));
    value >>= 7;
  }
  out.put(static_cast<char>(value));
}

void write_u16(std::ostream &out, std::uint16_t value) {
  out.put(static_cast<char>(value & 0xffu));
  out.put(static_cast<char>((value >> 8) & 0xffu));
}

void write_u32(std::ostream &out, std::uint32_t value) {
  for (int shift = 0; shift < 32; shift += 8) {
    out.put(static_cast<char>((value >> shift) & 0xffu));
  }
}

std::int64_t delta(std::uint64_t current, std::uint64_t previous) {
  return static_cast<std::int64_t>(current) - static_cast<std::int64_t>(previous);
}

} // namespace

BinaryMtraceWriter::BinaryMtraceWriter(std::ostream &out, bool flush_each_line)
    : out_(out), flush_each_line_(flush_each_line) {}

BinaryMtraceWriter::~BinaryMtraceWriter() {
  if (!finished_) finish();
}

void BinaryMtraceWriter::write_header() {
  if (header_written_) return;
  static constexpr std::uint8_t magic[4] = {'M', 'T', 'R', 'B'};
  out_.write(reinterpret_cast<const char *>(magic), 4);
  write_u16(out_, 1);  // format version
  write_u16(out_, 1);  // flags: little-endian
  write_u32(out_, 16); // header size
  write_u32(out_, 0);  // reserved
  header_written_ = true;
  if (flush_each_line_) out_.flush();
}

void BinaryMtraceWriter::write_record(std::uint8_t type, const Buffer &payload) {
  out_.put(static_cast<char>(type));
  write_varint(out_, payload.size());
  if (!payload.empty()) {
    out_.write(reinterpret_cast<const char *>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
  }
}

void BinaryMtraceWriter::write_definition(std::uint8_t type, std::uint32_t id,
                                          std::string_view name) {
  Buffer payload;
  append_varint(payload, id);
  append_varint(payload, name.size());
  for (const char c : name)
    payload.push_back(static_cast<std::uint8_t>(c));
  write_record(type, payload);
}

std::uint32_t BinaryMtraceWriter::op_id(std::string_view name) {
  const std::string key(name);
  if (const auto it = op_ids_.find(key); it != op_ids_.end()) return it->second;
  const auto id = static_cast<std::uint32_t>(op_ids_.size());
  op_ids_.emplace(key, id);
  write_definition(REC_OP_DEF, id, name);
  return id;
}

std::uint32_t BinaryMtraceWriter::part_id(std::string_view part) {
  if (part.empty()) return 0;
  const std::string key(part);
  if (const auto it = part_ids_.find(key); it != part_ids_.end()) return it->second;
  const auto id = static_cast<std::uint32_t>(part_names_.size());
  part_names_.emplace_back(part);
  part_ids_.emplace(key, id);
  write_definition(REC_PART_DEF, id, part);
  return id;
}

void BinaryMtraceWriter::write_endpoint(Buffer &payload, const TraceEndpoint &endpoint) {
  const auto kind = static_cast<std::uint8_t>(endpoint_kind_code(endpoint.kind) & 0x7u);
  const auto index = static_cast<std::uint8_t>(endpoint.index & 0x1fu);
  append_u8(payload, static_cast<std::uint8_t>(kind | (index << 3)));
  append_varint(payload, endpoint.bytes);
  append_u8(payload, endpoint.elem);

  const auto dims = std::min<std::uint8_t>(endpoint.dims, 2);
  append_u8(payload, dims);
  for (std::uint8_t i = 0; i < dims; ++i)
    append_varint(payload, endpoint.shape[i]);
  append_varint(payload, part_id(endpoint.part));
}

void BinaryMtraceWriter::write_memory(Buffer &payload, const TraceMemory &memory) {
  std::uint8_t flags = static_cast<std::uint8_t>(memory_kind_code(memory.kind) & 0x3u);
  flags = static_cast<std::uint8_t>(flags | ((region_code(memory.region) & 0x3u) << 2));
  flags = static_cast<std::uint8_t>(flags | ((descriptor_kind_code(memory.desc_kind) & 0x3u) << 4));
  if (memory.col_order) flags = static_cast<std::uint8_t>(flags | 0x40u);
  append_u8(payload, flags);
  append_u8(payload, static_cast<std::uint8_t>(memory.desc_index & 0x1fu));
  append_svarint(payload, delta(memory.addr, prev_addr_));
  prev_addr_ = memory.addr;
  append_u8(payload, memory.elem);

  if (memory.kind == TraceMemoryKind::linear) {
    append_varint(payload, memory.count);
    append_svarint(payload, memory.stride);
  } else if (memory.kind == TraceMemoryKind::block) {
    append_u8(payload, memory.rows);
    append_u8(payload, memory.cols);
    append_svarint(payload, memory.row_stride);
    append_svarint(payload, memory.col_stride);
  }
}

void BinaryMtraceWriter::operator()(const TraceEvent &event) {
  if (finished_) return;
  if (!header_written_) write_header();

  const auto op = op_id(event.op);
  payload_.clear();

  const bool has_mem = event.mem.kind != TraceMemoryKind::none;
  const bool has_macs = event.macs != 0;
  std::uint8_t flags = 0;
  if (event.type == TraceEventType::compute)
    flags = static_cast<std::uint8_t>(flags | EVENT_COMPUTE);
  if (has_mem) flags = static_cast<std::uint8_t>(flags | EVENT_HAS_MEM);
  if (has_macs) flags = static_cast<std::uint8_t>(flags | EVENT_HAS_MACS);

  append_u8(payload_, flags);
  append_svarint(payload_, delta(event.instr, prev_instr_));
  append_svarint(payload_, delta(event.pc, prev_pc_));
  prev_instr_ = event.instr;
  prev_pc_ = event.pc;
  append_varint(payload_, op);
  append_u8(payload_, event.src_count);
  append_u8(payload_, event.dst_count);

  for (std::uint8_t i = 0; i < event.src_count; ++i)
    write_endpoint(payload_, event.src[i]);
  for (std::uint8_t i = 0; i < event.dst_count; ++i)
    write_endpoint(payload_, event.dst[i]);
  if (has_mem) write_memory(payload_, event.mem);
  if (has_macs) append_varint(payload_, event.macs);

  write_record(REC_EVENT, payload_);
  ++events_;
  if (flush_each_line_) {
    out_.flush();
  } else if ((++records_ % TRACE_FLUSH_RECORDS) == 0) {
    out_.flush();
  }
}

void BinaryMtraceWriter::flush() { out_.flush(); }

void BinaryMtraceWriter::finish() {
  if (finished_) return;
  if (!header_written_) write_header();
  Buffer payload;
  append_varint(payload, events_);
  write_record(REC_END, payload);
  finished_ = true;
  out_.flush();
}

} // namespace tile_npu::trace
