#pragma once

#include "events.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tile_npu::trace {

// Streams the logical MTRACE event stream in the versioned MTRB binary
// format. Opcode and part strings are interned in dictionary records; events
// use delta-coded instruction/PC/address fields and LEB128 varints. See
// FORMAT.md for the on-disk layout.
class BinaryMtraceWriter final {
public:
  explicit BinaryMtraceWriter(std::ostream &out, bool flush_each_line = false);
  ~BinaryMtraceWriter();

  void write_header();
  void operator()(const TraceEvent &event);
  void flush();
  void finish();

private:
  using Buffer = std::vector<std::uint8_t>;

  void write_record(std::uint8_t type, const Buffer &payload);
  void write_definition(std::uint8_t type, std::uint32_t id, std::string_view name);
  std::uint32_t op_id(std::string_view name);
  std::uint32_t part_id(std::string_view part);
  void write_endpoint(Buffer &payload, const TraceEndpoint &endpoint);
  void write_memory(Buffer &payload, const TraceMemory &memory);

  std::ostream &out_;
  bool flush_each_line_ = false;
  bool header_written_ = false;
  bool finished_ = false;
  std::uint64_t records_ = 0;
  std::uint64_t events_ = 0;
  std::uint64_t prev_instr_ = 0;
  std::uint64_t prev_pc_ = 0;
  std::uint64_t prev_addr_ = 0;
  std::unordered_map<std::string, std::uint32_t> op_ids_;
  std::unordered_map<std::string, std::uint32_t> part_ids_;
  std::vector<std::string> part_names_{std::string{}};
  Buffer payload_;
};

} // namespace tile_npu::trace
