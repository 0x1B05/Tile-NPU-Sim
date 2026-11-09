#pragma once

#include "uarch/dyn_inst.h"

#include <cstdint>
#include <fstream>
#include <string>

namespace tile_npu {

// Versioned binary trace of DynInst records. It is intentionally separate
// from MTRACE: MTRACE is the ISA-level logical event stream, while this trace
// preserves the timing model's unit/shape/address/gather inputs so a replay
// does not need the functional simulator or architectural memory image.
enum class UarchTraceType : std::uint8_t { inst = 0, fence = 1 };

class UarchTraceWriter {
public:
  // start_instruction is the first DynInst id to record; max_records == 0
  // means "to the end". A non-zero start is a window, not a full checkpoint:
  // dependencies from before the window are treated as already satisfied.
  explicit UarchTraceWriter(const std::string &path, std::uint64_t start_instruction = 0,
                            std::uint64_t max_records = 0);
  ~UarchTraceWriter();

  UarchTraceWriter(const UarchTraceWriter &) = delete;
  UarchTraceWriter &operator=(const UarchTraceWriter &) = delete;

  bool good() const { return out_.good(); }

  void write(const DynInst &inst);
  void close();

private:
  void write_record(const DynInst &inst, UarchTraceType type);
  void write_header();

  std::ofstream out_;
  std::uint64_t start_instruction_ = 0;
  std::uint64_t max_records_ = 0;
  std::uint64_t count_ = 0;
  std::uint64_t previous_id_ = 0;
  std::uint64_t previous_pc_ = 0;
  std::uint64_t previous_addr_ = 0;
  bool closed_ = false;
};

class UarchTraceReader {
public:
  explicit UarchTraceReader(const std::string &path);

  bool good() const { return error_.empty(); }

  const std::string &error() const { return error_; }

  std::uint64_t remaining() const { return count_ - read_; }

  // Reads the next record. Returns false at end of trace or on error; check
  // error() to distinguish the two.
  bool next(DynInst &inst, UarchTraceType &type);

private:
  bool read_header();
  bool read_record(DynInst &inst, UarchTraceType &type);
  bool fail(std::string message);

  std::ifstream in_;
  std::uint64_t count_ = 0;
  std::uint64_t read_ = 0;
  std::uint64_t previous_id_ = 0;
  std::uint64_t previous_pc_ = 0;
  std::uint64_t previous_addr_ = 0;
  std::string error_;
};

} // namespace tile_npu
