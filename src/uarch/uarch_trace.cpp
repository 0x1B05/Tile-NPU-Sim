#include "uarch/uarch_trace.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <istream>
#include <limits>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>

namespace tile_npu {
namespace {

constexpr std::uint32_t TRACE_MAGIC = 0x55545243u; // "UTRC"
constexpr std::uint16_t TRACE_VERSION = 2;
constexpr std::uint16_t TRACE_HEADER_BYTES = 36;

enum TraceFlags : std::uint8_t {
  FLAG_STORE = 1u << 0,
  FLAG_TRANSPOSE = 1u << 1,
  FLAG_KERNEL_END = 1u << 2,
  FLAG_GATHER = 1u << 3,
  FLAG_MEMORY = 1u << 4,
  FLAG_WORK = 1u << 5,
};

template <typename T> void write_pod(std::ostream &out, const T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  out.write(reinterpret_cast<const char *>(&value), sizeof(T));
}

template <typename T> bool read_pod(std::istream &in, T &value) {
  static_assert(std::is_trivially_copyable_v<T>);
  in.read(reinterpret_cast<char *>(&value), sizeof(T));
  return static_cast<bool>(in);
}

void write_varint(std::ostream &out, std::uint64_t value) {
  while (value >= 0x80) {
    write_pod(out, static_cast<std::uint8_t>((value & 0x7f) | 0x80));
    value >>= 7;
  }
  write_pod(out, static_cast<std::uint8_t>(value));
}

bool read_varint(std::istream &in, std::uint64_t &value) {
  value = 0;
  for (unsigned shift = 0; shift < 64; shift += 7) {
    std::uint8_t byte = 0;
    if (!read_pod(in, byte)) return false;
    value |= static_cast<std::uint64_t>(byte & 0x7f) << shift;
    if ((byte & 0x80) == 0) return true;
  }
  return false;
}

std::uint64_t zigzag(std::int64_t value) {
  return (static_cast<std::uint64_t>(value) << 1) ^ static_cast<std::uint64_t>(value >> 63);
}

std::int64_t unzigzag(std::uint64_t value) {
  return static_cast<std::int64_t>((value >> 1) ^ (~(value & 1) + 1));
}

void write_ref(std::ostream &out, const DataRef &ref) {
  write_pod(out, ref.domain);
  write_pod(out, ref.index);
}

bool read_ref(std::istream &in, DataRef &ref) {
  return read_pod(in, ref.domain) && read_pod(in, ref.index);
}

bool has_work(const DynInst &inst) {
  return inst.rows != 0 || inst.cols != 0 || inst.k != 0 || inst.len != 0 ||
         inst.work_elements != 0 || inst.bytes != 0 || inst.macs != 0;
}

bool has_memory(const DynInst &inst) {
  return inst.op == DynOp::load || inst.op == DynOp::store || inst.op == DynOp::gather;
}

} // namespace

UarchTraceWriter::UarchTraceWriter(const std::string &path, std::uint64_t start_instruction,
                                   std::uint64_t max_records)
    : out_(path, std::ios::binary | std::ios::out | std::ios::trunc),
      start_instruction_(start_instruction), max_records_(max_records) {
  if (out_) write_header();
}

UarchTraceWriter::~UarchTraceWriter() { close(); }

void UarchTraceWriter::write_header() {
  write_pod(out_, TRACE_MAGIC);
  write_pod(out_, TRACE_VERSION);
  write_pod(out_, TRACE_HEADER_BYTES);
  write_pod(out_, std::uint32_t{0});
  write_pod(out_, std::uint64_t{0});
  write_pod(out_, std::uint64_t{0});
  write_pod(out_, std::uint64_t{0});
}

void UarchTraceWriter::write(const DynInst &inst) {
  if (inst.id < start_instruction_) return;
  if (max_records_ != 0 && count_ >= max_records_) return;
  const auto type = inst.op == DynOp::fence && inst.unit == Unit::sync ? UarchTraceType::fence
                                                                       : UarchTraceType::inst;
  write_record(inst, type);
}

void UarchTraceWriter::write_record(const DynInst &inst, UarchTraceType type) {
  if (!out_) return;

  const auto work = has_work(inst);
  const auto memory = has_memory(inst);
  const auto gather = inst.gather.has_value();
  std::uint8_t flags = 0;
  if (inst.is_store) flags |= FLAG_STORE;
  if (inst.transpose) flags |= FLAG_TRANSPOSE;
  if (inst.kernel_end) flags |= FLAG_KERNEL_END;
  if (gather) flags |= FLAG_GATHER;
  if (memory) flags |= FLAG_MEMORY;
  if (work) flags |= FLAG_WORK;

  write_pod(out_, type);
  write_pod(out_, flags);
  write_varint(out_, inst.id - previous_id_);
  write_varint(
      out_, zigzag(static_cast<std::int64_t>(inst.pc) - static_cast<std::int64_t>(previous_pc_)));
  write_pod(out_, inst.op);
  write_pod(out_, inst.unit);
  write_pod(out_, inst.src_count);
  write_pod(out_, inst.dst_count);
  for (std::uint8_t i = 0; i < inst.src_count; ++i)
    write_ref(out_, inst.src[i]);
  for (std::uint8_t i = 0; i < inst.dst_count; ++i)
    write_ref(out_, inst.dst[i]);

  if (work) {
    write_varint(out_, inst.rows);
    write_varint(out_, inst.cols);
    write_varint(out_, inst.k);
    write_varint(out_, inst.len);
    write_varint(out_, inst.work_elements);
    write_varint(out_, inst.bytes);
    write_varint(out_, inst.macs);
    write_pod(out_, inst.dtype);
    write_pod(out_, inst.elem_bytes);
  }

  if (memory) {
    write_varint(out_, zigzag(static_cast<std::int64_t>(inst.addr) -
                              static_cast<std::int64_t>(previous_addr_)));
    write_varint(out_, zigzag(inst.row_stride));
    write_varint(out_, zigzag(inst.col_stride));
  }

  if (type == UarchTraceType::fence) write_pod(out_, inst.fence);

  if (gather) {
    const auto count = inst.gather->count;
    write_varint(out_, count);
    for (std::uint8_t i = 0; i < count; ++i) {
      write_varint(out_, inst.gather->indices[i]);
    }
  }

  previous_id_ = inst.id;
  previous_pc_ = inst.pc;
  if (memory) previous_addr_ = inst.addr;
  ++count_;
}

void UarchTraceWriter::close() {
  if (closed_) return;
  closed_ = true;
  if (!out_) return;
  // Header layout: magic[4], version[2], header_bytes[2], reserved[4],
  // record_count[8], isa_hash[8], workload_hash[8].
  out_.seekp(12);
  write_pod(out_, count_);
  out_.flush();
  out_.close();
}

UarchTraceReader::UarchTraceReader(const std::string &path) : in_(path, std::ios::binary) {
  if (!in_) {
    error_ = "cannot open trace: " + path;
    return;
  }
  read_header();
}

bool UarchTraceReader::fail(std::string message) {
  error_ = std::move(message);
  return false;
}

bool UarchTraceReader::read_header() {
  std::uint32_t magic = 0;
  std::uint16_t version = 0;
  std::uint16_t header_bytes = 0;
  std::uint32_t reserved = 0;
  std::uint64_t isa_hash = 0;
  std::uint64_t workload_hash = 0;
  if (!read_pod(in_, magic) || !read_pod(in_, version) || !read_pod(in_, header_bytes) ||
      !read_pod(in_, reserved) || !read_pod(in_, count_) || !read_pod(in_, isa_hash) ||
      !read_pod(in_, workload_hash)) {
    return fail("truncated trace header");
  }
  if (magic != TRACE_MAGIC) return fail("bad trace magic");
  if (version != TRACE_VERSION) return fail("unsupported trace version");
  if (header_bytes != TRACE_HEADER_BYTES) return fail("unsupported trace header size");
  return true;
}

bool UarchTraceReader::read_record(DynInst &inst, UarchTraceType &type) {
  std::uint8_t flags = 0;
  std::uint64_t id_delta = 0;
  std::uint64_t pc_delta = 0;
  if (!read_pod(in_, type) || !read_pod(in_, flags) || !read_varint(in_, id_delta) ||
      !read_varint(in_, pc_delta)) {
    return fail("truncated trace record");
  }

  inst.id = previous_id_ + id_delta;
  inst.pc =
      static_cast<std::uint64_t>(static_cast<std::int64_t>(previous_pc_) + unzigzag(pc_delta));
  if (!read_pod(in_, inst.op) || !read_pod(in_, inst.unit) || !read_pod(in_, inst.src_count) ||
      !read_pod(in_, inst.dst_count)) {
    return fail("truncated trace record header");
  }
  if (inst.src_count > inst.src.size() || inst.dst_count > inst.dst.size()) {
    return fail("trace operand count out of range");
  }
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    if (!read_ref(in_, inst.src[i])) return fail("truncated trace source");
  }
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    if (!read_ref(in_, inst.dst[i])) return fail("truncated trace destination");
  }

  inst.is_store = (flags & FLAG_STORE) != 0;
  inst.transpose = (flags & FLAG_TRANSPOSE) != 0;
  inst.kernel_end = (flags & FLAG_KERNEL_END) != 0;

  if ((flags & FLAG_WORK) != 0) {
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t k = 0;
    std::uint64_t len = 0;
    std::uint64_t work_elements = 0;
    std::uint64_t bytes = 0;
    std::uint64_t macs = 0;
    if (!read_varint(in_, rows) || !read_varint(in_, cols) || !read_varint(in_, k) ||
        !read_varint(in_, len) || !read_varint(in_, work_elements) || !read_varint(in_, bytes) ||
        !read_varint(in_, macs) || !read_pod(in_, inst.dtype) || !read_pod(in_, inst.elem_bytes)) {
      return fail("truncated trace work fields");
    }
    inst.rows = static_cast<std::uint32_t>(rows);
    inst.cols = static_cast<std::uint32_t>(cols);
    inst.k = static_cast<std::uint32_t>(k);
    inst.len = static_cast<std::uint32_t>(len);
    inst.work_elements = static_cast<std::uint32_t>(work_elements);
    inst.bytes = static_cast<std::uint32_t>(bytes);
    inst.macs = static_cast<std::uint32_t>(macs);
  }

  if ((flags & FLAG_MEMORY) != 0) {
    std::uint64_t addr_delta = 0;
    std::uint64_t row_stride = 0;
    std::uint64_t col_stride = 0;
    if (!read_varint(in_, addr_delta) || !read_varint(in_, row_stride) ||
        !read_varint(in_, col_stride)) {
      return fail("truncated trace address fields");
    }
    inst.addr = static_cast<std::uint64_t>(static_cast<std::int64_t>(previous_addr_) +
                                           unzigzag(addr_delta));
    inst.row_stride = static_cast<std::int32_t>(unzigzag(row_stride));
    inst.col_stride = static_cast<std::int32_t>(unzigzag(col_stride));
    previous_addr_ = inst.addr;
  }

  if (type == UarchTraceType::fence) {
    if (!read_pod(in_, inst.fence)) return fail("truncated trace fence");
  }

  if ((flags & FLAG_GATHER) != 0) {
    std::uint64_t count = 0;
    if (!read_varint(in_, count) || count > VECTOR_LANES) {
      return fail("trace gather count out of range");
    }
    GatherInfo gather;
    gather.count = static_cast<std::uint8_t>(count);
    for (std::uint8_t i = 0; i < gather.count; ++i) {
      std::uint64_t index = 0;
      if (!read_varint(in_, index)) return fail("truncated gather index");
      gather.indices[i] = static_cast<std::uint32_t>(index);
    }
    inst.gather = gather;
  }

  previous_id_ = inst.id;
  previous_pc_ = inst.pc;
  return true;
}

bool UarchTraceReader::next(DynInst &inst, UarchTraceType &type) {
  if (!error_.empty()) return false;
  if (read_ >= count_) return false;
  inst = DynInst{};
  if (!read_record(inst, type)) return false;
  ++read_;
  return true;
}

} // namespace tile_npu
