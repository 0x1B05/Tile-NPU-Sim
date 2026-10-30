#include "mtrace_writer.h"

#ifdef CONFIG_MTRACE

#include "utils/format.h"

#include <ostream>
#include <string_view>

namespace tile_npu::monitor {
namespace {

std::string_view endpoint_name(TraceEndpointKind kind) {
  switch (kind) {
  case TraceEndpointKind::memory: return "mem";
  case TraceEndpointKind::tile: return "tile";
  case TraceEndpointKind::acc: return "acc";
  case TraceEndpointKind::vec32: return "vec32";
  case TraceEndpointKind::immediate: return "imm";
  }
  return "unknown";
}

std::string_view region_name(TraceRegion region) {
  switch (region) {
  case TraceRegion::ram: return "ram";
  case TraceRegion::serial: return "serial";
  case TraceRegion::mmio: return "mmio";
  case TraceRegion::unknown: return "unknown";
  }
  return "unknown";
}

void write_descriptor(std::ostream &out, TraceDescriptorKind kind, std::uint8_t index) {
  if (kind == TraceDescriptorKind::none) return;
  out << ",\"desc\":\"";
  switch (kind) {
  case TraceDescriptorKind::tm: out << 't'; break;
  case TraceDescriptorKind::am: out << 'a'; break;
  case TraceDescriptorKind::vm: out << 'v'; break;
  default: break;
  }
  out << "m" << static_cast<unsigned>(index) << '"';
}

void write_endpoint(std::ostream &out, const TraceEndpoint &endpoint) {
  out << "{\"k\":\"" << endpoint_name(endpoint.kind) << '"';
  if (endpoint.kind != TraceEndpointKind::memory && endpoint.kind != TraceEndpointKind::immediate) {
    out << ",\"r\":" << static_cast<unsigned>(endpoint.index);
  }
  out << ",\"bytes\":" << endpoint.bytes;
  if (endpoint.elem != 0) out << ",\"elem\":" << static_cast<unsigned>(endpoint.elem);
  if (endpoint.dims == 1) {
    out << ",\"shape\":[" << static_cast<unsigned>(endpoint.shape[0]) << ']';
  } else if (endpoint.dims == 2) {
    out << ",\"shape\":[" << static_cast<unsigned>(endpoint.shape[0]) << ','
        << static_cast<unsigned>(endpoint.shape[1]) << ']';
  }
  if (!endpoint.part.empty()) out << ",\"part\":\"" << endpoint.part << '"';
  out << '}';
}

void write_memory(std::ostream &out, const TraceMemory &memory) {
  if (memory.kind == TraceMemoryKind::none) return;
  out << ",\"mem\":{\"addr\":\"" << hex64(memory.addr) << "\",\"region\":\""
      << region_name(memory.region) << "\",\"elem\":" << static_cast<unsigned>(memory.elem);
  if (memory.kind == TraceMemoryKind::linear) {
    out << ",\"pattern\":\"linear\",\"count\":" << memory.count << ",\"stride\":" << memory.stride;
  } else {
    out << ",\"pattern\":\"block\",\"shape\":[" << static_cast<unsigned>(memory.rows) << ','
        << static_cast<unsigned>(memory.cols) << "],\"stride\":[" << memory.row_stride << ','
        << memory.col_stride << "],\"order\":\"" << (memory.col_order ? "col" : "row") << '"';
  }
  write_descriptor(out, memory.desc_kind, memory.desc_index);
  out << '}';
}

} // namespace

MtraceWriter::~MtraceWriter() { out_.flush(); }

void MtraceWriter::write_header() {
  out_ << "{\"t\":\"header\",\"v\":1,\"arch\":\"tile-npu\",\"format\":\"jsonl\","
          "\"time\":\"instruction\"}\n";
}

void MtraceWriter::operator()(const TraceEvent &event) {
  out_ << "{\"t\":\"" << (event.type == TraceEventType::move ? "move" : "compute")
       << "\",\"seq\":" << event.seq << ",\"i\":" << event.instr << ",\"pc\":\"" << hex64(event.pc)
       << "\",\"op\":\"" << event.op << '"';

  if (event.type == TraceEventType::move) {
    if (event.src_count == 1) {
      out_ << ",\"src\":";
      write_endpoint(out_, event.src[0]);
    } else if (event.src_count > 1) {
      out_ << ",\"src\":[";
      for (std::uint8_t i = 0; i < event.src_count; ++i) {
        if (i != 0) out_ << ',';
        write_endpoint(out_, event.src[i]);
      }
      out_ << ']';
    }
    if (event.dst_count == 1) {
      out_ << ",\"dst\":";
      write_endpoint(out_, event.dst[0]);
    } else if (event.dst_count > 1) {
      out_ << ",\"dst\":[";
      for (std::uint8_t i = 0; i < event.dst_count; ++i) {
        if (i != 0) out_ << ',';
        write_endpoint(out_, event.dst[i]);
      }
      out_ << ']';
    }
  } else {
    out_ << ",\"src\":[";
    for (std::uint8_t i = 0; i < event.src_count; ++i) {
      if (i != 0) out_ << ',';
      write_endpoint(out_, event.src[i]);
    }
    out_ << "],\"dst\":[";
    for (std::uint8_t i = 0; i < event.dst_count; ++i) {
      if (i != 0) out_ << ',';
      write_endpoint(out_, event.dst[i]);
    }
    out_ << ']';
  }

  write_memory(out_, event.mem);
  if (event.macs != 0) out_ << ",\"macs\":" << event.macs;
  out_ << "}\n";

  if ((++events_ & 0xfff) == 0) out_.flush();
}

} // namespace tile_npu::monitor

#endif // CONFIG_MTRACE
