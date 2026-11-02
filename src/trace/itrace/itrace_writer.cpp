#include "itrace_writer.h"

#include "disasm.h"
#include "utils/format.h"

static_assert(TRACE_FLUSH_RECORDS > 0, "TRACE_FLUSH_RECORDS must be positive");

namespace tile_npu::trace {

ItraceWriter::~ItraceWriter() { out_.flush(); }

void ItraceWriter::operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                              StepStatus status) {
  out_ << "[ITRACE] " << hex64(pc) << ": " << hex32(instr) << "  " << disassemble(pc, decoded);
  if (status != StepStatus::retired) {
    out_ << "  ; " << status_name(status);
  }
  out_ << '\n';
  if (flush_each_line_) {
    out_.flush();
  } else if ((++records_ % TRACE_FLUSH_RECORDS) == 0) {
    out_.flush();
  }
}

} // namespace tile_npu::trace
