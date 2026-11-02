#include "itrace_writer.h"

#include "disasm.h"
#include "utils/format.h"

namespace tile_npu::trace {

ItraceWriter::~ItraceWriter() { out_.flush(); }

void ItraceWriter::operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                              StepStatus status) {
  if (!window_.contains(pc)) return;
  out_ << "[ITRACE] " << hex64(pc) << ": " << hex32(instr) << "  " << disassemble(pc, decoded);
  if (status != StepStatus::retired) {
    out_ << "  ; " << status_name(status);
  }
  out_ << '\n';
  if (flush_each_line_) {
    out_.flush();
  } else if ((++lines_ & 0xfff) == 0) {
    out_.flush();
  }
}

} // namespace tile_npu::trace
