#include "itrace_writer.h"

#include "disasm.h"
#include "utils/format.h"

namespace tile_npu::trace {

void ItraceWriter::operator()(std::uint64_t pc, std::uint32_t instr, const DecodeResult &decoded,
                              StepStatus status) {
  out_ << "[ITRACE] " << hex64(pc) << ": " << hex32(instr) << "  " << disassemble(pc, decoded);
  if (status != StepStatus::retired) {
    out_ << "  ; " << status_name(status);
  }
  out_ << '\n' << std::flush;
}

} // namespace tile_npu::trace
