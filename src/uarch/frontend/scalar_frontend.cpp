#include "uarch/frontend/scalar_frontend.h"

#include "execute/execute.h"
#include "uarch/dataflow/npu_config.h"
#include "uarch/dataflow/npu_instruction.h"
#include "uarch/dataflow/npu_operands.h"
#include "uarch/frontend/execute/scalar_memory.h"
#include "uarch/frontend/scalar_hazards.h"
#include "utils/endian.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <variant>

namespace tile_npu {

ScalarFrontend::ScalarFrontend(const UarchClock &clock, State &state, AddressSpace &memory,
                               InstBuffer &inst_buffer)
    : clock_(clock), state_(state), memory_(memory), writeback_(state_, scoreboard_),
      inst_buffer_(inst_buffer) {}

void ScalarFrontend::reset(std::uint64_t entry_pc) {
  // The owner must reset the NPU executor first so no in-flight writeback can
  // arrive after the scoreboard is cleared.
  if_id_.reset();
  id_ex_.reset();
  ex_mem_.reset();
  mem_wb_.reset();
  fetch_pc_ = entry_pc;
  fetch_stopped_ = false;
  halted_ = false;
  stop_.reset();
  writeback_.reset();
  state_.scalar.pc = entry_pc;
}

std::optional<PipelineEntry> ScalarFrontend::fetch() {
  if (fetch_stopped_ || halted_) return std::nullopt;

  PipelineEntry entry;
  entry.pc = fetch_pc_;

  std::array<std::byte, INSTR_BYTES> bytes{};
  if (!memory_.read(entry.pc, bytes)) {
    entry.fault = StepStatus::fetch_fault;
    fetch_stopped_ = true;
    return entry;
  }

  entry.instr = load_le<std::uint32_t>(bytes);
  fetch_pc_ += INSTR_BYTES;
  return entry;
}

bool ScalarFrontend::prepare_id(PipelineEntry &entry) {
  if (entry.fault) return false;

  if (!entry.decoded) {
    auto decoded = decode(entry.instr);
    if (std::holds_alternative<DecodeError>(decoded)) {
      entry.fault = StepStatus::illegal_instruction;
      fetch_stopped_ = true;
      return false;
    }
    entry.decoded = std::move(decoded);
  }

  const auto busy = [this](RegisterIndex index) { return index != 0 && writeback_.busy(index); };

  const auto *scalar = std::get_if<DecodedScalarInstr>(&*entry.decoded);
  if (scalar == nullptr) {
    const auto *npu = std::get_if<DecodedNpuInstr>(&*entry.decoded);
    if (npu == nullptr) return false;
    if (reads_config(*npu) && has_config_stall(id_ex_, ex_mem_)) return true;

    const auto reads = scalar_reads(*npu);
    for (std::size_t i = 0; i < reads.count; ++i) {
      if (busy(reads.values[i])) return true;
    }

    const auto dest = scalar_write_dest(*npu);
    if (dest && busy(*dest)) return true;

    if (is_scalar_npu(*npu)) {
      if (dest) {
        writeback_.mark_busy(*dest);
        entry.busy_dest = dest;
      }
      return false;
    }

    if (has_stall_hazard(id_ex_, ex_mem_, reads)) return true;

    entry.scalar = {};
    for (std::size_t i = 0; i < reads.count; ++i) {
      entry.scalar.values[i] = ScalarOperand{
          .index = reads.values[i],
          .value = read_scalar_value(reads.values[i], id_ex_, ex_mem_, state_),
      };
    }
    entry.scalar.count = reads.count;
    entry.config = capture_config_operands(*npu, state_);

    // Only reserve the destination after every stall check. Otherwise a
    // retried instruction observes its own busy bit and stalls forever.
    if (dest) {
      writeback_.mark_busy(*dest);
      entry.busy_dest = dest;
    }
    return false;
  }
  if (busy(scalar->regs.src1) || busy(scalar->regs.src2) || busy(scalar->regs.dest)) return true;
  if (has_stall_hazard(id_ex_, ex_mem_, *scalar)) return true;

  entry.lhs = read_scalar_value(scalar->regs.src1, id_ex_, ex_mem_, state_);
  entry.rhs = read_scalar_value(scalar->regs.src2, id_ex_, ex_mem_, state_);
  return false;
}

void ScalarFrontend::execute_ex(PipelineEntry &entry) {
  if (entry.fault || !entry.decoded) return;

  const auto *scalar = std::get_if<DecodedScalarInstr>(&*entry.decoded);
  if (scalar == nullptr || !is_scalar_compute_op(scalar->op)) return;

  entry.compute = execute_scalar_compute(*scalar, entry.lhs, entry.rhs, entry.pc);
  if (entry.compute->status != StepStatus::retired) {
    entry.fault = entry.compute->status;
    entry.fault_address = entry.compute->fault_address;
    fetch_stopped_ = true;
    discard(if_id_);
    return;
  }

  if (entry.compute->redirect) {
    fetch_pc_ = *entry.compute->redirect;
    fetch_stopped_ = false;
    discard(if_id_);
  }
}

void ScalarFrontend::execute_mem(PipelineEntry &entry) {
  if (entry.fault || !entry.decoded) return;

  const auto *scalar = std::get_if<DecodedScalarInstr>(&*entry.decoded);
  if (scalar == nullptr || !is_scalar_memory_op(scalar->op)) return;

  entry.memory = execute_scalar_memory(*scalar, entry.lhs, entry.rhs, memory_);
  if (entry.memory->status != StepStatus::retired) {
    entry.fault = entry.memory->status;
    entry.fault_address = entry.memory->fault_address;
    fetch_stopped_ = true;
    discard(id_ex_);
    discard(if_id_);
  }
}

ScalarFrontend::WbOutcome ScalarFrontend::execute_wb(const PipelineEntry &entry) {
  if (entry.fault) {
    StepResult stop{.status = *entry.fault, .pc = entry.pc, .fault_address = entry.fault_address};
    if (*entry.fault != StepStatus::fetch_fault) stop.instr = entry.instr;
    return {.stop = stop};
  }

  if (!entry.decoded || std::holds_alternative<DecodeError>(*entry.decoded)) {
    return {.stop = StepResult{
                .status = StepStatus::illegal_instruction, .pc = entry.pc, .instr = entry.instr}};
  }

  if (const auto *npu = std::get_if<DecodedNpuInstr>(&*entry.decoded)) {
    if (is_scalar_npu(*npu)) {
      if (!is_config_npu(*npu)) {
        return {.stop = StepResult{.status = StepStatus::execution_not_implemented,
                                   .pc = entry.pc,
                                   .instr = entry.instr}};
      }

      state_.scalar.pc = entry.pc;
      ExecContext ctx{state_, memory_};
      const auto execution = exec_npu_instr(*npu, ctx);
      if (execution.status != StepStatus::retired) {
        return {.stop = StepResult{.status = execution.status,
                                   .pc = entry.pc,
                                   .instr = entry.instr,
                                   .fault_address = execution.fault_address}};
      }

      state_.scalar.pc = entry.pc + INSTR_BYTES;
      if (const auto dest = scalar_write_dest(*npu)) writeback_.cancel(*dest);
      return {.retired = true};
    }

    const InstBuffer::Entry command{
        .pc = entry.pc,
        .instr = entry.instr,
        .decoded = *npu,
        .scalar = entry.scalar,
        .config = entry.config,
    };
    if (!inst_buffer_.enqueue(command)) return {.stalled = true};

    state_.scalar.pc = entry.pc + INSTR_BYTES;
    return {.retired = true};
  }

  const auto &scalar = std::get<DecodedScalarInstr>(*entry.decoded);

  if (entry.memory) {
    if (entry.memory->status != StepStatus::retired) {
      return {.stop = StepResult{.status = entry.memory->status,
                                 .pc = entry.pc,
                                 .instr = entry.instr,
                                 .fault_address = entry.memory->fault_address}};
    }

    state_.scalar.pc = entry.pc + INSTR_BYTES;
    if (entry.memory->writes_dest) {
      state_.scalar.write_x(scalar.regs.dest, entry.memory->value);
    }
    return {.retired = true};
  }

  if (entry.compute) {
    if (entry.compute->status != StepStatus::retired) {
      return {.stop = StepResult{.status = entry.compute->status,
                                 .pc = entry.pc,
                                 .instr = entry.instr,
                                 .fault_address = entry.compute->fault_address}};
    }

    state_.scalar.pc = entry.compute->redirect.value_or(entry.pc + INSTR_BYTES);
    if (entry.compute->writes_dest) {
      state_.scalar.write_x(scalar.regs.dest, entry.compute->value);
    }
    return {.retired = true};
  }

  state_.scalar.pc = entry.pc;

  ExecContext ctx{state_, memory_};
  const auto execution = exec_scalar_instr(scalar, ctx);
  if (execution.status != StepStatus::retired) {
    return {.stop = StepResult{.status = execution.status,
                               .pc = entry.pc,
                               .instr = entry.instr,
                               .fault_address = execution.fault_address}};
  }

  state_.scalar.pc = execution.redirect.value_or(entry.pc + INSTR_BYTES);
  return {.redirect = execution.redirect, .retired = true};
}

void ScalarFrontend::discard(std::optional<PipelineEntry> &entry) {
  if (entry && entry->busy_dest) writeback_.cancel(*entry->busy_dest);
  entry.reset();
}

void ScalarFrontend::clear_pipeline() {
  discard(if_id_);
  discard(id_ex_);
  discard(ex_mem_);
  discard(mem_wb_);
}

std::size_t ScalarFrontend::process_cur_cycle() {
  std::size_t retired = 0;
  if (halted_) return retired;

  if (mem_wb_) {
    auto outcome = execute_wb(*mem_wb_);
    if (outcome.stop) {
      halted_ = true;
      fetch_stopped_ = true;
      stop_ = std::move(outcome.stop);
      clear_pipeline();
      return retired;
    }

    if (outcome.stalled) return retired;

    if (outcome.retired) ++retired;

    if (outcome.redirect) {
      fetch_pc_ = *outcome.redirect;
      fetch_stopped_ = false;
      clear_pipeline();
      return retired;
    }
  }

  if (id_ex_) execute_ex(*id_ex_);

  if (ex_mem_) execute_mem(*ex_mem_);

  const bool id_stalled = if_id_ && prepare_id(*if_id_);

  mem_wb_ = std::move(ex_mem_);
  ex_mem_ = std::move(id_ex_);
  if (id_stalled) {
    // Freeze IF/ID and insert a bubble into EX.
    id_ex_.reset();
  } else {
    id_ex_ = std::move(if_id_);
    if_id_ = fetch();
  }

  return retired;
}

std::optional<StepResult> ScalarFrontend::take_stop() {
  auto stop = stop_;
  stop_.reset();
  return stop;
}

std::optional<UarchClock::Time> ScalarFrontend::next_activity_time() const {
  if (halted_) return std::nullopt;
  return clock_.now() + 1;
}

} // namespace tile_npu
