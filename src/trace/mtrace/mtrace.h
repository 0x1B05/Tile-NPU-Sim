#pragma once

#include "tile_npu/decode.h"
#include "tile_npu/mtrace.h"

#include <cstdint>

namespace tile_npu {

// Dataflow events for instructions that do not touch memory. Memory moves are
// emitted by the M_BLK executor while the descriptor and access pattern are
// still live.
void trace_npu_dataflow(const DecodedNpuInstr &instr, const State &state, MtraceRecorder &trace);

// Memory-side events. The caller passes the pre-access descriptor values so
// tracing does not observe the committed base.
void trace_matrix_move(MtraceRecorder &trace, const MblkInstr &instr,
                       const MatrixDescriptor &descriptor, std::uint8_t rows, std::uint8_t cols,
                       std::uint8_t elem, bool transpose);

void trace_matrix_row_move(MtraceRecorder &trace, const MblkInstr &instr,
                           const MatrixDescriptor &descriptor, std::uint8_t cols,
                           std::uint8_t elem);

void trace_vector_move(MtraceRecorder &trace, const MblkInstr &instr,
                       const VectorDescriptor &descriptor, std::uint8_t len, std::uint8_t elem);

void trace_gather_tile(MtraceRecorder &trace, const MblkInstr &instr,
                       const MatrixDescriptor &descriptor, std::uint8_t indexed_rows,
                       std::uint8_t cols, std::uint8_t elem, const Vec32 &index);

void trace_gather_vector(MtraceRecorder &trace, const MblkInstr &instr,
                         const VectorDescriptor &descriptor, std::uint8_t indexed,
                         std::uint8_t elem, const Vec32 &index);

} // namespace tile_npu
