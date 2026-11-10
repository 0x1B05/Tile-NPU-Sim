#include "uarch/dataflow/dyn_inst_builder.h"

#include "framework.h"
#include "uarch/dyn_inst.h"
#include "uarch/frontend.h"

#include <cstdint>

using namespace tile_npu;

namespace {

bool has_src(const DynInst &inst, NpuDomain domain, RegisterIndex index) {
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    if (inst.src[i].domain == domain && inst.src[i].index == index) return true;
  }
  return false;
}

bool has_dst(const DynInst &inst, NpuDomain domain, RegisterIndex index) {
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    if (inst.dst[i].domain == domain && inst.dst[i].index == index) return true;
  }
  return false;
}

} // namespace

TEST_CASE(dyn_inst_builder_mma_shape_and_accumulator) {
  State state{};
  state.tc[1] = TileConfig{.dtype = DType::i8, .rows = 4, .cols = 8};
  state.tc[2] = TileConfig{.dtype = DType::i8, .rows = 5, .cols = 8};

  const MmaInstr mma{.op = MmaOp::nt_acc,
                     .dest = NpuRegister{.domain = NpuDomain::acc, .index = 3},
                     .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                     .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 2}};
  const auto inst = make_npu_dyn_inst(7, 0x80000000, DecodedNpuInstr{mma}, state);

  check_eq(inst.op, DynOp::mma, "mma op");
  check_eq(inst.unit, Unit::mma, "mma unit");
  check_eq(inst.rows, std::uint32_t{4}, "mma M");
  check_eq(inst.cols, std::uint32_t{5}, "mma N");
  check_eq(inst.k, std::uint32_t{8}, "mma K");
  check_eq(inst.macs, std::uint32_t{160}, "mma MAC count");
  check_eq(inst.dtype, static_cast<std::uint8_t>(DType::i8), "mma dtype");
  check_eq(inst.elem_bytes, std::uint8_t{1}, "mma element bytes");
  check_eq(inst.src_count, std::uint8_t{3}, "mma sources include the accumulator");
  check(has_src(inst, NpuDomain::tile, 1), "mma lhs source");
  check(has_src(inst, NpuDomain::tile, 2), "mma rhs source");
  check(has_src(inst, NpuDomain::acc, 3), "mma accumulator source");
  check_eq(inst.dst_count, std::uint8_t{1}, "mma has one destination");
  check_eq(inst.dst[0].domain, NpuDomain::acc, "mma destination domain");
  check_eq(inst.dst[0].index, RegisterIndex{3}, "mma destination index");
}

TEST_CASE(dyn_inst_builder_vmma_shape) {
  State state{};
  state.tc[1] = TileConfig{.dtype = DType::i8, .rows = 1, .cols = 8};
  state.tc[2] = TileConfig{.dtype = DType::i8, .rows = 5, .cols = 8};

  const VmmaInstr vmma{.op = VmmaOp::nt_zero,
                       .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                       .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                       .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 2},
                       .row = 0};
  const auto inst = make_npu_dyn_inst(8, 0x80000004, DecodedNpuInstr{vmma}, state);

  check_eq(inst.op, DynOp::vmma, "vmma op");
  check_eq(inst.unit, Unit::mma, "vmma unit");
  check_eq(inst.rows, std::uint32_t{1}, "vmma M");
  check_eq(inst.cols, std::uint32_t{5}, "vmma N");
  check_eq(inst.k, std::uint32_t{8}, "vmma K");
  check_eq(inst.macs, std::uint32_t{40}, "vmma MAC count");
  check_eq(inst.src_count, std::uint8_t{2}, "vmma has two sources without accumulate");
  check_eq(inst.dst_count, std::uint8_t{1}, "vmma has one destination");
  check_eq(inst.dst[0].domain, NpuDomain::vec32, "vmma destination domain");

  const VmmaInstr vmma_acc{.op = VmmaOp::nt_acc,
                           .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                           .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                           .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 2},
                           .row = 0};
  const auto acc_inst = make_npu_dyn_inst(9, 0, DecodedNpuInstr{vmma_acc}, state);
  check_eq(acc_inst.src_count, std::uint8_t{3}, "vmma accumulate reads its destination");
  check(has_src(acc_inst, NpuDomain::vec32, 2), "vmma accumulate source");
}

TEST_CASE(dyn_inst_builder_mblk_descriptor_and_direction) {
  State state{};
  state.tc[4] = TileConfig{.dtype = DType::i8, .rows = 3, .cols = 5};
  state.tm[4] = MatrixDescriptor{.base_addr = 0x2000,
                                 .row_stride_bytes = 8,
                                 .col_stride_bytes = 1,
                                 .transform = 0,
                                 .flags = 0};

  const MblkInstr load{.op = MblkOp::load,
                       .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4}};
  const auto load_inst = make_npu_dyn_inst(9, 0x80000008, DecodedNpuInstr{load}, state);
  check_eq(load_inst.op, DynOp::load, "load op");
  check_eq(load_inst.unit, Unit::dma, "load unit");
  check_eq(load_inst.rows, std::uint32_t{3}, "load rows");
  check_eq(load_inst.cols, std::uint32_t{5}, "load cols");
  check_eq(load_inst.bytes, std::uint32_t{15}, "load bytes");
  check_eq(hex(load_inst.addr), hex(std::uint64_t{0x2000}), "load base address");
  check_eq(load_inst.row_stride, std::int32_t{8}, "load row stride");
  check_eq(load_inst.col_stride, std::int32_t{1}, "load col stride");
  check(!load_inst.is_store, "load is not a store");
  check_eq(load_inst.src_count, std::uint8_t{0}, "load has no register source");
  check(has_dst(load_inst, NpuDomain::tile, 4), "load destination");

  const MblkInstr store{.op = MblkOp::store,
                        .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4}};
  const auto store_inst = make_npu_dyn_inst(10, 0x8000000c, DecodedNpuInstr{store}, state);
  check(store_inst.is_store, "store flag");
  check_eq(store_inst.src_count, std::uint8_t{1}, "store has a register source");
  check(has_src(store_inst, NpuDomain::tile, 4), "store source");
  check_eq(store_inst.dst_count, std::uint8_t{0}, "store has no destination");

  state.ac[3] = AccConfig{.dtype = DType::f32, .rows = 2, .cols = 4};
  state.am[3] = MatrixDescriptor{.base_addr = 0x3000,
                                 .row_stride_bytes = 16,
                                 .col_stride_bytes = 4,
                                 .transform = 1,
                                 .flags = 0};
  const MblkInstr acc_load{.op = MblkOp::load,
                           .reg = NpuRegister{.domain = NpuDomain::acc, .index = 3}};
  const auto acc_load_inst = make_npu_dyn_inst(11, 0x80000010, DecodedNpuInstr{acc_load}, state);
  check_eq(acc_load_inst.bytes, std::uint32_t{32}, "acc load bytes");
  check_eq(acc_load_inst.elem_bytes, std::uint8_t{4}, "acc load element bytes");
  check(acc_load_inst.transpose, "acc load transpose flag");
  check(has_dst(acc_load_inst, NpuDomain::acc, 3), "acc load destination");

  state.vc[2] = Vec32Config{.dtype = DType::i8, .len = 8};
  state.vm[2] = VectorDescriptor{.base_addr = 0x4000, .stride_bytes = 1, .flags = 0};
  const MblkInstr vec_load{.op = MblkOp::load,
                           .reg = NpuRegister{.domain = NpuDomain::vec32, .index = 2}};
  const auto vec_load_inst = make_npu_dyn_inst(12, 0x80000014, DecodedNpuInstr{vec_load}, state);
  check_eq(vec_load_inst.len, std::uint32_t{8}, "vector load length");
  check_eq(vec_load_inst.bytes, std::uint32_t{8}, "vector load bytes");
  check(has_dst(vec_load_inst, NpuDomain::vec32, 2), "vector load destination");
}

TEST_CASE(dyn_inst_builder_gather_clips_to_index_length) {
  State state{};
  state.tc[4] = TileConfig{.dtype = DType::i8, .rows = 4, .cols = 8};
  state.tm[4] = MatrixDescriptor{.base_addr = 0x2000,
                                 .row_stride_bytes = 8,
                                 .col_stride_bytes = 1,
                                 .transform = 0,
                                 .flags = 0};
  state.vc[1] = Vec32Config{.dtype = DType::u32, .len = 2};
  state.vc[2] = Vec32Config{.dtype = DType::u32, .len = 32};

  const MblkInstr short_tile{.op = MblkOp::gather,
                             .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4},
                             .gather_src = 1};
  const auto short_tile_inst = make_npu_dyn_inst(13, 0, DecodedNpuInstr{short_tile}, state);
  check_eq(short_tile_inst.rows, std::uint32_t{2}, "tile gather rows clipped to indices");
  check_eq(short_tile_inst.cols, std::uint32_t{8}, "tile gather keeps row width");
  check_eq(short_tile_inst.bytes, std::uint32_t{16}, "tile gather bytes");
  check(short_tile_inst.gather.has_value(), "tile gather records indices");
  check_eq(short_tile_inst.gather->count, std::uint8_t{2}, "tile gather index count");
  check(has_src(short_tile_inst, NpuDomain::vec32, 1), "tile gather index source");
  check(has_dst(short_tile_inst, NpuDomain::tile, 4), "tile gather destination");

  const MblkInstr long_tile{.op = MblkOp::gather,
                            .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4},
                            .gather_src = 2};
  const auto long_tile_inst = make_npu_dyn_inst(14, 0, DecodedNpuInstr{long_tile}, state);
  check_eq(long_tile_inst.rows, std::uint32_t{4}, "tile gather rows clipped to tile shape");
  check_eq(long_tile_inst.bytes, std::uint32_t{32}, "extra indices add no transfers");

  state.vc[3] = Vec32Config{.dtype = DType::i8, .len = 8};
  state.vm[3] = VectorDescriptor{.base_addr = 0x4000, .stride_bytes = 1, .flags = 0};
  const MblkInstr short_vec{.op = MblkOp::gather,
                            .reg = NpuRegister{.domain = NpuDomain::vec32, .index = 3},
                            .gather_src = 1};
  const auto short_vec_inst = make_npu_dyn_inst(15, 0, DecodedNpuInstr{short_vec}, state);
  check_eq(short_vec_inst.len, std::uint32_t{2}, "vector gather length clipped to indices");
  check_eq(short_vec_inst.rows, std::uint32_t{2}, "vector gather rows follow indices");
  check_eq(short_vec_inst.bytes, std::uint32_t{2}, "vector gather bytes");
  check(has_dst(short_vec_inst, NpuDomain::vec32, 3), "vector gather destination");
}

TEST_CASE(frontend_npu_classification) {
  check(is_frontend_npu(DecodedNpuInstr{CfgSetiInstr{}}), "cfg.seti is frontend");
  check(is_frontend_npu(DecodedNpuInstr{CfgGetInstr{}}), "cfg.get is frontend");
  check(is_frontend_npu(DecodedNpuInstr{SysInstr{.op = SysOp::fence_sa}}), "fence is frontend");
  check(!is_frontend_npu(DecodedNpuInstr{EbinInstr{}}), "elementwise is backend");
  check(!is_frontend_npu(DecodedNpuInstr{MblkInstr{}}), "memory is backend");
}

TEST_CASE(dyn_inst_builder_move_fill_and_lane) {
  State state{};
  state.ac[1] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 3};
  state.vc[1] = Vec32Config{.dtype = DType::i32, .len = 4};
  state.vc[2] = Vec32Config{.dtype = DType::i32, .len = 4};

  const FillInstr fill{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                       .value = ScalarRegister{.index = 5}};
  const auto fill_inst = make_npu_dyn_inst(20, 0, DecodedNpuInstr{fill}, state);
  check_eq(fill_inst.op, DynOp::fill, "fill op");
  check_eq(fill_inst.unit, Unit::vec32, "fill unit");
  check_eq(fill_inst.rows, std::uint32_t{2}, "fill rows");
  check_eq(fill_inst.cols, std::uint32_t{3}, "fill cols");
  check_eq(fill_inst.bytes, std::uint32_t{24}, "fill bytes");
  check_eq(fill_inst.src_count, std::uint8_t{0}, "scalar fill has no NPU source");
  check(has_dst(fill_inst, NpuDomain::acc, 1), "fill destination");

  const FillInstr fill_imm{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                           .value = FillImm16{.bits = 0x3c00}};
  const auto fill_imm_inst = make_npu_dyn_inst(21, 0, DecodedNpuInstr{fill_imm}, state);
  check_eq(fill_imm_inst.src_count, std::uint8_t{0}, "immediate fill has no source");

  const MoveInstr move{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                       .src = NpuRegister{.domain = NpuDomain::vec32, .index = 2}};
  const auto move_inst = make_npu_dyn_inst(22, 0, DecodedNpuInstr{move}, state);
  check_eq(move_inst.op, DynOp::move, "move op");
  check_eq(move_inst.len, std::uint32_t{4}, "move length");
  check(has_src(move_inst, NpuDomain::vec32, 2), "move source");
  check(has_dst(move_inst, NpuDomain::vec32, 1), "move destination");

  const RowInstr row_insert{.op = RowOp::insert,
                            .matrix = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                            .vector = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                            .index = ScalarRegister{.index = 6}};
  const auto row_insert_inst = make_npu_dyn_inst(23, 0, DecodedNpuInstr{row_insert}, state);
  check_eq(row_insert_inst.op, DynOp::row, "row insert op");
  check_eq(row_insert_inst.rows, std::uint32_t{1}, "row insert rows");
  check_eq(row_insert_inst.cols, std::uint32_t{3}, "row insert cols");
  check(has_src(row_insert_inst, NpuDomain::vec32, 1), "row insert vector source");
  check(has_dst(row_insert_inst, NpuDomain::acc, 1), "row insert destination");

  const RowInstr row_extract{.op = RowOp::extract,
                             .matrix = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                             .vector = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                             .index = ImmediateIndex{.value = 0}};
  const auto row_extract_inst = make_npu_dyn_inst(24, 0, DecodedNpuInstr{row_extract}, state);
  check(has_src(row_extract_inst, NpuDomain::acc, 1), "row extract matrix source");
  check(has_dst(row_extract_inst, NpuDomain::vec32, 1), "row extract destination");

  const LaneExtractInstr extract{.dest = ScalarRegister{.index = 9},
                                 .src = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                                 .index = ScalarRegister{.index = 10}};
  const auto extract_inst = make_npu_dyn_inst(25, 0, DecodedNpuInstr{extract}, state);
  check_eq(extract_inst.op, DynOp::lane, "lane extract op");
  check(has_src(extract_inst, NpuDomain::vec32, 1), "lane extract vector source");
  check_eq(extract_inst.dst_count, std::uint8_t{0}, "lane extract has no NPU destination");

  const LaneInsertInstr insert{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                               .src = ScalarRegister{.index = 9},
                               .index = ImmediateIndex{.value = 0}};
  const auto insert_inst = make_npu_dyn_inst(26, 0, DecodedNpuInstr{insert}, state);
  check_eq(insert_inst.src_count, std::uint8_t{0}, "lane insert has no NPU source");
  check(has_dst(insert_inst, NpuDomain::vec32, 1), "lane insert destination");

  const LaneBroadcastInstr broadcast{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                                     .src = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                                     .index = ScalarRegister{.index = 10}};
  const auto broadcast_inst = make_npu_dyn_inst(27, 0, DecodedNpuInstr{broadcast}, state);
  check(has_src(broadcast_inst, NpuDomain::vec32, 2), "lane broadcast source");
  check(has_dst(broadcast_inst, NpuDomain::vec32, 1), "lane broadcast destination");
}

TEST_CASE(dyn_inst_builder_elementwise_and_matrix_ops) {
  State state{};
  state.ac[1] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 4};
  state.ac[2] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 4};
  state.vc[1] = Vec32Config{.dtype = DType::i32, .len = 8};
  state.vc[2] = Vec32Config{.dtype = DType::i32, .len = 8};
  state.vc[3] = Vec32Config{.dtype = DType::i32, .len = 8};
  state.vc[4] = Vec32Config{.dtype = DType::i32, .len = 8};

  const EbinInstr ebin{.op = EbinOp::add,
                       .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                       .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                       .src2 = ScalarRegister{.index = 8}};
  const auto ebin_inst = make_npu_dyn_inst(30, 0, DecodedNpuInstr{ebin}, state);
  check_eq(ebin_inst.op, DynOp::elementwise, "ebin op");
  check_eq(ebin_inst.unit, Unit::vec32, "ebin unit");
  check_eq(ebin_inst.len, std::uint32_t{8}, "ebin length");
  check_eq(ebin_inst.src_count, std::uint8_t{1}, "ebin records only the NPU source");
  check(has_src(ebin_inst, NpuDomain::vec32, 2), "ebin vector source");
  check(has_dst(ebin_inst, NpuDomain::vec32, 1), "ebin destination");

  const EunInstr eun{.op = EunOp::abs,
                     .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                     .src = NpuRegister{.domain = NpuDomain::vec32, .index = 2}};
  check_eq(make_npu_dyn_inst(31, 0, DecodedNpuInstr{eun}, state).op, DynOp::elementwise, "eun op");

  const CmpInstr cmp{.op = CmpOp::lt,
                     .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                     .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                     .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto cmp_inst = make_npu_dyn_inst(32, 0, DecodedNpuInstr{cmp}, state);
  check_eq(cmp_inst.op, DynOp::compare, "cmp op");
  check_eq(cmp_inst.src_count, std::uint8_t{2}, "cmp sources");

  const BcastInstr bcast{.op = EbinOp::mul,
                         .dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                         .src1 = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                         .vector = NpuRegister{.domain = NpuDomain::vec32, .index = 3},
                         .by_col = false};
  const auto bcast_inst = make_npu_dyn_inst(33, 0, DecodedNpuInstr{bcast}, state);
  check_eq(bcast_inst.op, DynOp::broadcast, "bcast op");
  check_eq(bcast_inst.rows, std::uint32_t{2}, "bcast rows");
  check_eq(bcast_inst.cols, std::uint32_t{4}, "bcast cols");
  check(has_src(bcast_inst, NpuDomain::acc, 2), "bcast matrix source");
  check(has_src(bcast_inst, NpuDomain::vec32, 3), "bcast vector source");

  const Er4Instr er4{.op = Er4Op::select,
                     .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                     .arg = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                     .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 3},
                     .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 4}};
  const auto er4_inst = make_npu_dyn_inst(34, 0, DecodedNpuInstr{er4}, state);
  check_eq(er4_inst.op, DynOp::select, "er4 op");
  check_eq(er4_inst.src_count, std::uint8_t{3}, "er4 source count");

  const MaskInstr mask{.op = MaskOp::tail,
                       .dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                       .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                       .bound = 6,
                       .fill = ScalarRegister{.index = 7}};
  const auto mask_inst = make_npu_dyn_inst(35, 0, DecodedNpuInstr{mask}, state);
  check_eq(mask_inst.op, DynOp::mask, "mask op");
  check_eq(mask_inst.src_count, std::uint8_t{1}, "mask source count");
  check(has_src(mask_inst, NpuDomain::acc, 2), "mask source");
  check(has_dst(mask_inst, NpuDomain::acc, 1), "mask destination");
}

TEST_CASE(dyn_inst_builder_reduce_and_quant) {
  State state{};
  state.ac[2] = AccConfig{.dtype = DType::f32, .rows = 4, .cols = 32};
  state.ac[1] = AccConfig{.dtype = DType::f32, .rows = 2, .cols = 2};
  state.tc[1] = TileConfig{.dtype = DType::i8, .rows = 2, .cols = 2};
  state.vc[1] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[3] = Vec32Config{.dtype = DType::f32, .len = 32};
  state.vc[4] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[5] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[6] = Vec32Config{.dtype = DType::f32, .len = 4};

  const AreduceInstr areduce{.op = ReduceOp::sum,
                             .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                             .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  const auto areduce_inst = make_npu_dyn_inst(40, 0, DecodedNpuInstr{areduce}, state);
  check_eq(areduce_inst.op, DynOp::reduce, "areduce op");
  check_eq(areduce_inst.unit, Unit::reduce, "areduce unit");
  check_eq(areduce_inst.reduce_src_elements, std::uint32_t{128}, "areduce work");
  check_eq(areduce_inst.len, std::uint32_t{4}, "areduce destination length");
  check(has_src(areduce_inst, NpuDomain::acc, 2), "areduce source");
  check(has_dst(areduce_inst, NpuDomain::vec32, 1), "areduce destination");

  const VreduceInstr vreduce{.op = ReduceOp::sum,
                             .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 4},
                             .src = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto vreduce_inst = make_npu_dyn_inst(41, 0, DecodedNpuInstr{vreduce}, state);
  check_eq(vreduce_inst.reduce_src_elements, std::uint32_t{32}, "vreduce work");
  check_eq(vreduce_inst.len, std::uint32_t{4}, "vreduce destination length");

  const AreduceArgmaxInstr argmax{.index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 5},
                                  .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 6},
                                  .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  const auto argmax_inst = make_npu_dyn_inst(42, 0, DecodedNpuInstr{argmax}, state);
  check_eq(argmax_inst.dst_count, std::uint8_t{2}, "argmax has two destinations");
  check_eq(argmax_inst.reduce_src_elements, std::uint32_t{128}, "argmax work");
  check(has_dst(argmax_inst, NpuDomain::vec32, 5), "argmax index destination");
  check(has_dst(argmax_inst, NpuDomain::vec32, 6), "argmax value destination");

  const VreduceArgmaxInstr vargmax{
      .index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 5},
      .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 6},
      .src = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto vargmax_inst = make_npu_dyn_inst(47, 0, DecodedNpuInstr{vargmax}, state);
  check_eq(vargmax_inst.dst_count, std::uint8_t{2}, "vreduce argmax destinations");
  check_eq(vargmax_inst.reduce_src_elements, std::uint32_t{32}, "vreduce argmax work");
  check(has_src(vargmax_inst, NpuDomain::vec32, 3), "vreduce argmax source");

  const AquantInstr aquant{.dest = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                           .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                           .scale = 3};
  const auto aquant_inst = make_npu_dyn_inst(43, 0, DecodedNpuInstr{aquant}, state);
  check_eq(aquant_inst.op, DynOp::quant, "aquant op");
  check_eq(aquant_inst.unit, Unit::quant, "aquant unit");
  check(has_src(aquant_inst, NpuDomain::acc, 2), "aquant source");
  check(has_src(aquant_inst, NpuDomain::vec32, 3), "aquant scale");
  check(has_dst(aquant_inst, NpuDomain::tile, 1), "aquant destination");

  const TdequantInstr tdequant{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                               .src = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                               .scale = 3};
  const auto tdequant_inst = make_npu_dyn_inst(44, 0, DecodedNpuInstr{tdequant}, state);
  check_eq(tdequant_inst.op, DynOp::quant, "tdequant op");
  check(has_src(tdequant_inst, NpuDomain::tile, 1), "tdequant source");
  check(has_src(tdequant_inst, NpuDomain::vec32, 3), "tdequant scale");

  const AdequantInstr adequant{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                               .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                               .col_scale = 3,
                               .row_scale = 4};
  const auto adequant_inst = make_npu_dyn_inst(45, 0, DecodedNpuInstr{adequant}, state);
  check_eq(adequant_inst.src_count, std::uint8_t{3}, "adequant sources");
  check(has_src(adequant_inst, NpuDomain::acc, 2), "adequant source");
  check(has_src(adequant_inst, NpuDomain::vec32, 3), "adequant col scale");
  check(has_src(adequant_inst, NpuDomain::vec32, 4), "adequant row scale");

  const VdequantInstr vdequant{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                               .src = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                               .col_scale = 3,
                               .row_scale = 4};
  const auto vdequant_inst = make_npu_dyn_inst(46, 0, DecodedNpuInstr{vdequant}, state);
  check_eq(vdequant_inst.src_count, std::uint8_t{3}, "vdequant sources");
  check(has_src(vdequant_inst, NpuDomain::vec32, 2), "vdequant source");
  check(has_src(vdequant_inst, NpuDomain::vec32, 3), "vdequant col scale");
  check(has_src(vdequant_inst, NpuDomain::vec32, 4), "vdequant row scale");
}

TEST_CASE(dyn_inst_builder_units) {
  State state{};
  state.ac[2] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 8};
  state.vc[1] = Vec32Config{.dtype = DType::i32, .len = 8};

  const EbinInstr ebin{.op = EbinOp::add,
                       .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                       .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                       .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  check_eq(make_npu_dyn_inst(60, 0, DecodedNpuInstr{ebin}, state).unit, Unit::vec32,
           "elementwise unit");

  const AreduceInstr reduce{.op = ReduceOp::sum,
                            .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                            .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  check_eq(make_npu_dyn_inst(61, 0, DecodedNpuInstr{reduce}, state).unit, Unit::reduce,
           "reduction unit");

  const AquantInstr quant{.dest = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                          .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                          .scale = 1};
  check_eq(make_npu_dyn_inst(62, 0, DecodedNpuInstr{quant}, state).unit, Unit::quant,
           "quantisation unit");

  const MmaInstr mma{.op = MmaOp::nt_zero,
                     .dest = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                     .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 0},
                     .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 1}};
  check_eq(make_npu_dyn_inst(63, 0, DecodedNpuInstr{mma}, state).unit, Unit::mma, "mma unit");

  const MblkInstr load{.op = MblkOp::load,
                       .reg = NpuRegister{.domain = NpuDomain::tile, .index = 0}};
  check_eq(make_npu_dyn_inst(64, 0, DecodedNpuInstr{load}, state).unit, Unit::dma, "load unit");
}
