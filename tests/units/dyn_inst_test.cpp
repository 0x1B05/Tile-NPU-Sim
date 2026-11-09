#include "uarch/dataflow/dyn_inst_builder.h"
#include "framework.h"

#include "uarch/dyn_inst.h"
#include "tile_npu/instruction.h"

using namespace tile_npu;

namespace {

bool has_src(const DynInst &inst, Domain domain, std::uint8_t index) {
  for (std::uint8_t i = 0; i < inst.src_count; ++i) {
    if (inst.src[i].domain == domain && inst.src[i].index == index) return true;
  }
  return false;
}

bool has_dst(const DynInst &inst, Domain domain, std::uint8_t index) {
  for (std::uint8_t i = 0; i < inst.dst_count; ++i) {
    if (inst.dst[i].domain == domain && inst.dst[i].index == index) return true;
  }
  return false;
}

} // namespace

TEST_CASE(dyn_inst_mma_shape_and_dependencies) {
  State state{};
  state.tc[1] = TileConfig{.dtype = DType::i8, .rows = 4, .cols = 8};
  state.tc[2] = TileConfig{.dtype = DType::i8, .rows = 5, .cols = 8};
  state.ac[3] = AccConfig{.dtype = DType::i32, .rows = 4, .cols = 5};

  const MmaInstr mma{.op = MmaOp::nt_acc,
                     .dest = NpuRegister{.domain = NpuDomain::acc, .index = 3},
                     .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                     .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 2}};
  const auto inst = make_dyn_inst(7, 0x80000000, DecodedNpuInstr{mma}, state);

  check_eq(inst.id, std::uint64_t{7}, "dyn inst id");
  check_eq(hex(inst.pc), hex(0x80000000), "dyn inst pc");
  check_eq(inst.op, DynOp::mma, "mma op");
  check_eq(inst.unit, Unit::mma, "mma unit");
  check_eq(inst.rows, std::uint32_t{4}, "mma M");
  check_eq(inst.cols, std::uint32_t{5}, "mma N");
  check_eq(inst.k, std::uint32_t{8}, "mma K");
  check_eq(inst.macs, std::uint32_t{160}, "mma MAC count");
  check_eq(inst.src_count, std::uint8_t{3}, "mma sources include the accumulator");
  check_eq(inst.src[0].domain, Domain::tile, "mma lhs domain");
  check_eq(inst.src[1].domain, Domain::tile, "mma rhs domain");
  check_eq(inst.src[2].domain, Domain::acc, "mma acc domain");
  check_eq(inst.dst_count, std::uint8_t{1}, "mma has one destination");
  check_eq(inst.dst[0].domain, Domain::acc, "mma destination domain");
  check_eq(inst.dst[0].index, std::uint8_t{3}, "mma destination index");
  check_eq(inst.config.tile_a.rows, std::uint8_t{4}, "snapshot lhs rows");
  check_eq(inst.config.tile_b.rows, std::uint8_t{5}, "snapshot rhs rows");
  check_eq(inst.config.acc_d.cols, std::uint8_t{5}, "snapshot acc cols");
}

TEST_CASE(dyn_inst_mblk_descriptor_snapshot) {
  State state{};
  state.tc[4] = TileConfig{.dtype = DType::i8, .rows = 3, .cols = 5};
  state.tm[4] = MatrixDescriptor{.base_addr = 0x2000,
                                 .row_stride_bytes = 8,
                                 .col_stride_bytes = 1,
                                 .transform = 0,
                                 .flags = 0};

  const MblkInstr load{.op = MblkOp::load,
                       .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4}};
  const auto inst = make_dyn_inst(8, 0x80000004, DecodedNpuInstr{load}, state);

  check_eq(inst.op, DynOp::load, "load op");
  check_eq(inst.unit, Unit::dma, "load unit");
  check_eq(inst.rows, std::uint32_t{3}, "load rows");
  check_eq(inst.cols, std::uint32_t{5}, "load cols");
  check_eq(inst.bytes, std::uint32_t{15}, "load bytes");
  check_eq(hex(inst.addr), hex(std::uint64_t{0x2000}), "load base address");
  check_eq(inst.row_stride, std::int32_t{8}, "load row stride");
  check_eq(inst.col_stride, std::int32_t{1}, "load col stride");
  check(!inst.is_store, "load is not a store");
  check_eq(inst.src_count, std::uint8_t{0}, "load has no register source");
  check_eq(inst.dst_count, std::uint8_t{1}, "load has a destination");
  check_eq(inst.dst[0].domain, Domain::tile, "load destination domain");
  check_eq(inst.config.tile_d.rows, std::uint8_t{3}, "snapshot tile rows");
  check_eq(hex(inst.config.matrix.base_addr), hex(std::uint64_t{0x2000}),
           "snapshot descriptor base");

  const MblkInstr store{.op = MblkOp::store,
                        .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4}};
  const auto store_inst = make_dyn_inst(9, 0x80000008, DecodedNpuInstr{store}, state);
  check(store_inst.is_store, "store flag");
  check_eq(store_inst.src_count, std::uint8_t{1}, "store has a register source");
  check_eq(store_inst.dst_count, std::uint8_t{0}, "store has no destination");
}

TEST_CASE(dyn_inst_sync_and_scalar) {
  State state{};
  const SysInstr fence{.op = SysOp::fence_sa};
  const auto fence_inst = make_dyn_inst(10, 0x8000000c, DecodedNpuInstr{fence}, state);
  check_eq(fence_inst.op, DynOp::fence, "fence op");
  check_eq(fence_inst.unit, Unit::sync, "fence unit");
  check_eq(fence_inst.fence, FenceKind::sa, "fence kind");
  check(!fence_inst.kernel_end, "fence is not kernel end");

  const SysInstr end{.op = SysOp::kernel_end};
  const auto end_inst = make_dyn_inst(11, 0x80000010, DecodedNpuInstr{end}, state);
  check_eq(end_inst.op, DynOp::kernel_end, "kernel end op");
  check(end_inst.kernel_end, "kernel end flag");
  check_eq(end_inst.fence, FenceKind::all, "kernel end implies fence.all");

  const DecodedScalarInstr addi{
      .op = ScalarOp::addi, .regs = ScalarRegs{.dest = 1, .src1 = 0, .src2 = 0}, .imm = 7};
  const auto scalar = make_scalar_dyn_inst(12, 0x80000014, addi);
  check_eq(scalar.op, DynOp::scalar, "scalar op");
  check_eq(scalar.unit, Unit::scalar, "scalar unit");
  check_eq(scalar.src_count, std::uint8_t{0}, "x0 operands are not recorded");
  check_eq(scalar.dst_count, std::uint8_t{1}, "scalar destination is recorded");
  check_eq(scalar.dst[0].domain, Domain::scalar, "scalar destination domain");
  check_eq(scalar.dst[0].index, std::uint8_t{1}, "scalar destination index");

  const DecodedScalarInstr add{.op = ScalarOp::add,
                               .regs = ScalarRegs{.dest = 3, .src1 = 1, .src2 = 2}};
  const auto add_inst = make_scalar_dyn_inst(13, 0x80000018, add);
  check_eq(add_inst.src_count, std::uint8_t{2}, "scalar sources are recorded");
  check_eq(add_inst.src[0].index, std::uint8_t{1}, "first scalar source");
  check_eq(add_inst.src[1].index, std::uint8_t{2}, "second scalar source");
  check_eq(add_inst.dst[0].index, std::uint8_t{3}, "scalar add destination");
}

TEST_CASE(dyn_inst_records_scalar_side_operands) {
  State state{};

  const FillInstr fill{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                       .value = ScalarRegister{.index = 5}};
  const auto fill_inst = make_dyn_inst(20, 0x80000020, DecodedNpuInstr{fill}, state);
  check_eq(fill_inst.src_count, std::uint8_t{1}, "fill records the register value");
  check(has_src(fill_inst, Domain::scalar, 5), "fill reads its scalar value");
  check(has_dst(fill_inst, Domain::acc, 1), "fill writes its matrix destination");

  const MaskInstr mask{.op = MaskOp::tail,
                       .dest = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                       .src = NpuRegister{.domain = NpuDomain::acc, .index = 3},
                       .bound = 6,
                       .fill = ScalarRegister{.index = 7}};
  const auto mask_inst = make_dyn_inst(21, 0x80000024, DecodedNpuInstr{mask}, state);
  check_eq(mask_inst.src_count, std::uint8_t{3}, "mask records source, bound and fill");
  check(has_src(mask_inst, Domain::acc, 3), "mask reads its source matrix");
  check(has_src(mask_inst, Domain::scalar, 6), "mask reads the bound register");
  check(has_src(mask_inst, Domain::scalar, 7), "mask reads the fill register");

  const EbinInstr ebin{.op = EbinOp::add,
                       .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                       .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                       .src2 = ScalarRegister{.index = 8}};
  const auto ebin_inst = make_dyn_inst(22, 0x80000028, DecodedNpuInstr{ebin}, state);
  check_eq(ebin_inst.src_count, std::uint8_t{2}, "ebin records both operands");
  check(has_src(ebin_inst, Domain::vec32, 2), "ebin reads its vector operand");
  check(has_src(ebin_inst, Domain::scalar, 8), "ebin reads its scalar operand");

  const LaneExtractInstr extract{.dest = ScalarRegister{.index = 9},
                                 .src = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                                 .index = ScalarRegister{.index = 10}};
  const auto extract_inst = make_dyn_inst(23, 0x8000002c, DecodedNpuInstr{extract}, state);
  check_eq(extract_inst.src_count, std::uint8_t{2}, "lane extract records source and index");
  check(has_dst(extract_inst, Domain::scalar, 9), "lane extract writes a scalar");
  check(has_src(extract_inst, Domain::vec32, 1), "lane extract reads the vector");
  check(has_src(extract_inst, Domain::scalar, 10), "lane extract reads the index");

  const CfgSetxInstr setx{
      .reg_type = ConfigRegType::tc, .dest = 2, .field = ConfigField::rows, .src = 11, .n = 0};
  const auto setx_inst = make_dyn_inst(24, 0x80000030, DecodedNpuInstr{setx}, state);
  check_eq(setx_inst.dst_count, std::uint8_t{0}, "cfg.setx writes no scalar register");
  check_eq(setx_inst.src_count, std::uint8_t{1}, "cfg.setx reads its scalar source");
  check(has_src(setx_inst, Domain::scalar, 11), "cfg.setx source");

  const CfgGetInstr get{
      .reg_type = ConfigRegType::tc, .dest = 12, .src = 2, .field = ConfigField::rows};
  const auto get_inst = make_dyn_inst(25, 0x80000034, DecodedNpuInstr{get}, state);
  check_eq(get_inst.src_count, std::uint8_t{0}, "cfg.get reads only config state");
  check(has_dst(get_inst, Domain::scalar, 12), "cfg.get writes its scalar destination");

  const CfgSetwInstr setw{.reg_type = ConfigRegType::ac, .dest = 3, .src = 13, .n = 0};
  const auto setw_inst = make_dyn_inst(26, 0x80000038, DecodedNpuInstr{setw}, state);
  check(has_src(setw_inst, Domain::scalar, 13), "cfg.setw source");

  const CfgGetwInstr getw{.reg_type = ConfigRegType::ac, .dest = 14, .src = 3};
  const auto getw_inst = make_dyn_inst(27, 0x8000003c, DecodedNpuInstr{getw}, state);
  check(has_dst(getw_inst, Domain::scalar, 14), "cfg.getw destination");
}

TEST_CASE(dyn_inst_records_matrix_side_operands) {
  State state{};

  const BcastInstr bcast{.op = EbinOp::mul,
                         .dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                         .src1 = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                         .vector = NpuRegister{.domain = NpuDomain::vec32, .index = 3},
                         .by_col = false};
  const auto bcast_inst = make_dyn_inst(30, 0x80000040, DecodedNpuInstr{bcast}, state);
  check_eq(bcast_inst.src_count, std::uint8_t{2}, "bcast records matrix and vector");
  check(has_src(bcast_inst, Domain::acc, 2), "bcast matrix operand");
  check(has_src(bcast_inst, Domain::vec32, 3), "bcast vector operand");

  const Er4Instr er4{.op = Er4Op::select,
                     .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                     .arg = NpuRegister{.domain = NpuDomain::vec32, .index = 4},
                     .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 5},
                     .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 6}};
  const auto er4_inst = make_dyn_inst(31, 0x80000044, DecodedNpuInstr{er4}, state);
  check_eq(er4_inst.src_count, std::uint8_t{3}, "er4 records arg and both sources");
  check(has_src(er4_inst, Domain::vec32, 4), "er4 arg operand");
  check(has_src(er4_inst, Domain::vec32, 5), "er4 first source");
  check(has_src(er4_inst, Domain::vec32, 6), "er4 second source");

  const AquantInstr aquant{.dest = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                           .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                           .scale = 3};
  const auto aquant_inst = make_dyn_inst(32, 0x80000048, DecodedNpuInstr{aquant}, state);
  check_eq(aquant_inst.op, DynOp::quant, "aquant op");
  check(has_src(aquant_inst, Domain::acc, 2), "aquant source");
  check(has_src(aquant_inst, Domain::vec32, 3), "aquant scale is a vector register");
  check(has_dst(aquant_inst, Domain::tile, 1), "aquant destination");

  const TdequantInstr tdequant{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                               .src = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                               .scale = 3};
  const auto tdequant_inst = make_dyn_inst(33, 0x8000004c, DecodedNpuInstr{tdequant}, state);
  check(has_src(tdequant_inst, Domain::tile, 1), "tdequant source");
  check(has_src(tdequant_inst, Domain::vec32, 3), "tdequant scale is a vector register");

  const AdequantInstr adequant{.dest = NpuRegister{.domain = NpuDomain::acc, .index = 1},
                               .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                               .col_scale = 3,
                               .row_scale = 4};
  const auto adequant_inst = make_dyn_inst(34, 0x80000050, DecodedNpuInstr{adequant}, state);
  check_eq(adequant_inst.src_count, std::uint8_t{3}, "adequant records source and scales");
  check(has_src(adequant_inst, Domain::acc, 2), "adequant source");
  check(has_src(adequant_inst, Domain::vec32, 3), "adequant column scale");
  check(has_src(adequant_inst, Domain::vec32, 4), "adequant row scale");

  const VdequantInstr vdequant{.dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                               .src = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                               .col_scale = 3,
                               .row_scale = 4};
  const auto vdequant_inst = make_dyn_inst(35, 0x80000054, DecodedNpuInstr{vdequant}, state);
  check_eq(vdequant_inst.src_count, std::uint8_t{3}, "vdequant records source and scales");
  check(has_src(vdequant_inst, Domain::vec32, 2), "vdequant source");
  check(has_src(vdequant_inst, Domain::vec32, 3), "vdequant column scale");
  check(has_src(vdequant_inst, Domain::vec32, 4), "vdequant row scale");

  const AreduceArgmaxInstr areduce{
      .index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
      .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
      .src = NpuRegister{.domain = NpuDomain::acc, .index = 3}};
  const auto areduce_inst = make_dyn_inst(36, 0x80000058, DecodedNpuInstr{areduce}, state);
  check_eq(areduce_inst.dst_count, std::uint8_t{2}, "areduce.argmax writes two registers");
  check(has_dst(areduce_inst, Domain::vec32, 1), "areduce.argmax index destination");
  check(has_dst(areduce_inst, Domain::vec32, 2), "areduce.argmax value destination");
  check(has_src(areduce_inst, Domain::acc, 3), "areduce.argmax source");

  const VreduceArgmaxInstr vreduce{
      .index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
      .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
      .src = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto vreduce_inst = make_dyn_inst(37, 0x8000005c, DecodedNpuInstr{vreduce}, state);
  check_eq(vreduce_inst.dst_count, std::uint8_t{2}, "vreduce.argmax writes two registers");
  check(has_dst(vreduce_inst, Domain::vec32, 1), "vreduce.argmax index destination");
  check(has_dst(vreduce_inst, Domain::vec32, 2), "vreduce.argmax value destination");
  check(has_src(vreduce_inst, Domain::vec32, 3), "vreduce.argmax source");
}

TEST_CASE(dyn_inst_records_reduce_work_elements) {
  State state{};
  state.ac[2] = AccConfig{.dtype = DType::f32, .rows = 4, .cols = 32};
  state.vc[1] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[3] = Vec32Config{.dtype = DType::f32, .len = 32};
  state.vc[4] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[5] = Vec32Config{.dtype = DType::f32, .len = 4};
  state.vc[6] = Vec32Config{.dtype = DType::f32, .len = 4};

  const AreduceInstr areduce{.op = ReduceOp::sum,
                             .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                             .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  const auto areduce_inst = make_dyn_inst(40, 0x80000060, DecodedNpuInstr{areduce}, state);
  check_eq(areduce_inst.work_elements, std::uint32_t{128},
           "areduce work is the source rows times columns");
  check_eq(areduce_inst.cols, std::uint32_t{4}, "areduce keeps the destination shape");

  const VreduceInstr vreduce{.op = ReduceOp::sum,
                             .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 4},
                             .src = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto vreduce_inst = make_dyn_inst(41, 0x80000064, DecodedNpuInstr{vreduce}, state);
  check_eq(vreduce_inst.work_elements, std::uint32_t{32},
           "vreduce work is the source vector length");
  check_eq(vreduce_inst.len, std::uint32_t{4}, "vreduce keeps the destination length");

  const AreduceArgmaxInstr areduce_argmax{
      .index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 5},
      .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 6},
      .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  const auto argmax_inst = make_dyn_inst(42, 0x80000068, DecodedNpuInstr{areduce_argmax}, state);
  check_eq(argmax_inst.work_elements, std::uint32_t{128},
           "areduce.argmax work is the source rows times columns");

  const VreduceArgmaxInstr vreduce_argmax{
      .index_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 5},
      .value_dest = NpuRegister{.domain = NpuDomain::vec32, .index = 6},
      .src = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  const auto vargmax_inst = make_dyn_inst(43, 0x8000006c, DecodedNpuInstr{vreduce_argmax}, state);
  check_eq(vargmax_inst.work_elements, std::uint32_t{32},
           "vreduce.argmax work is the source vector length");
}

TEST_CASE(dyn_inst_maps_current_isa_units) {
  State state{};
  state.ac[2] = AccConfig{.dtype = DType::i32, .rows = 2, .cols = 8};
  state.vc[1] = Vec32Config{.dtype = DType::i32, .len = 8};

  const EbinInstr ebin{.op = EbinOp::add,
                       .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                       .src1 = NpuRegister{.domain = NpuDomain::vec32, .index = 2},
                       .src2 = NpuRegister{.domain = NpuDomain::vec32, .index = 3}};
  check_eq(make_dyn_inst(60, 0, DecodedNpuInstr{ebin}, state).unit, Unit::vec32,
           "elementwise uses the Vec32 ALU");

  const AreduceInstr reduce{.op = ReduceOp::sum,
                            .dest = NpuRegister{.domain = NpuDomain::vec32, .index = 1},
                            .src = NpuRegister{.domain = NpuDomain::acc, .index = 2}};
  check_eq(make_dyn_inst(61, 0, DecodedNpuInstr{reduce}, state).unit, Unit::reduce,
           "reduction has its own queue");

  const AquantInstr quant{.dest = NpuRegister{.domain = NpuDomain::tile, .index = 1},
                          .src = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                          .scale = 1};
  check_eq(make_dyn_inst(62, 0, DecodedNpuInstr{quant}, state).unit, Unit::quant,
           "quantisation has its own queue");

  const MmaInstr mma{.op = MmaOp::nt_zero,
                     .dest = NpuRegister{.domain = NpuDomain::acc, .index = 2},
                     .src_a = NpuRegister{.domain = NpuDomain::tile, .index = 0},
                     .src_b = NpuRegister{.domain = NpuDomain::tile, .index = 1}};
  check_eq(make_dyn_inst(63, 0, DecodedNpuInstr{mma}, state).unit, Unit::mma,
           "MMA uses the matrix unit");

  const MblkInstr load{.op = MblkOp::load,
                       .reg = NpuRegister{.domain = NpuDomain::tile, .index = 0}};
  check_eq(make_dyn_inst(64, 0, DecodedNpuInstr{load}, state).unit, Unit::dma, "load uses DMA");

  const SysInstr fence{.op = SysOp::fence_all};
  check_eq(make_dyn_inst(65, 0, DecodedNpuInstr{fence}, state).unit, Unit::sync,
           "fence uses the sync unit");
}

TEST_CASE(dyn_inst_gather_uses_effective_length) {
  State state{};
  state.tc[4] = TileConfig{.dtype = DType::i8, .rows = 4, .cols = 8};
  state.vc[1] = Vec32Config{.dtype = DType::u32, .len = 2};
  state.vc[2] = Vec32Config{.dtype = DType::u32, .len = 32};
  state.vc[3] = Vec32Config{.dtype = DType::i8, .len = 8};

  const auto tile = NpuRegister{.domain = NpuDomain::tile, .index = 4};
  const MblkInstr short_tile{.op = MblkOp::gather, .reg = tile, .gather_src = 1};
  const auto short_tile_inst = make_dyn_inst(50, 0x80000070, DecodedNpuInstr{short_tile}, state);
  check_eq(short_tile_inst.rows, std::uint32_t{2}, "tile gather is bounded by the index length");
  check_eq(short_tile_inst.cols, std::uint32_t{8}, "tile gather keeps the row width");
  check_eq(short_tile_inst.bytes, std::uint32_t{16}, "tile gather counts only indexed rows");

  const MblkInstr long_tile{.op = MblkOp::gather, .reg = tile, .gather_src = 2};
  const auto long_tile_inst = make_dyn_inst(51, 0x80000074, DecodedNpuInstr{long_tile}, state);
  check_eq(long_tile_inst.rows, std::uint32_t{4}, "tile gather is bounded by the tile rows");
  check_eq(long_tile_inst.bytes, std::uint32_t{32}, "extra indices do not add transfers");

  const auto vector = NpuRegister{.domain = NpuDomain::vec32, .index = 3};
  const MblkInstr short_vector{.op = MblkOp::gather, .reg = vector, .gather_src = 1};
  const auto short_vector_inst =
      make_dyn_inst(52, 0x80000078, DecodedNpuInstr{short_vector}, state);
  check_eq(short_vector_inst.len, std::uint32_t{2}, "vector gather is bounded by the index length");
  check_eq(short_vector_inst.bytes, std::uint32_t{2}, "vector gather counts indexed lanes");
  check(has_dst(short_vector_inst, Domain::vec32, 3), "vector gather writes its destination");

  const MblkInstr long_vector{.op = MblkOp::gather, .reg = vector, .gather_src = 2};
  const auto long_vector_inst = make_dyn_inst(53, 0x8000007c, DecodedNpuInstr{long_vector}, state);
  check_eq(long_vector_inst.len, std::uint32_t{8},
           "vector gather is bounded by the destination length");
  check_eq(long_vector_inst.bytes, std::uint32_t{8}, "extra indices do not add transfers");
}
