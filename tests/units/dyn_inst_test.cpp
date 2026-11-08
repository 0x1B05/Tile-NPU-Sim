#include "dataflow/dyn_inst_builder.h"
#include "framework.h"

#include "tile_npu/dyn_inst.h"
#include "tile_npu/instruction.h"

using namespace tile_npu;

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
  check(inst.dst.has_value(), "mma has a destination");
  check_eq(inst.dst->domain, Domain::acc, "mma destination domain");
  check_eq(inst.dst->index, std::uint8_t{3}, "mma destination index");
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
  check(inst.dst.has_value(), "load has a destination");
  check_eq(inst.dst->domain, Domain::tile, "load destination domain");
  check_eq(inst.config.tile_d.rows, std::uint8_t{3}, "snapshot tile rows");
  check_eq(hex(inst.config.matrix.base_addr), hex(std::uint64_t{0x2000}),
           "snapshot descriptor base");

  const MblkInstr store{.op = MblkOp::store,
                        .reg = NpuRegister{.domain = NpuDomain::tile, .index = 4}};
  const auto store_inst = make_dyn_inst(9, 0x80000008, DecodedNpuInstr{store}, state);
  check(store_inst.is_store, "store flag");
  check_eq(store_inst.src_count, std::uint8_t{1}, "store has a register source");
  check(!store_inst.dst.has_value(), "store has no destination");
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
  check(scalar.dst.has_value(), "scalar destination is recorded");
  check_eq(scalar.dst->domain, Domain::scalar, "scalar destination domain");
  check_eq(scalar.dst->index, std::uint8_t{1}, "scalar destination index");

  const DecodedScalarInstr add{.op = ScalarOp::add,
                               .regs = ScalarRegs{.dest = 3, .src1 = 1, .src2 = 2}};
  const auto add_inst = make_scalar_dyn_inst(13, 0x80000018, add);
  check_eq(add_inst.src_count, std::uint8_t{2}, "scalar sources are recorded");
  check_eq(add_inst.src[0].index, std::uint8_t{1}, "first scalar source");
  check_eq(add_inst.src[1].index, std::uint8_t{2}, "second scalar source");
  check_eq(add_inst.dst->index, std::uint8_t{3}, "scalar add destination");
}
