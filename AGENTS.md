# Tile-NPU-Sim 开发约定 (供 AI 编码代理阅读)

## 项目概览

Tile NPU 指令集的功能模拟器 (functional simulator), 独立 git 仓库, 远端为 `git@github.com:0x1B05/Tile-NPU-Sim`。当前为单 hart, 按程序序同步执行的库 (`tile_npu_sim` 静态库) 加 monitor 程序 `sim_run` (默认运行程序, `-i` 进入交互 sdb shell); 译码与执行已覆盖 32-bit RV64IM 整数指令、逐元素二元 (`E_BIN`)、逐元素一元 (`E_UNARY`)、比较 (`E_CMP`)、向量广播 (`E_BCAST`)、四寄存器 (`E_R4`: select/fmadd)、mask (`E_MASK`: tail/tril/triu)、类型转换 (`CVT`: acvt/vcvt/twiden)、规约 (`AREDUCE` 行规约, `VREDUCE` 向量规约与 argmax)、量化/反量化 (`QNT`)、矩阵乘/点积 (`MMA_BDOT`) 和同步/结束 (`SYS`) 各族、配置指令 (`cfg.seti/setx/copy/get`)、`M_BLK` 访存族 (block/row load/store, gather load)、`INIT_MOVE` (fill/fillx/copy/ttranspose) 和行/lane 搬运族 (`ROW_MOVE` insert/extract.row, `LANE_MOVE` extract/insert/broadcast); 手册 major4 表中的指令族已全部覆盖。

配套的 ISA 手册仓库在 `../Tile-NPU-ISA/` (另一个独立 git 仓库)。

**注意**: `sim_run` 使用 CLI11 解析命令行, 由 xmake 管理依赖; 本仓库无 CI, 验证方式就是 `xmake test` 通过。

## 技术栈

- 语言: C++20 (`namespace tile_npu`), `sim_run` 依赖 CLI11
- 构建: xmake (编译器缓存在本目录 `.cache/ccache`, 已 gitignore)
- 文档语言: `README.md` 与代码注释为英文; 本文件为中文

## 构建与测试命令

```bash
xmake                                  # 构建 tile_npu_sim 静态库与 sim_run
xmake test                             # 构建并运行全部测试 (sim_tests + guest 镜像)
xmake run sim_run program.bin          # 默认运行, ebreak/kernel.end 退出
xmake run sim_run -i program.bin       # sdb 交互模式 (si/c/info r/x/q)
xmake f --monitor=n && xmake           # 关闭 monitor, 只构建库
scripts/fetch_riscv_tests.sh           # 拉取官方 riscv-tests 到 external/ (一次性)
xmake f --rvtests=y && xmake test      # 额外跑官方 rv64ui/rv64um 套件
xmake f --qwen3test=y && xmake test    # 额外跑 tiny Qwen3 guest/numpy 对齐回归 (需 python3+numpy)
xmake project -k compile_commands      # 为 clangd 生成编译数据库
```

构建配置用 xmake 原生 option (`xmake f --xxx=y/n`, 定义在 xmake.lua 顶部), 不引入 Kconfig 工具链。

## 文件组织

| 文件 | 作用 |
| ---- | ---- |
| `include/tile_npu/state.h` | 架构状态: 五个数据域的寄存器堆 (Tile/Acc/Vec8/Vec32/Scalar), 计算配置 (TC/AC/BC/VC) 与访存描述符 (TM/AM/BM/VM), 物理尺寸常量; 以及步进流水线的状态词表 `StepStatus` (模拟器诊断, 非 ISA trap, 不属于架构状态) |
| `include/tile_npu/platform/address_space.h` | 可寻址空间接口 `AddressSpace` (`contains`/`read`/`write`, 含多元素传输的原子性预检契约); 主存, MMIO 设备与总线共用 |
| `include/tile_npu/platform/ram.h`, `src/platform/ram.cpp` | 连续主存 `Ram` (基地址 + 平坦字节数组) |
| `include/tile_npu/platform/bus.h` | 平台总线 `SystemBus`: 地址按注册设备的窗口路由 (先声明先中, 设备窗口不得重叠), 未命中进主存; 新设备注册即用, 总线不变 |
| `include/tile_npu/platform/serial.h`, `src/platform/serial.cpp` | 平台串口控制台 `SerialConsole` (持有自己的地址窗口 `0x10000000`, 16550-lite: THR 写即宿主打印, LSR 恒读就绪) |
| `include/tile_npu/instruction.h` | typed instruction IR: scalar operands, NPU domains/registers (`domain_register_count`), `IndexOperand` (index5 操作数: `ImmediateIndex` 或 `ScalarRegister` 的 variant; `MblkInstr` 按方向拆 `dest`/`src`, 方向判读共享 `is_store_form`/`accessed_reg`), elementwise (`EbinInstr` + `EBIN_BY_FUNCT4` 操作元数据表, `EunInstr`, `CmpInstr` + `CMP_BY_FUNCT3` 条件元数据表, `BcastInstr`, `Er4Instr`, `MaskInstr`, `CvtInstr`, `ReduceInstr`, `QntInstr` + QNT 域配对谓词, `MmaInstr` + `MMA_BY_FUNCT3` 元数据表, `SysInstr`), config (`CfgInstr`, `ConfigRegType`), 访存 (`MblkInstr`), 初始化 (`InitInstr`) 与行/lane 搬运 (`RowInstr`, `LaneInstr`) 指令类型 |
| `include/tile_npu/decode.h` | 译码结果类型: typed scalar/NPU instructions, `DecodeError`, `DecodeResult` |
| `include/tile_npu/config.h`, `src/config.cpp` | 统一配置/描述符字段访问、字段编号映射与写入校验 |
| `include/tile_npu/matrix_view.h` | 固定尺寸矩阵寄存器的非拥有视图 `MatrixView`: `(row, col)` 到 canonical 行主序存储偏移的映射, 不携带 shape/layout; 构造工厂 `matrix_view(reg)` 只接受 `MATRIX_ELEMENTS` 大小的寄存器 |
| `include/tile_npu/descriptor_view.h` | TM/AM/BM/VM 地址计算、保留位校验、`incremented_base` 预检与访存后 `base_addr` 自增 |
| `include/tile_npu/simulator.h` | `Simulator` 单步 API: `reset(entry_pc)`, `step()` → `StepResult` |
| `src/utils/endian.h` | 内部工具: `load_le`/`store_le` 小端字节转换 (取指/load/store 共享); 不属于模型接口, `include/tile_npu/` 只放架构状态与 API |
| `src/utils/format.h` | 内部工具: `hex32`/`hex64` 将整数格式化为带 `0x` 前缀、补零至 8/16 位的十六进制字符串, monitor 与 SDB 共用 |
| `src/utils/unreachable.h` | 内部工具: 不变式违背时当场中止的 `unreachable()` (始终生效, 不随 NDEBUG 消失); guest 可见错误不走这里 |
| `src/decode/encoding.h` | 位段提取 (`field<Hi,Lo>`) 与 RISC-V R/I/S/B/U/J/Shift, NPU 各格式的字段解析 |
| `src/decode/pattern.h` | INSTPAT 风格译码机制: 编译期 `Pattern` 位型解析, `DecodeEntry` 译码表, `match` 匹配与 `fail` 错误 handler |
| `src/decode/decode.cpp` | 顶层译码分派: `q = 11` 走 RV64IM 译码表, 其余 quadrant 走 NPU 译码表 |
| `src/decode/decode_npu.cpp` | NPU 译码表 (q != 11): major4/sel3/funct4 由 pattern 直接匹配, 首条命中生效 |
| `src/decode/decode_rv64im.cpp` | RV64IM 译码表 (q = 11): 一行一条指令的 pattern; 文件内的 `make_instr`/`as`/`no_operands` 构造标量 IR, 表尾 catch-all 拒绝非法编码 |
| `src/execute/execute.cpp` | 执行分派 (标量 / NPU 各域执行器); `exec_npu_instr` 通过文件内的 `FpEnvGuard` 管理浮点环境, 先做结构校验再分派 |
| `src/execute/validate.h` | 执行入口的统一结构校验 `valid_structure` (每个 IR 类型一个 overload): op code/domain/操作数 variant/寄存器号是否可能来自某个真实编码 |
| `src/execute/config_check.h` | 各执行器共享的配置合法性 (`valid_domain_config` / `valid_elementwise_config`) 与同源同形约束 (`configs_match`) |
| `src/execute/execute_cfg.cpp` | 配置指令 (`cfg.*`) 执行器: 经 `config.h` 字段访问器读写, 字段值校验失败上报 `invalid_configuration` (CFG_ERROR) |
| `src/execute/execute_ebin.cpp` | 逐元素二元 (E_BIN) 族执行器: 公共约束校验与矩阵/向量循环抽象为模板, 操作语义集中在 `apply_op` |
| `src/execute/execute_eun.cpp` | 逐元素一元 (E_UNARY) 族执行器: not/abs/neg 为精确整数语义, exp2/rcp/rsqrt/silu 用 host libm 近似 (特殊值表现由测试锁定) |
| `src/execute/execute_ecmp.cpp` | 比较 (E_CMP) 族执行器: Acc/Vec32 only, 目的为同形 u32 mask (全一/零); i32 有符号, u32 无符号, f32 用 IEEE 语义 (NaN 使有序条件为假, unord 查 NaN) |
| `src/execute/ebin_op.h` | E_BIN/E_BCAST 共享的逐元素语义: 操作 dtype 合法性 (`ebin_op_supports_dtype`, 读 `EBIN_BY_FUNCT4`), 溢出模式查询 (`arith_mode_of`) 与单元素操作 (`apply_ebin_op`) |
| `src/execute/f32_util.h` | 执行器共享的 f32 工具: 位模式转换 `f32_bits` 与 absmax 扫描 (`f32_absmax_step`/`f32_absmax`, NaN 候选不胜出), 规约与量化族共用 |
| `src/execute/operand.h` | 执行器共享的操作数读取: 立即数/Scalar 索引解析 (`resolve_index`), 双源逐元素族 (E_BIN/E_CMP) 的 xS 低位截取 (`scalar_operand`) 与 src2 到元素访问器的解析 (`with_src2_matrix`/`with_src2_vector`) |
| `src/execute/execute_ebcast.cpp` | 向量广播 (E_BCAST) 族执行器: 矩阵二元操作, 第二源为配对向量按行/列广播; 向量 dtype 同矩阵, len 等于 rows (byrow) 或 cols (bycol) |
| `src/execute/execute_er4.cpp` | 四寄存器 (E_R4) 族执行器: select 按 mask 位选 (mask dtype 固定 u8/u32, 数据 dtype 无关), fmadd 为 f32 单次舍入融合乘加 (`std::fma`) |
| `src/execute/execute_emask.cpp` | mask (E_MASK) 族执行器: tail 按打包 nRows/nCols (矩阵) 或 xLen (向量) 保留头部, tril/triu 按有符号 xDelta 保留下/上三角; 保留区域之外写 fill (fill5 码表或 xFill 低元素位宽位), fill5 码 5..8 (±0.5/±inf) 仅 f32 目的 |
| `src/execute/execute_cvt.cpp` | 类型转换 (CVT) 族执行器: acvt/vcvt 要求两侧 i32 同形, 逐元素转 f32 并在成功时更新目的 dtype (允许原地); twiden 把 i8 Tile 符号扩展到同形 i32 Acc |
| `src/execute/execute_reduce.cpp` | 规约 (AREDUCE/VREDUCE) 族执行器: 行规约要求 f32 源且目的 len 等于行数 (argmax 目的为 i32); vreduce 结果 f32 位模式写 xD 低 32 位; argmax 按手册 NaN 规则扫描 (NaN 候选胜出, 已有 NaN 不败, 并列取小索引) |
| `src/execute/execute_qnt.cpp` | 量化/反量化 (QNT) 族执行器: tquant/vquant 动态 scale 为 absmax/127 (忽略 NaN), 量化用 RNE + 饱和 [-127,127] (NaN 商得 0); tquant 外部 scale 形式不写 vScale; 反量化为 f32(bits) x scale; 矩阵形式 vScale 逐行, 向量形式按 index5/xS 选单个 scale lane |
| `src/execute/execute_mma.cpp` | 矩阵乘/点积 (MMA_BDOT) 族执行器: mma.{nn,nt}.{zero,acc} 为 i8 符号扩展后的 i32 乘加 (acc 形式 wrap32 累加到目的旧值), nt 对右 Tile 做逻辑转置; bdot.{nn,nt} 为 Vec8 与 Tile 列/行的点积, 只有 zero 形式 |
| `src/execute/execute_sys.cpp` | 同步/结束 (SYS) 族执行器: fence.* 在单 hart 按序模型下为 no-op; kernel.end 上报 `StepStatus::kernel_end` 且 PC 停在指令上 (模拟器约定, 手册未定义 kernel 完成后的架构行为) |
| `src/execute/execute_mblk.cpp` | M_BLK 访存族执行器: 先完成全部地址计算 (溢出报 `invalid_configuration`), 再做传输; load 先 stage 寄存器副本, store 先 `contains` 全探测, fault 不改变架构状态 |
| `src/execute/execute_init.cpp` | INIT_MOVE 族执行器: fill/fillx 按目的 dtype 截取元素位宽 (f32 目的的 fill 立即数按 bf16 位模式解释), copy 要求同 dtype 同有效 shape, ttranspose 先 stage 源再写回并更新目的配置行列 |
| `src/execute/execute_row_lane.cpp` | ROW_MOVE/LANE_MOVE 执行器: 行搬运要求向量长度等于矩阵列数且行号有效, lane extract 按源 dtype 扩展, insert 截取低位, broadcast 写入目的有效 lane |
| `src/execute/execute_rv64i.cpp`, `src/execute/execute_rv64m.cpp` | RV64I 与 M 扩展语义 |
| `src/execute/rv64_util.h` | RV64 执行器共享的 *W 结果符号扩展 (`signed_word`) |
| `src/simulator.cpp`, `src/platform/ram.cpp` | 取指 (小端 4 字节拼字) 与步进循环, 主存实现 |
| `src/main.cpp` | `sim_run` 入口: CLI11 参数解析; 编进 sim_run 而非静态库 (xmake.lua 用 `\|main.cpp` 排除) |
| `tests/*.cpp` | 全部编入 `sim_tests` 一个测试程序; 手卷迷你框架, 见"测试约定" |
| `tests/guest/` | guest 测试: C 程序 (共享 `crt0.S`) 与裸 `.S`, 经 `link.ld` 链接成扁平 bin 到 `build/guest/`; `include/npu.inc` 为共享的 NPU 编码宏与配置常量 (供 tests/guest 与 apps/qwen3 共同使用) |
| `tests/riscv-tests-env/riscv_test.h` | 官方 riscv-tests 的自定义最小 env: 无特权态/tohost, PASS/FAIL 走 `ebreak`/`a0` 约定 |
| `scripts/fetch_riscv_tests.sh` | 拉取官方 riscv-tests (pinned commit) 到 `external/` (gitignored) |
| `apps/qwen3/` | Qwen3-0.6B 推理 guest 程序 (rv64im 裸机, NPU kernel 库 + forward 编排, 末位 prompt token 上带数值对齐探针); `dims.h` 为 forward.c 与 kernels.S 共享的缓冲区容量常量 (纯 #define, C 与汇编两侧包含, 模型维度校验据此拒绝超限); `xmake build qwen3_app` 交叉编译, 模型 blob 由 `scripts/qwen3/` 生成 |
| `scripts/qwen3/` | Qwen3 宿主工具: `fetch_model.sh` 下载模型权重/分词器到 `external/qwen3` (sha256 固定), `convert.py` safetensors→q8s32 模型 blob (含 `--tiny` 随机小模型), `ref.py` tiny 模型 numpy 对齐参考, `tokenizer.py` 纯 Python 分词器, `tiny_check.py` guest 输出与 ref.py 的逐探针对齐比较 (由 `--qwen3test=y` 的 xmake 测试驱动) |
| `xmake/guest.lua`, `xmake/riscv_tests.lua`, `xmake/apps.lua` | 按关注点拆出的构建脚本 (guest 镜像 / 官方套件 / Qwen3 应用与 tiny 回归), 由根 `xmake.lua` includes; 共享汇编逻辑在 `xmake/modules/guestimg.lua`; 增量重建用 `depend.on_changed` 的 `lastmtime` 锚定产物 mtime (产物缺失即重建) |
| `src/monitor/monitor.h`, `src/monitor/monitor.cpp` | monitor 平台胶水与运行层: 镜像装载 (`image@0xADDR`), 统一步进循环 `npu_exec` (步数预算, 返回停机的 `StepResult`, 预算耗尽返回 `std::nullopt`), 运行/交互分派与停机解释; ebreak/kernel.end 停机时 a0 作退出码, 属模拟器约定; 入口 main 在 `src/main.cpp` |
| `src/monitor/parse.h` | 命令行参数与 sdb 命令共享的数值解析 `parse_u64` (base 0, 全串消费) |
| `src/monitor/sdb.cpp` | sdb 交互 shell: 命令表驱动 (`SDB_COMMANDS`, help 与分派同表), handler 返回 false 退出循环; 空行重复上一条命令 |

## 建模约定 (修改前必读)

- 取指固定 32-bit 小端拼字; 不启用 RVC。`[1:0] = 11` 为 RV64IM 标量指令, 其余 quadrant 为 NPU 指令。
- 配置字段以 C++ 值存储 (如 `DType` 枚举), 不是打包的寄存器位型; 与指令编码间的位打包映射是后续工作。
- Tile 按行主序存储, 对应 canonical layout。
- 标量执行: 64-bit 回绕算术, 小端访存, load/store 要求自然对齐; 非对齐访问只报告状态, 不改变架构状态。
- 单 hart 模型下 RV64 `FENCE` 与 NPU `fence.*` 均为 no-op; `ECALL`/`EBREAK`/`kernel.end` 使 `step()` 停在当前 PC, 交给调用方处理。
- 主机浮点策略: 构建经 `-ffp-contract=off` (xmake.lua) 禁止编译器隐式乘加融合, 手册逐次舍入的运算 (如 sumsq 先乘积舍入再累加) 依赖此约定; `fmadd` 的单次舍入经显式 `std::fma` 表达。`FpEnvGuard` (`src/execute/execute.cpp`) 管运行时浮点环境 (舍入模式/异常标志), 约束不了编译期融合, 两者各管一层。
- NPU 指令统一在默认浮点环境 (RNE 舍入, 默认 subnormal 处理) 下执行: `exec_npu_instr` 入口的 `FpEnvGuard` (`src/execute/execute.cpp`) 保存并恢复宿主的浮点环境, 执行结果只取决于架构输入。
- 状态与诊断的区分是刻意的: `illegal` (译码失败的唯一类别: 手册预留编码, 以及本 hart 未实现的 RV64 扩展编码——按 RISC-V 语义后者同为非法指令) / `execution_not_implemented` (新增指令的开发中间态) / `invalid_configuration` 都是**模拟器诊断, 不是 ISA 定义的 trap**。理论上不可达的路径 (不变式被破坏, 即模拟器自身 bug) 用 `unreachable()` 当场中止, 不走 `StepStatus`。
- 结构合法性只校验一次: `decode()` 产生的 IR 保证结构合法 (op code、domain、操作数 variant、寄存器号都在某个真实编码允许的范围内); `execute_npu` 入口用 `valid_structure` (`src/execute/validate.h`) 统一复查, 手工构造的非法 IR (embedder bug) 报 `illegal_instruction`。各执行器不再重复结构检查, 只负责配置与数据的动态约束 (dtype/shape/len/描述符), 失败报 `invalid_configuration`。
- `Simulator::reset(entry_pc)` 清模拟器状态并设 PC, 不清内存; ISA 手册未定义架构复位值与入口地址。`sim_run` 的装载地址 (`0x80000000`), RAM 大小 (128 MiB), 串口控制台 MMIO 地址 (`0x10000000`) 与 `EBREAK` 时 `a0 == 0` 判成功, 都是 **runner 约定, 不是架构契约**。
- 模拟器自有约定 (复位, 装载, 停机判定, 串口) 统一在本文档说明 (见本节与文件组织表); 代码注释只陈述当前行为, 不反复标注 "convention / not ISA" 公式句。

## ISA 语义同步规则 (重要)

**最终事实来源是 ISA 手册仓库** (`../Tile-NPU-ISA/`): 规范语义以 `yai-isa.typ` 为准, 编码细节以 `编码设计规则.md` 与 `rivet/*.yaml` 为准。

- 模拟器不得实现手册未定义的行为; 手册语焉不详时, 先在手册仓库澄清/修正, 模拟器再跟随。
- 编码或语义变更需三处同步: `yai-isa.typ` (+ 重新编译 PDF 确认), `编码设计规则.md`, 本仓库 (译码/执行/测试)。
- 发现手册与模拟器行为不一致时, 默认按手册修模拟器; 认为手册有误时, 先改手册再改模拟器, 不要只改一边。

## 新增指令的实现流程

1. `src/decode/encoding.h` 补充字段提取 (若无对应格式)。
2. 译码: `include/tile_npu/instruction.h` 增加带数据域和 typed operand 的指令类型, 在对应译码表 (`src/decode/decode_npu.cpp` 或 `src/decode/decode_rv64im.cpp`) 加一行 `{pattern, handler}`; pattern 无法表达的剩余检查 (如寄存器号边界) 写进自定义 handler, 保留编码必须报 `illegal`。
3. 执行: `src/execute/validate.h` 为新指令类型加 `valid_structure` overload (结构契约与译码 handler 保持一致), `src/execute/execute.cpp` 分派到对应执行器, 语义严格按手册伪代码; 未实现的执行报 `execution_not_implemented`, 不要留静默路径。
4. 测试: 每新增操作必须补三类用例 — 合法编码, 保留/非法编码, 边界值 (立即数边界, 寄存器编号边界, 有效 shape 边界); NPU 正例优先写 guest 镜像, 负例留 host 侧表驱动 (见"测试约定")。

## 测试约定

- 无第三方测试框架: `tests/framework.h` 提供手卷迷你框架 — `TEST_CASE(name)` 静态自动注册, `tests/main.cpp` 按名字排序后逐用例跑并汇总 (重名直接报错; `sim_tests --list` 列出全部用例, `sim_tests --filter SUBSTR` 只跑名字匹配的用例); `check_eq(actual, expected, what)` 失败时带 actual/expected 值与 file:line, 指令字/地址等数值用 `hex()` 包装后按十六进制打印。host 侧执行测试的共享 fixture (`CODE_BASE`/`load_one`/`tile_cfg`/`acc_cfg`/`vec_cfg`/`f32_bits`/`i8_bits`/`write_word`) 集中在 `tests/support/execute_helpers.h`, 纯译码测试只需 `framework.h`。
- 指令级测试直接内联机器码常数, 行尾注释对应的汇编形式 (如 `0x002081b3u, // add x3, x1, x2`)。
- guest 镜像: `tests/guest/**.c` 由 RISC-V 工具链交叉编译 (共享 `crt0.S` 起栈并调 `main`, 裸 `.S` 自带 `_start`), 链接成扁平 bin (构建产物, 不入库); `rv64im/` 下为自校验程序 (`main` 返回 0 即 PASS, 经 `ebreak`/`a0` 约定上报), 逐个注册为 `sim_run` 的 xmake 测试。需要 riscv64 工具链, 前缀可用 `xmake f --riscv_cross=` 改。镜像依赖挂在 `sim_tests` 上 (`sim_run` 二进制本身不需要交叉工具链); 过滤单跑 guest 测试前先 `xmake build guest_images` 或跑一遍完整 `xmake test`。
- NPU 指令正例写 guest 镜像: `tests/guest/npu/*.S` 用 `npu.inc` 的编码宏 (字段布局与 `src/decode/encoding.h` 一一对应), host 侧 driver (`tests/npu_guest_test.cpp`) 装载镜像、设配置、逐条步进断言; "跑到停止并检查 a0" 的自校验镜像统一走 `check_guest_image()` helper (停止状态默认 `breakpoint`, `kernel.end` 见 sys 用例), 步进循环本身由 `run_until_stop()` 提供 — 超过步数预算会报告 timeout, 步数, PC 与 PC 处指令字; 非法编码负例留在 host 侧表驱动。
- 官方套件 (可选): `xmake f --rvtests=y` 后 `riscv-tests` 的 rv64ui/rv64um 逐个注册为 `sim_run` 测试; `fence_i` (Zifencei) 与 `ma_data` (依赖 trap 投递) 在 xmake.lua 里显式排除。
- tiny Qwen3 对齐回归 (可选): `xmake f --qwen3test=y` 后注册 `qwen3_tiny/forward` 测试 — `convert.py --tiny` 生成固定 seed 随机模型, `sim_run` 跑 guest forward, `scripts/qwen3/tiny_check.py` 与 `ref.py` 的 numpy 参考逐探针对齐 (整数探针位精确, f32 探针走容差, 生成 token 精确); python3 或 numpy 不可用时打印警告并跳过注册, 不影响其余测试。
- 外部 `.bin` 测试镜像不入库, 路径由命令行提供, 不作为构建依赖。

## 代码风格

- 格式化以仓库根目录 `.clang-format` (LLVM 派生) 为准; 改动过的 `.h`/`.cpp` 提交前跑一遍 `clang-format -i`, 保证各机器输出一致。clang-format 只作用于 C++ 源文件, 命令必须带类型过滤 (`git ls-files '*.h' '*.cpp' | xargs clang-format -i`), 不带过滤会把 YAML/Markdown 等非 C++ 文件当 C++ 改坏。`tests/riscv-tests-env/riscv_test.h` 的宏体是 GNU as 汇编, 靠文件内的 `/* clang-format off */` 豁免, 不得删除; `npu.inc`/`*.S` 等其余汇编内容按扩展名天然不在过滤范围。配置含 clang-format 14 引入的键, 旧版本会因不认识的键拒绝整个配置, 故要求 clang-format >= 14。
- 遵循现有代码: `#pragma once`, `snake_case` 函数/变量, `enum class` 强类型枚举, 尽可能 `constexpr`。
- 常量名用 `ALL_CAPS` (如 `MATRIX_EXTENT`), 不用 `k` 前缀。
- 查表性质的 switch、位段拼装和编码枚举/元数据表用对齐排版 (编码枚举可按编码空间分组、一行多个枚举项), 并以 `// clang-format off`/`on` 保护 (格式化器不会动保护区)。
- 位段一律通过 `encoding::field<Hi,Lo>` 提取, 不在别处散落魔数移位。
- 注释为英文, 只写"为什么/哪一层约定", 不复述代码; 见名知意处不写注释; 陈述现状, 不用否定对比句式 (不写"而不是 X / 不再 X / X 不会到这里"), 不描述历史行为。

## 架构扩展约定

- 译码结果必须表达语义操作数的域和类型; 不要为新指令复用一个含义不明确的 `dest/src1/src2` 三元组。
- 矩阵执行通过 `MatrixView` 访问逻辑坐标, 描述符访存通过 `DescriptorView` 计算地址与提交 `base_addr` 自增; 不在具体指令中散落行主序索引或描述符位标志解释。
- 配置指令通过 `read_config_field`/`write_config_field` 和 `config_field_from_code` 访问字段, 不在各个执行器中复制字段编号与范围检查。
- 配置类型与字段编号的组合未定义时, `cfg.seti`/`cfg.setx`/`cfg.get` 在译码阶段报 `illegal`; 执行入口对手工构造的 IR 复查同一规则, 并要求 `cfg.copy` 的字段编号为零。
- 写入校验的拒绝规则有手册依据: 合法字段的非法写入值对应 `cfg.seti`/`cfg.setx` 的 `CFG_ERROR`, 模拟器上报 `invalid_configuration`。手册未定义 `CFG_ERROR` 的后续架构行为, `invalid_configuration` 是模拟器诊断, 具体 ISA trap 行为以手册为准。

## Git 工作流

- 主分支 `main`; 开发走特性分支 (如 `feat/rv64im-sim`) 再合入。
- 提交信息用英文 Conventional Commits: `<type>(<scope>): <imperative summary>`; 常用 type: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`; 摘要简短, 不以句号结尾。
- 不提交构建产物与外部套件: `.xmake/`, `build/`, `.cache/`, `compile_commands.json`, `external/` 均已 gitignore。

## 安全注意事项

- 本仓库无网络服务, 无密钥; 唯一外部输入是测试用 `.bin` 镜像, 不提交, 不运行来源不明的镜像。
