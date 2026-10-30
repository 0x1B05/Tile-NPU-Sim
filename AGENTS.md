# Tile-NPU-Sim 开发约定 (供 AI 编码代理阅读)

## 项目与职责

本仓库为 C++20 功能模拟器：单 hart、按程序序同步执行 RV64IM 标量与 NPU 指令。核心为内部静态库 `sim_core`；默认 target `sim` 是用于程序运行和调试的二进制，命令行由 CLI11 解析。使用方法见 `README.md`；本文件维护开发约定和模块边界。

ISA 手册位于相邻仓库 `../Tile-NPU-ISA/`，指令语义与编码位段以 `tile-npu-isa.typ` 和 `rivet/*.yaml` 为准。发现手册与实现不一致时默认修模拟器；需要调整 ISA 时先澄清手册，再同步实现与测试。只有编码或布局变化才更新编码设计说明和 Rivet 文件。

## 构建与验证

在本目录执行：

```bash
xmake                                    # 构建默认 sim 二进制
xmake f --menu                           # 按 Build、Debug、Guests 分组配置
xmake test                               # 只运行 host 单元测试，不需要交叉工具链
python3 tests/run.py guest --build       # 构建并运行 guest 测试
python3 tests/run.py riscv-tests --build # 构建并运行官方套件
python3 tests/run.py qwen3 --build       # 构建并运行 Qwen3 回归
python3 tests/run.py all --build         # 运行全部可用套件
xmake run sim program.bin                # 默认运行程序
xmake f --sdb=y && xmake                 # 构建带 sdb 的二进制
xmake run sim -i program.bin             # 进入交互 sdb
xmake project -k compile_commands        # 为 clangd 生成 compile_commands.json
```

- 修改后至少运行 `xmake test`。`sim_tests/core` 只包含 host 用例；guest 测试在独立二进制 `sim_guest_tests` 中，由 `tests/run.py` 调用。
- 构建选项在 `xmake.lua`，使用 `xmake f --xxx=y/n`；编译器缓存位于仓库内 `.cache/ccache`。`--itrace=y` 启用 ITRACE 与反汇编。
- 宿主构建启用 `set_warnings("all", "error")`，所有已启用的警告都按错误处理。
- guest 镜像需要 RISC-V 交叉工具链，默认前缀 `riscv64-linux-gnu-`，可用 `xmake f --riscv_cross=<prefix>` 覆盖；构建 `sim` 和 `sim_tests` 本身不需要交叉工具链。
- Qwen3 的 Q3B/Q3P 格式、prompt 分词与 `qwen3_run` 用法见 `README.md`。tiny 回归由 `tests/run.py qwen3` 驱动：`qwen3_snapshot_app`、模型、prompt 与 NumPy 参考均由脚本生成。
- `tests/run.py` 负责 suite 选择、过滤、构建缺失 artifact 和结果汇总；xmake 只负责生成 `sim_tests`、`sim_guest_tests`、`guest_images` 及可选 app/image artifact。

## 模块导航

| 路径 | 职责 |
| ---- | ---- |
| `include/tile_npu/state.h` | 寄存器堆、配置、描述符与物理尺寸常量 |
| `include/tile_npu/instruction.h` | 按功能组织的标量/NPU 语义 IR 与操作数类型，语义类型不要求与 major 一一对应 |
| `src/op_info.h` | 译码与结构检查共享的操作元数据：操作/数据域合法性与 funct 表 |
| `include/tile_npu/decode.h` | `DecodeResult` 与译码接口 |
| `include/tile_npu/simulator.h`、`src/simulator.cpp` | `Simulator::reset` / `step`、取指与步进状态词表 `StepStatus` |
| `src/config.h`、`src/config.cpp` | 配置字段编号映射、字段读写、值转换与写入校验 |
| `src/execute/matrix_view.h` | 固定大小矩阵寄存器的非拥有视图，只负责物理坐标映射 |
| `src/execute/descriptor_view.h` | 描述符地址计算、保留位检查及访存后基地址自增 |
| `include/tile_npu/platform/`、`src/platform/` | `AddressSpace` 接口、`Ram`、`SystemBus` 与 `SerialConsole` |
| `src/decode/encoding.h`、`pattern.h` | 位段提取、格式解析、编译期 pattern 解析与按序匹配 |
| `src/decode/decode*.cpp` | 顶层 q 分派，以及 RV64IM / NPU 译码表与 IR 构造 |
| `src/decode/disasm.*` | ITRACE 用反汇编：从解码 IR 生成 RV64IM 与 NPU 汇编文本 |
| `src/execute/execute.h`、`execute.cpp` | 执行上下文、结果类型及标量/NPU 分派 |
| `src/execute/validate.h`、`validate.cpp` | IR 的统一结构合法性检查，头文件仅声明 `valid_structure(const DecodedNpuInstr &)` |
| `src/execute/config_check.h` | dtype、shape、layout 等配置约束与配置匹配 |
| `src/execute/execute_*.cpp` | 按指令族实现执行语义；`execute_cfg.cpp` 负责配置指令取值和执行结果转换 |
| `src/execute/execute_move.cpp` | `FillInstr` 执行立即数/Scalar 填充，`MoveInstr` 执行同域复制 |
| `src/domain.h` | 数据域工具、寄存器堆/配置/描述符绑定、域分派与有效区域遍历 |
| `src/execute/elementwise_op.h`、`operand.h` | 执行域共享的元素运算、操作数读取和浮点工具 |
| `src/execute/reduce_util.h` | 规约树、absmax 叶映射与跨指令族共享的规约工具 |
| `src/utils/bits.h` | f32 位模式解释、整数/浮点存储位转换与符号扩展工具 |
| `src/utils/format.h` | 十六进制格式化工具 (hex16/hex32/hex64) |
| `src/utils/` | 通用内部工具：位模式、小端转换、variant visitor、十六进制格式化与 `unreachable` |
| `src/main.cpp`、`src/monitor/` | CLI、镜像装载、`npu_exec` 步进循环和 sdb 命令 |
| `tests/` | `units/` host 用例与入口、`guest/` guest driver 与 guest 源码、`include/` 测试头文件 |
| `tests/run.py` | host/guest/riscv-tests/Qwen3 suite runner |
| `apps/qwen3/`、`scripts/qwen3/` | Qwen3 guest、Q3B/Q3P 输入、权重转换、分词、NumPy 参考、运行与对齐检查工具 |
| `xmake/guest.lua`、`riscv_tests.lua` | guest 与官方套件镜像的 build-only target；交叉编译公共逻辑在 `xmake/modules/guestimg.lua` |
| `apps/qwen3/xmake.lua` | Qwen3 app 与 snapshot app 的构建，以及 `qwen3_run` 注册 |

公开模型类型和 API 放在 `include/tile_npu/`；实现内部头文件按所属模块放在 `src/`。只有跨模块通用工具放入 `src/utils/`，单个文件使用的 helper 优先留在该文件的匿名命名空间。

## 建模与执行契约

- 指令定长 32-bit，按小端取指；`q[1:0] = 11` 走 RV64IM，其余走 NPU，不启用 RVC。
- 寄存器元素保存位模式；配置字段使用 C++ 类型保存 (如 `DType` 枚举)，字段编号与值的转换由配置接口处理。
- 标量运算使用 64-bit 回绕语义；load/store 要求自然对齐，非对齐访问报告状态并保持架构状态。
- `MatrixView` 将逻辑坐标映射到固定尺寸、canonical 行主序存储，不携带有效 shape/layout；`matrix_view(reg)` 只接受 `MATRIX_ELEMENTS` 大小的寄存器。有效区域与 layout 由执行器按配置校验。
- 描述符访存经 `DescriptorView` 计算地址与自增。load 先暂存结果，store 先完整探测；访问失败时不提交寄存器、内存或描述符更新。
- `AddressSpace::contains` 为多元素传输提供预检：已接受范围内的 read/write 必须成功；失败的 read/write 不改变缓冲区或地址空间。新设备必须遵守此契约。
- `SystemBus` 优先将访问路由到接受该范围的设备，否则交给主存；设备按注册顺序匹配，设备窗口之间不得重叠。
- `decode()` 保证输出 IR 的结构合法；`exec_npu_instr` 通过 `valid_structure` 统一复查手工构造的 IR。各族执行器只检查 dtype/shape/描述符等动态约束，不逐层重复寄存器号和操作数 variant 检查。
- 译码失败使用 `DecodeError::illegal`；手工构造的非法 NPU IR 返回 `StepStatus::illegal_instruction`，非法配置返回 `invalid_configuration`。这些是模拟器诊断，尚未建模 ISA trap 投递；不变式被破坏属于实现错误，使用始终生效的 `unreachable()`。
- 浮点规则同时由编译与执行保证：`-ffp-contract=off` 禁止隐式乘加融合，`std::fma` 实现显式单次舍入；`FpEnvGuard` 在 NPU 执行入口设置默认浮点环境 (RNE、默认 subnormal 处理)，结束后恢复宿主环境。
- 规约按相邻两两二叉树执行，奇数节点直通下一层；`sum`/`sumsq` 的每个加法节点执行一次 f32 RNE，`sumsq` 先对乘积做 f32 舍入。`max`/`min`/`absmax`/`argmax` 的 NaN、带符号零与并列规则遵循手册，不使用单位元填充非空规约树。
- 单 hart 同步模型中 RV64 `FENCE` 与 NPU `fence.*` 为 no-op。`ECALL`、`EBREAK`、`kernel.end` 停在当前 PC，交给调用方处理。
- `reset(entry_pc)` 将状态恢复为 `State{}` 并设置 PC，不清内存。runner 默认入口、RAM 容量、串口窗口以及 `ebreak/kernel.end` 以 `a0` 为退出码的约定见 README；这些平台约定与 ISA 指令语义分开维护。

## 配置访问契约

- 配置访问使用 `decode_config_field`、`read_config_field`、`write_config_field_range`、`copy_config_range`，字段编号和写入范围检查集中在 `config.cpp`。
- 配置类型与字段编号的组合未定义时，`cfg.seti/setx/get` 在译码阶段报 `illegal`；对应 IR 的 `field` 只保存合法字段。执行入口会复查手工构造的 IR。`CFG_REG` 的写操作 (`setx/copy/setw`) 以 `n` 直接保存 `N5` 的零基目的偏移，`copy` 先读源再广播；`cfg.setw/getw` 只接受计算配置 (TC/AC/VC)，经 `write_compute_config_range`/`read_compute_config` 按 32-bit 配置字读写；配置字的保留位必须为零。
- 字段值以 ISA 传递的 64-bit 原始值进出配置接口：枚举字段为架构编码，带符号步长为符号扩展的 i32。`read_config_field` 返回 `cfg.get` 写入 `xD` 的值，要求寄存器类型、编号和字段组合已合法；非法调用进入 `unreachable()`。原始字段编号解析返回 `std::optional`，配置写入保留可检查的错误状态。
- `write_config_field_range`、`copy_config_range` 和 `write_compute_config_range` 的 `ConfigAccessStatus::success` 表示整个范围写入成功；合法字段的非法值对应手册的 `CFG_ERROR`，执行器映射为 `invalid_configuration`。模拟器不自行定义 `CFG_ERROR` 的 trap 行为。

## 新增指令与测试

1. 在 `instruction.h` 添加带域与语义操作数的 IR，操作元数据与合法性规则放在 `src/op_info.h`；已有 `IndexOperand` / `FillOperand` 等 variant 可表达立即数与 Scalar 来源。避免用裸整数加 bool 维护二选一关系。
2. 必要时补充 `encoding.h` 的格式解析，在译码表增加 pattern 与 handler。首条匹配生效，需检查编码重叠；保留编码和超出实际寄存器数量的编号报 `illegal`。
3. 在 `validate.cpp` 补充结构检查，在 `execute.cpp` 接入分派，并在所属指令族实现语义。开发中尚未实现的执行显式返回 `execution_not_implemented`。
4. 每个新增操作覆盖合法编码、保留/非法编码和边界值，包括立即数、寄存器号与有效 shape 边界。

测试放置与运行约定：

- `tests/include/framework.h` 提供 `TEST_CASE`、`check_eq` 与 `i8_bits`；`tests/units/main.cpp` 是 host 入口，`tests/guest/main.cpp` 是 guest 入口。`sim_tests --list` 列出 host 用例，`sim_guest_tests --list` 列出 guest 用例，二者都支持 `--filter SUBSTR`；只有 guest 入口接受 `--guest-dir`。host 执行 fixture 在 `tests/include/units/execute_helpers.h`，各指令族的编码构造器在 `tests/include/units/<族>_helpers.h`；guest 图片加载接口在 `tests/include/guest/guest_image.h`。
- host 译码测试使用内联机器码常量或 `tests/include/units/<族>_helpers.h` 的构造器，并标注对应汇编；指令字和地址的断言用 `hex()` 包装。
- NPU 正例优先放在 `tests/guest/npu/*.S`，复用 `tests/guest/include/npu.inc` 的编码宏；`tests/guest/npu_guest_test.cpp` 负责装载、执行与断言，镜像目录通过 `--guest-dir` 传入。复用 `run_until_stop` 的步数上限和停机诊断，或 `check_guest_image` 的自校验流程；非法编码负例放在 `tests/units`。
- guest C 程序链接共享 `crt0.S`，裸 `.S` 提供自己的 `_start`，生成到构建目录的 `guest/`。标量自校验程序放在 `tests/guest/rv64im/`，按 `ebreak/a0` 约定退出。
- 官方套件的 env 在 `tests/guest/riscv-tests-env/`；排除项在 `xmake/riscv_tests.lua`，对应尚未建模的 Zifencei 与 trap 投递。镜像由 `xmake build riscv_tests` 生成，由 `tests/run.py riscv-tests` 执行。
- tiny Qwen3 回归以固定 seed 模型对照 NumPy 参考：整数探针与 token 精确匹配，f32 探针使用容差。
- 仓库内 guest 源码生成的镜像属于构建依赖；用户提供的外部镜像和下载的模型不加入构建依赖或提交。

## 代码风格与 Git

- `README.md` 与代码注释使用英文，本文件使用中文。注释说明设计原因、契约和必要的位段布局，避免复述显然的代码或历史变更。
- 不在执行器中重复声明“执行失败不改变架构状态”等通用保证；需要注释时，说明具体的暂存、预检或提交顺序及其原因。
- 沿用 `#pragma once`、`snake_case`、`enum class` 和 `constexpr`；常量用 `ALL_CAPS`。
- 操作码合法性用元数据表或显式枚举分支判断。校验和执行分派的 `switch` 显式写出 `default`，分别拒绝非法输入或报告已验证输入违反不变式；数据域查询保留约定的无效值返回。新增枚举时同步检查所有相关分支，带 `default` 的分支不能依赖普通 `-Wswitch` 提醒遗漏。
- 使用仓库 `.clang-format`，要求 clang-format >= 14。提交前只格式化本次修改的 C++ 文件，例如 `clang-format -i src/decode/decode_npu.cpp`；不对 Markdown、YAML 或汇编源码运行它。
- `tests/guest/riscv-tests-env/riscv_test.h` 的宏体是 GNU as 汇编，保留其 `/* clang-format off */` 保护。对齐的译码表、位段表和元数据表用 `// clang-format off/on` 保持排版。
- 指令位段通过 `encoding::field<Hi,Lo>` 与格式解析器提取，避免重复魔数位移。译码位图按 Rivet 的具体字段命名 (如 `dom2`、`IS`、`BX`)，与实际位段对齐。
- 主分支为 `main`，开发使用特性分支；提交遵循英文 Conventional Commits：`<type>(<scope>): <imperative summary>`，摘要简短、不以句号结尾。
- 不提交构建产物和下载内容：`.xmake/`、`build/`、`.cache/`、`compile_commands.json`、`external/` 已忽略。推送远端需用户明确要求。
