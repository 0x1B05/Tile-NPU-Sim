# Tile-NPU-Sim 开发约定 (供 AI 编码代理阅读)

## 项目与职责

本仓库为 C++20 功能模拟器：单 hart、按程序序同步执行 RV64IM 标量与 NPU 指令。核心为 `tile_npu_sim` 静态库，`sim_run` 提供程序运行和交互调试，命令行由 CLI11 解析。使用方法与 Qwen3 模型准备流程见 `README.md`；本文件维护开发约定和模块边界。

ISA 手册位于相邻仓库 `../ISA/`，指令语义以 `yai-isa.typ` 为准；`rivet/*.yaml` 定义编码图的位段，`编码设计规则.md` 记录指令空间与字段布局设计。

- 发现手册与实现不一致时，默认按手册修模拟器；若需要调整 ISA，先澄清手册，再同步受影响的实现与测试。
- 手册未定义的行为须先明确设计。仅在编码或布局变化时更新相应的编码设计说明和 Rivet 文件，不要求每次语义修改都改这些文件。

## 构建与验证

在本目录执行：

```bash
xmake                                  # 构建静态库与 sim_run
xmake test                             # 构建并运行当前配置启用的全部测试
xmake run sim_run program.bin          # 默认运行程序
xmake run sim_run -i program.bin       # 进入交互 sdb
xmake f --monitor=n                    # 关闭 runner，随后 xmake 只构建库
xmake project -k compile_commands      # 为 clangd 生成 compile_commands.json
```

- 修改后运行 `xmake test`。默认的 `sim_tests/core` 运行全部 host 用例 (含 NPU guest driver)，`sim_run/guest/*` 运行标量自校验程序；xmake 测试项数量不等于指令用例数量。
- 构建选项在 `xmake.lua`，使用 `xmake f --xxx=y/n`；编译器缓存位于仓库内 `.cache/ccache`。
- guest 镜像需要 RISC-V 交叉工具链，默认前缀 `riscv64-linux-gnu-`，可用 `xmake f --riscv_cross=<prefix>` 覆盖；构建 `sim_run` 本身不需要交叉工具链。
- 可选官方套件：先运行 `scripts/fetch_riscv_tests.sh`，再启用 `--rvtests=y`。可选 tiny Qwen3 回归：启用 `--qwen3test=y`，需要 Python 3 和 NumPy；缺少依赖时会警告并跳过该回归。两项均要求 `--monitor=y`。
- 过滤运行 guest 测试前，先构建对应镜像目标 (`guest_images` 或 `riscv_tests`)，或运行一次完整的 `xmake test`；镜像依赖挂在 `sim_tests` 上。

## 模块导航

| 路径 | 职责 |
| ---- | ---- |
| `include/tile_npu/state.h` | 寄存器堆、配置、描述符与物理尺寸常量 |
| `include/tile_npu/instruction.h` | 标量/NPU 语义 IR 与操作数类型 |
| `src/op_info.h` | 译码与结构检查共享的操作元数据：寄存器数量、操作/数据域合法性与 funct 表 |
| `include/tile_npu/decode.h` | `DecodeResult` 与译码接口 |
| `include/tile_npu/simulator.h`、`src/simulator.cpp` | `Simulator::reset` / `step`、取指与步进状态词表 `StepStatus` |
| `src/config.h`、`src/config.cpp` | 配置字段编号映射、字段读写、值转换与写入校验 |
| `src/execute/matrix_view.h` | 固定大小矩阵寄存器的非拥有视图，只负责物理坐标映射 |
| `src/execute/descriptor_view.h` | 描述符地址计算、保留位检查及访存后基地址自增 |
| `include/tile_npu/platform/`、`src/platform/` | `AddressSpace` 接口、`Ram`、`SystemBus` 与 `SerialConsole` |
| `src/decode/encoding.h`、`pattern.h` | 位段提取、格式解析、编译期 pattern 解析与按序匹配 |
| `src/decode/decode*.cpp` | 顶层 q 分派，以及 RV64IM / NPU 译码表与 IR 构造 |
| `src/execute/execute.h`、`execute.cpp` | 执行上下文、结果类型及标量/NPU 分派 |
| `src/execute/validate.h` | IR 的统一结构合法性检查 |
| `src/execute/config_check.h` | dtype、shape、layout 等配置约束与配置匹配 |
| `src/execute/execute_*.cpp` | 按指令族实现执行语义；`execute_cfg.cpp` 负责配置指令取值和执行结果转换 |
| `src/execute/domain.h` | 数据域到寄存器堆/配置/描述符的绑定、域分派与有效区域遍历 |
| `src/execute/ebin_op.h`、`operand.h`、`f32_util.h`、`rv64_util.h` | 执行域共享的元素运算、操作数读取和数值工具 |
| `src/utils/` | 通用内部工具：小端转换、十六进制格式化、variant visitor 与 `unreachable` |
| `src/main.cpp`、`src/monitor/` | CLI、镜像装载、`npu_exec` 步进循环和 sdb 命令 |
| `tests/` | host 测试框架、译码/执行测试及 guest 源码 |
| `apps/qwen3/`、`scripts/qwen3/` | Qwen3 guest 与权重转换、分词、NumPy 参考及对齐检查工具 |
| `xmake/guest.lua`、`riscv_tests.lua`、`apps.lua` | guest、官方套件、Qwen3 构建与测试注册；交叉编译公共逻辑在 `xmake/modules/guestimg.lua` |

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
- 单 hart 同步模型中 RV64 `FENCE` 与 NPU `fence.*` 为 no-op。`ECALL`、`EBREAK`、`kernel.end` 停在当前 PC，交给调用方处理。
- `reset(entry_pc)` 将状态恢复为 `State{}` 并设置 PC，不清内存。runner 默认入口、RAM 容量、串口窗口以及 `ebreak/kernel.end` 以 `a0` 为退出码的约定见 README；这些平台约定与 ISA 指令语义分开维护。

## 配置访问契约

- 配置访问使用 `config_field_from_code`、`read_config_field`、`write_config_field`，字段编号和写入范围检查集中在 `config.cpp`。
- 配置类型与字段编号的组合未定义时，`cfg.seti/setx/get` 在译码阶段报 `illegal`；执行入口对手工构造的 IR 复查，并要求 `cfg.copy` 的字段编号为零。
- 字段值以 ISA 传递的 64-bit 原始值进出配置接口：枚举字段为架构编码，带符号步长为符号扩展的 i32。`read_config_field` 返回 `cfg.get` 写入 `xD` 的值，要求寄存器类型、编号和字段组合已合法；非法调用进入 `unreachable()`。字段编号解析和配置写入保留可检查的错误状态。
- `write_config_field` 的 `ConfigAccessStatus::success` 表示写入成功；合法字段的非法值对应手册的 `CFG_ERROR`，执行器映射为 `invalid_configuration`。模拟器不自行定义 `CFG_ERROR` 的 trap 行为。

## 新增指令与测试

1. 在 `instruction.h` 添加带域与语义操作数的 IR，操作元数据与合法性规则放在 `src/op_info.h`；已有 `IndexOperand` / `FillOperand` 等 variant 可表达立即数与 Scalar 来源。避免用裸整数加 bool 维护二选一关系。
2. 必要时补充 `encoding.h` 的格式解析，在译码表增加 pattern 与 handler。首条匹配生效，需检查编码重叠；保留编码和超出实际寄存器数量的编号报 `illegal`。
3. 在 `validate.h` 补充结构检查，在 `execute.cpp` 接入分派，并在所属指令族实现语义。开发中尚未实现的执行显式返回 `execution_not_implemented`。
4. 每个新增操作覆盖合法编码、保留/非法编码和边界值，包括立即数、寄存器号与有效 shape 边界。

测试放置与运行约定：

- `tests/framework.h` 提供 `TEST_CASE` 与 `check_eq`；`sim_tests --list` 列出用例，`--filter SUBSTR` 按名称筛选。公共执行 fixture 在 `tests/support/execute_helpers.h`，各指令族的编码构造器在 `tests/support/<族>_helpers.h`。
- host 译码测试直接内联机器码常量，行尾标注汇编；指令字和地址的断言用 `hex()` 包装。
- NPU 正例优先放在 `tests/guest/npu/*.S`，复用 `tests/guest/include/npu.inc` 的编码宏；`tests/npu_guest_test.cpp` 负责装载、执行与断言。复用 `run_until_stop` 的步数上限和停机诊断，或 `check_guest_image` 的自校验流程；非法编码负例放在 host 测试。
- guest C 程序链接共享 `crt0.S`，裸 `.S` 提供自己的 `_start`，生成到构建目录的 `guest/`。标量自校验程序放在 `tests/guest/rv64im/`，按 `ebreak/a0` 约定退出。
- 官方套件的 env 在 `tests/riscv-tests-env/`；排除项在 `xmake/riscv_tests.lua`，对应尚未建模的 Zifencei 与 trap 投递。
- tiny Qwen3 回归以固定 seed 模型对照 NumPy 参考：整数探针与 token 精确匹配，f32 探针使用容差。配置和驱动在 `xmake/apps.lua` 与 `scripts/qwen3/tiny_check.py`。
- 仓库内 guest 源码生成的镜像属于构建依赖；用户提供的外部镜像和下载的模型不加入构建依赖或提交。

## 代码风格与 Git

- `README.md` 与代码注释使用英文，本文件使用中文。注释说明设计原因、契约和必要的位段布局，避免复述显然的代码或历史变更。
- 不在执行器中重复声明“执行失败不改变架构状态”等通用保证；需要注释时，说明具体的暂存、预检或提交顺序及其原因。
- 沿用 `#pragma once`、`snake_case`、`enum class` 和 `constexpr`；常量用 `ALL_CAPS`。
- 使用仓库 `.clang-format`，要求 clang-format >= 14。提交前只格式化本次修改的 C++ 文件，例如 `clang-format -i src/decode/decode_npu.cpp`；不对 Markdown、YAML 或汇编源码运行它。
- `tests/riscv-tests-env/riscv_test.h` 的宏体是 GNU as 汇编，保留其 `/* clang-format off */` 保护。对齐的译码表、位段表和元数据表用 `// clang-format off/on` 保持排版。
- 指令位段通过 `encoding::field<Hi,Lo>` 与格式解析器提取，避免重复魔数位移。译码位图按 Rivet 的具体字段命名 (如 `dom2`、`IS`、`BX`)，与实际位段对齐。
- 主分支为 `main`，开发使用特性分支；提交遵循英文 Conventional Commits：`<type>(<scope>): <imperative summary>`，摘要简短、不以句号结尾。
- 不提交构建产物和下载内容：`.xmake/`、`build/`、`.cache/`、`compile_commands.json`、`external/` 已忽略。推送远端需用户明确要求。
