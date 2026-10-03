# Tile-NPU-Sim 开发约定 (供 AI 编码代理阅读)

## 项目概览

Tile NPU 指令集的功能模拟器 (functional simulator), 独立 git 仓库, 远端为 `git@github.com:0x1B05/Tile-NPU-Sim`。当前为单 hart, 按程序序同步执行的库 (`tile_npu_sim` 静态库) 加 monitor 程序 `sim_run` (交互 sdb shell + batch 模式); 译码与执行已覆盖 32-bit RV64IM 整数指令和 Tile `tadd`; 其余 NPU 指令待逐族补充。

配套的 ISA 手册仓库在 `../Tile-NPU-ISA/` (另一个独立 git 仓库)。

**注意**: 本仓库无包管理清单以外的第三方依赖, 无 CI; 验证方式就是 `xmake test` 通过。

## 技术栈

- 语言: C++20 (`namespace tile_npu`), 无第三方库依赖
- 构建: xmake (编译器缓存在本目录 `.cache/ccache`, 已 gitignore)
- 文档语言: `README.md` 与代码注释为英文; 本文件为中文

## 构建与测试命令

```bash
xmake                                  # 构建 tile_npu_sim 静态库与 sim_run
xmake test                             # 构建并运行全部测试 (sim_tests + guest 镜像)
xmake run sim_run program.bin          # sdb 交互模式 (si/c/info r/x/q)
xmake run sim_run -b a.bin b.bin       # batch 模式, 逐镜像报 PASS/FAIL
xmake f --monitor=n && xmake           # 关闭 monitor, 只构建库
scripts/fetch_riscv_tests.sh           # 拉取官方 riscv-tests 到 external/ (一次性)
xmake f --rvtests=y && xmake test      # 额外跑官方 rv64ui/rv64um 套件
xmake project -k compile_commands      # 为 clangd 生成编译数据库
```

构建配置用 xmake 原生 option (`xmake f --xxx=y/n`, 定义在 xmake.lua 顶部), 不引入 Kconfig 工具链。

## 文件组织

| 文件 | 作用 |
| ---- | ---- |
| `include/tile_npu/state.h` | 架构状态: 五个数据域的寄存器堆 (Tile/Acc/Vec8/Vec32/Scalar), 计算配置 (TC/AC/BC/VC) 与访存描述符 (TM/AM/BM/VM), 物理尺寸常量 |
| `include/tile_npu/memory.h` | 字节寻址内存接口; `FlatMemory` 为平坦 RAM 实现 |
| `include/tile_npu/instruction.h` | typed instruction IR: scalar operands, NPU domains/registers, and elementwise operation types |
| `include/tile_npu/decode.h` | 译码结果类型: typed scalar/NPU instructions, `DecodeError`, `DecodeResult` |
| `include/tile_npu/config.h`, `src/config.cpp` | 统一配置/描述符字段访问、字段编号映射与写入校验 |
| `include/tile_npu/matrix_view.h` | 逻辑矩阵坐标到物理寄存器存储的 view, 当前实现 canonical layout |
| `include/tile_npu/descriptor_view.h` | TM/AM/BM/VM 地址计算、保留位校验与访存后 `base_addr` 自增 |
| `include/tile_npu/simulator.h` | `Simulator` 单步 API: `reset(entry_pc)`, `step()` → `StepResult`/`StepStatus` |
| `src/decode/encoding.h` | 位段提取 (`field<Hi,Lo>`) 与 RISC-V R/I/S/B/U/J/Shift, NPU 各格式的字段解析 |
| `src/decode/pattern.h` | INSTPAT 风格译码机制: 编译期 `Pattern` 位型解析, `DecodeEntry` 译码表, `as`/`fail`/`match` 通用 handler |
| `src/decode/decode.cpp` | NPU 译码表 (q != 11): major4/sel3/funct4 由 pattern 直接匹配, 首条命中生效 |
| `src/decode/decode_rv64im.cpp` | RV64IM 译码表 (q = 11): 一行一条指令的 pattern, 表尾 catch-all 划分 illegal / not_implemented |
| `src/execute/execute.cpp` | 执行分派 (标量 / NPU 各域执行器) |
| `src/execute/execute_elementwise.cpp` | 逐元素 (E_BIN) 族执行器: 公共约束校验与逐元素循环抽象为模板, 每个 op 只提供算术 lambda |
| `src/execute/execute_rv64i.cpp`, `src/execute/execute_rv64m.cpp` | RV64I 与 M 扩展语义 |
| `src/simulator.cpp`, `src/memory.cpp` | 取指 (小端 4 字节拼字) 与步进循环, 内存实现 |
| `tests/*.cpp` | 全部编入 `sim_tests` 一个测试程序; 手卷迷你框架, 见"测试约定" |
| `tests/guest/` | guest 测试: C 程序 (共享 `crt0.S`) 与裸 `.S` (NPU 编码宏在 `npu.inc`), 经 `link.ld` 链接成扁平 bin 到 `build/guest/` |
| `tests/riscv-tests-env/riscv_test.h` | 官方 riscv-tests 的自定义最小 env: 无特权态/tohost, PASS/FAIL 走 `ebreak`/`a0` 约定 |
| `scripts/fetch_riscv_tests.sh` | 拉取官方 riscv-tests (pinned commit) 到 `external/` (gitignored) |
| `src/monitor/monitor.cpp` | `sim_run` 主程序: 参数解析, 镜像装载, batch 模式 |
| `src/monitor/sdb.cpp` | sdb 交互 shell: `si`/`c`/`info r`/`x`/`q`, 空行重复上一条命令 |

## 建模约定 (修改前必读)

- 取指固定 32-bit 小端拼字; 不启用 RVC。`[1:0] = 11` 为 RV64IM 标量指令, 其余 quadrant 为 NPU 指令。
- 配置字段以 C++ 值存储 (如 `DType` 枚举), 不是打包的寄存器位型; 与指令编码间的位打包映射是后续工作。
- Tile 按行主序存储, 对应 canonical layout。
- 标量执行: 64-bit 回绕算术, 小端访存, load/store 要求自然对齐; 非对齐访问只报告状态, 不改变架构状态。
- 单 hart 模型下 `FENCE` 为 no-op; `ECALL`/`EBREAK` 使 `step()` 停在当前 PC, 交给调用方处理。
- 状态与诊断的区分是刻意的: `decode_not_implemented` (编码可能合法但未译码) / `illegal` (保留编码) / `execution_not_implemented` / `invalid_configuration` 都是**模拟器诊断, 不是 ISA 定义的 trap**。
- `Simulator::reset(entry_pc)` 清模拟器状态并设 PC, 不清内存; ISA 手册未定义架构复位值与入口地址。`sim_run` 的装载地址 (`0x80000000`), RAM 大小 (128 MiB) 与 `EBREAK` 时 `a0 == 0` 判成功, 都是 **runner 约定, 不是架构契约**。
- 凡是模拟器自有约定 (复位, 装载, 停机判定), 必须在代码注释或文档中明确标注为模拟器约定, 不得写成 ISA 行为。

## ISA 语义同步规则 (重要)

**最终事实来源是 ISA 手册仓库** (`../Tile-NPU-ISA/`): 规范语义以 `yai-isa.typ` 为准, 编码细节以 `编码设计规则.md` 与 `rivet/*.yaml` 为准。

- 模拟器不得实现手册未定义的行为; 手册语焉不详时, 先在手册仓库澄清/修正, 模拟器再跟随。
- 编码或语义变更需三处同步: `yai-isa.typ` (+ 重新编译 PDF 确认), `编码设计规则.md`, 本仓库 (译码/执行/测试)。
- 发现手册与模拟器行为不一致时, 默认按手册修模拟器; 认为手册有误时, 先改手册再改模拟器, 不要只改一边。

## 新增指令的实现流程

1. `src/decode/encoding.h` 补充字段提取 (若无对应格式)。
2. 译码: `include/tile_npu/instruction.h` 增加带数据域和 typed operand 的指令类型, 在对应译码表 (`src/decode/decode.cpp` 或 `src/decode/decode_rv64im.cpp`) 加一行 `{pattern, handler}`; pattern 无法表达的剩余检查 (如寄存器号边界) 写进自定义 handler, 保留编码必须报 `illegal`。
3. 执行: `src/execute/execute.cpp` 分派到对应执行器, 语义严格按手册伪代码; 未实现的执行报 `execution_not_implemented`, 不要留静默路径。
4. 测试: 每新增操作必须补三类用例 — 合法编码, 保留/非法编码, 边界值 (立即数边界, 寄存器编号边界, 有效 shape 边界); NPU 正例优先写 guest 镜像, 负例留 host 侧表驱动 (见"测试约定")。

## 测试约定

- 无第三方测试框架: `tests/check.h` 提供手卷迷你框架 — `TEST_CASE(name)` 静态自动注册, `tests/main.cpp` 逐用例跑并汇总; `check_eq(actual, expected, what)` 失败时带 actual/expected 值与 file:line, 指令字/地址等数值用 `hex()` 包装后按十六进制打印。
- 指令级测试直接内联机器码常数, 行尾注释对应的汇编形式 (如 `0x002081b3u, // add x3, x1, x2`)。
- guest 镜像: `tests/guest/**.c` 由 RISC-V 工具链交叉编译 (共享 `crt0.S` 起栈并调 `main`, 裸 `.S` 自带 `_start`), 链接成扁平 bin (构建产物, 不入库); `rv64im/` 下为自校验程序 (`main` 返回 0 即 PASS, 经 `ebreak`/`a0` 约定上报), 逐个注册为 `sim_run -b` 的 xmake 测试。需要 riscv64 工具链, 前缀可用 `xmake f --riscv_cross=` 改。
- NPU 指令正例写 guest 镜像: `tests/guest/npu/*.S` 用 `npu.inc` 的编码宏 (字段布局与 `src/decode/encoding.h` 一一对应), host 侧 driver (`tests/npu_guest_test.cpp`) 装载镜像、设配置、逐条步进断言; 非法编码负例留在 host 侧表驱动。
- 官方套件 (可选): `xmake f --rvtests=y` 后 `riscv-tests` 的 rv64ui/rv64um 逐个注册为 `sim_run -b` 测试; `fence_i` (Zifencei) 与 `ma_data` (依赖 trap 投递) 在 xmake.lua 里显式排除。
- 外部 `.bin` 测试镜像不入库, 路径由命令行提供, 不作为构建依赖。

## 代码风格

- 格式化以仓库根目录 `.clang-format` (LLVM 派生) 为准; 改动过的文件提交前跑一遍 `clang-format -i`, 保证各机器输出一致。
- 遵循现有代码: `#pragma once`, `snake_case` 函数/变量, `enum class` 强类型枚举, 尽可能 `constexpr`。
- 常量名用 `ALL_CAPS` (如 `MATRIX_EXTENT`), 不用 `k` 前缀。
- 查表性质的 switch 和位段拼装用对齐排版, 并以 `// clang-format off`/`// clang-format on` 保护 (格式化器不会动保护区)。
- 位段一律通过 `encoding::field<Hi,Lo>` 提取, 不在别处散落魔数移位。
- 注释为英文, 只写"为什么/哪一层约定", 不复述代码。

## 架构扩展约定

- 译码结果必须表达语义操作数的域和类型; 不要为新指令复用一个含义不明确的 `dest/src1/src2` 三元组。
- 矩阵执行通过 `MatrixView` 访问逻辑坐标, 描述符访存通过 `DescriptorView` 计算地址与提交 `base_addr` 自增; 不在具体指令中散落行主序索引或描述符位标志解释。
- 配置指令通过 `read_config_field`/`write_config_field` 和 `config_field_from_code` 访问字段, 不在各个执行器中复制字段编号与范围检查。
- 写入校验的拒绝规则有手册依据: `ConfigAccessError` 对应手册中 `cfg.seti`/`cfg.setx` 校验失败时的 `CFG_ERROR`; 手册未定义 `CFG_ERROR` 的后续架构行为, 模拟器以错误状态上报属于模拟器约定。`invalid_configuration` 仍是模拟器诊断, 具体 ISA trap 行为以手册为准。

## Git 工作流

- 主分支 `main`; 开发走特性分支 (如 `feat/rv64im-sim`) 再合入。
- 提交信息用英文 Conventional Commits: `<type>(<scope>): <imperative summary>`; 常用 type: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`; 摘要简短, 不以句号结尾。
- 不提交构建产物与外部套件: `.xmake/`, `build/`, `.cache/`, `compile_commands.json`, `external/` 均已 gitignore。

## 安全注意事项

- 本仓库无网络服务, 无密钥; 唯一外部输入是测试用 `.bin` 镜像, 不提交, 不运行来源不明的镜像。
