# Sim 开发约定

- 本目录是独立 git 仓库, 使用 C++20 和 xmake; 从本目录运行 `xmake test`.
- 提交信息使用英文 Conventional Commits: `<type>(<scope>): <imperative summary>`. 例如 `feat(sim): decode RV64IM instructions`. 常用 type 为 `feat`, `fix`, `refactor`, `test`, `docs`, `chore`; 摘要简短, 不以句号结尾.
- `src/encoding.h` 集中提取指令字段和立即数; `src/decode.cpp` 分派 NPU 与标量指令; `src/decode_rv64im.cpp` 负责 RV64IM 编码合法性和操作数解码. 执行语义留在 `src/execute.cpp`.
- 区分已解码但未实现执行、未知扩展和非法编码. 新增操作时补充合法编码、保留编码和边界立即数测试.
- `ysyx-workbench` 的 cpu-tests 依赖 AM/NEMU 运行环境, 不要把该工作区路径作为本仓库的构建依赖.
