set_project("tile-npu-sim")
set_languages("c++20")
set_warnings("all", "error")
-- The ISA fixes the rounding of each operation, so the host build forbids
-- implicit FP contraction (e.g. sumsq's mul+add must stay two roundings).
-- The fmadd instruction keeps its single rounding via explicit std::fma.
add_cxflags("-ffp-contract=off", {tools = {"gcc", "gxx", "clang", "clangxx"}})
set_config("ccachedir", path.join(os.projectdir(), ".cache", "ccache"))
add_rules("mode.debug", "mode.release")
add_moduledirs("xmake/modules")

option("monitor")
    set_default(true)
    set_showmenu(true)
    set_description("Build sim_run, the guest runner and interactive monitor")
option_end()

option("riscv_cross")
    set_default("riscv64-linux-gnu-")
    set_showmenu(true)
    set_description("Prefix of the RISC-V toolchain used to assemble guest test images")
option_end()

option("rvtests")
    set_default(false)
    set_showmenu(true)
    set_description("Run the official riscv-tests suite (needs external/riscv-tests, see scripts/fetch_riscv_tests.sh)")
option_end()

option("qwen3test")
    set_default(false)
    set_showmenu(true)
    set_description("Run the tiny Qwen3 guest/numpy alignment test (needs python3 with numpy)")
option_end()

target("tile_npu_sim")
    set_kind("static")
    add_files("src/*.cpp|main.cpp", "src/decode/*.cpp", "src/execute/*.cpp", "src/platform/*.cpp")
    add_includedirs("include", {public = true})
    add_includedirs("src")
target_end()

target("sim_tests")
    set_kind("binary")
    set_default(false)
    add_deps("tile_npu_sim")
    add_files("tests/*.cpp")
    add_includedirs("src", "tests")
    add_tests("core")
target_end()

if has_config("monitor") then
    add_requires("cli11 2.7.2")
    target("sim_run")
        set_kind("binary")
        add_deps("tile_npu_sim")
        add_files("src/main.cpp", "src/monitor/*.cpp")
        add_includedirs("src")
        add_packages("cli11")
    target_end()
end

-- Guest images and the optional conformance suite attach themselves to the
-- sim_tests / sim_run targets defined above.
includes("xmake/guest.lua")
includes("xmake/riscv_tests.lua")
includes("xmake/apps.lua")
