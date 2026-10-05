set_project("tile-npu-sim")
set_languages("c++20")
set_config("ccachedir", path.join(os.projectdir(), ".cache", "ccache"))
add_rules("mode.debug", "mode.release")
add_moduledirs("xmake/modules")

option("monitor")
    set_default(true)
    set_showmenu(true)
    set_description("Build sim_run, the interactive monitor and batch runner")
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

target("tile_npu_sim")
    set_kind("static")
    add_files("src/*.cpp", "src/decode/*.cpp", "src/execute/*.cpp")
    add_includedirs("include", {public = true})
    -- Internal helpers (src/utils) are private to the implementation tree;
    -- tests and the monitor opt into the same root below.
    add_includedirs("src")
target_end()

target("sim_tests")
    set_kind("binary")
    set_default(false)
    add_deps("tile_npu_sim")
    add_files("tests/*.cpp")
    add_includedirs("src")
    add_tests("core")
target_end()

if has_config("monitor") then
    target("sim_run")
        set_kind("binary")
        add_deps("tile_npu_sim")
        add_files("src/monitor/*.cpp")
        add_includedirs("src")
    target_end()
end

-- Guest images and the optional conformance suite attach themselves to the
-- sim_tests / sim_run targets defined above.
includes("xmake/guest.lua")
includes("xmake/riscv_tests.lua")
