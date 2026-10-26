set_project("tile-npu-sim")
set_languages("c++20")
set_warnings("all", "error")
-- Disable implicit FP contraction so that `a * b + c` keeps separate mul/add roundings
-- Explicit std::fma still emits a single-rounding FMA.
add_cxflags("-ffp-contract=off", { tools = { "gcc", "gxx", "clang", "clangxx" } })
set_config("ccachedir", path.join(os.projectdir(), ".cache", "ccache"))
add_rules("mode.debug", "mode.release")
add_moduledirs("xmake/modules")

option("native", function()
	set_default(false)
	set_showmenu(true)
	set_description("Build with -march=native for host-specific ISA extensions")
end)

if has_config("native") then
	add_cxflags("-march=native", { tools = { "gcc", "gxx", "clang", "clangxx" } })
end

option("monitor", function()
	set_default(true)
	set_showmenu(true)
	set_description("Build sim_run, the guest runner and interactive monitor")
end)

if has_config("monitor") then
	add_requires("cli11 2.7.2")
	target("sim_run", function()
		set_kind("binary")
		add_deps("tile_npu_sim")
		add_files("src/main.cpp", "src/monitor/*.cpp")
		add_includedirs("src")
		add_packages("cli11")
	end)
end

option("riscv_cross", function()
	set_default("riscv64-linux-gnu-")
	set_showmenu(true)
	set_description("Prefix of the RISC-V toolchain used to assemble guest test images")
end)

option("rvtests", function()
	set_default(false)
	set_showmenu(true)
	set_description("Run the official riscv-tests suite (needs external/riscv-tests, see scripts/fetch_riscv_tests.sh)")
end)

option("qwen3test", function()
	set_default(false)
	set_showmenu(true)
	set_description("Run the tiny Qwen3 guest/numpy alignment test (needs python3 with numpy)")
end)

target("tile_npu_sim", function()
	set_kind("static")
	add_files("src/*.cpp|main.cpp", "src/decode/*.cpp", "src/execute/*.cpp", "src/platform/*.cpp")
	add_includedirs("include", { public = true })
	add_includedirs("src")
end)

target("sim_tests", function()
	set_kind("binary")
	set_default(false)
	add_deps("tile_npu_sim")
	add_files("tests/*.cpp")
	add_includedirs("src", "tests")
	add_tests("core")
end)

target("sim_guest_tests", function()
	set_kind("binary")
	set_default(false)
	add_deps("tile_npu_sim")
	add_files("tests/main.cpp", "tests/guest/npu_guest_test.cpp")
	add_includedirs("src", "tests")
end)

-- Guest images and the optional conformance suite are separate build-only
-- targets; test execution is owned by tests/run.py.
includes("xmake/guest.lua")
includes("xmake/riscv_tests.lua")
for _, appfile in ipairs(os.files("apps/*/xmake.lua")) do
	includes(path.directory(appfile))
end
