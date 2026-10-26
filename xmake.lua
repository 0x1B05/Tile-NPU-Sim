set_project("tile-npu-sim")
set_languages("c++20")
set_warnings("all", "error")
-- Disable implicit FP contraction so that `a * b + c` keeps separate mul/add roundings
-- Explicit std::fma still emits a single-rounding FMA.
add_cxflags("-ffp-contract=off", { tools = { "gcc", "gxx", "clang", "clangxx" } })
set_config("ccachedir", path.join(os.projectdir(), ".cache", "ccache"))
add_rules("mode.debug", "mode.release")
add_moduledirs("xmake/modules")

option("monitor", function()
	set_default(true)
	set_showmenu(true)
	set_description("Build sim_run, the guest runner and interactive monitor")
end)

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

option("native", function()
	set_default(false)
	set_showmenu(true)
	set_description("Build with -march=native for host-specific ISA extensions")
end)

option("qwen3probes", function()
	set_default(true)
	set_showmenu(true)
	set_description("Compile Qwen3 numerical probes (disable for benchmarking)")
end)

if has_config("native") then
	add_cxflags("-march=native", { tools = { "gcc", "gxx", "clang", "clangxx" } })
end

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

-- Guest images and the optional conformance suite attach themselves to the
-- sim_tests / sim_run targets defined above.
includes("xmake/guest.lua")
includes("xmake/riscv_tests.lua")
for _, appfile in ipairs(os.files("apps/*/xmake.lua")) do
	includes(path.directory(appfile))
end
