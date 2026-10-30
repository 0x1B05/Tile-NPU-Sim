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
	set_category("Build")
	set_description("Build with -march=native for host-specific ISA extensions")
end)

if has_config("native") then
	add_cxflags("-march=native", { tools = { "gcc", "gxx", "clang", "clangxx" } })
end

-- Internal static library shared by the runner and the host test binaries.
target("sim_core", function()
	set_kind("static")
	set_default(false)
	add_files("src/*.cpp|main.cpp", "src/decode/*.cpp|disasm.cpp", "src/execute/*.cpp", "src/platform/*.cpp")
	if has_config("itrace") then
		add_defines("CONFIG_ITRACE", { public = true })
		add_files("src/decode/disasm.cpp")
	end
	if has_config("mtrace") then
		add_defines("CONFIG_MTRACE", { public = true })
		add_files("src/trace/mtrace.cpp")
	end
	add_includedirs("include", { public = true })
	add_includedirs("src")
end)

add_requires("cli11 2.7.2")

-- The default target is the runnable simulator.
target("sim", function()
	set_kind("binary")
	set_default(true)
	add_deps("sim_core")
	add_files("src/main.cpp", "src/monitor/monitor.cpp")
	add_includedirs("src")
	add_packages("cli11")
	if has_config("sdb") then
		add_files("src/monitor/sdb.cpp")
		add_defines("CONFIG_SDB")
	end
	if has_config("mtrace") then
		add_files("src/monitor/mtrace_writer.cpp")
	end
end)

option("sdb", function()
	set_default(false)
	set_showmenu(true)
	set_category("Debug")
	set_description("Enable the interactive sdb debugger in sim")
end)

option("itrace", function()
	set_default(false)
	set_showmenu(true)
	set_category("Debug")
	set_description("Enable instruction tracing with disassembly (--itrace)")
end)

option("mtrace", function()
	set_default(false)
	set_showmenu(true)
	set_category("Debug")
	set_description("Enable NPU data-path tracing (--mtrace)")
end)

target("sim_tests", function()
	set_kind("binary")
	set_default(false)
	add_deps("sim_core")
	add_files("tests/units/*.cpp")
	add_includedirs("src", "tests/include")
	add_tests("core")
end)

target("sim_guest_tests", function()
	set_kind("binary")
	set_default(false)
	add_deps("sim_core")
	add_files("tests/guest/main.cpp", "tests/guest/npu_guest_test.cpp")
	add_includedirs("src", "tests/include")
end)

option("riscv_cross", function()
	set_default("riscv64-linux-gnu-")
	set_showmenu(true)
	set_category("Guests")
	set_description("Prefix of the RISC-V toolchain used to assemble guest test images")
end)

-- Guest images and the optional conformance suite are separate build-only
-- targets; test execution is owned by tests/run.py.
includes("xmake/guest.lua")
includes("xmake/riscv_tests.lua")
for _, appfile in ipairs(os.files("apps/*/xmake.lua")) do
	includes(path.directory(appfile))
end
