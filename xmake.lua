set_project("tile-npu-sim")
set_languages("c++20")
set_config("ccachedir", path.join(os.projectdir(), ".cache", "ccache"))
add_rules("mode.debug", "mode.release")

option("monitor")
    set_default(true)
    set_showmenu(true)
    set_description("Build sim_run, the interactive monitor and batch runner")

option("riscv_cross")
    set_default("riscv64-linux-gnu-")
    set_showmenu(true)
    set_description("Prefix of the RISC-V toolchain used to assemble guest test images")

option("rvtests")
    set_default(false)
    set_showmenu(true)
    set_description("Run the official riscv-tests suite (needs external/riscv-tests, see scripts/fetch_riscv_tests.sh)")

local guest_srcdir = path.join(os.projectdir(), "tests", "guest")
local guest_bindir = path.join(path.absolute(get_config("buildir") or "build"), "guest")

-- Assemble every tests/guest/**.{c,S} into a flat binary under
-- <buildir>/guest/. Images are build products of in-repo sources, never
-- committed. C programs link the shared crt0; bare .S images provide their
-- own _start.
target("guest_images")
    set_kind("phony")
    set_default(false)
    set_values("guest_srcdir", guest_srcdir)
    set_values("guest_bindir", guest_bindir)
    on_build(function(target)
        import("lib.detect.find_program")
        local cross = get_config("riscv_cross")
        local gcc = find_program(cross .. "gcc")
        local objcopy = find_program(cross .. "objcopy")
        if not gcc or not objcopy then
            raise("RISC-V toolchain '" .. cross .. "{gcc,objcopy}' not found; " ..
                  "install it or point --riscv_cross= at another prefix")
        end
        local srcdir = target:values("guest_srcdir")
        local bindir = target:values("guest_bindir")
        os.mkdir(bindir)
        local linkscript = path.join(srcdir, "link.ld")
        local crt0 = path.join(srcdir, "crt0.S")
        local function assemble(src, extra)
            local elf = path.join(bindir, path.basename(src) .. ".elf")
            local bin = path.join(bindir, path.basename(src) .. ".bin")
            local args = {"-march=rv64im", "-mabi=lp64", "-O2", "-ffreestanding",
                          "-nostdlib", "-static", "-no-pie", "-I", srcdir,
                          "-T", linkscript, "-o", elf}
            table.insert(args, src)
            table.join2(args, extra)
            os.vrunv(gcc, args)
            os.vrunv(objcopy, {"-O", "binary", elf, bin})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.c"))) do
            assemble(src, {crt0})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.S"))) do
            if path.filename(src) ~= "crt0.S" then
                assemble(src, {})
            end
        end
    end)

target("tile_npu_sim")
    set_kind("static")
    add_files("src/*.cpp", "src/decode/*.cpp", "src/execute/*.cpp")
    add_includedirs("include", {public = true})

target("sim_tests")
    set_kind("binary")
    set_default(false)
    add_deps("tile_npu_sim", "guest_images")
    add_files("tests/*.cpp")
    add_defines("GUEST_IMAGE_DIR=\"" .. guest_bindir .. "\"")
    add_tests("core")

-- Official riscv-tests suite, opt-in. Each rv64ui/rv64um test is assembled
-- against our minimal env (tests/riscv-tests-env) and reported as its own
-- sim_run batch test. Excluded: fence_i (Zifencei, not modeled) and ma_data
-- (requires trap delivery, which the simulator does not implement).
local rvtests = {}
if has_config("rvtests") then
    local rvtests_srcdir = path.join(os.projectdir(), "external", "riscv-tests")
    if not os.isdir(rvtests_srcdir) then
        raise("external/riscv-tests not found; run scripts/fetch_riscv_tests.sh first")
    end
    local excluded = {fence_i = true, ma_data = true}
    for _, group in ipairs({"rv64ui", "rv64um"}) do
        for _, src in ipairs(os.files(path.join(rvtests_srcdir, "isa", group, "*.S"))) do
            if not excluded[path.basename(src)] then
                table.insert(rvtests, group .. "/" .. path.basename(src))
            end
        end
    end
    table.sort(rvtests)

    target("riscv_tests")
        set_kind("phony")
        set_default(false)
        set_values("rvtests_srcdir", rvtests_srcdir)
        set_values("rvtests_bindir", path.join(guest_bindir, "riscv-tests"))
        set_values("envdir", path.join(os.projectdir(), "tests", "riscv-tests-env"))
        set_values("linkscript", path.join(guest_srcdir, "link.ld"))
        set_values("passlist", table.unpack(rvtests))
        on_build(function(target)
            import("lib.detect.find_program")
            local cross = get_config("riscv_cross")
            local gcc = find_program(cross .. "gcc")
            local objcopy = find_program(cross .. "objcopy")
            if not gcc or not objcopy then
                raise("RISC-V toolchain '" .. cross .. "{gcc,objcopy}' not found")
            end
            local srcdir = target:values("rvtests_srcdir")
            local bindir = target:values("rvtests_bindir")
            os.mkdir(bindir)
            for _, name in ipairs(target:values("passlist")) do
                local src = path.join(srcdir, "isa", name .. ".S")
                local elf = path.join(bindir, name:gsub("/", "-") .. ".elf")
                local bin = path.join(bindir, name:gsub("/", "-") .. ".bin")
                os.vrunv(gcc, {"-march=rv64im", "-mabi=lp64", "-nostdlib",
                               "-static", "-no-pie",
                               "-I", target:values("envdir"),
                               "-I", path.join(srcdir, "isa", "macros", "scalar"),
                               "-T", target:values("linkscript"), "-o", elf, src})
                os.vrunv(objcopy, {"-O", "binary", elf, bin})
            end
        end)
end

if has_config("monitor") then
    target("sim_run")
        set_kind("binary")
        add_deps("tile_npu_sim", "guest_images")
        add_files("src/monitor/*.cpp")
        -- Self-checking guest images: sim_run -b reports PASS/FAIL per image.
        for _, ext in ipairs({"*.c", "*.S"}) do
            for _, src in ipairs(os.files(path.join(guest_srcdir, "rv64im", ext))) do
                add_tests("guest/" .. path.basename(src),
                          {runargs = {"-b", path.join(guest_bindir, path.basename(src) .. ".bin")}})
            end
        end
        if has_config("rvtests") then
            add_deps("riscv_tests")
            for _, name in ipairs(rvtests) do
                add_tests("riscv-tests/" .. name:gsub("/", "-"),
                          {runargs = {"-b", path.join(guest_bindir, "riscv-tests", name:gsub("/", "-") .. ".bin")}})
            end
        end
end
