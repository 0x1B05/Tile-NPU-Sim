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

local guest_srcdir = path.join(os.projectdir(), "tests", "guest")
local guest_bindir = path.join(path.absolute(get_config("buildir") or "build"), "guest")

-- Assemble every tests/guest/**.S into a flat binary under <buildir>/guest/.
-- Images are build products of in-repo assembly sources, never committed.
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
        -- C programs link the shared crt0; bare .S images provide their own _start.
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
end
