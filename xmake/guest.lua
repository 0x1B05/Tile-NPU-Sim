-- Guest test images: in-repo C/.S sources under tests/guest/ cross-compiled
-- to flat binaries in <buildir>/guest/. C programs link the shared crt0;
-- bare .S images provide their own _start. Images are build products of
-- in-repo sources, never committed.

local guest_srcdir = path.join(os.projectdir(), "tests", "guest")
-- Anchor at the project root: path.absolute alone would resolve against the
-- included script's own directory.
local buildir = get_config("buildir") or "build"
local guest_bindir = path.join(path.is_absolute(buildir) and buildir
                               or path.join(os.projectdir(), buildir), "guest")

target("guest_images")
    set_kind("phony")
    set_default(false)
    set_values("guest_srcdir", guest_srcdir)
    set_values("guest_bindir", guest_bindir)
    on_build(function(target)
        import("core.project.depend")
        import("guestimg")
        local gcc, objcopy = guestimg.toolchain()
        local srcdir = target:values("guest_srcdir")
        local bindir = target:values("guest_bindir")
        os.mkdir(bindir)
        local linkscript = path.join(srcdir, "link.ld")
        local crt0 = path.join(srcdir, "crt0.S")
        local npumacros = path.join(srcdir, "npu.inc")
        -- Per-image incremental assembly: rebuild only when the source, the
        -- shared include/linker inputs, or the toolchain change.
        local function assemble(src, extra, deps)
            local elf, bin = guestimg.output_paths(bindir, path.basename(src))
            depend.on_changed(function()
                guestimg.assemble(gcc, objcopy, linkscript, src, elf, bin, extra)
            end, {dependfile = target:dependfile(path.basename(src)),
                  files = deps,
                  values = {gcc, "-march=rv64im -mabi=lp64 -O2"}})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.c"))) do
            assemble(src, {crt0, "-I", srcdir}, {src, linkscript, crt0})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.S"))) do
            if path.filename(src) ~= "crt0.S" then
                assemble(src, {"-I", srcdir}, {src, linkscript, npumacros})
            end
        end
    end)
target_end()

target("sim_tests")
    add_deps("guest_images")
    add_defines("GUEST_IMAGE_DIR=\"" .. guest_bindir .. "\"")
target_end()

if has_config("monitor") then
    target("sim_run")
        -- Self-checking guest images: sim_run -b reports PASS/FAIL per image.
        for _, ext in ipairs({"*.c", "*.S"}) do
            for _, src in ipairs(os.files(path.join(guest_srcdir, "rv64im", ext))) do
                add_tests("guest/" .. path.basename(src),
                          {runargs = {"-b", path.join(guest_bindir, path.basename(src) .. ".bin")}})
            end
        end
    target_end()
end
