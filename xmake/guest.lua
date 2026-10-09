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
        -- Shared NPU bindings (encoding macros + config constants), also
        -- used by apps/qwen3.
        local incdir = path.join(srcdir, "include")
        local npumacros = path.join(incdir, "npu.inc")
        -- Per-image incremental assembly: rebuild only when the source, the
        -- shared include/linker inputs, or the toolchain change. lastmtime
        -- anchors on the outputs: a missing one (mtime 0) forces a rebuild.
        local function build_guest_image(src, extra, deps)
            local elf, bin = guestimg.output_paths(bindir, path.basename(src))
            depend.on_changed(function()
                guestimg.build_guest_image(gcc, objcopy, linkscript, src, elf, bin, extra)
            end, {dependfile = target:dependfile(path.basename(src)),
                  files = deps,
                  changed = target:is_rebuilt(),
                  lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
                  values = {gcc, "-march=rv64im -mabi=lp64 -O2"}})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.c"))) do
            build_guest_image(src, {crt0, "-I", srcdir}, {src, linkscript, crt0})
        end
        for _, src in ipairs(os.files(path.join(srcdir, "**.S"))) do
            if path.filename(src) ~= "crt0.S" then
                build_guest_image(src, {"-I", incdir}, {src, linkscript, npumacros})
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
        -- Each self-checking guest image runs as a separate xmake test.
        for _, ext in ipairs({"*.c", "*.S"}) do
            for _, src in ipairs(os.files(path.join(guest_srcdir, "rv64im", ext))) do
                add_tests("guest/" .. path.basename(src),
                          {runargs = {path.join(guest_bindir, path.basename(src) .. ".bin")}})
            end
        end
    target_end()
end
