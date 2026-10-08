-- The Qwen3 inference app: apps/qwen3 cross-compiled to a flat binary in
-- <buildir>/apps/. Built explicitly (xmake build qwen3_app); running it
-- needs a converted model blob, so it is not part of xmake test.

local app_srcdir = path.join(os.projectdir(), "apps", "qwen3")
local buildir = get_config("buildir") or "build"
local app_bindir = path.join(path.is_absolute(buildir) and buildir
                             or path.join(os.projectdir(), buildir), "apps")

target("qwen3_app")
    set_kind("phony")
    set_default(false)
    on_build(function(target)
        import("core.project.depend")
        import("guestimg")
        local gcc, objcopy = guestimg.toolchain()
        os.mkdir(app_bindir)
        local linkscript = path.join(app_srcdir, "link.ld")
        local sources = os.files(path.join(app_srcdir, "*.c"))
        table.join2(sources, os.files(path.join(app_srcdir, "*.S")))
        local elf = path.join(app_bindir, "qwen3.elf")
        local bin = path.join(app_bindir, "qwen3.bin")
        -- Shared NPU bindings live with the guest tests (see xmake/guest.lua).
        local guest_incdir = path.join(os.projectdir(), "tests", "guest",
                                       "include")
        local deps = {linkscript, path.join(app_srcdir, "qwen3.h"),
                      path.join(guest_incdir, "npu.inc")}
        table.join2(deps, sources)
        depend.on_changed(function()
            local args = {"-march=rv64im", "-mabi=lp64", "-O2",
                          "-ffreestanding", "-nostdlib", "-static",
                          "-no-pie", "-I", app_srcdir, "-I", guest_incdir,
                          "-T", linkscript, "-o", elf}
            table.join2(args, sources)
            os.vrunv(gcc, args)
            os.vrunv(objcopy, {"-O", "binary", elf, bin})
        end, {dependfile = target:dependfile("qwen3"),
              files = deps,
              values = {gcc, "-march=rv64im -mabi=lp64 -O2"}})
    end)
target_end()
