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
        -- lastmtime anchors on the outputs: a missing one forces a rebuild.
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
              changed = target:is_rebuilt(),
              lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
              values = {gcc, "-march=rv64im -mabi=lp64 -O2"}})
    end)
target_end()

-- Optional tiny-model alignment test (--qwen3test=y): convert.py --tiny
-- emits a fixed-seed random model, sim_run executes the guest forward on it,
-- and scripts/qwen3/tiny_check.py compares the probe stream and generated
-- token ids against ref.py's numpy reference. Skipped with a warning when
-- python3 with numpy is unavailable.
if has_config("qwen3test") then
    local scripts_dir = path.join(os.projectdir(), "scripts", "qwen3")
    target("qwen3_tiny")
        set_kind("phony")
        set_default(false)
        add_deps("qwen3_app")
        if has_config("monitor") then
            add_deps("sim_run")
        end
        on_load(function(target)
            import("lib.detect.find_program")
            local python = find_program("python3")
            local log = os.tmpfile()
            local numpy_ok = python and
                os.execv(python, {"-c", "import numpy"},
                         {try = true, stdout = log, stderr = log}) == 0
            os.rm(log)
            if not numpy_ok then
                print("warning: --qwen3test=y needs python3 with numpy; " ..
                      "skipping the tiny Qwen3 test")
                target:set("tests", {})
                target:data_set("skip", true)
                return
            end
            target:data_set("python", python)
        end)
        on_config(function(target)
            -- The test runs through sim_run; without the runner it would be
            -- silently skipped, so reject the combination (as rvtests does).
            if not has_config("monitor") then
                raise("--qwen3test=y requires the runner; configure with --monitor=y")
            end
        end)
        on_build(function(target)
            if target:data("skip") then
                return
            end
            import("core.project.depend")
            os.mkdir(app_bindir)
            local convert = path.join(scripts_dir, "convert.py")
            local blob = path.join(app_bindir, "tiny.q3b")
            local ref = path.join(app_bindir, "tiny_ref.npz")
            depend.on_changed(function()
                os.vrunv(target:data("python"), {convert, "--tiny", blob, ref})
            end, {dependfile = target:dependfile("tiny"),
                  files = {convert},
                  changed = target:is_rebuilt(),
                  lastmtime = math.min(os.mtime(blob), os.mtime(ref)),
                  values = {target:data("python")}})
        end)
        add_tests("forward")
        on_test(function(target)
            local sim_run = target:dep("sim_run"):targetfile()
            return os.execv(target:data("python"),
                            {path.join(scripts_dir, "tiny_check.py"), sim_run,
                             path.join(app_bindir, "qwen3.bin"),
                             path.join(app_bindir, "tiny.q3b"),
                             path.join(app_bindir, "tiny_ref.npz")},
                            {try = true}) == 0
        end)
    target_end()
end
