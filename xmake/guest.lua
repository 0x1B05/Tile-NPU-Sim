-- Guest image artifacts: in-repo C/.S sources under tests/guest/ cross-compiled
-- to flat binaries in <builddir>/guest/. C programs link the shared crt0;
-- bare .S images provide their own _start. Images are build products of
-- in-repo sources, never committed. tests/run.py consumes the outputs.

local guest_srcdir = path.join(os.projectdir(), "tests", "guest")
-- Anchor a relative --builddir at the project root; keep an absolute one as-is.
local builddir = get_config("builddir") or "build"
local guest_bindir = path.join(path.absolute(builddir, os.projectdir()), "guest")

target("guest_images", function()
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
			-- A fresh build directory may not have the target dependency directory yet.
			local dependfile = target:dependfile(path.basename(src))
			local depend_dir = path.directory(path.normalize(dependfile))
			if not os.isdir(depend_dir) then
				os.mkdir(depend_dir)
			end
			depend.on_changed(function()
				guestimg.build_guest_image(gcc, objcopy, linkscript, src, elf, bin, extra)
			end, {
				dependfile = dependfile,
				files = deps,
				changed = target:is_rebuilt(),
				lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
				values = { gcc, "-march=rv64im -mabi=lp64 -O2" },
			})
		end
		for _, src in ipairs(os.files(path.join(srcdir, "**.c"))) do
			build_guest_image(src, { crt0, "-I", srcdir }, { src, linkscript, crt0 })
		end
		for _, src in ipairs(os.files(path.join(srcdir, "**.S"))) do
			if path.filename(src) ~= "crt0.S" then
				build_guest_image(src, { "-I", incdir }, { src, linkscript, npumacros })
			end
		end
	end)
end)
