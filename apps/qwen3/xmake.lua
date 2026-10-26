-- The Qwen3 inference app: apps/qwen3 cross-compiled to a flat binary in
-- <buildir>/apps/qwen3/. Built explicitly (xmake build qwen3_app); running it
-- needs a converted model blob, so it is not part of xmake test.

local app_srcdir = os.scriptdir()
local project_root = path.directory(path.directory(app_srcdir))
local app_includedir = path.join(app_srcdir, "include")
local buildir = get_config("buildir") or "build"
local base = path.absolute(buildir, os.projectdir())
local app_bindir = path.join(base, "apps", "qwen3")

local function define_qwen3_app(name, base_name, snapshot)
	target(name, function()
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
			local elf = path.join(app_bindir, base_name .. ".elf")
			local bin = path.join(app_bindir, base_name .. ".bin")
			-- Shared NPU bindings live with the guest tests (see xmake/guest.lua).
			local guest_incdir = path.join(project_root, "tests", "guest", "include")
			local deps = { linkscript, path.join(guest_incdir, "npu.inc") }
			table.join2(deps, os.files(path.join(app_includedir, "*.h")))
			table.join2(deps, sources)
			-- lastmtime anchors on the outputs: a missing one forces a rebuild.
			depend.on_changed(function()
				local args = {
					"-march=rv64im",
					"-mabi=lp64",
					"-O2",
					"-ffreestanding",
					"-nostdlib",
					"-static",
					"-no-pie",
					"-I",
					app_includedir,
					"-I",
					guest_incdir,
					"-T",
					linkscript,
					"-o",
					elf,
				}
				if snapshot then
					table.insert(args, "-DQWEN3_SNAPSHOT=1")
				end
				table.join2(args, sources)
				os.vrunv(gcc, args)
				os.vrunv(objcopy, { "-O", "binary", elf, bin })
			end, {
				dependfile = target:dependfile(base_name),
				files = deps,
				changed = target:is_rebuilt(),
				lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
				values = { gcc, "-march=rv64im -mabi=lp64 -O2", snapshot },
			})
		end)
	end)
end

define_qwen3_app("qwen3_app", "qwen3", false)
define_qwen3_app("qwen3_snapshot_app", "qwen3_snapshot", true)

-- Generate apps/qwen3/compile_commands.json for clangd without touching the
-- root compilation database. Run: xmake build qwen3_compile_commands
target("qwen3_compile_commands", function()
	set_kind("phony")
	set_default(false)
	on_build(function(target)
		import("lib.detect.find_program")
		local python = find_program("python3")
		if not python then
			raise("python3 is required to generate the Qwen3 compile database")
		end
		os.vrunv(python, { path.join(app_srcdir, "gen_compile_commands.py") })
	end)
end)

if has_config("monitor") then
	target("qwen3_run", function()
		set_kind("phony")
		set_default(false)
		add_deps("qwen3_app", "sim_run")
		on_run(function(target)
			import("core.base.option")
			import("lib.detect.find_program")
			local args = option.get("arguments") or {}
			if args[1] == "--" then
				table.remove(args, 1)
			end
			local prompt = args[1]
			if not prompt then
				raise('usage: xmake run qwen3_run "prompt" [model]')
			end
			local python = find_program("python3")
			if not python then
				raise("python3 is required to run the Qwen3 app")
			end
			local app = path.join(app_bindir, "qwen3.bin")
			local sim_run = target:dep("sim_run"):targetfile()
			if not os.isfile(app) or not os.isfile(sim_run) then
				raise("run `xmake build qwen3_run` before `xmake run qwen3_run`")
			end
			local model = path.absolute(args[2] or path.join(app_bindir, "model.q3b"), project_root)
			if not os.isfile(model) then
				raise(
					"model blob not found: "
						.. model
						.. "; run `python3 scripts/qwen3/convert.py real external/qwen3 "
						.. model
						.. "` or pass an existing model as the second argument"
				)
			end
			os.execv(python, {
				path.join(project_root, "scripts", "qwen3", "run.py"),
				"--sim-run",
				sim_run,
				"--app",
				app,
				"--model",
				model,
				"--prompt-out",
				path.join(app_bindir, "prompt.bin"),
				"--tokenizer",
				path.join(project_root, "external", "qwen3", "tokenizer.json"),
				"--",
				prompt,
			})
		end)
	end)
end
