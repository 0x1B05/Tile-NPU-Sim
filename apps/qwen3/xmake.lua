-- The Qwen3 inference app: apps/qwen3 cross-compiled to a flat binary in
-- <buildir>/apps/qwen3/. Built explicitly (xmake build qwen3_app); running it
-- needs a converted model blob, so it is not part of xmake test.

local app_srcdir = os.scriptdir()
local project_root = path.directory(path.directory(app_srcdir))
local app_includedir = path.join(app_srcdir, "include")
local buildir = get_config("buildir") or "build"
local base = path.absolute(buildir, os.projectdir())
local app_bindir = path.join(base, "apps", "qwen3")

target("qwen3_app", function()
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
			if has_config("qwen3probes") == false then
				table.insert(args, "-DQWEN3_PROBES=0")
			end
			table.join2(args, sources)
			os.vrunv(gcc, args)
			os.vrunv(objcopy, { "-O", "binary", elf, bin })
		end, {
			dependfile = target:dependfile("qwen3"),
			files = deps,
			changed = target:is_rebuilt(),
			lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
			values = { gcc, "-march=rv64im -mabi=lp64 -O2", has_config("qwen3probes") ~= false },
		})
	end)
end)

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

-- Optional tiny-model alignment test (--qwen3test=y): convert.py tiny
-- emits a fixed-seed random model, sim_run executes the guest forward on it,
-- and scripts/qwen3/tiny_check.py compares the probe stream and generated
-- token ids against ref.py's numpy reference. Skipped with a warning when
-- python3 with numpy is unavailable.
if has_config("qwen3test") then
	local scripts_dir = path.join(project_root, "scripts", "qwen3")
	target("qwen3_tiny", function()
		set_kind("phony")
		set_default(false)
		add_deps("qwen3_app")
		if has_config("monitor") then
			add_deps("sim_run")
		end
		on_load(function(target)
			if has_config("qwen3probes") == false then
				print("warning: --qwen3test=y needs probes; skipping the tiny Qwen3 test")
				target:set("tests", {})
				target:data_set("skip", true)
				return
			end
			import("lib.detect.find_program")
			local python = find_program("python3")
			local log = os.tmpfile()
			local numpy_ok = python
				and os.execv(python, { "-c", "import numpy" }, { try = true, stdout = log, stderr = log }) == 0
			os.rm(log)
			if not numpy_ok then
				print("warning: --qwen3test=y needs python3 with numpy; " .. "skipping the tiny Qwen3 test")
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
			local convert = path.join(scripts_dir, "convert.py")
			local blob = path.join(app_bindir, "tiny.q3b")
			local prompt = path.join(app_bindir, "tiny_prompt.bin")
			local ref = path.join(app_bindir, "tiny_ref.npz")
			depend.on_changed(function()
				os.vrunv(target:data("python"), {
					convert,
					"tiny",
					"--out",
					blob,
					"--prompt-out",
					prompt,
					"--ref",
					ref,
				})
			end, {
				dependfile = target:dependfile("tiny"),
				files = os.files(path.join(scripts_dir, "*.py")),
				changed = target:is_rebuilt(),
				lastmtime = math.min(os.mtime(blob), os.mtime(prompt), os.mtime(ref)),
				values = { target:data("python") },
			})
		end)
		add_tests("forward")
		on_test(function(target)
			local sim_run = target:dep("sim_run"):targetfile()
			return os.execv(target:data("python"), {
				path.join(scripts_dir, "tiny_check.py"),
				sim_run,
				path.join(app_bindir, "qwen3.bin"),
				path.join(app_bindir, "tiny.q3b"),
				path.join(app_bindir, "tiny_prompt.bin"),
				path.join(app_bindir, "tiny_ref.npz"),
			}, { try = true }) == 0
		end)
	end)
end

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
