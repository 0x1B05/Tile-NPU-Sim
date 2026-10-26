-- Official riscv-tests image artifacts, opt-in (--rvtests=y, fetched by
-- scripts/fetch_riscv_tests.sh). Each rv64ui/rv64um test is assembled
-- against our minimal env (tests/riscv-tests-env); tests/run.py executes the
-- resulting images. Excluded: fence_i (Zifencei, not modeled) and ma_data
-- (requires trap delivery, which the simulator does not implement).

if has_config("rvtests") then
	local rvtests_srcdir = path.join(os.projectdir(), "external", "riscv-tests")
	local buildir = get_config("buildir") or "build"
	local rvtests_bindir = path.join(
		path.join(path.is_absolute(buildir) and buildir or path.join(os.projectdir(), buildir), "guest"),
		"riscv-tests"
	)

	local rvtests = {}
	if os.isdir(rvtests_srcdir) then
		local excluded = { fence_i = true, ma_data = true }
		for _, group in ipairs({ "rv64ui", "rv64um" }) do
			for _, src in ipairs(os.files(path.join(rvtests_srcdir, "isa", group, "*.S"))) do
				if not excluded[path.basename(src)] then
					table.insert(rvtests, group .. "/" .. path.basename(src))
				end
			end
		end
		table.sort(rvtests)
	end

	target("riscv_tests", function()
		set_kind("phony")
		set_default(false)
		set_values("rvtests_srcdir", rvtests_srcdir)
		set_values("rvtests_bindir", rvtests_bindir)
		set_values("envdir", path.join(os.projectdir(), "tests", "riscv-tests-env"))
		set_values("linkscript", path.join(os.projectdir(), "tests", "guest", "link.ld"))
		set_values("passlist", table.unpack(rvtests))
		-- Validation lives in on_config because raise() is not available in
		-- description scope.
		on_config(function(target)
			if #(target:values("passlist") or {}) == 0 then
				raise(
					"no rv64ui/rv64um sources under external/riscv-tests; " .. "run scripts/fetch_riscv_tests.sh first"
				)
			end
		end)
		on_build(function(target)
			import("core.project.depend")
			import("guestimg")
			local gcc, objcopy = guestimg.toolchain()
			local srcdir = target:values("rvtests_srcdir")
			local bindir = target:values("rvtests_bindir")
			os.mkdir(bindir)
			local macrosdir = path.join(srcdir, "isa", "macros", "scalar")
			local envheader = path.join(target:values("envdir"), "riscv_test.h")
			local linkscript = target:values("linkscript")
			for _, name in ipairs(target:values("passlist")) do
				local src = path.join(srcdir, "isa", name .. ".S")
				local elf, bin = guestimg.output_paths(bindir, name:gsub("/", "-"))
				-- lastmtime anchors on the outputs: a missing one rebuilds.
				depend.on_changed(function()
					guestimg.build_guest_image(
						gcc,
						objcopy,
						linkscript,
						src,
						elf,
						bin,
						{ "-I", target:values("envdir"), "-I", macrosdir }
					)
				end, {
					dependfile = target:dependfile(name:gsub("/", "-")),
					files = {
						src,
						linkscript,
						envheader,
						path.join(macrosdir, "test_macros.h"),
					},
					changed = target:is_rebuilt(),
					lastmtime = math.min(os.mtime(elf), os.mtime(bin)),
					values = { gcc, "-march=rv64im -mabi=lp64" },
				})
			end
		end)
	end)

	-- The images are consumed by tests/run.py; no test target or sim_run
	-- registration is attached here.
end
