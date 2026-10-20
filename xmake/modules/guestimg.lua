-- Shared helpers for assembling RISC-V guest images. Importable from target
-- build scripts as import("guestimg").

import("lib.detect.find_program")

-- Returns gcc, objcopy of the configured cross toolchain, or raises.
function toolchain()
	local cross = get_config("riscv_cross")
	local gcc = find_program(cross .. "gcc")
	local objcopy = find_program(cross .. "objcopy")
	if not gcc or not objcopy then
		raise(
			"RISC-V toolchain '"
				.. cross
				.. "{gcc,objcopy}' not found; "
				.. "install it or point --riscv_cross= at another prefix"
		)
	end
	return gcc, objcopy
end

-- Output pair (elf, bin) for a guest image name under bindir.
function output_paths(bindir, name)
	return path.join(bindir, name .. ".elf"), path.join(bindir, name .. ".bin")
end

-- Compile+link src (with optional extra sources/includes), objcopy to a bin.
function build_guest_image(gcc, objcopy, linkscript, src, elf, bin, extra)
	local args = {
		"-march=rv64im",
		"-mabi=lp64",
		"-O2",
		"-ffreestanding",
		"-nostdlib",
		"-static",
		"-no-pie",
		"-T",
		linkscript,
		"-o",
		elf,
	}
	table.insert(args, src)
	table.join2(args, extra or {})
	os.vrunv(gcc, args)
	os.vrunv(objcopy, { "-O", "binary", elf, bin })
end
