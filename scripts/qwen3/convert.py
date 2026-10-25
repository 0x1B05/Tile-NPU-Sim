#!/usr/bin/env python3
"""Convert Qwen3 weights into the flat q8s32 blob consumed by apps/qwen3.

Weight matrices are quantized per output row (symmetric int8, RNE, range
[-127, 127]) with one f32 scale per row; RMSNorm gammas stay f32. The blob
carries a section directory so the guest locates tensors at runtime:

  header: magic "Q3B", arch dims, eps, section count   (all u32 LE)
  directory: n_sections x { name char[8], offset u32, size u32 }
  sections: 16-byte aligned tensor data

Commands:
  convert.py real MODEL_DIR OUT
  convert.py tiny --out OUT --ref REF [--prompt-out PROMPT]

Run `convert.py --help` or `convert.py real --help` for details.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

# These tools are run as scripts, so sibling modules are top-level imports.
from blob import Blob, check_output, write_npz
from prompt import write_prompt
from quant import quantize_rows
from safetensors import SafeTensors

LOGGER = logging.getLogger("qwen3.convert")


# ---- quantization stats ---------------------------------------------------


def quant_error(w, q, s):
    deq = q.astype(np.float32) * s[:, None]
    num = np.abs(w - deq).mean()
    den = np.abs(w).mean()
    return num / den if den > 0 else 0.0


# ---- rope ----------------------------------------------------------------


def rope_tables(max_pos, head_dim, theta):
    j = np.arange(head_dim // 2, dtype=np.float64)
    inv_freq = theta ** (-2.0 * j / head_dim)
    pos = np.arange(max_pos, dtype=np.float64)
    freqs = np.outer(pos, inv_freq)  # [max_pos, hd/2]
    return (np.cos(freqs).astype("<f4"), np.sin(freqs).astype("<f4"))


# ---- model tensors --------------------------------------------------------


LAYER_TENSORS = [
    # (section suffix, hf name template, kind)
    ("inln", "model.layers.{i}.input_layernorm.weight", "f32"),
    ("qw", "model.layers.{i}.self_attn.q_proj.weight", "q8"),
    ("kw", "model.layers.{i}.self_attn.k_proj.weight", "q8"),
    ("vw", "model.layers.{i}.self_attn.v_proj.weight", "q8"),
    ("ow", "model.layers.{i}.self_attn.o_proj.weight", "q8"),
    ("qn", "model.layers.{i}.self_attn.q_norm.weight", "f32"),
    ("kn", "model.layers.{i}.self_attn.k_norm.weight", "f32"),
    ("ptln", "model.layers.{i}.post_attention_layernorm.weight", "f32"),
    ("gw", "model.layers.{i}.mlp.gate_proj.weight", "q8"),
    ("uw", "model.layers.{i}.mlp.up_proj.weight", "q8"),
    ("dw", "model.layers.{i}.mlp.down_proj.weight", "q8"),
]


# ---- model conversion -----------------------------------------------------


def convert_real(model_dir, max_pos):
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if not config_path.is_file():
        raise ValueError(f"model config not found: {config_path}")
    if not weights_path.is_file():
        raise ValueError(f"weights not found: {weights_path}")

    with config_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        arch = {
            "n_layers": cfg["num_hidden_layers"],
            "hidden": cfg["hidden_size"],
            "inter": cfg["intermediate_size"],
            "n_q_heads": cfg["num_attention_heads"],
            "n_kv_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["head_dim"],
            "vocab": cfg["vocab_size"],
            "max_pos": max_pos,
            "eps": cfg["rms_norm_eps"],
        }
        rope_theta = cfg["rope_theta"]
    except KeyError as exc:
        raise ValueError(f"{config_path}: missing key {exc.args[0]!r}") from exc

    st = SafeTensors(weights_path)
    LOGGER.info(
        "model: %d layers, hidden=%d, vocab=%d",
        arch["n_layers"],
        arch["hidden"],
        arch["vocab"],
    )

    blob = Blob(arch)
    cos, sin = rope_tables(max_pos, arch["head_dim"], rope_theta)
    blob.add("rope_cos", cos.tobytes())
    blob.add("rope_sin", sin.tobytes())

    quant_errors = []
    embed = st.get("model.embed_tokens.weight")
    q, s = quantize_rows(embed)
    error = quant_error(embed, q, s)
    quant_errors.append(error)
    LOGGER.debug("embed %s rel-err %.4f", embed.shape, error)
    blob.add("embed", q.tobytes())
    blob.add("embed_s", s.tobytes())
    del embed, q, s

    n_layers = arch["n_layers"]
    for i in range(n_layers):
        LOGGER.info("[%02d/%d] layer %d", i + 1, n_layers, i)
        for suffix, templ, kind in LAYER_TENSORS:
            name = templ.format(i=i)
            w = st.get(name)
            sec = f"l{i:02d}_{suffix}"
            if kind == "f32":
                if w.ndim != 1:
                    raise ValueError(f"{name}: expected a 1-D tensor, got {w.shape}")
                blob.add(sec, w.astype("<f4").tobytes())
            else:
                q, s = quantize_rows(w)
                error = quant_error(w, q, s)
                quant_errors.append(error)
                LOGGER.debug("%s %s rel-err %.4f", name, w.shape, error)
                blob.add(sec, q.tobytes())
                blob.add(sec + "s", s.tobytes())

    norm = st.get("model.norm.weight")
    if norm.ndim != 1:
        raise ValueError(f"model.norm.weight: expected a 1-D tensor, got {norm.shape}")
    blob.add("fin_norm", norm.astype("<f4").tobytes())

    if quant_errors:
        LOGGER.info(
            "quantization: mean rel-err %.4f, max %.4f",
            np.mean(quant_errors),
            np.max(quant_errors),
        )
    return blob


def convert_tiny():
    LOGGER.info("generating tiny model")
    rng = np.random.default_rng(0)
    arch = {
        "n_layers": 2,
        "hidden": 64,
        "inter": 128,
        "n_q_heads": 2,
        "n_kv_heads": 1,
        "head_dim": 32,
        "vocab": 320,
        "max_pos": 64,
        "eps": 1e-6,
    }

    def mat(rows, cols):
        return rng.standard_normal((rows, cols), dtype=np.float32) * 0.02

    def vec(n):
        return 1.0 + 0.1 * rng.standard_normal(n, dtype=np.float32)

    prompt_ids = [1, 5, 9, 42, 100, 7]
    blob = Blob(arch)
    cos, sin = rope_tables(arch["max_pos"], arch["head_dim"], 1000000.0)
    blob.add("rope_cos", cos.tobytes())
    blob.add("rope_sin", sin.tobytes())
    ref = {"rope_cos": cos, "rope_sin": sin}

    def emit(sec, w, kind):
        if kind == "f32":
            LOGGER.debug("%s %s", sec, w.shape)
            blob.add(sec, w.astype("<f4").tobytes())
            ref[sec] = w
        else:
            q, s = quantize_rows(w)
            LOGGER.debug("%s %s rel-err %.4f", sec, w.shape, quant_error(w, q, s))
            blob.add(sec, q.tobytes())
            blob.add("embed_s" if sec == "embed" else sec + "s", s.tobytes())
            # the reference consumes the same dequantized weights as the guest
            ref[sec] = q.astype(np.float32) * s[:, None]

    emit("embed", mat(arch["vocab"], arch["hidden"]), "q8")

    hf_shapes = {
        "qw": (arch["n_q_heads"] * arch["head_dim"], arch["hidden"]),
        "kw": (arch["n_kv_heads"] * arch["head_dim"], arch["hidden"]),
        "vw": (arch["n_kv_heads"] * arch["head_dim"], arch["hidden"]),
        "ow": (arch["hidden"], arch["n_q_heads"] * arch["head_dim"]),
        "gw": (arch["inter"], arch["hidden"]),
        "uw": (arch["inter"], arch["hidden"]),
        "dw": (arch["hidden"], arch["inter"]),
        "inln": (arch["hidden"],),
        "ptln": (arch["hidden"],),
        "qn": (arch["head_dim"],),
        "kn": (arch["head_dim"],),
    }
    for i in range(arch["n_layers"]):
        for suffix, _, kind in LAYER_TENSORS:
            sec = f"l{i:02d}_{suffix}"
            w = vec(hf_shapes[suffix][0]) if kind == "f32" else mat(*hf_shapes[suffix])
            emit(sec, w, kind)
    fin = vec(arch["hidden"])
    blob.add("fin_norm", fin.astype("<f4").tobytes())
    ref["fin_norm"] = fin
    ref["tokens"] = np.asarray(prompt_ids, "<u4")
    return blob, ref


# ---- command line ---------------------------------------------------------


def positive_int(value):
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from exc
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def add_verbosity(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print per-tensor quantization errors",
    )
    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print errors",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3 weights into the flat q8s32 blob consumed by apps/qwen3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "The blob is little-endian: header, section directory, "
            "then 16-byte aligned sections."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    real = subparsers.add_parser(
        "real",
        help="convert Hugging Face safetensors",
        description="Convert a Hugging Face Qwen3 model directory.",
    )
    real.add_argument(
        "model_dir",
        type=Path,
        metavar="MODEL_DIR",
        help="directory containing config.json and model.safetensors",
    )
    real.add_argument("out", type=Path, metavar="OUT", help="output .q3b blob")
    real.add_argument(
        "--max-pos",
        type=positive_int,
        default=512,
        metavar="N",
        help="maximum context positions (default: 512)",
    )
    add_verbosity(real)

    tiny = subparsers.add_parser(
        "tiny",
        help="generate a fixed-seed random tiny model",
        description="Generate a fixed-seed random tiny model and its f32 reference.",
    )
    tiny.add_argument(
        "--out",
        type=Path,
        required=True,
        metavar="OUT",
        help="output .q3b blob",
    )
    tiny.add_argument(
        "--ref",
        type=Path,
        required=True,
        metavar="REF",
        help="output .npz reference",
    )
    tiny.add_argument(
        "--prompt-out",
        type=Path,
        metavar="PROMPT",
        help="optional Q3P prompt image for the fixed tiny prompt",
    )
    add_verbosity(tiny)
    return parser


def run_real(args):
    check_output(args.out)
    blob = convert_real(args.model_dir, args.max_pos)
    size = blob.write(args.out)
    LOGGER.info("blob: %.1f MB -> %s", size / 1e6, args.out)
    return 0


def run_tiny(args):
    check_output(args.out)
    check_output(args.ref)
    if args.out.resolve() == args.ref.resolve():
        raise ValueError("--out and --ref must be different paths")
    blob, ref = convert_tiny()
    size = blob.write(args.out)
    write_npz(args.ref, ref)
    if args.prompt_out is not None:
        if args.prompt_out.resolve() in (args.out.resolve(), args.ref.resolve()):
            raise ValueError("--prompt-out must differ from --out and --ref")
        prompt_size = write_prompt(args.prompt_out, ref["tokens"])
        LOGGER.info("tiny prompt: %d bytes -> %s", prompt_size, args.prompt_out)
    LOGGER.info("tiny blob: %d bytes -> %s, reference -> %s", size, args.out, args.ref)
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(message)s")

    if args.command == "real":
        return run_real(args)
    if args.command == "tiny":
        return run_tiny(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
