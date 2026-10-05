#!/usr/bin/env python3
"""Convert Qwen3 weights into the flat q8s32 blob consumed by apps/qwen3.

Weight matrices are quantized per output row (symmetric int8, RNE, range
[-127, 127]) with one f32 scale per row; RMSNorm gammas stay f32. The blob
carries a section directory so the guest locates tensors at runtime:

  header: magic "Q3B1", arch dims, eps, section count   (all u32 LE)
  directory: n_sections x { name char[8], offset u32, size u32 }
  sections: 16-byte aligned tensor data

Modes:
  qwen3_convert.py MODEL_DIR out.q3b --ids "1,2,3"      real weights
  qwen3_convert.py MODEL_DIR out.q3b --prompt "..." [--chat]
  qwen3_convert.py --tiny out.q3b out_ref.npz           random tiny model
                                                      (+ f32 reference dump)
"""

import json
import struct
import sys

import numpy as np

MAGIC = b"Q3B1"


# ---- safetensors ------------------------------------------------------


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(
        "<f4")


class SafeTensors:
    """Lazily materializes tensors; peak memory stays near one tensor."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
            self.data_start = 8 + n

    def get(self, name):
        info = self.header[name]
        begin, end = info["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + begin)
            raw = f.read(end - begin)
        if info["dtype"] != "BF16":
            raise ValueError(f"{name}: unexpected dtype {info['dtype']}")
        return bf16_to_f32(raw).reshape(info["shape"])


# ---- quantization -------------------------------------------------------


def quantize_rows(w):
    """Per output row symmetric int8; returns (q i8 [out,in], s f32 [out])."""
    absmax = np.abs(w).max(axis=1)
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype("<f4")
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return q, scale


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


# ---- blob writer ----------------------------------------------------------


class Blob:
    def __init__(self, arch, n_prompt):
        self.arch = arch  # dict of u32 arch fields
        self.n_prompt = n_prompt
        self.sections = []  # (name, bytes)

    def add(self, name, data):
        assert len(name) <= 8, name
        self.sections.append((name, data))

    def write(self, path):
        header_size = 12 * 4
        directory_size = len(self.sections) * 16
        offset = header_size + directory_size
        entries = []
        payload = bytearray()
        for name, data in self.sections:
            pad = (-offset) % 16
            payload += b"\0" * pad
            offset += pad
            entries.append((name, offset, len(data)))
            payload += data
            offset += len(data)
        a = self.arch
        header = struct.pack(
            "<4s11I", MAGIC, a["n_layers"], a["hidden"], a["inter"],
            a["n_q_heads"], a["n_kv_heads"], a["head_dim"], a["vocab"],
            a["max_pos"], self.n_prompt,
            np.frombuffer(np.float32(a["eps"]).tobytes(), "<u4")[0],
            len(self.sections))
        with open(path, "wb") as f:
            f.write(header)
            for name, off, size in entries:
                f.write(struct.pack("<8sII", name.encode(), off, size))
            f.write(payload)
        return offset


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


def convert_real(model_dir, prompt_ids, max_pos):
    cfg = json.load(open(f"{model_dir}/config.json"))
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
    st = SafeTensors(f"{model_dir}/model.safetensors")
    blob = Blob(arch, len(prompt_ids))
    blob.add("tokens", np.asarray(prompt_ids, "<u4").tobytes())
    cos, sin = rope_tables(max_pos, arch["head_dim"], cfg["rope_theta"])
    blob.add("rope_cos", cos.tobytes())
    blob.add("rope_sin", sin.tobytes())

    embed = st.get("model.embed_tokens.weight")
    q, s = quantize_rows(embed)
    print(f"embed {embed.shape} rel-err {quant_error(embed, q, s):.4f}")
    blob.add("embed", q.tobytes())
    blob.add("embed_s", s.tobytes())
    del embed, q, s

    for i in range(arch["n_layers"]):
        for suffix, templ, kind in LAYER_TENSORS:
            name = templ.format(i=i)
            w = st.get(name)
            sec = f"l{i:02d}_{suffix}"
            if kind == "f32":
                assert w.ndim == 1, name
                blob.add(sec, w.astype("<f4").tobytes())
            else:
                q, s = quantize_rows(w)
                if i == 0:
                    print(f"{name} {w.shape} rel-err "
                          f"{quant_error(w, q, s):.4f}")
                blob.add(sec, q.tobytes())
                blob.add(sec + "s", s.tobytes())
    norm = st.get("model.norm.weight")
    blob.add("fin_norm", norm.astype("<f4").tobytes())
    return blob


def convert_tiny(ref_path):
    rng = np.random.default_rng(0)
    arch = {"n_layers": 2, "hidden": 64, "inter": 128, "n_q_heads": 4,
            "n_kv_heads": 2, "head_dim": 16, "vocab": 320, "max_pos": 64,
            "eps": 1e-6}

    def mat(rows, cols):
        return rng.standard_normal((rows, cols), dtype=np.float32) * 0.02

    def vec(n):
        return (1.0 + 0.1 * rng.standard_normal(n, dtype=np.float32))

    prompt_ids = [1, 5, 9, 42, 100, 7]
    blob = Blob(arch, len(prompt_ids))
    blob.add("tokens", np.asarray(prompt_ids, "<u4").tobytes())
    cos, sin = rope_tables(arch["max_pos"], arch["head_dim"], 1000000.0)
    blob.add("rope_cos", cos.tobytes())
    blob.add("rope_sin", sin.tobytes())
    ref = {"rope_cos": cos, "rope_sin": sin}

    def emit(sec, w, kind):
        if kind == "f32":
            blob.add(sec, w.astype("<f4").tobytes())
            ref[sec] = w
        else:
            q, s = quantize_rows(w)
            blob.add(sec, q.tobytes())
            blob.add(sec + "s", s.tobytes())
            ref[sec] = w  # f32 golden weights for the numpy reference

    emit("embed", mat(arch["vocab"], arch["hidden"]), "q8")

    hf_shapes = {
        "qw": (arch["n_q_heads"] * arch["head_dim"], arch["hidden"]),
        "kw": (arch["n_kv_heads"] * arch["head_dim"], arch["hidden"]),
        "vw": (arch["n_kv_heads"] * arch["head_dim"], arch["hidden"]),
        "ow": (arch["hidden"], arch["n_q_heads"] * arch["head_dim"]),
        "gw": (arch["inter"], arch["hidden"]),
        "uw": (arch["inter"], arch["hidden"]),
        "dw": (arch["hidden"], arch["inter"]),
        "inln": (arch["hidden"],), "ptln": (arch["hidden"],),
        "qn": (arch["head_dim"],), "kn": (arch["head_dim"],),
    }
    for i in range(arch["n_layers"]):
        for suffix, _, kind in LAYER_TENSORS:
            sec = f"l{i:02d}_{suffix}"
            w = vec(hf_shapes[suffix][0]) if kind == "f32" else mat(
                *hf_shapes[suffix])
            emit(sec, w, kind)
    fin = vec(arch["hidden"])
    blob.add("fin_norm", fin.astype("<f4").tobytes())
    ref["fin_norm"] = fin
    ref["tokens"] = np.asarray(prompt_ids, "<u4")
    np.savez(ref_path, **ref)
    return blob


def main():
    args = sys.argv[1:]
    max_pos = 512
    if "--max-pos" in args:
        k = args.index("--max-pos")
        max_pos = int(args[k + 1])
        del args[k:k + 2]
    if args and args[0] == "--tiny":
        blob = convert_tiny(args[2])
        size = blob.write(args[1])
        print(f"tiny blob: {size} bytes -> {args[1]}, reference -> {args[2]}")
        return
    if len(args) < 2:
        print(__doc__)
        return
    model_dir, out = args[0], args[1]
    chat = "--chat" in args
    if chat:
        args.remove("--chat")
    if "--prompt" in args:
        k = args.index("--prompt")
        sys.path.insert(0, "scripts")
        from qwen3_tokenize import CHAT_WRAP, Tokenizer
        text = args[k + 1]
        ids = Tokenizer().encode(CHAT_WRAP.format(text) if chat else text)
    elif "--ids" in args:
        k = args.index("--ids")
        ids = [int(x) for x in args[k + 1].split(",") if x]
    else:
        print("error: --prompt or --ids required")
        return
    print(f"prompt: {len(ids)} tokens")
    blob = convert_real(model_dir, ids, max_pos)
    size = blob.write(out)
    print(f"blob: {size / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()
