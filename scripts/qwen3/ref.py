#!/usr/bin/env python3
"""Numpy reference forward for the tiny Qwen3 model (numeric alignment).

Mirrors the guest pipeline's quantized math: per-row int8 weights, dynamic
uniform activation quantization, int64 accumulation, f32 elementwise ops.
Compares against guest output token ids.

Usage: ref.py build/qwen3/tiny_ref.npz [n_tokens]
"""

import sys
from dataclasses import dataclass

import numpy as np

from quant import quantize_rows


@dataclass(frozen=True)
class MatvecResult:
    """Matvec output plus the intermediate values used by alignment probes."""

    value: np.ndarray
    activation_q: np.ndarray
    accumulator: np.ndarray


def matvec_ref(w_f32, x):
    q, s = quantize_rows(w_f32)
    # activation quantization in f32, mirroring the guest kernel:
    # xs = absmax * fl32(1/127), q = RNE(x / xs) saturated to [-127, 127]
    absmax = np.abs(x).max().astype(np.float32)
    inv127 = np.array([0x3C010204], dtype=np.uint32).view(np.float32)[0]
    xs = absmax * inv127
    xq = np.rint((x / xs).astype(np.float32)).clip(-127, 127).astype(np.int64)
    acc = q.astype(np.int64) @ xq
    value = acc.astype(np.float32) * s * xs
    return MatvecResult(value, xq, acc)


def rmsnorm(x: np.ndarray, gamma: np.ndarray, eps: float) -> np.ndarray:
    xf = x.astype(np.float32)
    mean_sq = np.mean(xf * xf, axis=-1, keepdims=True)
    inv = np.float32(1.0) / np.sqrt(mean_sq + np.float32(eps))
    return xf * inv * gamma


def rope(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin])


class Ref:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.d = d
        self.embed = d["embed"]
        self.cos = d["rope_cos"]
        self.sin = d["rope_sin"]
        self.fin = d["fin_norm"]
        self.tokens = d["tokens"]
        # tiny arch constants (mirror scripts/qwen3/convert.py tiny)
        self.n_layers = 2
        self.n_q = 2
        self.n_kv = 1
        self.hd = 32
        self.eps = 1e-6

    def layer(self, i, suffix):
        return self.d[f"l{i:02d}_{suffix}"]

    def forward(self, token, pos, kv_cache, want_logits=True):
        probeme = want_logits and pos == len(self.tokens) - 1

        def probe(tag, v):
            if probeme:
                bits = " ".join(f"{x.view(np.uint32):08x}" for x in v[:4].astype("<f4"))
                vals = " ".join(f"{x:.6f}" for x in v[:4])
                print(f"{tag} {bits}  ({vals})")

        x = self.embed[token].copy()
        probe("embed", x)
        q_dim = self.n_q * self.hd
        for L in range(self.n_layers):
            h = rmsnorm(x, self.layer(L, "inln"), self.eps)
            if L == 0:
                probe("ln0", h)
            q_result = matvec_ref(self.layer(L, "qw"), h)
            q = q_result.value
            if L == 0 and probeme:
                packed = q_result.activation_q.astype("<i1").tobytes()
                words = np.frombuffer(packed, "<u4")
                print("qx0 " + " ".join(f"{w:08x}" for w in words))
                accw = np.frombuffer(
                    q_result.accumulator.astype("<i4").tobytes(), "<u4"
                )
                print("ac0 " + " ".join(f"{w:08x}" for w in accw[:8]))
                probe("q0", q)
            k = matvec_ref(self.layer(L, "kw"), h).value
            v = matvec_ref(self.layer(L, "vw"), h).value
            for hh in range(self.n_q):
                q[hh * self.hd : (hh + 1) * self.hd] = rmsnorm(
                    q[hh * self.hd : (hh + 1) * self.hd], self.layer(L, "qn"), self.eps
                )
            if L == 0:
                probe("q0n", q)
            for hh in range(self.n_kv):
                k[hh * self.hd : (hh + 1) * self.hd] = rmsnorm(
                    k[hh * self.hd : (hh + 1) * self.hd], self.layer(L, "kn"), self.eps
                )
            cos_p, sin_p = self.cos[pos], self.sin[pos]
            for hh in range(self.n_q):
                q[hh * self.hd : (hh + 1) * self.hd] = rope(
                    q[hh * self.hd : (hh + 1) * self.hd], cos_p, sin_p
                )
            if L == 0:
                probe("q0r", q)
                probe("q1r", q[self.hd :])
            for hh in range(self.n_kv):
                k[hh * self.hd : (hh + 1) * self.hd] = rope(
                    k[hh * self.hd : (hh + 1) * self.hd], cos_p, sin_p
                )
            kv_cache[L][0][pos] = k
            kv_cache[L][1][pos] = v
            attn = np.zeros(q_dim, dtype=np.float32)
            group = self.n_q // self.n_kv
            for hh in range(self.n_q):
                kvh = hh // group
                keys = kv_cache[L][0][: pos + 1, kvh * self.hd : (kvh + 1) * self.hd]
                vals = kv_cache[L][1][: pos + 1, kvh * self.hd : (kvh + 1) * self.hd]
                s = keys @ q[hh * self.hd : (hh + 1) * self.hd]
                if L == 0 and probeme:
                    print(
                        f"sc_h{hh} "
                        + " ".join(f"{x.view(np.uint32):08x}" for x in s.astype("<f4"))
                    )
                s = s / np.sqrt(np.float32(self.hd))
                p = np.exp(s - s.max()).astype(np.float32)
                if L == 0 and probeme:
                    rcp = np.float32(1.0) / p.sum()
                    print(
                        f"pb_h{hh} "
                        + " ".join(f"{x.view(np.uint32):08x}" for x in p.astype("<f4"))
                        + f" {rcp.view(np.uint32):08x}"
                    )
                p /= p.sum()
                attn[hh * self.hd : (hh + 1) * self.hd] = p @ vals
            if L == 0:
                probe("attn0", attn)
                probe("attn1", attn[self.hd :])
            o_result = matvec_ref(self.layer(L, "ow"), attn)
            o = o_result.value
            if L == 0 and probeme:
                packed = o_result.activation_q.astype("<i1").tobytes()
                words = np.frombuffer(packed, "<u4")
                print("qxa " + " ".join(f"{w:08x}" for w in words))
                accw = np.frombuffer(
                    o_result.accumulator.astype("<i4").tobytes(), "<u4"
                )
                print("aca " + " ".join(f"{w:08x}" for w in accw[:8]))
                probe("o0", o)
            x = x + o
            if L == 0:
                probe("xr0", x)
            h = rmsnorm(x, self.layer(L, "ptln"), self.eps)
            if L == 0:
                probe("ln1", h)
            gate = matvec_ref(self.layer(L, "gw"), h).value
            up = matvec_ref(self.layer(L, "uw"), h).value
            if L == 0:
                probe("g0", gate)
                probe("u0", up)
            act = (gate / (1.0 + np.exp(-gate.astype(np.float64)))).astype(
                np.float32
            ) * up
            if L == 0:
                probe("m0", act)
            d = matvec_ref(self.layer(L, "dw"), act).value
            if L == 0:
                probe("d0", d)
            x = x + d
            if L == 0:
                probe("x_l0", x)
        if not want_logits:
            return 0, None
        h = rmsnorm(x, self.fin, self.eps)
        probe("lnf", h)
        logits = matvec_ref(self.embed, h).value
        return int(np.argmax(logits)), logits


def main():
    ref = Ref(sys.argv[1])
    n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    max_pos = ref.cos.shape[0]
    kv_cache = [
        [
            np.zeros((max_pos, ref.n_kv * ref.hd), np.float32),
            np.zeros((max_pos, ref.n_kv * ref.hd), np.float32),
        ]
        for _ in range(ref.n_layers)
    ]
    tokens = list(ref.tokens)
    for pos in range(len(tokens) - 1):
        ref.forward(tokens[pos], pos, kv_cache, want_logits=False)
    nxt, logits = ref.forward(tokens[-1], len(tokens) - 1, kv_cache)
    top = np.argsort(logits)[::-1][:5]
    print("ref logits top5:", [(int(i), float(logits[i])) for i in top])
    gen = []
    pos = len(tokens)
    for _ in range(n_gen):
        gen.append(nxt)
        nxt, _ = ref.forward(nxt, pos, kv_cache)
        pos += 1
    print("ref generated:", " ".join(str(t) for t in gen))


if __name__ == "__main__":
    main()
