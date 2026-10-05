#!/bin/sh
# Fetches the Qwen3-0.6B weights and tokenizer into external/qwen3
# (pinned by sha256). The download is a build input, not committed source.
# Override HF_BASE to use another mirror, e.g.
#   HF_BASE=https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main
set -e

base="${HF_BASE:-https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main}"
dir="$(cd "$(dirname "$0")/../.." && pwd)/external/qwen3"

mkdir -p "$dir"
cd "$dir"

fetch() { # name sha256
  if [ -f "$1" ] && echo "$2  $1" | sha256sum -c - >/dev/null 2>&1; then
    echo "$1 already present"
    return
  fi
  echo "downloading $1"
  curl -fL --retry 3 -o "$1" "$base/$1"
  echo "$2  $1" | sha256sum -c -
}

fetch config.json 660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd
fetch tokenizer.json aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4
fetch model.safetensors f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
echo "Qwen3-0.6B ready in $dir"
