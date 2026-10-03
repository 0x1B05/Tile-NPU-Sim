#!/bin/sh
# Fetches the official riscv-tests suite (pinned commit) into
# external/riscv-tests. The checkout is a build input, not committed source.
set -e

commit=bcffa2b3188b040c611f90dc0b6e422f54775a09
dir="$(cd "$(dirname "$0")/.." && pwd)/external/riscv-tests"

mkdir -p "$dir"
cd "$dir"
if [ ! -d .git ]; then
  git init -q
  git remote add origin https://github.com/riscv-software-src/riscv-tests
fi
git fetch -q --depth 1 origin "$commit"
git checkout -q FETCH_HEAD
echo "riscv-tests @ $commit ready in $dir"
