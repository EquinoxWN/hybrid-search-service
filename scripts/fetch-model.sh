#!/usr/bin/env bash
# Downloads sentence-transformers/all-MiniLM-L6-v2 (Apache-2.0) at a pinned revision into
# .tmp/models and checks every file's SHA-256. `make clean-model` deletes it after a run.
# Add `--reference` to also fetch tokenizer.json and the ONNX export used by make reference.
set -euo pipefail
cd "$(dirname "$0")/.."

REV=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
DIR=.tmp/models/all-MiniLM-L6-v2
BASE=https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$REV
FILES=(
  "config.json 953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41"
  "vocab.txt 07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3"
  "model.safetensors 53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
if [ "${1:-}" = "--reference" ]; then
  FILES+=(
    "tokenizer.json be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037"
    "onnx/model.onnx 6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452"
  )
fi
for entry in "${FILES[@]}"; do
  read -r name sha <<<"$entry"
  out=$DIR/$name
  mkdir -p "$(dirname "$out")"
  if [ ! -f "$out" ]; then
    curl -fsSL --proto '=https' --retry 3 -o "$out.part" "$BASE/$name"
    mv "$out.part" "$out"
  fi
  echo "$sha  $out" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $out" >&2; rm -f "$out"; exit 1; }
done
echo "model all-MiniLM-L6-v2@${REV:0:12} verified in $DIR"
