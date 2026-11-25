#!/usr/bin/env bash
# 下载 HotpotQA distractor validation 与模型权重（经 hf-mirror 镜像）
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MIRROR="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$ROOT/data" "$ROOT/models"

echo "[1/2] 下载 HotpotQA distractor validation..."
curl -sL --retry 3 --max-time 900 -o "$ROOT/data/hotpot_qa_distractor_validation.parquet" \
  "$MIRROR/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet"
ls -la "$ROOT/data/hotpot_qa_distractor_validation.parquet"

echo "[2/2] 下载 Qwen2.5-Coder-3B-Instruct 权重..."
MODEL_DIR="$ROOT/models/Qwen2.5-Coder-3B-Instruct"
mkdir -p "$MODEL_DIR"
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json \
         vocab.json model.safetensors.index.json \
         model-00001-of-00002.safetensors model-00002-of-00002.safetensors; do
  echo "  - $f"
  curl -sL --retry 3 --max-time 3000 -o "$MODEL_DIR/$f" \
    "$MIRROR/Qwen/Qwen2.5-Coder-3B-Instruct/resolve/main/$f"
done
ls -la "$MODEL_DIR"
echo "完成。"
