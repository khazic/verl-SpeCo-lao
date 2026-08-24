#!/bin/bash
# Build the real-data train/held-out feature stores from the Qwen3-8B-regenerated
# PerfectBlend conversations, using Qwen3-8B as the target (on-policy).
source /llm-align/liuchonghan/speco_env.sh
EXP=/llm-align/liuchonghan/speco_exp
cd "$EXP"
export PYTHONPATH="$EXP"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export PYTHONUNBUFFERED=1

DATA=/llm-align/liuchonghan/datasets/perfectblend-qwen3-8b-regen/data
RUN=/llm-align/liuchonghan/runs/dflash2_ab
mkdir -p "$RUN"

python _exp/build_real_feature_store.py \
  --target /llm-align/liuchonghan/Qwen3-8B \
  --parquet "$DATA/train-00000-of-00041.parquet" \
  --out-train "$RUN/features_train" \
  --out-heldout "$RUN/features_heldout" \
  --num-train 768 \
  --num-heldout 256 \
  --max-len 384
echo "FEATURES_EXIT=$?"
