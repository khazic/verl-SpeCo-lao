#!/bin/bash
# Train DFlash and DFlash2 on identical real-data feature stores and evaluate
# both on the disjoint held-out store.
source /llm-align/liuchonghan/speco_env.sh
EXP=/llm-align/liuchonghan/speco_exp
cd "$EXP"
export PYTHONPATH="$EXP"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export PYTHONUNBUFFERED=1

RUN=/llm-align/liuchonghan/runs/dflash2_ab
git -C "$EXP" log -1 --oneline

python _exp/dflash2_ab.py \
  --target /llm-align/liuchonghan/Qwen3-8B \
  --train-store "$RUN/features_train" \
  --heldout-store "$RUN/features_heldout" \
  --steps "${STEPS:-1600}" \
  --eval-every "${EVAL_EVERY:-200}" \
  --eval-samples 128 \
  --out "$RUN/ab_report.json"
echo "AB_EXIT=$?"
