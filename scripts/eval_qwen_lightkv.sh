#!/bin/bash
# Evaluate Qwen2.5-VL with LightKV, using lmms-eval.
#
# Usage:
#   bash scripts/eval_qwen_lightkv.sh <MODEL_KEY> <CKPT> <MASTER_PORT>
# Example:
#   bash scripts/eval_qwen_lightkv.sh qwen2_5_vl_lightkv Qwen/Qwen2.5-VL-7B-Instruct 12345

set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

MODEL="${1:-qwen2_5_vl_lightkv}"
CKPT_FULL="${2:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${3:-12345}"
CKPT="${CKPT_FULL#*/}"

# --- LightKV schedule (example; see the paper for the best per-model config) ---
MERGE_LAYERS=12-17-22
MERGE_WINDOWS=6-4-2
MERGE_RATIOS=0.5-0.5-0.5

# Paper benchmarks
TASKS="coco2017_cap_val,gqa,mme,nocaps_val,pope,seedbench_lite,vizwiz_vqa_val"

accelerate launch --num_processes=1 --main_process_port "$PORT" -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained=$CKPT_FULL,merge_layers=$MERGE_LAYERS,merge_windows=$MERGE_WINDOWS,merge_ratios=$MERGE_RATIOS \
    --gen_kwargs max_new_tokens=128 \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${MODEL}/${CKPT}__L-${MERGE_LAYERS}__W-${MERGE_WINDOWS}__R-${MERGE_RATIOS}"
