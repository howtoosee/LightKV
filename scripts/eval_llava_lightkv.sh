#!/bin/bash
# Evaluate LLaVA (v1.5 / NeXT) with LightKV, using lmms-eval.
#
# Usage:
#   bash scripts/eval_llava_lightkv.sh <CKPT> <MASTER_PORT>
# Example:
#   bash scripts/eval_llava_lightkv.sh liuhaotian/llava-v1.6-vicuna-7b 12345
#   bash scripts/eval_llava_lightkv.sh liuhaotian/llava-v1.5-7b        12345

set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

CKPT_FULL="${1:-liuhaotian/llava-v1.6-vicuna-7b}"
PORT="${2:-12345}"
CKPT="${CKPT_FULL#*/}"

MODEL=llava_lightkv

# --- LightKV schedule (example; see the paper for the best per-model config) ---
# merge_layers  (Lambda): decoder layers at which vision tokens are compressed
# merge_windows (W):       number of windows per side used for message passing
# merge_ratios  (P):       fraction of vision tokens dropped at each stage
# Multiple stages are joined with "-".
MERGE_LAYERS=12-18-24
MERGE_WINDOWS=4-2-1
MERGE_RATIOS=0.5-0.5-0.5

# Paper benchmarks
TASKS="coco2017_cap_val,mme,nocaps_val,pope,seedbench_lite,vizwiz_vqa_val,gqa"

accelerate launch --num_processes=1 --main_process_port "$PORT" -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained=$CKPT_FULL,merge_layers=$MERGE_LAYERS,merge_windows=$MERGE_WINDOWS,merge_ratios=$MERGE_RATIOS \
    --gen_kwargs max_new_tokens=128 \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${MODEL}/${CKPT}__L-${MERGE_LAYERS}__W-${MERGE_WINDOWS}__R-${MERGE_RATIOS}"
