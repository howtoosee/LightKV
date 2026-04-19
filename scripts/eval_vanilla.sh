#!/bin/bash
# Evaluate the uncompressed (Vanilla) baseline — no LightKV — for reference.
#
# Usage:
#   bash scripts/eval_vanilla.sh <MODEL_KEY> <CKPT> <MASTER_PORT>
# Examples:
#   bash scripts/eval_vanilla.sh llava        liuhaotian/llava-v1.6-vicuna-7b  12345
#   bash scripts/eval_vanilla.sh qwen2_5_vl   Qwen/Qwen2.5-VL-7B-Instruct      12345

set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_NET_GDR_LEVEL=0

MODEL="${1:-llava}"
CKPT_FULL="${2:-liuhaotian/llava-v1.6-vicuna-7b}"
PORT="${3:-12345}"
CKPT="${CKPT_FULL#*/}"

TASKS="coco2017_cap_val,gqa,mme,nocaps_val,pope,seedbench_lite,vizwiz_vqa_val"

accelerate launch --num_processes=1 --main_process_port "$PORT" -m lmms_eval \
    --model "$MODEL" \
    --model_args pretrained=$CKPT_FULL \
    --gen_kwargs max_new_tokens=128 \
    --tasks "$TASKS" \
    --batch_size 1 \
    --log_samples \
    --output_path "./logs/${MODEL}/${CKPT}__vanilla"
