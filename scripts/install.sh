#!/bin/bash
# Install a single environment that runs BOTH LLaVA and Qwen LightKV evaluation.
#
# Why this is not one `pip install -r requirements.txt`:
#   LLaVA-NeXT and lmms-eval disagree on pins. The important conflict is `transformers`:
#     - LLaVA-NeXT (its `train` extra): transformers >=4.53.0,<4.54.0
#     - lmms-eval:                       transformers >=4.39.2
#     - Qwen2.5-VL LightKV modeling:     needs Qwen2_5_VLConfig + dynamic_rope_update,
#                                        which exist only in transformers >=4.49
#   The single version that satisfies all three is transformers 4.53.x.
#
#   The trick that makes one env possible: LLaVA-NeXT's base package declares NO
#   dependencies (its pins live in optional [standalone]/[train] extras we skip). So we
#   install lmms-eval's dependency set, add the `llava` package with --no-deps, then pin
#   the shared packages to the resolved versions below.
#
# Usage:
#   bash scripts/install.sh                 # default: CUDA 12.1 wheels
#   CUDA=cu124 bash scripts/install.sh      # other CUDA build
#   CUDA=cpu   bash scripts/install.sh      # CPU-only (no GPU eval)
#   INSTALL_FLASH_ATTN=1 bash scripts/install.sh
#   SKIP_TORCH=1 bash scripts/install.sh    # keep an already-installed torch
#
# Run this from the repository root (the folder containing LLaVA-NeXT/ and lmms-eval/).

set -euo pipefail

# ----- resolved versions (the conflict resolution) -----
# torch must be >=2.6: transformers 4.53 blocks torch.load of non-safetensors (.bin)
# checkpoints on older torch (CVE fix), and the LLaVA-v1.5/1.6 weights are .bin.
CUDA="${CUDA:-cu124}"
TORCH="${TORCH:-2.6.0}"
TORCHVISION="${TORCHVISION:-0.21.0}"
TRANSFORMERS="${TRANSFORMERS:-transformers>=4.53.0,<4.54.0}"
NUMPY="${NUMPY:-numpy==1.26.4}"

HERE="$(cd "$(dirname "$0")/.." && pwd)"
LLAVA_DIR="$HERE/LLaVA-NeXT"
LMMS_DIR="$HERE/lmms-eval"

for d in "$LLAVA_DIR" "$LMMS_DIR"; do
    [ -d "$d" ] || { echo "ERROR: $d not found. Run this from the repo root."; exit 1; }
done

echo "==> [1/5] PyTorch ${TORCH} (${CUDA}) + torchvision ${TORCHVISION}"
if [ "${SKIP_TORCH:-0}" = "1" ]; then
    echo "    SKIP_TORCH=1 -> keeping the currently installed torch"
elif [ "$CUDA" = "cpu" ]; then
    pip install "torch==${TORCH}" "torchvision==${TORCHVISION}" \
        --index-url https://download.pytorch.org/whl/cpu
else
    pip install "torch==${TORCH}" "torchvision==${TORCHVISION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA}"
fi

echo "==> [2/5] lmms-eval (editable) — provides the shared dependency set"
pip install -e "$LMMS_DIR"

echo "==> [3/5] llava package (editable, --no-deps) — avoids LLaVA-NeXT's conflicting pins"
pip install -e "$LLAVA_DIR" --no-deps

echo "==> [4/5] Pin the shared packages to the resolved versions"
# transformers 4.53.x unifies LLaVA + Qwen2.5-VL + lmms-eval; it also pulls a matching
# tokenizers/huggingface_hub, overriding lmms-eval's older transitive pins.
pip install "$TRANSFORMERS" "$NUMPY" "qwen-vl-utils" "accelerate>=0.34.0"

if [ "${INSTALL_FLASH_ATTN:-0}" = "1" ]; then
    echo "    Installing flash-attn (optional; sdpa attention works without it)"
    pip install flash-attn --no-build-isolation
fi

echo "==> [5/5] Smoke test — imports and version checks"
python - <<'PY'
import torch, transformers
print("torch       :", torch.__version__, "| cuda:", torch.cuda.is_available())
print("transformers:", transformers.__version__)

# Qwen2.5-VL APIs that only exist in transformers >=4.49
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig  # noqa
from transformers.modeling_rope_utils import dynamic_rope_update  # noqa

# LightKV entry points resolve
from lmms_eval.mymodels.lightkv import LightKVModule, get_merge_modules  # noqa
from lmms_eval.models import AVAILABLE_MODELS
for k in ("llava_lightkv", "qwen2_5_vl_lightkv"):
    assert k in AVAILABLE_MODELS, f"missing model key: {k}"
import llava  # the LLaVA-NeXT package
print("OK: transformers has Qwen2.5-VL, LightKV models registered, llava importable")
PY

echo
echo "Done. Try:  cd lmms-eval && bash ../scripts/eval_qwen_lightkv.sh qwen2_5_vl_lightkv Qwen/Qwen2.5-VL-7B-Instruct 12345"
