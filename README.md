# LightKV: Make Your LVLM KV Cache More Lightweight

[![OpenReview](https://img.shields.io/badge/OpenReview-n77IeySrQl-8c1b13.svg)](https://openreview.net/forum?id=n77IeySrQl)
[![arXiv](https://img.shields.io/badge/arXiv-2605.00789-b31b1b.svg)](https://arxiv.org/abs/2605.00789)

Official code for the paper **"Make Your LVLM KV Cache More Lightweight"** (TMLR).

LightKV reduces the Key–Value (KV) cache size of Large Vision-Language Models (LVLMs)
by exploiting redundancy among vision-token embeddings. Guided by the text prompt,
LightKV performs **cross-modality message passing** to aggregate informative messages
across vision tokens and **progressively compresses them during prefill**. With only
55% of the original vision tokens, LightKV halves the vision-token KV cache, reduces
computation by up to 40%, and preserves general-purpose performance.

This release contains the code needed to reproduce the **LLaVA** and **Qwen** results.

---

## Repository layout

```
LightKV/
├── LLaVA-NeXT/          # the `llava` package — LLaVA-v1.5 / LLaVA-NeXT models + LightKV
│   └── llava/model/
│       ├── builder.py                       # loads the "-lightkv" model variants
│       └── language_model/
│           ├── llava_llama_lightkv.py        # LLaVA (LLaMA backbone) + LightKV
│           ├── llava_qwen_lightkv.py         # LLaVA (Qwen backbone)  + LightKV
│           ├── modeling_llama_lightkv.py     # LLaMA decoder w/ in-prefill merging
│           └── modeling_qwen2_lightkv.py     # Qwen2 decoder  w/ in-prefill merging
└── lmms-eval/          # evaluation harness + Qwen2.5-VL LightKV
    └── lmms_eval/
        ├── models/
        │   ├── llava_lightkv.py               # lmms-eval wrapper: --model llava_lightkv
        │   └── qwen2_5_vl_lightkv.py          #                    --model qwen2_5_vl_lightkv
        └── mymodels/
            ├── qwen2_5_vl/                     # Qwen2.5-VL modeling + LightKV
            └── lightkv/                        # core LightKV merging (the method)
                ├── module.py                   #   LightKVModule (the method)
                ├── constructor.py              #   get_merge_modules(...)
                └── base.py                     #   LightKVBase
```

The **core idea** (cross-modality message passing / progressive merging) lives in a
single shared module, `lmms-eval/lmms_eval/mymodels/lightkv/`
(`module.py` = `LightKVModule`, `constructor.py` = `get_merge_modules`). Both the LLaVA
decoders (`LLaVA-NeXT/llava/model/language_model/modeling_*_lightkv.py`) and the Qwen
decoders import the merging modules from there — so `lmms-eval` must be installed even
when evaluating LLaVA.

### Why the LLaVA model code is not under `mymodels/`

LLaVA and Qwen use two different loading paths, so their LightKV modeling lives in
different places:

- **LLaVA / LLaVA-NeXT** load through the `llava` package (from LLaVA-NeXT). Both the
  `llava` and `llava_lightkv` wrappers call `llava.model.builder.load_pretrained_model`,
  so the LightKV decoders live *inside that package* at
  `LLaVA-NeXT/llava/model/language_model/*_lightkv.py` — not under `mymodels/`.
- **Qwen2.5-VL** has no separate framework package; it loads through plain
  `transformers`. Its LightKV-modified modeling is therefore vendored into
  `lmms-eval/lmms_eval/mymodels/qwen2_5_vl/`.

That is why `mymodels/` holds `qwen2_5_vl/` (Qwen modeling) and `lightkv/` (the shared core
imported by both), but no `llava`/`llava_next` — the LLaVA modeling is carried by the
`llava` package instead.

---

## Installation

Use the provided script, which sets up a single environment that runs both LLaVA and
Qwen LightKV (from the repository root):

```bash
conda create -n lightkv python=3.10 -y
conda activate lightkv

bash scripts/install.sh                 # CUDA 12.1 wheels (default)
# CUDA=cu124 bash scripts/install.sh    # other CUDA build
# CUDA=cpu   bash scripts/install.sh    # CPU only
# INSTALL_FLASH_ATTN=1 bash scripts/install.sh
```

### What the script resolves

LLaVA-NeXT and lmms-eval disagree on pins; the decisive one is `transformers`:

| Source                         | `transformers` constraint |
|--------------------------------|---------------------------|
| LLaVA-NeXT (`train` extra)     | `>=4.53.0,<4.54.0`        |
| lmms-eval                      | `>=4.39.2`                |
| Qwen2.5-VL LightKV modeling    | `>=4.49` (needs `Qwen2_5_VLConfig`, `dynamic_rope_update`) |

**`transformers` 4.53.x** is the one version that satisfies all three. The script
installs lmms-eval's dependency set, adds the `llava` package with `--no-deps` (its base
package declares no dependencies, so this skips its conflicting pins), then pins
`transformers` 4.53.x, `numpy==1.26.4`, `qwen-vl-utils`, and `accelerate`. It ends with an
import smoke test.

Prefer to install by hand? `pip install -e lmms-eval`, then
`pip install -e LLaVA-NeXT --no-deps`, then
`pip install "transformers>=4.53.0,<4.54.0" numpy==1.26.4 qwen-vl-utils`.

---

## Available models

| `--model` key         | Description                                  |
|-----------------------|----------------------------------------------|
| `llava`               | LLaVA-v1.5 / LLaVA-NeXT — Vanilla (baseline) |
| `llava_lightkv`       | LLaVA-v1.5 / LLaVA-NeXT — **LightKV**        |
| `qwen2_5_vl`          | Qwen2.5-VL — Vanilla (baseline)              |
| `qwen2_5_vl_lightkv`  | Qwen2.5-VL — **LightKV**                     |

---

## LightKV hyperparameters

LightKV is configured through three `--model_args`, each a `-`-joined list where the
`i`-th entry describes the `i`-th compression stage:

| Arg             | Symbol | Meaning                                              |
|-----------------|--------|------------------------------------------------------|
| `merge_layers`  | Λ      | decoder layer indices at which vision tokens merge   |
| `merge_windows` | W      | number of windows per side for message passing       |
| `merge_ratios`  | P      | fraction of vision tokens dropped at that stage      |

Example: `merge_layers=12-18-24,merge_windows=4-2-1,merge_ratios=0.5-0.5-0.5`
compresses at layers 12/18/24, dropping 50% of the (remaining) vision tokens at each
stage. The optimal per-model schedule is reported in the paper (selected on COCO/MME);
the run scripts ship with a representative configuration.

---

## Running evaluation

Ready-to-use scripts are in `scripts/` (run them from the `lmms-eval/` directory so the
`./logs` output path resolves there):

```bash
cd lmms-eval

# LLaVA + LightKV  (LLaVA-NeXT-7B or LLaVA-v1.5-7B)
bash ../scripts/eval_llava_lightkv.sh liuhaotian/llava-v1.6-vicuna-7b 12345
bash ../scripts/eval_llava_lightkv.sh liuhaotian/llava-v1.5-7b        12345

# Qwen + LightKV
bash ../scripts/eval_qwen_lightkv.sh qwen2_5_vl_lightkv Qwen/Qwen2.5-VL-7B-Instruct 12345

# Vanilla (uncompressed) reference
bash ../scripts/eval_vanilla.sh qwen2_5_vl Qwen/Qwen2.5-VL-7B-Instruct 12345
```

Or invoke `lmms_eval` directly:

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl_lightkv \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,merge_layers=12-17-22,merge_windows=6-4-2,merge_ratios=0.5-0.5-0.5 \
    --gen_kwargs max_new_tokens=128 \
    --tasks mme,pope,gqa,seedbench_lite,vizwiz_vqa_val,nocaps_val,coco2017_cap_val \
    --batch_size 1 --log_samples --output_path ./logs/
```

The evaluated benchmarks (MME, POPE, GQA, SeedBench, VizWiz, NoCaps, COCO) follow the
standard [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) task definitions;
datasets are downloaded automatically on first use.

---

## Citation

```bibtex
@article{chen2026make,
  title   = {Make Your {LVLM} {KV} Cache More Lightweight},
  author  = {Xihao Chen and Yangyang Guo and Roger Zimmermann},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2026},
  url     = {https://openreview.net/forum?id=n77IeySrQl}
}
```

## Acknowledgements

This code builds on [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). We thank the authors for
releasing their code.
