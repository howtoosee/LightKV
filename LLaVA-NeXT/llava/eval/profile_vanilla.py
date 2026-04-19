import argparse
import json
import os
import pprint
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from torch.profiler import ProfilerActivity, profiler

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.eval.memory_utils import bytes_to_gb, get_curr_memory, get_curr_peak_memory, get_kv_memory, reset_peak_memory
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

warnings.filterwarnings("ignore")


def write_profile_trace(profiler, profile_trace_dir, qn_idx):
    if not profiler or not profile_trace_dir:
        return

    profile_trace_path = os.path.join(profile_trace_dir, f"qn_{qn_idx}.json")
    profiler.export_chrome_trace(profile_trace_path)


def get_data(image_processor, model_config, tokenizer, device, bsz=1):
    qs = "Describe this image in detail."
    # image_file = "/playground/demo/dog.jpeg"
    home = Path.home()
    image_file = os.path.join(home, "repository", "lvlm", "LLaVA-NeXT", "playground", "demo", "dog.jpeg")

    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["vicuna_v1"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    image = Image.open(image_file).convert("RGB")
    image_tensor = process_images([image], image_processor, model_config)[0]
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    input_ids = input_ids.to(device=device, non_blocking=True).unsqueeze(0)

    input_ids = input_ids.to(device="cuda", non_blocking=True)
    input_ids = torch.vstack([input_ids for _ in range(bsz)])
    image_tensor = image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True)
    image_tensor = torch.vstack([image_tensor.unsqueeze(0) for _ in range(bsz)])
    image_sizes = [image.size for _ in range(bsz)]

    return input_ids, image_tensor, image_sizes


def eval_model(args):
    pp = pprint.PrettyPrinter(indent=2, sort_dicts=False, compact=True)

    # Model
    disable_torch_init()
    # model_path = os.path.expanduser(args.model_path)
    # model_name = get_model_name_from_path(model_path)
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_name=model_name,
        model_base=None,
        device_map="cuda:0",
        attn_implementation=args.attn_impl,
    )

    model.eval()

    if args.profile_trace_dir:
        profile_trace = os.path.expanduser(args.profile_trace)
        os.makedirs(profile_trace, exist_ok=True)

    input_ids, image_tensor, image_sizes = get_data(image_processor, model.config, tokenizer, "cuda")

    reset_peak_memory()
    # base_memory = get_curr_peak_memory()
    base_memory = get_curr_memory()
    with (
        torch.inference_mode(),
        torch.no_grad(),
        profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            with_flops=True,
        ) as prof,
    ):
        output = model.generate(
            inputs=input_ids,
            pad_token_id=-999,
            eos_token_id=-999,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
        )
    peak_memory = get_curr_peak_memory() - base_memory
    # peak_memory = get_curr_peak_memory()

    kv_memory = get_kv_memory(output["past_key_values"])
    peak_memory_gb = bytes_to_gb(peak_memory)
    kv_memory_gb = bytes_to_gb(kv_memory)

    stats = prof.key_averages()
    flops = sum([s.flops for s in stats])

    num_tokens = output["sequences"].size()[1]
    # num_tokens_history += (num_tokens, )
    output = tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
    eval_logger.info(f"Output: {output}")

    time_to_first_token_mean = None
    time_to_first_token_std = None
    runtimes = list()
    if args.profile_time:
        with torch.inference_mode(), torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                output = model.generate(
                    inputs=input_ids,
                    pad_token_id=-999,
                    eos_token_id=-999,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=1,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                duration = time.perf_counter() - start_time
                runtimes.append(duration)
        time_to_first_token_mean = np.mean(runtimes)
        time_to_first_token_std = np.std(runtimes)

    gen_walltime_mean = None
    gen_walltime_std = None
    gen_walltimes = list()
    if args.profile_gen_latency:
        with torch.inference_mode(), torch.no_grad():
            for _ in range(10):
                start_time = time.perf_counter()
                output = model.generate(
                    inputs=input_ids,
                    pad_token_id=-999,
                    eos_token_id=-999,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    num_beams=1,
                    max_new_tokens=100,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                duration = time.perf_counter() - start_time
                gen_walltimes.append(duration)
        gen_walltime_mean = np.mean(gen_walltimes)
        gen_walltime_std = np.std(gen_walltimes)

    profile_dict = dict(
        config=dict(
            model_name=model_name,
            method="vanilla",
            attn_impl=args.attn_impl,
        ),
        profile=dict(
            flops=flops,
            peak_memory=peak_memory_gb,
            kv_memory=kv_memory_gb,
            memory_units="GB",
            num_tokens_generated=num_tokens,
            time_to_first_token=dict(
                mean=time_to_first_token_mean,
                std=time_to_first_token_std,
                runtimes=runtimes,
                unit="seconds",
            ),
            gen_walltime=dict(
                mean=gen_walltime_mean,
                std=gen_walltime_std,
                runtimes=gen_walltimes,
                unit="seconds",
            ),
        ),
    )

    print("*" * 10 + " PROFILE " + "*" * 10)
    print(f"Vanilla, Max new tokens={args.max_new_tokens}")
    pp.pprint(profile_dict)
    print("*" * 30)

    if args.info_file:
        info_file = os.path.abspath(args.info_file)
        os.makedirs(os.path.dirname(info_file), exist_ok=True)

        print(f"Writing profile to {info_file}")
        with open(info_file, "w") as f:
            json.dump(profile_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbosity", type=str, default="INFO")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--attn-impl", type=str, default="sdpa")
    parser.add_argument("--profile-time", action="store_true", default=False)
    parser.add_argument("--profile-gen-latency", action="store_true", default=False)

    parser.add_argument("--info-file", type=str, default=None)
    parser.add_argument("--profile-trace-dir", type=str, default=None)
    args = parser.parse_args()

    eval_logger.remove()
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity)

    eval_model(args)
