import math

import torch

def reset_peak_memory():
    torch.cuda.reset_max_memory_allocated()


def get_curr_memory():
    torch.cuda.synchronize()
    mem = torch.cuda.memory_allocated()
    return mem

def get_curr_peak_memory():
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated()
    return mem


def get_memory(tensor):
    memory_bytes = tensor.element_size() * tensor.nelement()
    return memory_bytes

def get_kv_memory(kv):
    kv_size = 0
    for layer_kv in kv:
        for head_kv in layer_kv:
            kv_size += get_memory(head_kv)
    return kv_size


def round_sigfigs(num, sigfigs=3):
    if num == 0:
        return 0.0
    return round(num, sigfigs - int(math.floor(math.log10(abs(num)))) - 1)


def bytes_to_gb(num_bytes, sigfigs=3):
    num_gb = num_bytes / (1024**3)
    return round_sigfigs(num_gb, sigfigs=sigfigs)
