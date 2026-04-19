import math
from typing import Optional, Tuple

import torch
from torch.nn import functional as F



def count_index_occurrences(index_tensor: torch.Tensor, m: int, replace_zeros: Optional[int] = None) -> int:
    """
    Count the occurrences of each index (from 0 to m-1) in the given index_tensor.

    Args:
        index_tensor (torch.Tensor): Tensor of shape (batch_size, n), where each element is an index in the range [0, m-1].
        m (int): The maximum value of the index range (exclusive, i.e., indices range from 0 to m-1).
        replace_zeros (int, optional): If provided, replace zero counts with this value. Defaults to None.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, m), where each element represents the count of occurrences of an index.
    """
    # batch_size, n = index_tensor.size()

    ## Create a one-hot representation of the indices
    one_hot = F.one_hot(index_tensor, num_classes=m).to(index_tensor.device)

    ## Sum over the second dimension (n) to count occurrences
    counts = one_hot.sum(dim=1)
    if replace_zeros is not None:
        counts[counts == 0] = replace_zeros

    return counts


def invert_indices(indices: Tuple[Tuple[int, int]], seq_len: int) -> Tuple[Tuple[int, int]]:
    inverted_indices = list()
    prev_start = 0
    for start_idx, end_idx in indices:
        inverted_indices += ((prev_start, start_idx),)
        prev_start = end_idx + 1
    if prev_start < seq_len:
        inverted_indices += ((prev_start, seq_len),)

    return tuple(inverted_indices)


def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    int_rt = math.isqrt(n)
    return int_rt ** 2 == n


def make_square(n: int) -> Tuple[int, int]:
    int_rt = math.isqrt(n)
    sq = (int_rt + 1) ** 2
    return sq, sq - n


def pad_to_square(hidden_states: torch.Tensor, value: int = -1) -> Tuple[torch.Tensor, int]:
    """
    Pad the hidden states tensor to make the sequence length a perfect square.

    Args:
        hidden_states: (bsz, seq_len, hidden_dim)
        value: Value to use for padding, default -1

    Returns:
        torch.Tensor: Padded hidden states tensor
        int: Amount of padding added
    """
    bsz, seq_len, hidden_dim = hidden_states.size()
    if is_perfect_square(seq_len):
        return hidden_states, 0
    sq, amt_to_pad = make_square(seq_len)

    hidden_states = F.pad(hidden_states, (0, 0, 0, amt_to_pad), mode="constant", value=value)

    return hidden_states, amt_to_pad


def unpad_from_square(hidden_states: torch.Tensor, amt_padded: int) -> torch.Tensor:
    """
    Remove padding from the hidden states tensor, assumes padding is added to the end.
    Args:
        hidden_states: (bsz, seq_len, hidden_dim)
        amt_padded: Amount of padding added
    Returns:
        torch.Tensor: Hidden states tensor with padding removed.
    """
    return hidden_states[:, :-amt_padded, ...]
