from abc import ABC
from typing import List, Optional, Tuple, Dict
from torch import nn

from loguru import logger as eval_logger

from .module import LightKVModule


class LightKVBase(nn.Module, ABC):
    def set_img_indices(self, img_indices: Tuple[Tuple[Tuple[int, int]]]) -> None:
        raise NotImplementedError("This method is not implemented yet.")

    def forward(self):
        raise NotImplementedError("This method is not implemented yet.")


class MergeModuleDict(Dict[int, LightKVBase]):
    def __init__(self, modules_dict, listeners_dict):
        super(MergeModuleDict, self).__init__(modules_dict)
        self.listeners = listeners_dict
        self.layer_idxs = tuple(sorted(modules_dict.keys()))

    def update_img_indices(self, layer_id: int) -> None:
        """
        Reads the new image indices from the listener at `layer_idx` and updates the image indices for the next module.
        """
        if self.listeners is None or layer_id not in self.listeners:
            return

        new_img_indices = self.listeners[layer_id]()["new_img_indices"]
        next_layer_lst_idx = self.layer_idxs.index(layer_id) + 1

        if next_layer_lst_idx < len(self.layer_idxs):
            next_layer_idx = self.layer_idxs[next_layer_lst_idx]
            self[next_layer_idx].set_img_indices(new_img_indices)


def extend_or_check_lengths(*lists) -> List:
    """
    Ensures all lists are the same length.
    If a list has length 1, its item is repeated to match the length of the longest list.
    Raises a ValueError if lists are of incompatible lengths.
    """
    max_length = max(len(lst) for lst in lists)
    extended_lists = []
    for lst in lists:
        if len(lst) == 1:
            # Repeat the single item to match the maximum length
            extended_lists.append(lst * max_length)
        elif len(lst) == max_length:
            # No need to modify the list
            extended_lists.append(lst)
        else:
            # Raise an error if lengths don't match and can't be extended
            raise ValueError(f"List {lst} has incompatible length: {len(lst)}")
    return extended_lists


LIST_DELIMITER = "-"


def get_merge_modules(
    method: str,
    merge_layers: str,
    merge_ratios: str,
    merge_windows: Optional[str] = None,
    trace_source: Optional[bool] = None,
    img_indices: Optional[Tuple[int, int]] = None,
    total_layers: Optional[int] = None,
) -> Dict[int, LightKVBase]:
    if method is None:
        method = "lightkv"

    if merge_layers is None:
        return MergeModuleDict({}, {})

    if method in ("lightkv", "lightkv_optimized"):
        return get_lightkv_modules(merge_layers, merge_ratios, merge_windows, img_indices, trace_source)

    raise ValueError(f"Invalid LightKV merge method: {method}")


def get_lightkv_modules(merge_layers, merge_ratios, merge_windows, img_indices, trace_source):
    assert merge_layers is not None and merge_ratios is not None and merge_windows is not None, "Merge layers, ratios, and windows must be provided."

    if isinstance(merge_layers, str):
        merge_layers = [int(x) for x in merge_layers.split(LIST_DELIMITER)]
    elif isinstance(merge_layers, int):
        merge_layers = [merge_layers]

    if isinstance(merge_ratios, str):
        merge_ratios = [float(x) for x in merge_ratios.split(LIST_DELIMITER)]
    elif isinstance(merge_ratios, float):
        merge_ratios = [merge_ratios]

    if isinstance(merge_windows, str):
        merge_windows = [int(x) for x in merge_windows.split(LIST_DELIMITER)]
    elif isinstance(merge_windows, int):
        merge_windows = [merge_windows]

    merge_layers, merge_ratios, merge_windows = extend_or_check_lengths(merge_layers, merge_ratios, merge_windows)

    def get_listener():
        data = dict()
        return lambda new_info: data.update(new_info), lambda: data

    merge_modules = {}
    get_info_fns = {}
    module_cls = LightKVModule

    for layer, num_win, ratio in zip(merge_layers, merge_windows, merge_ratios):
        event_hook, get_info = get_listener()
        get_info_fns[layer] = get_info

        merge_modules[layer] = module_cls(
            prune_layer=layer,
            n_parts_per_side=num_win,
            img_indices=img_indices,
            discard_ratio=ratio,
            event_hook=event_hook,
            trace_source=trace_source,
        )
    return MergeModuleDict(merge_modules, get_info_fns)
