import math
from typing import List, Optional, Tuple, Dict

from einops import rearrange, repeat

import torch
from torch import Tensor

from loguru import logger as eval_logger

from .base import LightKVBase
from .utils import is_perfect_square, pad_to_square, unpad_from_square


class LightKVModule(LightKVBase):
    """The LightKV in-prefill vision-token merging module (batched implementation).

    Windows can have different token counts; each is padded to the largest window
    size and explicit masks are carried through matching and weighting.
    """

    def __init__(
        self,
        prune_layer: int,
        n_parts_per_side: int,
        discard_ratio: float,
        img_indices: Optional[Tuple[Tuple[Tuple[int, int]]]] = None,
        trace_source: Optional[bool] = False,
        event_hook: Optional[callable] = None,
    ):
        super(LightKVModule, self).__init__()
        self.prune_layer = prune_layer
        self.n_parts_per_side = n_parts_per_side  # number of parts in each side of 2D grid
        ## total number of windows = num_parts_per_side ** 2
        self.img_indices = None
        self.set_img_indices(img_indices)  # Tuple[bsz x Tuple[ num_images x Tuple[start_idx, end_idx] ] ]
        self.discard_ratio = discard_ratio  # ratio of tokens to keep in each window

        self.event_hook = event_hook

        self.trace_source = trace_source
        if self.trace_source:
            eval_logger.info("Tracing source of merged tokens")

    def set_img_indices(self, img_indices: Tuple[Tuple[Tuple[int, int]]]) -> None:
        """
        Validates img_indices and set img_indices attribute.

        Args:
            img_indices (Tuple[bsz x Tuple[num_imgs x Tuple[int, int]]]): Tuple of (start_idx, end_idx) for each image for each batch
                Currently only batch size = 1 is supported
        """
        if img_indices is None:
            return

        bsz = len(img_indices)
        assert bsz == 1, f"Currently only batch size = 1 is supported, got {bsz}"

        all_has_images = all(len(batch) > 0 for batch in img_indices)
        assert all_has_images, f"Got batch with no images"

        correct_indices = all(all(len(idxs) == 2 for idxs in batch) for batch in img_indices)
        assert correct_indices, f"Image indices must be a tuple of (start_idx, end_idx) for each image"

        self.img_indices = img_indices

    @property
    def img_token_indices(self):
        return tuple(self.img_indices)

    def forward(self, layer_id: int, hidden_states: torch.Tensor, attentions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            layer_id (int): The layer index of the current transformer layer.
            hidden_states (torch.Tensor): size (bsz, seq_len, hidden_size)
                 (torch.Tensor): size(bsz, num_heads, seq_len, seq_len) FULL attention matrix
        """
        if layer_id != self.prune_layer:
            return hidden_states

        if self.discard_ratio == 0:
            return hidden_states

        if self.img_indices is None:
            raise ValueError("Image indices must be provided for merging operations.")

        if attentions is None:
            raise ValueError("Attention matrix must be provided for merging operations.")

        bsz, seq_len, hidden_dim = hidden_states.size()
        assert bsz == 1, f"Currently only batch size = 1 is supported, got {bsz}"

        event_info = dict()

        attentions = attentions.softmax(dim=2)

        # TODO - handle multiple batches
        img_indices = self.img_indices[0]

        # TODO - handle images of different token sizes
        img_token_sizes = tuple([idx[1] - idx[0] + 1 for idx in img_indices])
        if any(sz < 8 * (self.n_parts_per_side**2) for sz in img_token_sizes):
            eval_logger.debug("Image token sizes are too small for merging, skipping...")
            event_info["new_img_indices"] = self.img_indices
            self.event_hook(event_info)
            return hidden_states, torch.arange(seq_len, device=hidden_states.device), None

        # eval_logger.debug(f"Img_indices: {img_indices}")
        if any([idx[0] >= seq_len or idx[1] >= seq_len for idx in img_indices]):
            eval_logger.warning(f"Image indices out of bounds. Seq_len={seq_len}, indices: {img_indices}. Truncating image indices to fit seq_len.")
        img_indices = tuple([idx for idx in img_indices if idx[0] < seq_len and idx[1] < seq_len])

        image_tokens, image_attentions, non_image_tokens = self.split_tokens_attentions(hidden_states, attentions, img_indices)
        eval_logger.debug(f"Image_tokens: {image_tokens.size()}, Image_attentions: {image_attentions.size()}")
        merged_image_tokens, sorted_idxs, source_adj_matrix = self.compute(image_tokens, image_attentions)  # (num_imgs, new_seq_len, hidden_dim)
        eval_logger.debug(f"Merged_image_tokens: {merged_image_tokens.size()}")
        combined_tokens, new_img_indices = self.combine_all_tokens(
            merged_image_tokens, img_indices, non_image_tokens
        )  # Combine vision tokens with non-vision tokens
        eval_logger.debug(f"Combined tokens: {combined_tokens.size()}")
        eval_logger.debug(f"New_img_indices: {new_img_indices}")

        new_img_indices = (new_img_indices,)  # TODO: handle multiple batches
        event_info["new_img_indices"] = new_img_indices  # Tuple[bsz x Tuple[num_imgs x Tuple[new_img_start, new_img_end]]]

        self.event_hook(event_info)

        return combined_tokens, sorted_idxs, source_adj_matrix

    @staticmethod
    def combine_all_tokens(
        merged_image_tokens: torch.Tensor, img_indices: Tuple[Tuple[int, int]], non_image_token_list: Tuple[torch.Tensor]
    ) -> Tuple[Tensor, Tuple[Tuple[int, int]]]:
        """
        Comine vision tokens with non-vision tokens.
        """
        img_tokens_list = [
            rearrange(merged_image_tokens[i, :], "s h -> 1 s h") for i in range(merged_image_tokens.size(0))
        ]  # List[num_imgs x Tensor[bsz=1, seq_len, hidden_dim]]

        combined_tokens_lst = list()
        new_img_indices = tuple()
        curr_num_tokens = 0
        for i in range(len(non_image_token_list) + len(img_tokens_list)):
            if i % 2 == 0:  # text sequence
                next_tokens = non_image_token_list[i // 2]
            else:  # image sequence
                next_tokens = img_tokens_list[i // 2]
                img_start = curr_num_tokens
                img_end = curr_num_tokens + next_tokens.size(1) - 1
                new_img_indices += ((img_start, img_end),)

            combined_tokens_lst.append(next_tokens)
            curr_num_tokens += next_tokens.size(1)

        combined_tokens = torch.cat(combined_tokens_lst, dim=1)
        return combined_tokens, new_img_indices

    @staticmethod
    def invert_indices(indices: Tuple[Tuple[int, int]], seq_len: int) -> Tuple[Tuple[int, int]]:
        inverted_indices = tuple()
        prev_start = 0
        for start, end in indices:
            inverted_indices += ((prev_start, start),)
            prev_start = end + 1
        if prev_start < seq_len:
            inverted_indices += ((prev_start, seq_len),)

        return inverted_indices

    def split_tokens_attentions(
        self, hidden_states: torch.Tensor, attentions: torch.Tensor, img_indices: Tuple[Tuple[int, int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor]]:
        """
        Args:
            hidden_states (torch.Tensor): size (bsz, seq_len, hidden_size)
            attentions (torch.Tensor): size(bsz, num_heads, seq_len_src, seq_len_dst) lower triangular matrix
            img_indices (Tuple[Tuple[int, int]]): Tuple of (start_idx, end_idx) for each image

        Returns:
            torch.Tensor: image_tokens (bsz, seq_len, hidden_dim)
            torch.Tensor: image_attentions from image tokens to all other tokens (bsz, num_heads, seq_len, seq_len)
            Tuple[torch.Tensor]: non_image_tokens Tuple[sequences x (bsz, seq_len, hidden_dim)]
        """
        # bsz, seq_len, hidden_dim = hidden_states.size()
        device = hidden_states.device

        src_idxs = torch.cat([torch.arange(a, b + 1, device=device) for (a, b) in img_indices])

        all_image_tokens = hidden_states.index_select(dim=1, index=src_idxs)
        ### WARNING Batch size must be 1 !!!
        all_image_tokens = rearrange(all_image_tokens, "b (n s) h -> (b n) s h", n=len(img_indices))  # (num_imgs, seq_len, hidden_dim)

        other_token_indices = self.invert_indices(img_indices, hidden_states.size(1))  # Tuple[Tuple[int, int]]
        dst_idxs = torch.cat([torch.arange(a, b, device=device) for (a, b) in other_token_indices])

        all_non_image_tokens = tuple()
        for start_idx, end_idx in other_token_indices:
            all_non_image_tokens += (hidden_states[:, start_idx:end_idx, :],)

        if len(img_indices) == 1:
            a, b = img_indices[0]
            all_image_attns = attentions[:, :, a : b + 1, :].index_select(dim=3, index=dst_idxs)
        else:
            all_image_attns = attentions.index_select(dim=2, index=src_idxs).index_select(dim=3, index=dst_idxs)

        all_image_attns = rearrange(all_image_attns, "b n_heads (n_img src) dst -> (b n_img) n_heads src dst", n_img=len(img_indices))

        # Multiple images
        return all_image_tokens, all_image_attns, all_non_image_tokens

    def get_windows(self, image_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        """
        Split a square image into `num_segments` equal square segments, returning
        the label matrix and the set of indices for each segment.

        Args:
            image_tokens (torch.Tensor): A 3D tensor of shape (num_imgs, num_tokens, hidden_dim).
            # num_segments (int): The number of segments along each side.

        Returns:
            torch.Tensor: Labels indicating segment membership (bzs, num_tokens).
            torch.Tensor: 1D indices of each token in the original image.
            dict[int, torch.Tensor]: A dictionary where keys are segment labels and values are sets of indices in the form (row, col).
        """
        bsz, num_tokens, hidden_dim = image_tokens.size()

        if not is_perfect_square(num_tokens):
            image_tokens_padded, amt_padded = pad_to_square(image_tokens)
            return self.get_windows_square(image_tokens_padded, amt_padded)

        return self.get_windows_square(image_tokens)

    def get_windows_square(self, image_tokens: torch.Tensor, amt_padded: int = 0) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor]]:
        bsz, num_tokens, hidden_dim = image_tokens.size()  # num_tokens must be a square number
        device = image_tokens.device

        side_len = math.isqrt(num_tokens)  # Length of one side of the square image
        num_segments = self.n_parts_per_side**2

        partitions_per_side = self.n_parts_per_side  # Number of partitions along one side

        # Create grid of row and column indices
        indices_1d = repeat(torch.arange(num_tokens, device=device), "n -> b n", b=bsz)  # (bsz, num_tokens)

        partition_size = side_len // partitions_per_side
        rem = side_len % partitions_per_side

        partitions = torch.full((partitions_per_side,), partition_size, device=device)
        partitions[:rem] += 1

        labels_sq = torch.arange(partitions_per_side**2, device=device)
        labels_sq = rearrange(labels_sq, "(h w) -> h w", h=partitions_per_side)
        labels_sq = repeat(labels_sq, "h w -> b h w", b=bsz)

        labels_sq = labels_sq.repeat_interleave(partitions, dim=2)
        labels_sq = labels_sq.repeat_interleave(partitions, dim=1)

        labels = rearrange(labels_sq, "b h w -> b (h w)")  # (bsz, num_tokens)

        if amt_padded is not None and amt_padded > 0:
            labels = unpad_from_square(labels, amt_padded)
            indices_1d = unpad_from_square(indices_1d, amt_padded)

        # Generate index dictionary
        label_to_idx = dict()
        for label in range(num_segments):
            curr_indices_1d = indices_1d[labels == label]
            curr_indices_1d = rearrange(curr_indices_1d, "(b n) -> b n", b=bsz)
            label_to_idx[label] = curr_indices_1d

        return labels, indices_1d, label_to_idx

    def compute_source(
        self, all_unm_idxs: List[torch.Tensor], all_src_idxs: List[torch.Tensor], all_dst_idxs: List[torch.Tensor], partition_sizes: List[int], total: int,
    ) -> torch.Tensor:
        """
        Compute the source of the merged tokens.

        Args:
            unm_idxs (torch.Tensor): Unmerged indices (bsz, seq_len)
            src_idxs (torch.Tensor): Source indices (bsz, seq_len)
            dst_idxs (torch.Tensor): Destination indices (bsz, seq_len)
            total (int): Total number of tokens

        Returns:
            torch.Tensor: Source of the merged tokens
        """
        assert sum(partition_sizes) == total, f"Sum of partition sizes {sum(partition_sizes)} must match total {total}"
        
        # import pdb; pdb.set_trace()
        adj_matrices = list()

        for unm_idxs, src_idxs, dst_idxs, part_size in zip(all_unm_idxs, all_src_idxs, all_dst_idxs, partition_sizes):
            device = unm_idxs.device
            source = torch.eye(part_size, device=device).expand(1, part_size, part_size)

            src, dst = source[..., ::2, :], source[..., 1::2, :]
            n, t1, c = src.shape

            r = min(int(self.discard_ratio * part_size), part_size // 2)  # number of tokens involved in merging from either bi-part

            unm = src.gather(dim=-2, index=unm_idxs.expand(n, t1 - r, c))
            src = src.gather(dim=-2, index=src_idxs.expand(n, r, c))
            dst = dst.scatter_reduce(-2, dst_idxs.expand(n, r, c), src, reduce="amax")
            adj_matrices.append(torch.cat([unm, dst], dim=1))
        
        def diagonal_stack(matrices):
            bsz = matrices[0].shape[0]
            assert bsz == 1, f"Batch size must be 1, got {bsz}"

            # Calculate total required output size
            total_rows = sum(m.shape[1] for m in matrices)
            total_cols = sum(m.shape[2] for m in matrices)

            # Create the result matrix filled with zeros
            result = torch.zeros(bsz, total_rows, total_cols)

            # Offsets for placing each matrix
            row_offset = 0
            col_offset = 0

            for m in matrices:
                b, h, w = m.shape
                result[:, row_offset : row_offset + h, col_offset : col_offset + w] = m
                row_offset += h
                col_offset += w

            return result

        return diagonal_stack(adj_matrices)

    def __repr__(self):
        return f"ImageTokenMergingOptimized(layer={self.prune_layer}, win={self.n_parts_per_side}, ratio={self.discard_ratio}), img_indices={self.img_indices})"

    def compute(self, image_tokens: torch.Tensor, image_attentions: torch.Tensor) -> torch.Tensor:
        num_imgs, seq_len, hidden_dim = image_tokens.size()
        _, num_heads, _, att_dst_len = image_attentions.size()

        _labels, _idx_1d, label_to_idxs = self.get_windows(image_tokens)
        window_indices = tuple(label_to_idxs.values())
        partition_sizes = torch.tensor([curr_idxs.size(1) for curr_idxs in window_indices], device=image_tokens.device, dtype=torch.long)

        padded_window_indices = self._pad_window_indices(window_indices)
        window_tokens = self._gather_window_tokens(image_tokens, padded_window_indices)
        window_attentions = self._gather_window_attentions(image_attentions, padded_window_indices, num_heads, att_dst_len)

        flat_window_tokens = rearrange(window_tokens, "b w s h -> (b w) s h")
        flat_window_attentions = rearrange(window_attentions, "b w nh s d -> (b w) nh s d")
        flat_window_indices = rearrange(padded_window_indices, "b w s -> (b w) s")
        flat_partition_sizes = repeat(partition_sizes, "w -> (b w)", b=num_imgs)

        combined_tokens, combined_idxs, combined_mask, trace_info = self.bipartite_matching_padded(
            flat_window_tokens,
            flat_window_attentions,
            flat_window_indices,
            flat_partition_sizes,
        )

        combined_tokens = rearrange(combined_tokens, "(b w) s h -> b w s h", b=num_imgs)
        combined_idxs = rearrange(combined_idxs, "(b w) s -> b w s", b=num_imgs)
        combined_mask = rearrange(combined_mask, "(b w) s -> b w s", b=num_imgs)

        flat_combined_tokens = rearrange(combined_tokens, "b w s h -> b (w s) h")
        flat_combined_idxs = rearrange(combined_idxs, "b w s -> b (w s)")
        flat_combined_mask = rearrange(combined_mask, "b w s -> b (w s)")

        new_tokens = flat_combined_tokens.masked_select(flat_combined_mask.unsqueeze(-1)).view(num_imgs, -1, hidden_dim)
        new_idxs = flat_combined_idxs.masked_select(flat_combined_mask).view(num_imgs, -1)

        sorted_idxs = new_idxs.argsort(dim=1)
        reordered_tokens = new_tokens.gather(dim=1, index=repeat(sorted_idxs, "b n -> b n h", h=hidden_dim))

        if self.trace_source:
            partition_sizes_list = partition_sizes.tolist()
            all_unm_idxs, all_src_idxs, all_dst_idxs = self._split_trace_info(trace_info, num_imgs, len(window_indices), partition_sizes_list)
            source_matrix = self.compute_source(
                all_unm_idxs=all_unm_idxs,
                all_src_idxs=all_src_idxs,
                all_dst_idxs=all_dst_idxs,
                partition_sizes=partition_sizes_list,
                total=seq_len,
            )

            sorted_idxs = sorted_idxs.to(source_matrix.device)
            orig_idxs = torch.cat(window_indices, dim=1).to(source_matrix.device)
            reordered_source_matrix = torch.zeros_like(source_matrix, device=source_matrix.device).scatter(
                dim=1,
                index=repeat(sorted_idxs, "b n -> b n s", s=seq_len),
                src=source_matrix,
            )
            reordered_source_matrix = torch.zeros_like(source_matrix, device=source_matrix.device).scatter(
                dim=2,
                index=repeat(orig_idxs, "b s -> b n s", n=reordered_tokens.size(1)),
                src=reordered_source_matrix,
            )
            return reordered_tokens, new_idxs, reordered_source_matrix

        return reordered_tokens, new_idxs, None

    def _pad_window_indices(self, window_indices: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        num_imgs = window_indices[0].size(0)
        num_windows = len(window_indices)
        max_window_tokens = max(curr_idxs.size(1) for curr_idxs in window_indices)
        device = window_indices[0].device
        dtype = window_indices[0].dtype

        padded_window_indices = torch.empty((num_imgs, num_windows, max_window_tokens), device=device, dtype=dtype)
        for window_idx, curr_idxs in enumerate(window_indices):
            curr_window_tokens = curr_idxs.size(1)
            padded_window_indices[:, window_idx, :curr_window_tokens] = curr_idxs
            if curr_window_tokens < max_window_tokens:
                padded_window_indices[:, window_idx, curr_window_tokens:] = curr_idxs[:, -1:].expand(-1, max_window_tokens - curr_window_tokens)

        return padded_window_indices

    @staticmethod
    def _gather_window_tokens(image_tokens: torch.Tensor, padded_window_indices: torch.Tensor) -> torch.Tensor:
        hidden_dim = image_tokens.size(-1)
        image_tokens = image_tokens.unsqueeze(1).expand(-1, padded_window_indices.size(1), -1, -1)
        gather_index = padded_window_indices.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
        return image_tokens.gather(dim=2, index=gather_index)

    @staticmethod
    def _gather_window_attentions(
        image_attentions: torch.Tensor,
        padded_window_indices: torch.Tensor,
        num_heads: int,
        att_dst_len: int,
    ) -> torch.Tensor:
        image_attentions = image_attentions.unsqueeze(1).expand(-1, padded_window_indices.size(1), -1, -1, -1)
        gather_index = padded_window_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, num_heads, -1, att_dst_len)
        return image_attentions.gather(dim=3, index=gather_index)

    def bipartite_matching_padded(
        self,
        hidden_states: torch.Tensor,
        attentions: torch.Tensor,
        idxs: torch.Tensor,
        token_counts: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Batched LightKV matching for padded windows.

        The batch dimension is `num_imgs * num_windows`. `token_counts` contains the number of
        valid tokens in each padded sample.
        """
        device = hidden_states.device
        batch_size, max_tokens, hidden_dim = hidden_states.size()

        src = hidden_states[:, ::2, ...]
        dst = hidden_states[:, 1::2, ...]
        src_part_idxs_global = idxs[:, ::2]
        dst_part_idxs_global = idxs[:, 1::2]

        src_len_max = src.size(1)
        dst_len_max = dst.size(1)
        src_counts = (token_counts + 1) // 2
        dst_counts = token_counts // 2
        r_values = torch.minimum((self.discard_ratio * token_counts).to(torch.long), token_counts // 2)
        unm_counts = src_counts - r_values

        token_pos = torch.arange(max_tokens, device=device).unsqueeze(0)
        src_pos = torch.arange(src_len_max, device=device).unsqueeze(0)
        dst_pos = torch.arange(dst_len_max, device=device).unsqueeze(0)

        valid_token_mask = token_pos < token_counts.unsqueeze(1)
        valid_src_mask = src_pos < src_counts.unsqueeze(1)
        valid_dst_mask = dst_pos < dst_counts.unsqueeze(1)

        norm_src = torch.norm(src, p=2, dim=-1, keepdim=True)
        norm_dst = torch.norm(dst, p=2, dim=-1, keepdim=True)
        sim_scores = torch.bmm(src, rearrange(dst, "b s h -> b h s"))
        sim_scores = sim_scores / (torch.bmm(norm_src, rearrange(norm_dst, "b s h -> b h s")) + 1e-8)
        sim_scores = sim_scores.masked_fill(~(valid_src_mask.unsqueeze(-1) & valid_dst_mask.unsqueeze(1)), float("-inf"))

        node_max, node_idx = sim_scores.max(dim=-1)
        node_max = node_max.masked_fill(~valid_src_mask, float("-inf"))
        edge_idx = node_max.argsort(dim=-1, descending=True)

        max_r = int(r_values.max().item())
        max_unm = int(unm_counts.max().item())

        if max_r > 0:
            src_idxs_in_part = edge_idx[:, :max_r]
            merge_mask = torch.arange(max_r, device=device).unsqueeze(0) < r_values.unsqueeze(1)
            dst_idxs_in_part = node_idx.gather(dim=1, index=src_idxs_in_part)
        else:
            src_idxs_in_part = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            dst_idxs_in_part = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            merge_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)

        if max_unm > 0:
            unm_offsets = torch.arange(max_unm, device=device).unsqueeze(0) + r_values.unsqueeze(1)
            unm_idxs_in_part = edge_idx.gather(dim=1, index=unm_offsets.clamp_max(src_len_max - 1))
            unm_mask = torch.arange(max_unm, device=device).unsqueeze(0) < unm_counts.unsqueeze(1)
        else:
            unm_idxs_in_part = torch.empty(batch_size, 0, device=device, dtype=torch.long)
            unm_mask = torch.zeros(batch_size, 0, device=device, dtype=torch.bool)

        attn_sum = attentions.sum(dim=1).sum(dim=-1)
        attn_sum = attn_sum * valid_token_mask.to(attn_sum.dtype)
        attn_src = attn_sum[:, ::2]
        attn_dst = attn_sum[:, 1::2]

        dst_weights = torch.ones_like(attn_dst, device=device)
        if max_r > 0:
            src_weights = attn_src.gather(dim=1, index=src_idxs_in_part) * merge_mask.to(attn_src.dtype)
            dst_touched = torch.zeros(batch_size, dst_len_max, device=device, dtype=torch.long)
            dst_touched = dst_touched.scatter_add(dim=1, index=dst_idxs_in_part, src=merge_mask.to(torch.long))
            dst_weights = torch.where(dst_touched > 0, attn_dst, dst_weights)
            dst_weights_total = dst_weights.scatter_reduce(dim=1, index=dst_idxs_in_part, src=src_weights, reduce="sum")
            src_weights_total = dst_weights_total.gather(dim=1, index=dst_idxs_in_part)
            src_weights = torch.where(merge_mask, src_weights / src_weights_total.clamp_min(1e-8), torch.zeros_like(src_weights))

            src_merge = src.gather(dim=1, index=src_idxs_in_part.unsqueeze(-1).expand(-1, -1, hidden_dim))
            src_merge = src_merge * src_weights.unsqueeze(-1)
        else:
            dst_weights_total = dst_weights
            src_merge = torch.empty(batch_size, 0, hidden_dim, device=device, dtype=hidden_states.dtype)

        dst_weights = torch.where(valid_dst_mask, dst_weights / dst_weights_total.clamp_min(1e-8), torch.zeros_like(dst_weights))
        dst_merge = dst * dst_weights.unsqueeze(-1)
        if max_r > 0:
            dst_merge = dst_merge.scatter_reduce(
                dim=1,
                index=dst_idxs_in_part.unsqueeze(-1).expand(-1, -1, hidden_dim),
                src=src_merge,
                reduce="sum",
            )

        if max_unm > 0:
            unm = src.gather(dim=1, index=unm_idxs_in_part.unsqueeze(-1).expand(-1, -1, hidden_dim))
        else:
            unm = torch.empty(batch_size, 0, hidden_dim, device=device, dtype=hidden_states.dtype)

        unm_idxs_global = src_part_idxs_global.gather(dim=1, index=unm_idxs_in_part) if max_unm > 0 else src_part_idxs_global[:, :0]
        combined_tokens = torch.cat([unm, dst_merge], dim=1)
        combined_idxs_global = torch.cat([unm_idxs_global, dst_part_idxs_global], dim=1)
        combined_mask = torch.cat([unm_mask, valid_dst_mask], dim=1)

        trace_info = {
            "unm_idxs_in_part": unm_idxs_in_part.unsqueeze(-1),
            "src_idxs_in_part": src_idxs_in_part.unsqueeze(-1),
            "dst_idxs_in_part": dst_idxs_in_part.unsqueeze(-1),
            "r_values": r_values,
            "unm_counts": unm_counts,
        }
        return combined_tokens, combined_idxs_global, combined_mask, trace_info

    @staticmethod
    def _split_trace_info(
        trace_info: Dict[str, torch.Tensor],
        num_imgs: int,
        num_windows: int,
        partition_sizes: List[int],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        all_unm_idxs = rearrange(trace_info["unm_idxs_in_part"], "(b w) s one -> b w s one", b=num_imgs, w=num_windows)
        all_src_idxs = rearrange(trace_info["src_idxs_in_part"], "(b w) s one -> b w s one", b=num_imgs, w=num_windows)
        all_dst_idxs = rearrange(trace_info["dst_idxs_in_part"], "(b w) s one -> b w s one", b=num_imgs, w=num_windows)
        r_values = rearrange(trace_info["r_values"], "(b w) -> b w", b=num_imgs, w=num_windows)
        unm_counts = rearrange(trace_info["unm_counts"], "(b w) -> b w", b=num_imgs, w=num_windows)

        split_unm_idxs = []
        split_src_idxs = []
        split_dst_idxs = []
        for window_idx, _part_size in enumerate(partition_sizes):
            curr_unm = int(unm_counts[0, window_idx].item())
            curr_r = int(r_values[0, window_idx].item())
            split_unm_idxs.append(all_unm_idxs[:, window_idx, :curr_unm, :])
            split_src_idxs.append(all_src_idxs[:, window_idx, :curr_r, :])
            split_dst_idxs.append(all_dst_idxs[:, window_idx, :curr_r, :])

        return split_unm_idxs, split_src_idxs, split_dst_idxs
