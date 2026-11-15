# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend storage geometry for the explicit native KV allocation-policy API.

Allocation, prefix-cache and replay policy is declared by each layer's manager
config. These specs only describe the tensor pages consumed by Ascend kernels.
"""

from dataclasses import dataclass, replace

import torch
from typing_extensions import Self
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import AttentionSpec, MLAAttentionSpec

from vllm_ascend.core.kv_cache_interface import get_kv_cache_compression_ratio, get_storage_block_size


@dataclass(frozen=True, kw_only=True)
class AscendMLAAttentionSpec(MLAAttentionSpec):
    """MLA cache spec with Ascend-specific layout metadata.

    For SFA, this spec describes only the main MLA cache. The indexer K
    tensor, its quantization scale, and DCP replication are described by a
    separate :class:`AscendSFAIndexerCacheSpec`.
    """

    max_tp_shards: int | None = 1
    num_heads: int = 1
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_sfa_c8: bool = False
    cache_sparse_sfa_turboquant: bool = False
    store_on_host: bool = False
    indexes_kv_by_block_stride: bool = False
    block_table_block_size: int | None = None

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(
            super().copy_with_new_block_size(block_size),
            block_table_block_size=self.block_table_block_size or self.block_size,
        )

    @property
    def uses_packed_sfa_main_cache(self) -> bool:
        """Whether the SFA main cache is stored in one packed tensor."""
        return self.cache_sparse_sfa_c8 or self.cache_sparse_sfa_turboquant

    @property
    def supports_single_raw_backing(self) -> bool:
        """Whether this geometry can be exposed from one raw MLA backing."""
        return (
            get_kv_cache_compression_ratio(self) == 1
            and self.model_version is None
            and (not self.indexes_kv_by_block_stride)
            and (not self.cache_sparse_sfa_c8)
        )

    @property
    def real_page_size_bytes(self) -> int:
        return (
            get_storage_block_size(self)
            * self.num_kv_heads
            * (self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype))
        )

    @property
    def unpadded_page_size_bytes(self) -> int:
        return self.real_page_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSFAIndexerCacheSpec(MLAAttentionSpec):
    """KV cache spec for SFA indexer K/scale cache.

    The full-attention manager lets it share block ids with MLA layers having
    the same manager policy and logical block size. The
    indexer retains its own physical cache region, separate from the MLA cache.
    """

    max_tp_shards: int | None = 1
    scale_dim: int = 0
    scale_dtype: torch.dtype = torch.int8
    cache_sparse_li_c8: bool = False
    cache_sparse_li_c4: bool = False
    cache_dtype_str: str | None = None
    sfa_dcp_replicated_indexer_size: int = 1
    block_table_block_size: int | None = None

    def copy_with_new_block_size(self, block_size: int) -> Self:
        return replace(
            super().copy_with_new_block_size(block_size),
            block_table_block_size=self.block_table_block_size or self.block_size,
        )

    @property
    def state_content_size_bytes(self) -> int:
        return self.sfa_dcp_replicated_indexer_size * (
            self.head_size * get_dtype_size(self.dtype) + self.scale_dim * get_dtype_size(self.scale_dtype)
        )

    @property
    def page_size_bytes(self) -> int:
        return self.real_page_size_bytes

    @property
    def real_page_size_bytes(self) -> int:
        num_heads_per_page = self.block_size * self.num_kv_heads
        return num_heads_per_page * self.state_content_size_bytes


@dataclass(frozen=True, kw_only=True)
class AscendSlidingWindowMLASpec(MLAAttentionSpec):
    """Sliding window attention with MLA cache format."""

    sliding_window: int
    extra_retained_tokens: int = 0
    cache_dtype_str: str | None = None
    alignment: int | None = None
    compress_ratio: int = 1
    model_version: str | None = None
    indexes_kv_by_block_stride: bool = False
    bounded_replay: bool = False

    def __post_init__(self):
        # Ascend windows use their declared page padding, not MLA's automatic
        # alignment rounding. Preserve the layout consumed by Ascend kernels.
        AttentionSpec.__post_init__(self)

    @property
    def storage_block_size(self) -> int:
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        return self.storage_block_size * self.num_kv_heads * self.head_size * get_dtype_size(self.dtype)


@dataclass(frozen=True, kw_only=True)
class AscendIndexerKPoolTailSpec(AttentionSpec):
    """One fixed FP32 ``[ring_capacity, 2 * head_size]`` page per request.

    ``block_size`` is the logical pool size (``compress_ratio``), while
    ``ring_capacity`` also retains speculative rows. The dedicated manager
    owns one complete ring per request and never slides or grows it.
    """

    sliding_window: int
    compress_ratio: int
    ring_capacity: int
    dcp_sharded: bool = False
    model_version: str = "glm5_next"
    cache_role: str = "indexer_tail"
    indexes_kv_by_block_stride: bool = True

    @property
    def storage_block_size(self) -> int:
        return self.ring_capacity

    def get_num_kernel_states(self, kernel_block_size: int) -> int:
        return self.ring_capacity

    @property
    def unpadded_page_size_bytes(self) -> int:
        return 2 * self.ring_capacity * self.head_size * get_dtype_size(self.dtype)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.dtype != torch.float32:
            raise ValueError(f"Indexer K-pool tail must use FP32, got {self.dtype}.")
        if self.num_kv_heads != 1 or self.head_size <= 0:
            raise ValueError("Indexer K-pool tail requires one K/gate pair with positive head size.")
        if self.compress_ratio <= 1 or self.ring_capacity < self.compress_ratio:
            raise ValueError("Tail ring capacity must be at least the compression ratio, which must exceed one.")
        if self.block_size != self.compress_ratio:
            raise ValueError("Tail logical block size must equal the compression ratio.")
        if self.sliding_window != self.compress_ratio:
            raise ValueError("Tail sliding_window describes the logical pool size and must equal compress_ratio.")


@dataclass(frozen=True, kw_only=True)
class AscendCircularBufferSpec(AttentionSpec):
    """Circular state whose physical pages participate in Ascend KV zeroing."""
