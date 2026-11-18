# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ascend allocation notifications for circular state sharing physical pages."""

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from vllm.v1.core.block_pool import BlockPool

from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import KVCacheSpec


class AscendCircularBufferManager(CircularBufferManager):
    if vllm_version_is("0.33.0"):

        def __init__(
            self,
            kv_cache_spec: KVCacheSpec,
            block_pool: BlockPool,
            enable_caching: bool,
            kv_cache_group_id: int,
            scheduler_block_size: int,
            dcp_world_size: int = 1,
            pcp_world_size: int = 1,
            needs_kv_cache_zeroing: bool = False,
            max_admission_blocks_per_request: int | None = None,
        ) -> None:
            super().__init__(
                kv_cache_spec,
                block_pool,
                enable_caching,
                kv_cache_group_id,
                scheduler_block_size,
                dcp_world_size,
                pcp_world_size,
                needs_kv_cache_zeroing,
                max_admission_blocks_per_request,
            )
            # Ascend zeroes every physical backing of a recycled block ID. Ring
            # allocations must report IDs too, even when the last owner was an
            # attention group. The upstream ring manager excludes these IDs.
            self._record_new_block_ids = needs_kv_cache_zeroing
    else:
        zeroes_new_blocks = True
