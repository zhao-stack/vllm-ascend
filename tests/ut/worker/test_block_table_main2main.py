# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_ascend.worker.block_table as block_table_module
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.block_table import (
    BlockTable,
    get_max_num_blocks_per_req,
    is_encoder_only_kv_cache_spec,
)


def _group(world_size: int, rank: int = 0) -> MagicMock:
    group = MagicMock()
    group.world_size = world_size
    group.rank_in_group = rank
    return group


def test_mamba_block_table_row_length_matches_installed_contract():
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
        num_speculative_blocks=2,
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
        ),
    )

    with patch.object(block_table_module, "get_total_cp_world_size", return_value=4):
        row_length = get_max_num_blocks_per_req(spec, config, max_model_len=1024)

    with (
        patch.object(block_table_module, "get_dcp_group", return_value=_group(2)),
        patch.object(block_table_module, "get_pcp_group", return_value=_group(2)),
    ):
        block_table = BlockTable(
            block_size=16,
            max_num_reqs=1,
            max_num_blocks_per_req=row_length,
            max_num_batched_tokens=8,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[0],
            kv_cache_group=KVCacheGroupSpec(
                layer_names=["mamba"],
                kv_cache_spec=spec,
            ),
        )

    if vllm_version_is("0.23.0"):
        assert row_length == 18
        assert block_table.max_num_blocks_per_req == 72
    else:
        assert row_length == spec.max_num_blocks_per_req(config, 1024)
        assert block_table.max_num_blocks_per_req == row_length


def test_uniform_mamba_group_detection_matches_installed_contract():
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float32,),
        mamba_cache_mode="none",
    )
    group = KVCacheGroupSpec(
        layer_names=["mamba"],
        kv_cache_spec=UniformTypeKVCacheSpecs(
            block_size=16,
            kv_cache_specs={"mamba": spec},
        ),
    )
    config = SimpleNamespace(
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2,
            prefill_context_parallel_size=2,
        ),
    )

    with patch.object(block_table_module, "get_total_cp_world_size", return_value=4):
        row_length = get_max_num_blocks_per_req(
            group.kv_cache_spec,
            config,
            max_model_len=1024,
        )

    with (
        patch.object(block_table_module, "get_dcp_group", return_value=_group(1)),
        patch.object(block_table_module, "get_pcp_group", return_value=_group(1)),
    ):
        block_table = BlockTable(
            block_size=16,
            max_num_reqs=1,
            max_num_blocks_per_req=1,
            max_num_batched_tokens=1,
            pin_memory=False,
            device=torch.device("cpu"),
            kernel_sizes=[0],
            kv_cache_group=group,
        )

    assert block_table.is_mamba_group is (not vllm_version_is("0.23.0"))
    expected_row_length = 16 if vllm_version_is("0.23.0") else 1
    assert row_length == expected_row_length


def test_uniform_encoder_only_group_detection_matches_installed_contract():
    spec = EncoderOnlyAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
    )
    uniform_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={"encoder": spec},
    )

    assert is_encoder_only_kv_cache_spec(uniform_spec) is (not vllm_version_is("0.23.0"))


def test_slot_mapping_kernel_kwargs_match_installed_vllm(monkeypatch):
    monkeypatch.setattr(block_table_module, "get_dcp_group", lambda: _group(1))
    monkeypatch.setattr(block_table_module, "get_pcp_group", lambda: _group(1))
    kernel = MagicMock()
    monkeypatch.setattr(block_table_module, "_compute_slot_mapping_kernel", kernel)

    block_table = BlockTable(
        block_size=32,
        max_num_reqs=1,
        max_num_blocks_per_req=4,
        max_num_batched_tokens=8,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_sizes=[16],
    )
    block_table.compute_slot_mapping(
        1,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int64),
    )

    kwargs = kernel.__getitem__.return_value.call_args.kwargs
    if vllm_version_is("0.23.0"):
        assert "KV_CACHE_BLOCK_SIZE" not in kwargs
        assert "BLOCKS_PER_KV_BLOCK" not in kwargs
    else:
        assert kwargs["KV_CACHE_BLOCK_SIZE"] == 32
        assert kwargs["BLOCKS_PER_KV_BLOCK"] == 2
