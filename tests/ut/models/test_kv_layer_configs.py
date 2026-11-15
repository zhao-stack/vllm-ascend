# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom cache declarations retain Ascend geometry through native planning."""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.models.deepseek_v4.compressor import AscendCompressorStateCache
from vllm_ascend.models.deepseek_v4.indexer import AscendDeepseekV4IndexerCache
from vllm_ascend.models.deepseek_v4.model import AscendDeepseekV4SWACache
from vllm_ascend.models.layer.attention import layer as layer_module
from vllm_ascend.models.layer.attention.layer import DSAAttention
from vllm_ascend.models.minimax_m3.minimax_m3 import MiniMaxM3SparseAttention
from vllm_ascend.models.minimax_m3.msa_m3 import AscendMiniMaxM3IndexerCache
from vllm_ascend.utils import vllm_version_is


@pytest.mark.parametrize(
    "cache_cls,attrs,sliding_window",
    [
        (
            AscendCompressorStateCache,
            dict(state_dim=512, compress_ratio=4, block_size=8, sliding_window=8, dtype=torch.float32),
            8,
        ),
        (AscendDeepseekV4IndexerCache, dict(head_dim=128, compress_ratio=4, dtype=torch.bfloat16), None),
        (
            AscendDeepseekV4SWACache,
            dict(head_dim=512, window_size=128, block_size=128, dtype=torch.bfloat16),
            128,
        ),
        (DSAAttention, dict(compress_ratio=4, kv_cache_dtype="auto", head_size=128, n_local_heads=4), None),
        (AscendMiniMaxM3IndexerCache, dict(head_dim=128, dtype=torch.bfloat16, kv_cache_dtype="auto"), None),
        (
            MiniMaxM3SparseAttention,
            dict(num_kv_heads=4, head_dim=128, kv_cache_torch_dtype=torch.bfloat16, kv_cache_dtype="auto"),
            None,
        ),
    ],
)
def test_custom_cache_declaration_preserves_storage_and_policy(monkeypatch, cache_cls, attrs, sliding_window):
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"), model_config=SimpleNamespace()
    )
    monkeypatch.setattr(layer_module, "is_turboquant", lambda config: False)
    monkeypatch.setattr(layer_module, "is_a5_bf16_kv_enabled", lambda config: False)
    monkeypatch.setattr(layer_module, "kv_cache_dtype_str_to_dtype", lambda *args: torch.bfloat16)
    # Projection and checkpoint construction are irrelevant to cache declarations.
    cache = cache_cls.__new__(cache_cls)
    torch.nn.Module.__init__(cache)
    cache.cache_config = config.cache_config
    for name, value in attrs.items():
        setattr(cache, name, value)
    storage = cache.get_kv_cache_spec(config)
    assert storage.page_size_bytes > 0
    if vllm_version_is("0.33.0"):
        if sliding_window is not None:
            assert storage.sliding_window == sliding_window
        return

    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
    from vllm.v1.kv_cache_interface import KVCacheGroup

    layer_config = cache.get_kv_cache_config(config)
    assert layer_config.spec == storage
    group = KVCacheGroup.from_layers({"cache": layer_config})
    assert group.page_size_bytes == storage.page_size_bytes
    assert group.kv_cache_specs["cache"] == storage
    assert group.manager_cls is (FullAttentionManager if sliding_window is None else SlidingWindowManager)
    if sliding_window is not None:
        assert layer_config.manager_config.params.sliding_window == sliding_window


def test_uncompressed_dsa_has_no_separate_cache():
    cache = DSAAttention.__new__(DSAAttention)
    torch.nn.Module.__init__(cache)
    cache.compress_ratio = 1
    config = SimpleNamespace()
    assert cache.get_kv_cache_spec(config) is None
    if not vllm_version_is("0.33.0"):
        assert cache.get_kv_cache_config(config) is None
