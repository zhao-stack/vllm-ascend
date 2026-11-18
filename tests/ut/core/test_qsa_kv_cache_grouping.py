from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import torch

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.33.0"):
    from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import FullAttentionSpec as AttentionSpec
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as KVCacheGroup
    from vllm.v1.kv_cache_interface import KVCacheSpec
else:
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, KVCacheLayerConfig

if vllm_version_is("0.33.0"):
    from vllm_ascend.patch.platform import patch_kv_cache_utils
else:
    from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder


def _release__ordinary_specs() -> dict[str, KVCacheSpec]:
    ordinary = AttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": ordinary}


def _ordinary_specs() -> dict[str, KVCacheLayerConfig]:
    ordinary = AttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=64,
        head_size_v=64,
        dtype=torch.bfloat16,
    )
    return {"model.layers.0.self_attn": FullAttentionManager.layer_config(ordinary)}


if vllm_version_is("0.33.0"):
    _ordinary_specs = _release__ordinary_specs  # type: ignore[assignment, misc]


def test_non_qsa_grouping_delegates_to_upstream_vllm(monkeypatch) -> None:
    specs = _ordinary_specs()
    vllm_config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")))
    expected = [Mock(spec=KVCacheGroup)]
    upstream_grouping = Mock(return_value=expected)
    monkeypatch.setattr(
        patch_kv_cache_utils if vllm_version_is("0.33.0") else DefaultKVCacheConfigBuilder,
        "_orig_get_kv_cache_groups" if vllm_version_is("0.33.0") else "get_kv_cache_groups",
        upstream_grouping,
    )

    actual = (
        patch_kv_cache_utils._get_ascend_kv_cache_groups(vllm_config, specs)
        if vllm_version_is("0.33.0")
        else AscendKVCacheConfigBuilder().get_kv_cache_groups(vllm_config, specs)
    )

    assert actual is expected
    upstream_grouping.assert_called_once_with(vllm_config, specs)


def test_non_qsa_config_uses_upstream_vllm_planner(monkeypatch) -> None:
    if vllm_version_is("0.33.0"):
        ordinary = next(iter(_ordinary_specs().values()))

    groups = [
        KVCacheGroup(layer_names=["model.layers.0.self_attn"], kv_cache_spec=ordinary)
        if vllm_version_is("0.33.0")
        else KVCacheGroup.from_layers(_ordinary_specs())
    ]
    expected = SimpleNamespace()
    upstream_planner = Mock(return_value=expected)
    if vllm_version_is("0.33.0"):
        monkeypatch.setattr(
            patch_kv_cache_utils,
            "_orig_get_kv_cache_config_from_groups",
            upstream_planner,
        )
        monkeypatch.setattr(
            patch_kv_cache_utils,
            "_is_deepseek_v4_groups",
            lambda _: False,
        )

    monkeypatch.setattr(
        patch_kv_cache_utils if vllm_version_is("0.33.0") else DefaultKVCacheConfigBuilder,
        "_get_glm5_next_cache_layout" if vllm_version_is("0.33.0") else "get_kv_cache_config_from_groups",
        (lambda _: None) if vllm_version_is("0.33.0") else upstream_planner,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen3")),
        kv_transfer_config=None,
    )

    actual = (
        patch_kv_cache_utils._ascend_get_kv_cache_config_from_groups(
            vllm_config, groups, **cast(dict[str, Any], {"available_memory": 123456})
        )
        if vllm_version_is("0.33.0")
        else AscendKVCacheConfigBuilder().get_kv_cache_config_from_groups(
            vllm_config, groups, **cast(dict[str, Any], {"num_blocks": 123})
        )
    )

    assert actual is expected
    upstream_planner.assert_called_once_with(vllm_config, groups, 123456)
    assert actual.kv_transfer_config is None
