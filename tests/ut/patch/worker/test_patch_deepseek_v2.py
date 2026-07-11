# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm_ascend.patch.worker.patch_deepseek_v2 import (
    _resolve_mla_skip_topk,
    _should_skip_indexer_init,
)
from vllm_ascend.utils import vllm_version_is


def _config(**overrides) -> SimpleNamespace:
    values = {"num_hidden_layers": 80}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_glm51_skip_topk_keeps_per_layer_indexer():
    assert not _should_skip_indexer_init(
        _config(),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_glm52_shared_layer_skips_indexer_init():
    assert _should_skip_indexer_init(
        _config(indexer_types=["full", "full", "shared"]),
        "model.layers.2.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_keeps_indexer():
    indexer_types = ["full"] * 80 + ["shared"]
    assert not _should_skip_indexer_init(
        _config(indexer_types=indexer_types),
        "model.layers.80.self_attn",
        skip_topk=True,
    )


def test_mtp_layer_skip_topk_matches_installed_vllm_contract():
    assert _resolve_mla_skip_topk(True, is_mtp_layer=False)
    assert _resolve_mla_skip_topk(False, is_mtp_layer=True) is False
    assert _resolve_mla_skip_topk(True, is_mtp_layer=True) is vllm_version_is("0.23.0")
