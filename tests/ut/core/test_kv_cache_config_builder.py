# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
# ruff: noqa: E402
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    """Dispatch contract of ``AscendKVCacheConfigBuilder`` (main-only).

Locks the grouping precedence (GLM-Next, Ascend shared-tuple layouts, Kimi-K3
DSpark, then vLLM's generic grouping) and the TurboQuant pool accounting: the
surface most likely to drift between the pluggable builder on main and the
release-lane monkey patches.
"""
else:
    from dataclasses import replace


from types import SimpleNamespace

if vllm_version_is("0.33.0"):
    from unittest.mock import Mock


import pytest

if vllm_version_is("0.33.0"):
    pytest.importorskip(
        "vllm.v1.core.kv_cache_config_builder",
        reason="the pluggable KV cache builder is main-only",
    )


import torch
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder

if not vllm_version_is("0.33.0"):
    from vllm.v1.core.kv_cache_planning import _pool_bytes_per_block_from_builder

if vllm_version_is("0.33.0"):
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
else:
    from vllm.v1.core.single_type_kv_cache_manager import (
        CircularBufferManager,
        FullAttentionManager,
        MambaManager,
        SlidingWindowManager,
    )

from vllm.v1.kv_cache_interface import MambaSpec

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import FullAttentionSpec as AttentionSpec
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as KVCacheGroup
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
else:
    from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheGroup, MLAAttentionSpec

if vllm_version_is("0.33.0"):
    from vllm_ascend.core import kv_cache_config_builder as builder_module

from vllm_ascend.core.kv_cache_config_builder import AscendKVCacheConfigBuilder

if vllm_version_is("0.33.0"):
    from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
else:
    from vllm_ascend.core.kv_cache_interface import (
        AscendIndexerKPoolTailSpec,
        AscendMLAAttentionSpec,
        AscendSFAIndexerCacheSpec,
    )

if not vllm_version_is("0.33.0"):
    from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager
    from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_groups

from vllm_ascend.quantization.methods.kv_cache.turboquant import (
    SLOT_BYTES,
    TURBOQUANT_CACHE_DTYPE,
)

if vllm_version_is("0.33.0"):
    from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

    @pytest.fixture(autouse=True)
    def registered_specs():
        register_all_kvcache_specs(None)
        register_ascend_kv_cache_specs()

    def make_runtime():
        return SimpleNamespace(
            model_config=SimpleNamespace(
                supports_kv_cache_grouping=False, hf_config=SimpleNamespace(model_type="qwen3")
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            attention_config=SimpleNamespace(hisparse_config=None),
        )

    def generic_specs():
        spec = AttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        return {"layer.a": spec, "layer.b": spec}

    def test_kimi_k31_grouping_precedes_generic_hybrid_promotion(monkeypatch):
        runtime = make_runtime()
        runtime.model_config.hf_config.model_type = "kimi_k31"
        runtime.scheduler_config.disable_hybrid_kv_cache_manager = True
        specs = generic_specs()
        paired = Mock(side_effect=ValueError("Paired SFA cache requires the hybrid KV cache manager."))
        promote = Mock()
        monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", paired)
        monkeypatch.setattr(builder_module, "unify_hybrid_kv_cache_specs", promote)
        with pytest.raises(ValueError, match="requires the hybrid KV cache manager"):
            AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
        paired.assert_called_once_with(runtime, specs)
        promote.assert_not_called()

    def test_kimi_k31_builder_uses_paired_grouping_and_pool_accounting(monkeypatch):
        runtime = make_runtime()
        runtime.model_config.hf_config.model_type = "kimi_k31"
        specs, groups = generic_specs(), [Mock()]
        paired = Mock(return_value=groups)
        pool = Mock(return_value=14487552)
        monkeypatch.setattr(builder_module, "paired_sfa_cache_groups", paired)
        monkeypatch.setattr(builder_module, "paired_sfa_pool_bytes", pool)
        builder = AscendKVCacheConfigBuilder()
        assert builder.get_kv_cache_groups(runtime, specs) is groups
        assert builder._get_pool_bytes_per_block(runtime, groups) == 14487552
        paired.assert_called_once_with(runtime, specs)
        pool.assert_called_once_with(groups)

    @pytest.mark.parametrize("draft_sharded", [True, False])
    def test_kimi_grouping_requires_matching_dcp_layout(draft_sharded):
        target = AttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        draft = AttentionSpec(
            block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, dcp_sharded=draft_sharded
        )
        state = MambaSpec(block_size=16, shapes=((32, 16),), dtypes=(torch.bfloat16,))
        assert target.page_size_bytes == draft.page_size_bytes == state.page_size_bytes
        specs = {
            "language_model.model.layers.0.attn": target,
            "language_model.model.layers.1.state": state,
            "model.layers.0.attn": draft,
        }
        groups = builder_module._get_kimi_k3_dspark_mixed_kv_cache_groups(specs)
        if draft_sharded:
            assert groups is not None
            assert set(groups[0].layer_names) == {"language_model.model.layers.0.attn", "model.layers.0.attn"}
        else:
            assert groups is None

    def test_generic_uniform_spec_falls_back_to_default_grouping():
        runtime = make_runtime()
        specs = generic_specs()
        result = AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
        expected = DefaultKVCacheConfigBuilder().get_kv_cache_groups(runtime, specs)
        assert result == expected
        assert len(result) == 1
        assert set(result[0].layer_names) == {"layer.a", "layer.b"}

    def test_glm_next_layout_takes_precedence_over_ascend_grouping(monkeypatch):
        sentinel = [KVCacheGroup(["glm"], generic_specs()["layer.a"])]
        glm_mock = Mock(return_value=sentinel)
        ascend_mock = Mock()
        monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: True)
        monkeypatch.setattr(builder_module, "get_glm5_next_kv_cache_groups", glm_mock)
        monkeypatch.setattr(builder_module, "group_and_unify_kv_cache_specs", ascend_mock)
        assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
        ascend_mock.assert_not_called()

    def test_ascend_layout_takes_precedence_over_kimi_dspark(monkeypatch):
        sentinel = [KVCacheGroup(["tuple"], generic_specs()["layer.a"])]
        ascend_mock = Mock(return_value=sentinel)
        kimi_mock = Mock(return_value=None)
        monkeypatch.setattr(builder_module, "is_glm5_next_cache_spec", lambda spec: False)
        monkeypatch.setattr(builder_module, "is_deepseek_v41_cache", lambda spec: True)
        monkeypatch.setattr(builder_module, "group_cache_specs", Mock(return_value=[]))
        monkeypatch.setattr(builder_module, "make_cache_groups", ascend_mock)
        monkeypatch.setattr(builder_module, "_get_kimi_k3_dspark_mixed_kv_cache_groups", kimi_mock)
        assert AscendKVCacheConfigBuilder().get_kv_cache_groups(make_runtime(), generic_specs()) is sentinel
        kimi_mock.assert_not_called()

    def test_turboquant_groups_use_packed_pool_bytes():
        tq_spec = AscendSlidingWindowMLASpec(
            block_size=16,
            num_kv_heads=1,
            head_size=SLOT_BYTES,
            dtype=torch.uint8,
            sliding_window=128,
            cache_dtype_str=TURBOQUANT_CACHE_DTYPE,
            model_version="deepseek_v4",
        )
        uniform = UniformTypeKVCacheSpecs.from_specs({"l0": tq_spec, "l1": tq_spec})
        groups = [KVCacheGroup(layer_names=["l0", "l1"], kv_cache_spec=uniform)]
        builder = AscendKVCacheConfigBuilder()
        assert tq_cache.uses_turboquant_groups(groups)
        pool_bytes = builder._get_pool_bytes_per_block(make_runtime(), groups)
        assert pool_bytes == tq_cache.pool_bytes_per_block(groups)
        assert pool_bytes >= sum(spec.page_size_bytes for spec in uniform.kv_cache_specs.values())
else:

    @pytest.fixture
    def runtime():
        return SimpleNamespace(
            model_config=SimpleNamespace(
                supports_kv_cache_grouping=False, hf_config=SimpleNamespace(model_type="qwen3")
            ),
            scheduler_config=SimpleNamespace(disable_hybrid_kv_cache_manager=False),
            attention_config=SimpleNamespace(hisparse_config=None),
            cache_config=SimpleNamespace(prefix_cache_retention_interval=None, num_gpu_blocks_override=None),
            kv_transfer_config=None,
        )

    def test_generic_layer_configs_use_default_grouping(runtime):
        spec = AttentionSpec(block_size=16, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        layers = dict.fromkeys(("layer.a", "layer.b"), FullAttentionManager.layer_config(spec))
        expected = DefaultKVCacheConfigBuilder().get_kv_cache_groups(runtime, layers)
        assert AscendKVCacheConfigBuilder().get_kv_cache_groups(runtime, layers) == expected

    def test_deepseek_v4_shared_tuple_offsets(runtime):
        main = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.bfloat16,
            tokens_per_state=4,
            model_version="deepseek_v4",
        )
        index = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=16,
            dtype=torch.bfloat16,
            tokens_per_state=4,
            model_version="deepseek_v4",
        )
        swa = MLAAttentionSpec(
            block_size=32,
            num_kv_heads=1,
            head_size=12,
            dtype=torch.bfloat16,
            model_version="deepseek_v4",
        )
        layers = {
            "main": FullAttentionManager.layer_config(main),
            "index": FullAttentionManager.layer_config(index),
            "swa": SlidingWindowManager.layer_config(swa, sliding_window=64),
            "mtp.swa": SlidingWindowManager.layer_config(swa, sliding_window=64),
        }
        builder = AscendKVCacheConfigBuilder()
        groups = builder.get_kv_cache_groups(runtime, layers)
        config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
        placements = {name: tensor for tensor in config.kv_cache_tensors for name in tensor.layers}
        tuple_bytes = (main.page_size_bytes + index.page_size_bytes) * 3
        assert _pool_bytes_per_block_from_builder(builder, runtime, groups) == 2 * tuple_bytes // 3
        assert {t.size for t in placements.values()} == {2 * tuple_bytes}
        assert placements["main"].offset == 0
        assert placements["index"].offset == placements["swa"].offset == main.page_size_bytes * 3
        assert placements["index"].block_stride == placements["swa"].block_stride == index.page_size_bytes
        assert placements["mtp.swa"].offset == tuple_bytes + placements["swa"].offset

    def test_deepseek_v4_turboquant_places_whole_rows_in_shared_pages(runtime):
        spec = AscendMLAAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=SLOT_BYTES,
            dtype=torch.uint8,
            tokens_per_state=4,
            model_version="deepseek_v4",
            cache_dtype_str=TURBOQUANT_CACHE_DTYPE,
        )
        layer = FullAttentionManager.layer_config(spec)
        builder = AscendKVCacheConfigBuilder()
        groups = builder.get_kv_cache_groups(runtime, {"main": layer})
        stride = _pool_bytes_per_block_from_builder(builder, runtime, groups)
        config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
        (tensor,) = config.kv_cache_tensors
        assert stride >= spec.page_size_bytes
        assert stride % SLOT_BYTES == stride % 256 == 0
        assert (tensor.offset, tensor.block_stride, tensor.size) == (0, stride, 3 * stride)

    @pytest.mark.parametrize("draft_sharded", [False, True])
    def test_kimi_k31_native_draft_config_controls_placement(runtime, draft_sharded):
        runtime.model_config.hf_config.model_type = "kimi_k31"
        main = AscendMLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        index = AscendSFAIndexerCacheSpec(block_size=64, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
        draft = AttentionSpec(
            block_size=64, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, dcp_sharded=draft_sharded
        )
        main_name = "model.layers.0.attn"
        index_name = "model.layers.0.indexer.k_cache"
        draft_name = "model.layers.1.attn"
        layers = {
            main_name: FullAttentionManager.layer_config(main),
            index_name: FullAttentionManager.layer_config(index),
            draft_name: replace(FullAttentionManager.layer_config(draft), is_draft=True),
        }
        builder = AscendKVCacheConfigBuilder()
        groups = builder.get_kv_cache_groups(runtime, layers)
        assert len(groups) == (1 if draft_sharded else 2)
        assert groups[0].layer_names == [main_name, index_name] + ([draft_name] if draft_sharded else [])
        config = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
        placements = {name: tensor for tensor in config.kv_cache_tensors for name in tensor.layers}
        assert placements[draft_name].offset == 3 * (main.page_size_bytes + index.page_size_bytes)
        assert placements[draft_name].size == placements[draft_name].offset + 3 * draft.page_size_bytes

    def test_glm_next_groups_carry_ring_policy_and_share_paired_pages(runtime):
        main = AscendMLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=32, dtype=torch.bfloat16, model_version="glm5_next"
        )
        index = replace(main, head_size=8, tokens_per_state=4)
        tail = AscendIndexerKPoolTailSpec(
            block_size=4,
            ring_capacity=4,
            num_kv_heads=1,
            head_size=8,
            dtype=torch.float32,
            compress_ratio=4,
        )
        main_name = "model.layers.0.attn"
        index_name = "model.layers.0.indexer.k_cache"
        tail_name = "model.layers.0.indexer.tail_cache"
        layers = {
            main_name: FullAttentionManager.layer_config(main),
            index_name: FullAttentionManager.layer_config(index),
            tail_name: KpoolTailManager.layer_config(tail),
        }
        builder = AscendKVCacheConfigBuilder()
        full, ring = groups = builder.get_kv_cache_groups(runtime, layers)
        assert full.layer_names == [main_name, index_name]
        assert ring.manager_cls is KpoolTailManager
        assert ring.layer_names == [tail_name]
        plan = builder.get_kv_cache_config_from_groups(runtime, groups, num_blocks=3)
        assert any(tensor.layers == [index_name, tail_name] for tensor in plan.kv_cache_tensors)

    def test_qwen4_composite_group_uses_configs_and_preserves_recurrent_checkpoint_policy(runtime):
        main = AttentionSpec(block_size=128, num_kv_heads=1, head_size=16, dtype=torch.bfloat16)
        compressed = MLAAttentionSpec(
            block_size=128, num_kv_heads=1, head_size=16, dtype=torch.bfloat16, tokens_per_state=4
        )
        raw = replace(main, block_size=4, head_size_v=0)
        state = MambaSpec(block_size=128, shapes=((4, 8), (8, 8)), dtypes=(torch.bfloat16, torch.float32))
        main_name = "model.layers.0.attn"
        compressed_name = "model.layers.0.indexer.compressed_key_cache"
        raw_name = "model.layers.0.indexer.raw_key_cache"
        layers = {
            main_name: FullAttentionManager.layer_config(main),
            compressed_name: FullAttentionManager.layer_config(compressed),
            raw_name: CircularBufferManager.layer_config(raw),
            "model.layers.1.linear_attn": MambaManager.layer_config(state, mamba_cache_mode="align"),
            "model.layers.2.linear_attn": MambaManager.layer_config(
                state, mamba_cache_mode="align", num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=128
            ),
        }

        def group_by_layer(_config, configs):
            return [KVCacheGroup.from_layers({name: layer}) for name, layer in configs.items()]

        groups = get_qwen4_exp_kv_cache_groups(runtime, layers, group_by_layer)
        assert [group.layer_names for group in groups] == [
            [main_name, compressed_name],
            [raw_name],
            ["model.layers.1.linear_attn"],
            ["model.layers.2.linear_attn"],
        ]
        assert groups[3].manager_config == layers["model.layers.2.linear_attn"].manager_config
