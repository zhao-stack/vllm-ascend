# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from vllm_ascend.utils import vllm_version_is

if not vllm_version_is("0.33.0"):
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import FullAttentionSpec as AttentionSpec
    from vllm.v1.kv_cache_interface import SlidingWindowSpec
else:
    from vllm.v1.kv_cache_interface import AttentionSpec


from tests.ut.kvpp_utils import make_dspark_kvpp_case
from vllm_ascend.core.kv_cache_placement import create_kvpp_cache_allocation_plan
from vllm_ascend.worker import worker


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("draft_block_size", [1, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_worker_returns_the_specs_used_for_kvpp_budget(monkeypatch, rank, draft_block_size, enabled):
    config, specs, drafts = make_dspark_kvpp_case()
    full_plan = create_kvpp_cache_allocation_plan(config, specs, rank)
    config.additional_config["enable_kvpp"] = enabled
    if not vllm_version_is("0.33.0"):
        layers = {name: FullAttentionManager.layer_config(spec) for name, spec in specs.items()}

    for name in drafts:
        if vllm_version_is("0.33.0"):
            specs[name] = SlidingWindowSpec(
                block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16, sliding_window=4
            )
        else:
            layers[name] = SlidingWindowManager.layer_config(
                AttentionSpec(block_size=draft_block_size, num_kv_heads=2, head_size=8, dtype=torch.float16),
                sliding_window=4,
            )

    instance = SimpleNamespace(
        vllm_config=config,
        model_runner=SimpleNamespace(
            **cast(
                dict[str, Any],
                {"get_kv_cache_spec": lambda: specs}
                if vllm_version_is("0.33.0")
                else {"get_kv_cache_layer_configs": lambda: layers.copy()},
            ),
            speculator=SimpleNamespace(draft_attn_layer_names=set(drafts)),
        ),
        _kvpp_cache_allocation_plan=None,
    )
    monkeypatch.setattr(worker, "get_layerwise_reuse_config", lambda _, *, vllm_config: None)
    monkeypatch.setattr(worker, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank))
    monkeypatch.setattr(worker, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(worker, "get_pcp_group", lambda: SimpleNamespace(rank_in_group=0))
    monkeypatch.setattr(
        worker, "get_ascend_config", lambda: SimpleNamespace(sparse_kv_offload_config=SimpleNamespace(enabled=False))
    )
    returned = (
        worker.NPUWorker.get_kv_cache_spec(instance)
        if vllm_version_is("0.33.0")
        else worker.NPUWorker.get_kv_cache_layer_configs(instance)
    )
    for name in drafts:
        assert (
            isinstance(specs[name], SlidingWindowSpec)
            if vllm_version_is("0.33.0")
            else layers[name].manager_cls is SlidingWindowManager
        )
        assert (specs[name] if vllm_version_is("0.33.0") else layers[name].spec).block_size == draft_block_size
        assert (specs[name] if vllm_version_is("0.33.0") else layers[name].manager_config.params).sliding_window == 4
    if not enabled:
        assert returned is specs if vllm_version_is("0.33.0") else returned == layers
        assert instance._kvpp_cache_allocation_plan is None
        return
    plan = instance._kvpp_cache_allocation_plan
    assert (
        returned is not specs
        if vllm_version_is("0.33.0")
        else plan.logical_cache_spec == {name: owner.spec for name, owner in returned.items()}
    )
    assert (plan.logical_cache_spec if vllm_version_is("0.33.0") else plan.layer_owner_ranks) == (
        returned if vllm_version_is("0.33.0") else full_plan.layer_owner_ranks
    )
    assert (plan.layer_owner_ranks if vllm_version_is("0.33.0") else plan.tensor_sizes) == (
        full_plan.layer_owner_ranks if vllm_version_is("0.33.0") else full_plan.tensor_sizes
    )
    assert (plan.tensor_sizes if vllm_version_is("0.33.0") else plan.get_num_blocks(8192)) == (
        full_plan.tensor_sizes if vllm_version_is("0.33.0") else full_plan.get_num_blocks(8192)
    )
    if vllm_version_is("0.33.0"):
        assert plan.get_num_blocks(8192) == full_plan.get_num_blocks(8192)
        for name, spec in returned.items():
            if name in drafts:
                assert isinstance(spec, AttentionSpec)
                assert spec.block_size == 2
                assert name not in plan.layer_owner_ranks
            else:
                assert spec is specs[name]
    else:
        for name, owner in returned.items():
            spec = owner.spec
            if name in drafts:
                assert isinstance(spec, AttentionSpec)
                assert spec.block_size == 2
                assert name not in plan.layer_owner_ranks
            else:
                assert spec is specs[name]
