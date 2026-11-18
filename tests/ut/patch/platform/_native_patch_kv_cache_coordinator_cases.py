# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native grouping retains Ascend's producer/standalone EAGLE policy."""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, MambaManager
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheGroup, MambaSpec

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import AscendHybridKVCacheCoordinator


@pytest.mark.parametrize("role,skips", [(None, True), ("producer", True), ("consumer", False), ("both", False)])
def test_native_constructor_retains_role_policy_and_full_attention_fallback(monkeypatch, role, skips):
    attention = AttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
    state = MambaSpec(block_size=16, shapes=((1,),), dtypes=(torch.float32,))
    groups = [
        KVCacheGroup.from_layers({"attention": FullAttentionManager.layer_config(attention)}),
        KVCacheGroup.from_layers({"state": MambaManager.layer_config(state, mamba_cache_mode="align")}),
    ]
    config = KVCacheConfig(num_blocks=16, kv_cache_tensors=[], kv_cache_groups=groups)
    config.kv_transfer_config = (
        None
        if role is None
        else SimpleNamespace(is_kv_producer=role in ("producer", "both"), is_kv_consumer=role in ("consumer", "both"))
    )
    calls = []

    def initialize(self, native_config, *args):
        calls.append((native_config, args))
        self.kv_cache_config = native_config
        self.eagle_group_ids = {i for i, g in enumerate(native_config.kv_cache_groups) if g.is_eagle_group}
        self.drop_eagle_group_ids = self.eagle_group_ids.copy()
        # Exercise the same native class identity used by layer configs.
        self.single_type_managers = (SimpleNamespace(), MambaManager.__new__(MambaManager))

    base = AscendHybridKVCacheCoordinator.__bases__[0]
    monkeypatch.setattr(base, "__init__", initialize)
    regrouped = []
    monkeypatch.setattr(
        AscendHybridKVCacheCoordinator, "verify_and_split_kv_cache_groups", lambda self: regrouped.append(self)
    )
    coordinator = AscendHybridKVCacheCoordinator(config, 256, 16, True, True, False, 1, 1, 16, 16)

    assert len(calls) == 1
    native_config, args = calls[0]
    assert native_config is not config
    assert native_config.kv_cache_groups[0].manager_config == groups[0].manager_config
    assert native_config.kv_cache_groups[1].manager_config == groups[1].manager_config
    assert [g.is_eagle_group for g in native_config.kv_cache_groups] == [True, False]
    assert [g.is_eagle_group for g in config.kv_cache_groups] == [False, False]
    assert args[:5] == (256, 16, True, True, False)
    assert coordinator.skips_eagle_block_drop is skips
    assert coordinator.drop_eagle_group_ids == (set() if skips else {0})
    assert bool(regrouped) is skips
    assert coordinator.single_type_managers[1].drop_eagle_checkpoint_block is False
