# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_kv_cache_manager import (
    CPUKVCacheManager,
)


@pytest.mark.parametrize(
    ("is_v024", "expected_key", "unexpected_key"),
    [
        (True, "max_num_batched_tokens", "max_in_flight_tokens"),
        (False, "max_in_flight_tokens", "max_num_batched_tokens"),
    ],
)
def test_cpu_kv_cache_manager_uses_versioned_token_budget(is_v024, expected_key, unexpected_key):
    module = "vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_kv_cache_manager"
    kv_cache_spec = SimpleNamespace(block_size=16)

    with (
        patch(f"{module}.BlockPool"),
        patch(f"{module}.get_manager_for_kv_cache_spec") as get_manager,
        patch(f"{module}.vllm_version_is", return_value=is_v024) as version_is,
    ):
        CPUKVCacheManager(kv_cache_spec, num_cpu_blocks=4)

    manager_kwargs = get_manager.call_args.kwargs
    assert manager_kwargs[expected_key] == 64
    assert unexpected_key not in manager_kwargs
    assert manager_kwargs["scheduler_block_size"] == 16
    version_is.assert_called_once_with("0.24.0")
