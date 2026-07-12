# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _load_cpu_kv_cache_manager(monkeypatch):
    """Import through real package paths after ascend_store mock collection."""
    module_name = "vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_kv_cache_manager"
    package_name = "vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload"
    kv_pool_name = "vllm_ascend.distributed.kv_transfer.kv_pool"
    kv_pool_package = sys.modules.get(kv_pool_name)
    if kv_pool_package is not None:
        kv_pool_path = Path(__file__).resolve().parents[3] / "vllm_ascend/distributed/kv_transfer/kv_pool"
        monkeypatch.setattr(kv_pool_package, "__path__", [str(kv_pool_path)])

    manager_dependency = sys.modules.get("vllm_ascend.core.single_type_kv_cache_manager")
    if manager_dependency is not None and not hasattr(manager_dependency, "get_manager_for_kv_cache_spec"):
        monkeypatch.setattr(
            manager_dependency,
            "get_manager_for_kv_cache_spec",
            MagicMock(),
            raising=False,
        )

    saved_modules = {name: sys.modules.pop(name) for name in (module_name, package_name) if name in sys.modules}
    module = importlib.import_module(module_name)
    for name in (module_name, package_name):
        sys.modules.pop(name, None)
    sys.modules.update(saved_modules)
    return module


@pytest.mark.parametrize(
    ("is_v024", "expected_key", "unexpected_key"),
    [
        (True, "max_num_batched_tokens", "max_in_flight_tokens"),
        (False, "max_in_flight_tokens", "max_num_batched_tokens"),
    ],
)
def test_cpu_kv_cache_manager_uses_versioned_token_budget(
    monkeypatch,
    is_v024,
    expected_key,
    unexpected_key,
):
    module = _load_cpu_kv_cache_manager(monkeypatch)
    kv_cache_spec = SimpleNamespace(block_size=16)

    with (
        patch.object(module, "BlockPool"),
        patch.object(module, "get_manager_for_kv_cache_spec") as get_manager,
        patch.object(module, "vllm_version_is", return_value=is_v024) as version_is,
    ):
        module.CPUKVCacheManager(kv_cache_spec, num_cpu_blocks=4)

    manager_kwargs = get_manager.call_args.kwargs
    assert manager_kwargs[expected_key] == 64
    assert unexpected_key not in manager_kwargs
    assert manager_kwargs["scheduler_block_size"] == 16
    version_is.assert_called_once_with("0.24.0")
