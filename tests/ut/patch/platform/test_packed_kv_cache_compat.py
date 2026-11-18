# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from __future__ import annotations

from types import SimpleNamespace

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from unittest.mock import MagicMock, patch
else:
    from unittest.mock import Mock

if vllm_version_is("0.33.0"):
    from vllm_ascend.patch.platform import patch_kv_cache_utils as cache_patch
else:
    from vllm_ascend.core import kv_cache_config_builder as builder_module


def _release_test_packed_groups_leave_eagle_annotation_to_upstream():
    """Both supported planners annotate EAGLE groups in get_kv_cache_configs."""
    config = SimpleNamespace()
    specs = {"layer": object()}
    groups = [object()]
    annotator = MagicMock()
    warning = MagicMock()
    upstream = SimpleNamespace(
        _annotate_eagle_groups=annotator,
        _warn_if_unannotated_eagle_mamba=warning,
        _uses_trailing_mtp_layers=MagicMock(),
    )
    with (
        patch.object(cache_patch, "_kv_cache_planning", upstream),
        patch.object(cache_patch, "is_deepseek_v41_cache", return_value=True),
        patch.object(cache_patch, "group_cache_specs", return_value=specs),
        patch.object(cache_patch, "make_cache_groups", return_value=groups),
    ):
        assert cache_patch._ascend_get_packed_kv_cache_groups(config, specs) is groups

    annotator.assert_not_called()
    warning.assert_not_called()


def test_packed_groups_leave_eagle_annotation_to_upstream(monkeypatch):
    config = SimpleNamespace()
    layers = {"layer": SimpleNamespace(spec=object())}
    groups = [object()]
    group_layers = Mock(return_value=groups)
    monkeypatch.setattr(builder_module, "is_deepseek_v41_cache", lambda _: True)
    monkeypatch.setattr(builder_module, "group_layer_configs", group_layers)

    assert builder_module.AscendKVCacheConfigBuilder().get_kv_cache_groups(config, layers) is groups
    group_layers.assert_called_once_with(layers)


if vllm_version_is("0.33.0"):
    test_packed_groups_leave_eagle_annotation_to_upstream = (
        _release_test_packed_groups_leave_eagle_annotation_to_upstream  # type: ignore[assignment, misc]
    )
