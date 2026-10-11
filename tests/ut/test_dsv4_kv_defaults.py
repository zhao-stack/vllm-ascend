# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config.attention import AttentionConfig
from vllm.config.cache import CacheConfig
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype

from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile
from vllm_ascend.platform import NPUPlatform


@pytest.mark.parametrize("device_type", [AscendDeviceType.A2, AscendDeviceType.A3, AscendDeviceType.A5])
@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v3", "qwen3"])
@pytest.mark.parametrize(
    "main,index", [("auto", "auto"), ("auto", "fp8"), ("fp8", "auto"), ("bfloat16", "auto"), ("bfloat16", "fp8")]
)
def test_dsv4_kv_defaults_are_resolved_before_cache_consumers(device_type, model_type, main, index):
    config = SimpleNamespace(
        device_config=SimpleNamespace(device_type="npu"),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type=model_type), dtype=torch.bfloat16),
        cache_config=CacheConfig(cache_dtype=main),
        attention_config=AttentionConfig(indexer_kv_dtype=index),
    )
    expected_main, expected_index = main, index
    if device_type == AscendDeviceType.A5 and model_type == "deepseek_v4":
        expected_main = "fp8" if main == "auto" else main
        expected_index = "fp8"

    # Stop at the next real consumer: all worker/cache planning happens later.
    class Resolved(Exception):
        pass

    def inspect_resolved(actual):
        assert actual.cache_config.cache_dtype == expected_main
        assert actual.attention_config.indexer_kv_dtype == expected_index
        main_dtype = kv_cache_dtype_str_to_dtype(actual.cache_config.cache_dtype, actual.model_config)
        index_dtype = kv_cache_dtype_str_to_dtype(actual.attention_config.indexer_kv_dtype, actual.model_config)
        assert main_dtype == (torch.float8_e4m3fn if expected_main == "fp8" else torch.bfloat16)
        assert index_dtype == (torch.float8_e4m3fn if expected_index == "fp8" else torch.bfloat16)
        raise Resolved

    with (
        patch("vllm_ascend.platform.get_current_hardware_profile", return_value=get_hardware_profile(device_type)),
        patch("vllm_ascend.platform._validate_model_runner_config"),
        patch.object(NPUPlatform, "_validate_indexer_pp_config"),
        patch("vllm_ascend.platform._validate_routing_replay_config"),
        patch("vllm_ascend.platform._validate_pcp_dcp_config"),
        patch("vllm_ascend.platform._validate_draft_decode_context_parallel_config"),
        patch("vllm_ascend.platform._validate_parallel_config"),
        patch("vllm_ascend.platform._validate_engram_config"),
        patch("vllm_ascend.logger.configure_ascend_file_logging"),
        patch("vllm_ascend.logger.configure_ascend_logging"),
        patch("vllm_ascend.quantization.utils.maybe_auto_detect_quantization", side_effect=inspect_resolved),
        pytest.raises(Resolved),
    ):
        NPUPlatform.check_and_update_config(config)
