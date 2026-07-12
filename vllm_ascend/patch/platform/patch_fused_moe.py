#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Patch vllm's FusedMoE factory to use AscendMoERunner by default.
#
# vllm's FusedMoE is a factory function (not a class). deepseek_v2 and other
# models do `from vllm.model_executor.layers.fused_moe import FusedMoE` and
# call it directly, so we must patch the binding in the package __init__ as
# well as the layer module before any model is imported.
#
# Import order in worker.__init__:
#   1. adapt_patch()  ->  this file runs  ->  FusedMoE patched
#   2. from vllm_ascend import ops
#   3. model loading  ->  deepseek_v2 imported  ->  gets patched FusedMoE  ✓

import vllm.model_executor.layers.fused_moe as _fused_moe_pkg
import vllm.model_executor.layers.fused_moe.layer as _fused_moe_layer
import vllm.model_executor.layers.fused_moe.routed_experts as _routed_experts_module
from vllm.distributed import get_ep_group

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.utils import is_310p, vllm_version_is

# Capture the real original before fused_moe.py's module-level code runs.
_original_FusedMoE = _fused_moe_layer.FusedMoE
_original_make_expert_params_mapping = _fused_moe_layer.fused_moe_make_expert_params_mapping
_original_build_expert_params_mapping = (
    None if vllm_version_is("0.24.0") else _routed_experts_module.RoutedExperts.build_expert_params_mapping
)

if is_310p():
    from vllm_ascend._310p.fused_moe.fused_moe import AscendMoERunner310 as _DefaultAscendMoERunner
else:
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner as _DefaultAscendMoERunner


def _replace_positional_arg(args, kwargs, index, name, value):
    if len(args) <= index:
        kwargs[name] = value
        return args
    kwargs.pop(name, None)
    return (*args[:index], value, *args[index + 1 :])


def _ascend_eplb_overrides() -> tuple[int, bool]:
    ascend_config = get_ascend_config()
    eplb_config = ascend_config.eplb_config
    num_redundant_experts = int(eplb_config.num_redundant_experts)
    enable_eplb = (
        eplb_config.dynamic_eplb
        or eplb_config.expert_map_path is not None
        or num_redundant_experts > 0
        or ascend_config.mix_placement
    )
    return num_redundant_experts, enable_eplb


def _fused_moe_eplb_arg_indices() -> tuple[int, int]:
    # vLLM main added intermediate_pad before params_dtype after v0.24.0.
    return (24, 25) if vllm_version_is("0.24.0") else (25, 26)


def _fused_moe_n_shared_experts_arg_index() -> int:
    # vLLM main added both reduce_results and ckpt_names before this arg.
    return 29 if vllm_version_is("0.24.0") else 31


def _get_arg(args, kwargs, index, name, default):
    return args[index] if len(args) > index else kwargs.get(name, default)


def _ascend_mix_placement_allocation(args, kwargs) -> int:
    ascend_config = get_ascend_config()
    if not ascend_config.mix_placement or ascend_config.eplb_config.expert_map_path is not None:
        return 0

    n_shared_experts = _get_arg(
        args,
        kwargs,
        _fused_moe_n_shared_experts_arg_index(),
        "n_shared_experts",
        0,
    )
    if not n_shared_experts:
        return 0

    # Unified placement keeps one shared-expert replica on each EP rank.
    # Treat those slots as allocation-only redundancy until the Ascend runner
    # extends the logical map without changing the router's top-k domain.
    return get_ep_group().world_size


def _ascend_FusedMoE(*args, runner_cls=None, runner_args=None, **kwargs):
    if runner_cls is None:
        runner_cls = _DefaultAscendMoERunner
    num_redundant_experts, enable_eplb = _ascend_eplb_overrides()
    if enable_eplb:
        enable_eplb_index, num_redundant_experts_index = _fused_moe_eplb_arg_indices()
        num_redundant_experts += _ascend_mix_placement_allocation(args, kwargs)
        args = _replace_positional_arg(args, kwargs, enable_eplb_index, "enable_eplb", True)
        args = _replace_positional_arg(
            args,
            kwargs,
            num_redundant_experts_index,
            "num_redundant_experts",
            num_redundant_experts,
        )
    # 'hash' is a DeepSeek V4 flag already consumed before FusedMoE is called;
    # 'tid2eid' is Ascend-specific and must reach AscendMoERunner via runner_args.
    kwargs.pop("hash", None)
    tid2eid = kwargs.pop("tid2eid", None)
    if tid2eid is not None:
        runner_args = dict(runner_args) if runner_args is not None else {}
        runner_args["tid2eid"] = tid2eid
    return _original_FusedMoE(*args, runner_cls=runner_cls, runner_args=runner_args, **kwargs)


def _ascend_make_expert_params_mapping(*args, **kwargs):
    _, enable_eplb = _ascend_eplb_overrides()
    if enable_eplb:
        # Ascend's map indexes logical checkpoint experts into local physical
        # slots, so the checkpoint mapping must not add synthetic expert ids.
        args = _replace_positional_arg(args, kwargs, 5, "num_redundant_experts", 0)
    return _original_make_expert_params_mapping(*args, **kwargs)


def _ascend_build_expert_params_mapping(*args, **kwargs):
    assert _original_build_expert_params_mapping is not None
    _, enable_eplb = _ascend_eplb_overrides()
    if enable_eplb:
        args = _replace_positional_arg(args, kwargs, 4, "num_redundant_experts", 0)
    return _original_build_expert_params_mapping(*args, **kwargs)


_fused_moe_layer.FusedMoE = _ascend_FusedMoE
_fused_moe_pkg.FusedMoE = _ascend_FusedMoE
_fused_moe_layer.fused_moe_make_expert_params_mapping = _ascend_make_expert_params_mapping
_fused_moe_pkg.fused_moe_make_expert_params_mapping = _ascend_make_expert_params_mapping
if not vllm_version_is("0.24.0"):
    _routed_experts_module.RoutedExperts.build_expert_params_mapping = staticmethod(_ascend_build_expert_params_mapping)
