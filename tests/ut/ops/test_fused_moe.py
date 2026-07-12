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
# This file is a part of the vllm-ascend project.
#
import ast
import importlib
import inspect
import textwrap
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from vllm_ascend.ascend_forward_context import MoECommType
from vllm_ascend.ops.fused_moe import fused_moe as fused_moe_module
from vllm_ascend.ops.fused_moe.fused_moe import (
    AscendMoERunner,
    AscendUnquantizedFusedMoEMethod,
    _configure_eplb_expert_map,
)


@pytest.fixture(autouse=True)
def setup_vllm_config_mock(monkeypatch):
    model_config = SimpleNamespace(enable_return_routed_experts=False)
    vllm_config = SimpleNamespace(model_config=model_config)
    monkeypatch.setattr(
        fused_moe_module,
        "get_current_vllm_config",
        MagicMock(return_value=vllm_config),
    )


def _drop_self(signature: inspect.Signature) -> list[inspect.Parameter]:
    params = list(signature.parameters.values())
    if params and params[0].name == "self":
        return params[1:]
    return params


def _assert_child_signature_accepts_parent_interface(child_method, parent_method):
    child_params = _drop_self(inspect.signature(child_method))
    parent_params = _drop_self(inspect.signature(parent_method))
    child_by_name = {
        param.name: param
        for param in child_params
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    child_has_var_positional = any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in child_params)
    child_has_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in child_params)
    issues: list[str] = []

    for parent_param in parent_params:
        if parent_param.kind == inspect.Parameter.VAR_POSITIONAL:
            if not child_has_var_positional:
                issues.append("child is missing *args from parent")
            continue
        if parent_param.kind == inspect.Parameter.VAR_KEYWORD:
            if not child_has_var_keyword:
                issues.append("child is missing **kwargs from parent")
            continue

        child_param = child_by_name.get(parent_param.name)
        if child_param is None:
            if parent_param.kind == inspect.Parameter.KEYWORD_ONLY:
                if not child_has_var_keyword:
                    issues.append(f"missing keyword-only parameter {parent_param.name!r}")
            elif not child_has_var_positional and not child_has_var_keyword:
                issues.append(f"missing parameter {parent_param.name!r}")
            continue

        if parent_param.kind != child_param.kind:
            issues.append(
                f"parameter {parent_param.name!r} has kind {child_param.kind!s}, expected {parent_param.kind!s}"
            )

    parent_param_names = {param.name for param in parent_params}
    for child_param in child_params:
        if child_param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if child_param.name in parent_param_names:
            continue
        if child_param.default is inspect.Parameter.empty:
            issues.append(f"extra parameter {child_param.name!r} must be optional")

    assert not issues, f"{parent_method.__qualname__} signature is not aligned: " + "; ".join(issues)


def _method_uses_super(method) -> bool:
    source = inspect.getsource(method)
    tree = ast.parse(textwrap.dedent(source))
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "super"
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize(
    "child_cls,parent_cls,method_name",
    [
        (
            AscendUnquantizedFusedMoEMethod,
            fused_moe_module.UnquantizedFusedMoEMethod,
            "__init__",
        ),
        (
            AscendUnquantizedFusedMoEMethod,
            fused_moe_module.UnquantizedFusedMoEMethod,
            "process_weights_after_loading",
        ),
        (AscendMoERunner, fused_moe_module.MoERunner, "__init__"),
    ],
)
def test_super_calls_accept_parent_interface(child_cls, parent_cls, method_name):
    child_method = getattr(child_cls, method_name)
    assert _method_uses_super(child_method)
    _assert_child_signature_accepts_parent_interface(
        child_method,
        getattr(parent_cls, method_name),
    )


def test_ascend_unquantized_skips_upstream_modular_kernel_init():
    assert AscendUnquantizedFusedMoEMethod.maybe_make_prepare_finalize(object()) is None


class TestAscendUnquantizedFusedMoEMethod:
    @staticmethod
    def _build_layer(*, has_bias=True, zero_expert_num=0):
        layer = MagicMock()
        layer.w13_weight = nn.Parameter(torch.randn(2, 3, 4))
        layer.w2_weight = nn.Parameter(torch.randn(2, 4, 3))
        layer.w13_bias = torch.randn(2, 4) if has_bias else None
        layer.w2_bias = torch.randn(2, 3) if has_bias else None
        layer.zero_expert_num = zero_expert_num
        layer.zero_expert_type = "identity" if zero_expert_num else None
        layer.n_shared_experts = 0
        layer.moe_config = SimpleNamespace(num_logical_experts=None)
        layer.layer_id = 3
        return layer

    @pytest.mark.parametrize("enable_fused_mc2", [True, False])
    def test_process_weights_after_loading_transposes_and_formats(
        self,
        monkeypatch,
        enable_fused_mc2,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.dynamic_eplb = False
        method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
        layer = self._build_layer()
        original_w13 = layer.w13_weight.detach().clone()
        original_w2 = layer.w2_weight.detach().clone()
        format_cast = MagicMock(side_effect=lambda weight, _: weight)
        maybe_trans_nz = MagicMock(side_effect=lambda weight: weight)
        ascend_config = SimpleNamespace(enable_fused_mc2=enable_fused_mc2)

        monkeypatch.setattr(
            fused_moe_module,
            "get_ascend_config",
            MagicMock(return_value=ascend_config),
        )
        monkeypatch.setattr(
            fused_moe_module.torch_npu,
            "npu_format_cast",
            format_cast,
        )
        monkeypatch.setattr(fused_moe_module, "maybe_trans_nz", maybe_trans_nz)

        method.process_weights_after_loading(layer)

        torch.testing.assert_close(
            layer.w13_weight,
            original_w13.transpose(1, 2),
        )
        torch.testing.assert_close(
            layer.w2_weight,
            original_w2.transpose(1, 2),
        )
        if enable_fused_mc2:
            assert format_cast.call_count == 2
            maybe_trans_nz.assert_not_called()
        else:
            assert maybe_trans_nz.call_count == 2
            format_cast.assert_not_called()

    def test_process_weights_after_loading_splits_dynamic_eplb_weights(
        self,
        monkeypatch,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.dynamic_eplb = True
        method._maybe_pad_weight = MagicMock(side_effect=lambda weight: weight)
        layer = nn.Module()
        layer.w13_weight = nn.Parameter(torch.randn(2, 3, 4))
        layer.w2_weight = nn.Parameter(torch.randn(2, 4, 3))
        expected_w13 = layer.w13_weight.detach().clone().transpose(1, 2)
        expected_w2 = layer.w2_weight.detach().clone().transpose(1, 2)
        format_cast = MagicMock(side_effect=lambda weight, _: weight)
        empty_cache = MagicMock()

        monkeypatch.setattr(
            fused_moe_module,
            "get_ascend_config",
            MagicMock(return_value=SimpleNamespace(enable_fused_mc2=True)),
        )
        monkeypatch.setattr(
            fused_moe_module.torch_npu,
            "npu_format_cast",
            format_cast,
        )
        monkeypatch.setattr(
            fused_moe_module.torch,
            "npu",
            SimpleNamespace(empty_cache=empty_cache),
            raising=False,
        )

        method.process_weights_after_loading(layer)

        assert "w13_weight" not in layer._parameters
        assert "w2_weight" not in layer._parameters
        assert len(layer.w13_weight_list) == 2
        assert len(layer.w2_weight_list) == 2
        torch.testing.assert_close(layer.w13_weight_list[0], expected_w13[0])
        torch.testing.assert_close(layer.w2_weight_list[1], expected_w2[1])
        assert format_cast.call_count == 2
        empty_cache.assert_called_once()

    @pytest.mark.parametrize(
        "moe_comm_type",
        [MoECommType.MC2, MoECommType.FUSED_MC2],
    )
    def test_apply_builds_fused_experts_input(
        self,
        monkeypatch,
        moe_comm_type,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=True)
        method.dynamic_eplb = False
        method.tid2eid = None
        layer = self._build_layer(has_bias=True)
        hidden_states = torch.randn(2, 4, dtype=torch.float16)
        router_logits = torch.randn(2, 4)
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.6, 0.4]],
            dtype=torch.float32,
        )
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.ones_like(hidden_states)
        select_experts = MagicMock(return_value=(topk_weights, topk_ids))

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=moe_comm_type,
                moe_comm_method=moe_comm_method,
            ),
        )
        monkeypatch.setattr(fused_moe_module, "select_experts", select_experts)
        monkeypatch.setattr(
            fused_moe_module,
            "get_forward_context",
            MagicMock(return_value=SimpleNamespace(input_ids=None)),
        )

        result = method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=router_logits,
            renormalize=True,
            num_experts=4,
            apply_router_weight_on_input=True,
            activation="gelu",
            pertoken_scale=torch.ones(2),
            mc2_mask=torch.tensor([True, False]),
        )

        torch.testing.assert_close(result, torch.ones_like(hidden_states))
        select_experts.assert_called_once()
        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.hidden_states is hidden_states
        torch.testing.assert_close(
            fused_input.topk_weights,
            topk_weights.to(hidden_states.dtype),
        )
        assert torch.equal(fused_input.topk_ids, topk_ids)
        assert fused_input.weights.w1_bias is layer.w13_bias
        assert fused_input.weights.w2_bias is layer.w2_bias
        assert fused_input.routing.apply_router_weight_on_input
        assert fused_input.activation == "gelu"
        if moe_comm_type == MoECommType.FUSED_MC2:
            assert fused_input.weights.w1[0] is layer.w13_weight
            assert fused_input.weights.w2[0] is layer.w2_weight
            assert fused_input.weights.w1_scale[0].dtype == torch.int64
            assert fused_input.weights.w2_scale[0].dtype == torch.int64
        else:
            assert fused_input.weights.w1 is layer.w13_weight
            assert fused_input.weights.w2 is layer.w2_weight
            assert fused_input.weights.w1_scale is None
            assert fused_input.weights.w2_scale is None

    @pytest.mark.parametrize(
        "moe_comm_type",
        [MoECommType.MC2, MoECommType.FUSED_MC2],
    )
    def test_apply_uses_dynamic_eplb_weight_lists(
        self,
        monkeypatch,
        moe_comm_type,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.dynamic_eplb = True
        method.tid2eid = None
        layer = self._build_layer(has_bias=False)
        layer.w13_weight_list = [torch.randn(4, 6), torch.randn(4, 6)]
        layer.w2_weight_list = [torch.randn(3, 4), torch.randn(3, 4)]
        hidden_states = torch.randn(2, 4, dtype=torch.float16)
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.ones_like(hidden_states)

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=moe_comm_type,
                moe_comm_method=moe_comm_method,
            ),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "select_experts",
            MagicMock(
                return_value=(
                    torch.ones(2, 2),
                    torch.tensor([[0, 1], [1, 0]]),
                )
            ),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "get_forward_context",
            MagicMock(return_value=SimpleNamespace(input_ids=None)),
        )

        method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=torch.randn(2, 4),
            renormalize=True,
            num_experts=4,
        )

        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.weights.w1 is layer.w13_weight_list
        assert fused_input.weights.w2 is layer.w2_weight_list
        if moe_comm_type == MoECommType.FUSED_MC2:
            assert fused_input.weights.w1_scale[0].numel() == 0
            assert fused_input.weights.w2_scale[0].numel() == 0
        else:
            assert fused_input.weights.w1_scale is None
            assert fused_input.weights.w2_scale is None

    def test_apply_warns_for_unsplit_dynamic_eplb_fused_mc2_weights(
        self,
        monkeypatch,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.dynamic_eplb = True
        method.tid2eid = None
        layer = self._build_layer(has_bias=False)
        hidden_states = torch.randn(2, 4, dtype=torch.float16)
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = torch.ones_like(hidden_states)
        warning_once = MagicMock()

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=MoECommType.FUSED_MC2,
                moe_comm_method=moe_comm_method,
            ),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "select_experts",
            MagicMock(
                return_value=(
                    torch.ones(2, 2),
                    torch.tensor([[0, 1], [1, 0]]),
                )
            ),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "get_forward_context",
            MagicMock(return_value=SimpleNamespace(input_ids=None)),
        )
        monkeypatch.setattr(fused_moe_module.logger, "warning_once", warning_once)

        method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=torch.randn(2, 4),
            renormalize=True,
            num_experts=4,
        )

        warning_once.assert_called_once()
        assert "dynamic EPLB" in warning_once.call_args.args[0]

    def test_apply_adds_zero_expert_result_and_force_balances(
        self,
        monkeypatch,
    ):
        method = AscendUnquantizedFusedMoEMethod.__new__(AscendUnquantizedFusedMoEMethod)
        method.moe = SimpleNamespace(has_bias=False)
        method.dynamic_eplb = True
        method.tid2eid = None
        layer = self._build_layer(has_bias=False, zero_expert_num=1)
        hidden_states = torch.randn(2, 4)
        topk_weights = torch.ones(2, 2)
        topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
        zero_hidden = torch.full_like(hidden_states, 3.0)
        routed_hidden = torch.full_like(hidden_states, 5.0)
        expected_hidden = routed_hidden.clone() + zero_hidden
        moe_comm_method = MagicMock()
        moe_comm_method.fused_experts.return_value = routed_hidden
        zero_experts = MagicMock(return_value=(topk_ids, topk_weights, zero_hidden))

        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=MoECommType.MC2,
                moe_comm_method=moe_comm_method,
            ),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "select_experts",
            MagicMock(return_value=(topk_weights, topk_ids)),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "zero_experts_compute",
            zero_experts,
        )
        monkeypatch.setattr(
            fused_moe_module.torch,
            "rand",
            MagicMock(return_value=torch.tensor([[0.2, 0.1], [0.4, 0.3]])),
        )
        monkeypatch.setattr(
            fused_moe_module,
            "get_forward_context",
            MagicMock(return_value=SimpleNamespace(input_ids=None)),
        )

        result = method.apply(
            layer=layer,
            x=hidden_states,
            use_grouped_topk=False,
            top_k=2,
            router_logits=torch.randn(2, 2),
            renormalize=False,
            num_experts=2,
            enable_force_load_balance=True,
        )

        torch.testing.assert_close(result, expected_hidden)
        zero_experts.assert_called_once()
        fused_input = moe_comm_method.fused_experts.call_args.kwargs["fused_experts_input"]
        assert fused_input.dynamic_eplb
        assert fused_input.weights.w1_bias is None
        assert fused_input.weights.w2_bias is None


class TestAscendMoERunner:
    @pytest.mark.parametrize(
        "moe_comm_type,flash_comm_v1_enabled,expected",
        [
            (MoECommType.ALLTOALL, False, True),
            (MoECommType.MC2, False, True),
            (MoECommType.FUSED_MC2, False, True),
            (MoECommType.ALLGATHER, False, False),
            (MoECommType.ALLGATHER, True, True),
        ],
    )
    def test_reduction_properties(
        self,
        monkeypatch,
        moe_comm_type,
        flash_comm_v1_enabled,
        expected,
    ):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        monkeypatch.setattr(
            fused_moe_module,
            "_EXTRA_CTX",
            SimpleNamespace(
                moe_comm_type=moe_comm_type,
                flash_comm_v1_enabled=flash_comm_v1_enabled,
            ),
        )

        assert runner.use_dp_chunking is False
        assert runner._fused_output_is_reduced is expected
        assert runner._maybe_reduce_shared_expert_output("shared") == "shared"

    def test_maybe_reduce_final_output_uses_vllm_op(self, monkeypatch):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        reduced = torch.arange(12).reshape(2, 6)
        maybe_all_reduce = MagicMock(return_value=reduced)
        monkeypatch.setattr(
            fused_moe_module.torch.ops,
            "vllm",
            SimpleNamespace(maybe_all_reduce_tensor_model_parallel=maybe_all_reduce),
            raising=False,
        )
        states = torch.ones(2, 6)

        result = runner._maybe_reduce_final_output(states, 4)

        maybe_all_reduce.assert_called_once_with(states)
        torch.testing.assert_close(result, reduced[..., :4])

    @pytest.mark.parametrize("has_shared_experts", [False, True])
    def test_forward_impl_selects_current_runner_path(
        self,
        has_shared_experts,
    ):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        runner._shared_experts = MagicMock() if has_shared_experts else None
        runner._sequence_parallel_context = MagicMock(return_value=nullcontext())
        runner.no_shared_forward_impl = MagicMock(return_value="routed")
        runner.shared_forward_impl = MagicMock(return_value=("shared", "routed"))
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)

        result = runner._forward_impl(hidden_states, router_logits, None)

        if has_shared_experts:
            assert result == ("shared", "routed")
            runner.shared_forward_impl.assert_called_once_with(
                hidden_states,
                router_logits,
            )
            runner.no_shared_forward_impl.assert_not_called()
        else:
            assert result == "routed"
            runner.no_shared_forward_impl.assert_called_once_with(
                hidden_states,
                router_logits,
            )
            runner.shared_forward_impl.assert_not_called()

    def test_shared_experts_split_with_expert_gate(self):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        hidden_states = torch.tensor([[1.0, -1.0]])
        gate_up = torch.tensor([[2.0, -2.0]])
        down_out = torch.tensor([[3.0, 4.0]])
        gate_out = torch.tensor([[0.0, 2.0]])
        shared_experts = MagicMock()
        shared_experts.gate_up_proj.return_value = (gate_up, None)
        shared_experts.act_fn.side_effect = lambda tensor: tensor + 1
        shared_experts.down_proj.return_value = (down_out, None)
        shared_experts.expert_gate.return_value = (gate_out, None)
        runner._shared_experts = shared_experts

        part1_out = runner._shared_experts_part1(hidden_states)
        part2_out = runner._shared_experts_part2(hidden_states, part1_out)

        torch.testing.assert_close(part1_out, gate_up)
        torch.testing.assert_close(part2_out, torch.sigmoid(gate_out) * down_out)

    @pytest.mark.parametrize("has_shared_experts", [False, True])
    def test_shared_forward_impl_routes_shared_output(
        self,
        monkeypatch,
        has_shared_experts,
    ):
        runner = AscendMoERunner.__new__(AscendMoERunner)
        runner.shared_multistream_overlap_gate = False
        runner._shared_experts = MagicMock() if has_shared_experts else None
        hidden_states = torch.randn(2, 4)
        router_logits = torch.randn(2, 3)
        fused_result = fused_moe_module.FusedMoEResult(
            routed_out=torch.ones(2, 4),
            before_dispatch_evt=MagicMock(),
            before_combine_evt=MagicMock(),
        )
        runner.no_shared_forward_impl = MagicMock(return_value=fused_result)
        runner._forward_shared_experts = MagicMock(return_value="shared_out")
        stream = MagicMock()
        stream.record_event.side_effect = [MagicMock(), MagicMock()]
        monkeypatch.setattr(
            AscendMoERunner,
            "is_internal_router",
            property(lambda _: False),
        )
        monkeypatch.setattr(
            fused_moe_module.torch,
            "npu",
            SimpleNamespace(current_stream=MagicMock(return_value=stream)),
            raising=False,
        )

        result = runner.shared_forward_impl(hidden_states, router_logits)

        runner.no_shared_forward_impl.assert_called_once_with(
            hidden_states,
            router_logits,
            return_with_event=True,
        )
        if has_shared_experts:
            assert result == ("shared_out", fused_result.routed_out)
            runner._forward_shared_experts.assert_called_once()
        else:
            torch.testing.assert_close(result, fused_result.routed_out)
            runner._forward_shared_experts.assert_not_called()


def test_ascend_fused_moe_factory_injects_current_runner(monkeypatch):
    patch_module = importlib.import_module("vllm_ascend.patch.platform.patch_fused_moe")
    original_fused_moe = MagicMock(return_value="runner")
    monkeypatch.setattr(patch_module, "_original_FusedMoE", original_fused_moe)
    monkeypatch.setattr(
        patch_module,
        "_DefaultAscendMoERunner",
        AscendMoERunner,
    )
    monkeypatch.setattr(
        patch_module,
        "_ascend_eplb_overrides",
        MagicMock(return_value=(0, False)),
    )
    tid2eid = torch.tensor([0, 1])

    result = patch_module._ascend_FusedMoE(
        num_experts=2,
        top_k=1,
        hidden_size=4,
        intermediate_size=8,
        hash=True,
        tid2eid=tid2eid,
        runner_args={"n_shared_experts": 1},
    )

    assert result == "runner"
    kwargs = original_fused_moe.call_args.kwargs
    assert kwargs["runner_cls"] is AscendMoERunner
    assert kwargs["runner_args"]["n_shared_experts"] == 1
    assert kwargs["runner_args"]["tid2eid"] is tid2eid
    assert "hash" not in kwargs
    assert "tid2eid" not in kwargs


def test_configure_eplb_expert_map_does_not_double_count_redundancy():
    moe_config = SimpleNamespace(
        num_experts=10,
        num_logical_experts=8,
        num_local_experts=5,
        ep_size=2,
    )
    manager = SimpleNamespace(
        global_num_experts=10,
        _local_num_experts=5,
        _expert_map=None,
    )
    routed_experts = SimpleNamespace(
        global_num_experts=8,
        expert_map_manager=manager,
        update_expert_map_info=MagicMock(),
    )
    expert_map = torch.tensor([0, 1, 2, 3, -1, -1, -1, -1])

    local_num_experts, global_num_experts = _configure_eplb_expert_map(
        moe_config,
        routed_experts,
        expert_map,
        num_redundant_experts=2,
    )

    assert local_num_experts == 5
    assert global_num_experts == 10
    assert moe_config.num_experts == 10
    assert manager._local_num_experts == 5
    assert manager._expert_map is expert_map
    assert routed_experts.global_num_experts == 10
    routed_experts.update_expert_map_info.assert_called_once_with()


@pytest.mark.parametrize(
    ("is_v024", "enable_eplb_index", "num_redundant_experts_index"),
    [(True, 24, 25), (False, 25, 26)],
)
def test_ascend_fused_moe_factory_overrides_versioned_positional_eplb_args(
    monkeypatch,
    is_v024,
    enable_eplb_index,
    num_redundant_experts_index,
):
    patch_module = importlib.import_module("vllm_ascend.patch.platform.patch_fused_moe")
    original_fused_moe = MagicMock(return_value="runner")
    monkeypatch.setattr(patch_module, "_original_FusedMoE", original_fused_moe)
    monkeypatch.setattr(
        patch_module,
        "_ascend_eplb_overrides",
        MagicMock(return_value=(2, True)),
    )
    monkeypatch.setattr(
        patch_module,
        "_ascend_mix_placement_allocation",
        MagicMock(return_value=0),
    )
    monkeypatch.setattr(
        patch_module,
        "vllm_version_is",
        MagicMock(return_value=is_v024),
    )
    args = list(range(num_redundant_experts_index + 1))

    assert patch_module._ascend_FusedMoE(*args) == "runner"

    forwarded_args = original_fused_moe.call_args.args
    assert forwarded_args[enable_eplb_index] is True
    assert forwarded_args[num_redundant_experts_index] == 2


def test_ascend_fused_moe_factory_allocates_redundant_expert_slots(monkeypatch):
    patch_module = importlib.import_module("vllm_ascend.patch.platform.patch_fused_moe")

    def allocated_local_experts(*args, **kwargs):
        num_experts = kwargs["num_experts"]
        num_redundant_experts = kwargs["num_redundant_experts"]
        return (num_experts + num_redundant_experts) // 2

    monkeypatch.setattr(patch_module, "_original_FusedMoE", allocated_local_experts)
    monkeypatch.setattr(
        patch_module,
        "_ascend_eplb_overrides",
        MagicMock(return_value=(2, True)),
    )
    monkeypatch.setattr(
        patch_module,
        "_ascend_mix_placement_allocation",
        MagicMock(return_value=0),
    )

    result = patch_module._ascend_FusedMoE(num_experts=128)

    assert result == 65


@pytest.mark.parametrize(("is_v024", "n_shared_index"), [(True, 29), (False, 31)])
def test_mix_placement_reserves_one_shared_expert_slot_per_ep_rank(monkeypatch, is_v024, n_shared_index):
    patch_module = importlib.import_module("vllm_ascend.patch.platform.patch_fused_moe")
    ascend_config = SimpleNamespace(
        mix_placement=True,
        eplb_config=SimpleNamespace(expert_map_path=None),
    )
    monkeypatch.setattr(
        patch_module,
        "get_ascend_config",
        MagicMock(return_value=ascend_config),
    )
    monkeypatch.setattr(
        patch_module,
        "get_ep_group",
        MagicMock(return_value=SimpleNamespace(world_size=2)),
    )
    monkeypatch.setattr(
        patch_module,
        "vllm_version_is",
        MagicMock(return_value=is_v024),
    )
    args = [0] * (n_shared_index + 1)
    args[n_shared_index] = 1

    allocation = patch_module._ascend_mix_placement_allocation(tuple(args), {})

    assert allocation == 2


def test_ascend_expert_mappings_keep_logical_checkpoint_ids(monkeypatch):
    patch_module = importlib.import_module("vllm_ascend.patch.platform.patch_fused_moe")
    original_mapping = MagicMock(return_value="tag-mapping")
    original_build_mapping = MagicMock(return_value="main-mapping")
    monkeypatch.setattr(
        patch_module,
        "_ascend_eplb_overrides",
        MagicMock(return_value=(2, True)),
    )
    monkeypatch.setattr(
        patch_module,
        "_original_make_expert_params_mapping",
        original_mapping,
    )
    monkeypatch.setattr(
        patch_module,
        "_original_build_expert_params_mapping",
        original_build_mapping,
    )

    assert patch_module._ascend_make_expert_params_mapping(None, "gate", "down", "up", 8, 2) == "tag-mapping"
    assert original_mapping.call_args.args[5] == 0

    assert patch_module._ascend_build_expert_params_mapping("gate", "down", "up", 8, 2) == "main-mapping"
    assert original_build_mapping.call_args.args[4] == 0

    patch_module._ascend_eplb_overrides.return_value = (0, True)
    assert patch_module._ascend_build_expert_params_mapping("gate", "down", "up", 8, 1) == "main-mapping"
    assert original_build_mapping.call_args.args[4] == 0


def test_ascend_moe_runner_exposes_eplb_update_protocol():
    runner = AscendMoERunner.__new__(AscendMoERunner)
    manager = SimpleNamespace(_expert_map=None, expert_map=torch.tensor([0, 1]))
    runner.routed_experts = SimpleNamespace(
        expert_map_manager=manager,
        update_expert_map=MagicMock(),
        update_expert_map_info=MagicMock(),
    )
    new_expert_map = torch.tensor([1, 0])

    runner.update_expert_map(new_expert_map)

    assert runner._expert_map is new_expert_map
    assert manager._expert_map is new_expert_map
    runner.routed_experts.update_expert_map_info.assert_called_once_with()

    runner.routed_experts.update_expert_map_info.reset_mock()
    runner.update_expert_map()
    assert runner._expert_map is new_expert_map
    assert manager._expert_map is new_expert_map
    runner.routed_experts.update_expert_map.assert_not_called()
    runner.routed_experts.update_expert_map_info.assert_called_once_with()

    runner.log2phy = torch.tensor([1, 0])
    assert runner.get_log2phy_map() is runner.log2phy
    runner.moe_load = MagicMock()
    runner.multi_stage = True
    runner.load_counter = MagicMock()
    runner.clear_moe_load()
    runner.moe_load.zero_.assert_called_once_with()
    runner.load_counter.zero_.assert_called_once_with()
