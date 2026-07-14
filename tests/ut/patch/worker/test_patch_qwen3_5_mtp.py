# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5DecoderLayer
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention, Qwen3NextDecoderLayer
from vllm.sequence import IntermediateTensors

from vllm_ascend.ops.gdn import AscendGatedDeltaNetAttention
from vllm_ascend.patch.worker import patch_qwen3_5
from vllm_ascend.utils import vllm_version_is


def test_ascend_gdn_forward_uses_active_vllm_output_protocol():
    hidden_states = torch.zeros(2, 3)
    projected = torch.ones(2, 3)
    forward_impl = MagicMock(return_value=projected)
    layer = SimpleNamespace(_forward_ascend=forward_impl)

    if vllm_version_is("0.23.0"):
        output = torch.zeros(4, 3)
        result = AscendGatedDeltaNetAttention.forward(layer, hidden_states, output)

        assert result is None
        assert torch.equal(output[:2], projected)
        assert torch.equal(output[2:], torch.zeros(2, 3))
    else:
        result = AscendGatedDeltaNetAttention.forward(layer, hidden_states)

        assert result is projected

    forward_impl.assert_called_once_with(hidden_states)
    if not patch_qwen3_5.is_310p():
        assert patch_qwen3_5._GDN_PATCH_TARGET.forward is AscendGatedDeltaNetAttention.forward


def test_ascend_qwen_attention_uses_active_vllm_output_protocol():
    positions = torch.arange(2)
    hidden_states = torch.zeros(2, 3)
    projected = torch.ones(2, 3)
    forward_impl = MagicMock(return_value=projected)
    layer = SimpleNamespace(_forward_ascend=forward_impl)

    if vllm_version_is("0.23.0"):
        output = torch.zeros_like(projected)
        result = patch_qwen3_5.AscendQwen3NextAttention.forward(layer, positions, output, hidden_states)

        assert result is None
        assert torch.equal(output, projected)
    else:
        result = patch_qwen3_5.AscendQwen3NextAttention.forward(layer, positions, hidden_states)

        assert result is projected

    forward_impl.assert_called_once_with(positions, hidden_states)
    assert Qwen3NextAttention.forward is patch_qwen3_5.AscendQwen3NextAttention.forward


def test_qwen3_5_decoder_patch_tracks_active_vllm_protocol():
    if vllm_version_is("0.23.0"):
        assert Qwen3_5DecoderLayer.forward is patch_qwen3_5.AscendQwen3_5DecoderLayer.forward
    else:
        assert Qwen3_5DecoderLayer.forward is Qwen3NextDecoderLayer.forward


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_uses_local_inputs_on_last_pp_rank(monkeypatch):
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 2
    predictor.config = SimpleNamespace(hidden_size=4)
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(2, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x + 1)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x + 2)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4] + x[:, 4:])
    layer0 = MagicMock(return_value=(torch.full((2, 4), 3.0), torch.full((2, 4), 4.0)))
    layer1 = MagicMock(return_value=(torch.full((2, 4), 5.0), torch.full((2, 4), 6.0)))
    layer0.use_attn_reduce_scatter_for_moe = False
    layer1.use_attn_reduce_scatter_for_moe = False
    predictor.layers = [layer0, layer1]
    predictor.norm = MagicMock(return_value=(torch.full((2, 4), 7.0), None))
    gather = MagicMock()
    if not patch_qwen3_5._IS_VLLM_RELEASE:
        monkeypatch.setattr(patch_qwen3_5, "_all_gather_hidden_and_residual", gather)

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1, 2]),
            positions=torch.tensor([0, 1]),
            hidden_states=torch.zeros(2, 4),
            intermediate_tensors=IntermediateTensors({"hidden_states": torch.full((2, 4), 99.0)}),
            spec_step_idx=3,
        )

    predictor.embed_input_ids.assert_called_once()
    layer1.assert_called_once()
    predictor.norm.assert_called_once()
    gather.assert_not_called()
    assert torch.equal(output, torch.full((2, 4), 7.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None or patch_qwen3_5._IS_VLLM_RELEASE,
    reason="Qwen3.5 MTP sequence-parallel gather is main-only.",
)
def test_qwen3_5_mtp_forward_gathers_sequence_parallel_output(monkeypatch):
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.config = SimpleNamespace(hidden_size=4)
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(4, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])

    sharded_hidden = torch.full((2, 4), 3.0)
    sharded_residual = torch.full((2, 4), 4.0)
    full_hidden = torch.full((4, 4), 5.0)
    full_residual = torch.full((4, 4), 6.0)
    mtp_layer = MagicMock(return_value=(sharded_hidden, sharded_residual))
    mtp_layer.use_attn_reduce_scatter_for_moe = True
    predictor.layers = [mtp_layer]
    predictor.norm = MagicMock(return_value=(torch.full((4, 4), 7.0), None))
    gather = MagicMock(return_value=(full_hidden, full_residual))
    monkeypatch.setattr(patch_qwen3_5, "_all_gather_hidden_and_residual", gather)

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = predictor.forward(
            input_ids=torch.arange(4),
            positions=torch.arange(4),
            hidden_states=torch.zeros(4, 4),
        )

    gather.assert_called_once_with(sharded_hidden, sharded_residual, 4, 4)
    predictor.norm.assert_called_once_with(full_hidden, full_residual)
    assert torch.equal(output, torch.full((4, 4), 7.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None or not patch_qwen3_5._IS_VLLM_RELEASE,
    reason="This regression covers the v0.23.0 MTP layer protocol.",
)
def test_qwen3_5_mtp_release_does_not_read_sequence_parallel_flag():
    class ReleaseLayer:
        def __call__(self, **_kwargs):
            return torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)

    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(1, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    predictor.layers = [ReleaseLayer()]
    predictor.norm = MagicMock(return_value=(torch.full((1, 4), 7.0), None))

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=True),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.zeros(1, 4),
        )

    assert torch.equal(output, torch.full((1, 4), 7.0))


@pytest.mark.skipif(
    patch_qwen3_5.Qwen3_5MultiTokenPredictor is None,
    reason="Qwen3.5 MTP model is not available in this vLLM version.",
)
def test_qwen3_5_mtp_forward_returns_intermediate_tensors_on_non_last_pp_rank(monkeypatch):
    predictor = patch_qwen3_5.Qwen3_5MultiTokenPredictor.__new__(patch_qwen3_5.Qwen3_5MultiTokenPredictor)
    predictor.num_mtp_layers = 1
    predictor.embed_input_ids = MagicMock(return_value=torch.ones(1, 4))
    predictor.pre_fc_norm_embedding = MagicMock(side_effect=lambda x: x)
    predictor.pre_fc_norm_hidden = MagicMock(side_effect=lambda x: x)
    predictor.fc = MagicMock(side_effect=lambda x: x[:, :4])
    mtp_layer = MagicMock(return_value=(torch.full((1, 4), 3.0), torch.full((1, 4), 4.0)))
    mtp_layer.use_attn_reduce_scatter_for_moe = True
    predictor.layers = [mtp_layer]
    predictor.norm = MagicMock()
    gather = MagicMock()
    if not patch_qwen3_5._IS_VLLM_RELEASE:
        monkeypatch.setattr(patch_qwen3_5, "_all_gather_hidden_and_residual", gather)

    with patch(
        "vllm_ascend.patch.worker.patch_qwen3_5.get_pp_group",
        return_value=SimpleNamespace(is_last_rank=False),
    ):
        output = predictor.forward(
            input_ids=torch.tensor([1]),
            positions=torch.tensor([0]),
            hidden_states=torch.zeros(1, 4),
        )

    assert isinstance(output, IntermediateTensors)
    assert torch.equal(output["hidden_states"], torch.full((1, 4), 3.0))
    assert torch.equal(output["residual"], torch.full((1, 4), 4.0))
    predictor.norm.assert_not_called()
    gather.assert_not_called()
