# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType
from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.sample.rejection_sampler import PLACEHOLDER_TOKEN_ID

from vllm_ascend.core.recompute_scheduler import RecomputeScheduler
from vllm_ascend.utils import vllm_version_is


def test_finished_request_transfer_params_follow_active_vllm_protocol():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    request = MagicMock(spec=Request)
    kv_transfer_params = {"kv": "params"}
    ec_transfer_params = {"ec": "params"}
    scheduler._free_request = MagicMock(
        return_value=(kv_transfer_params if vllm_version_is("0.23.0") else (kv_transfer_params, ec_transfer_params))
    )

    actual_kv_params, actual_ec_params = scheduler._free_request_transfer_params(request)

    assert actual_kv_params == kv_transfer_params
    assert actual_ec_params == (None if vllm_version_is("0.23.0") else ec_transfer_params)
    scheduler._free_request.assert_called_once_with(request)


def test_engine_core_output_propagates_ec_params_on_main():
    kv_transfer_params = {"kv": "params"}
    ec_transfer_params = {"ec": "params"}

    output = RecomputeScheduler._make_engine_core_output(
        request_id="transfer-output",
        new_token_ids=[],
        kv_transfer_params=kv_transfer_params,
        ec_transfer_params=ec_transfer_params,
    )

    assert output.kv_transfer_params == kv_transfer_params
    if not vllm_version_is("0.23.0"):
        assert output.ec_transfer_params == ec_transfer_params


def test_pd_consumer_first_step_injects_placeholder_spec_tokens():
    scheduler = RecomputeScheduler.__new__(RecomputeScheduler)
    scheduler.requests = {}
    scheduler.is_kv_producer = False
    scheduler.is_hybrid_model = False
    scheduler.is_mtp_kv_consumer = True
    scheduler.num_spec_tokens = 1
    scheduler.max_model_len = 1024
    scheduler.log_stats = False
    scheduler.connector = None

    enqueued_requests = []

    def enqueue_waiting_request(self, request):
        enqueued_requests.append(request)

    scheduler._enqueue_waiting_request = MethodType(enqueue_waiting_request, scheduler)

    request = Request(
        request_id="pd-consumer-first-step",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
    )

    scheduler.add_request(request)

    assert enqueued_requests == [request]
    assert scheduler.requests[request.request_id] is request
    assert request.spec_token_ids == [PLACEHOLDER_TOKEN_ID]
    assert request.num_tokens_with_spec == request.num_tokens + 1
