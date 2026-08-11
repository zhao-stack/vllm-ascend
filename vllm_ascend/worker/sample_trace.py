# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
from threading import Lock
from typing import Any

import torch
from vllm.logger import logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput

SAMPLE_TRACE_POOL_SIZE = 4
SAMPLE_TRACE_TOKEN_STAGE_COUNT = 2
SAMPLE_TRACE_PROBE_STAGE_COUNT = 2
SAMPLE_TRACE_MAX_PROBES = 8

TOKEN_STAGE_AFTER_SAMPLE = 0
TOKEN_STAGE_BEFORE_ASYNC_OUTPUT = 1
PROBE_STAGE_HIDDEN_STATES = 0
PROBE_STAGE_LOGITS = 1


def normalize_eos_token_ids(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, int):
        return (value,)
    if isinstance(value, (list, tuple, set)):
        return tuple(token_id for token_id in value if isinstance(token_id, int))
    return ()


def build_probe_indices(
    width: int,
    eos_token_ids: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Return a small deterministic set of columns without touching device data."""
    if width <= 0:
        return ()

    candidates = [0, width // 3, (2 * width) // 3, width - 1]
    candidates.extend(token_id for token_id in eos_token_ids if 0 <= token_id < width)

    indices: list[int] = []
    for index in candidates:
        if index not in indices:
            indices.append(index)
        if len(indices) == SAMPLE_TRACE_MAX_PROBES:
            break
    return tuple(indices)


class SampleTraceBufferPool:
    """Preallocated NPU/host buffers for non-synchronizing sample tracing."""

    def __init__(
        self,
        max_num_reqs: int,
        max_sample_tokens: int,
        device: torch.device,
        eos_token_ids: tuple[int, ...],
    ) -> None:
        token_shape = (
            SAMPLE_TRACE_POOL_SIZE,
            SAMPLE_TRACE_TOKEN_STAGE_COUNT,
            max_num_reqs,
            max_sample_tokens,
        )
        probe_shape = (
            SAMPLE_TRACE_POOL_SIZE,
            SAMPLE_TRACE_PROBE_STAGE_COUNT,
            max_num_reqs,
            SAMPLE_TRACE_MAX_PROBES,
        )
        self.token_ids_device = torch.empty(token_shape, dtype=torch.int64, device=device)
        self.token_ids_host = torch.empty(
            token_shape,
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        self.probes_device = torch.empty(probe_shape, dtype=torch.float32, device=device)
        self.probes_host = torch.empty(
            probe_shape,
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        )

        self.max_num_reqs = max_num_reqs
        self.max_sample_tokens = max_sample_tokens
        self.eos_token_ids = eos_token_ids
        self._free_slots = deque(range(SAMPLE_TRACE_POOL_SIZE))
        self._lock = Lock()
        self._next_step_id = 0
        self._warned_pool_exhausted = False

    def acquire(self, req_ids: list[str]) -> "SampleTraceHandle | None":
        with self._lock:
            if not self._free_slots:
                if not self._warned_pool_exhausted:
                    logger.warning(
                        "SAMPLE_TRACE skipped because all %d trace slots are in flight",
                        SAMPLE_TRACE_POOL_SIZE,
                    )
                    self._warned_pool_exhausted = True
                return None
            slot = self._free_slots.popleft()
            step_id = self._next_step_id
            self._next_step_id += 1

        return SampleTraceHandle(self, slot, step_id, req_ids)

    def release(self, slot: int) -> None:
        with self._lock:
            self._free_slots.append(slot)


class SampleTraceHandle:
    """One in-flight trace. It never retains a reference to source tensors."""

    def __init__(
        self,
        pool: SampleTraceBufferPool,
        slot: int,
        step_id: int,
        req_ids: list[str],
    ) -> None:
        self.pool = pool
        self.slot = slot
        self.step_id = step_id
        self.req_ids = req_ids
        self.token_shapes: dict[int, tuple[int, int]] = {}
        self.probe_indices: dict[int, tuple[int, ...]] = {}
        self.tensor_metadata: dict[str, dict[str, Any]] = {}
        self._released = False

    def snapshot_tensor_probes(
        self,
        stage: int,
        name: str,
        tensor: torch.Tensor,
        include_eos: bool = False,
    ) -> None:
        if tensor.ndim != 2:
            self.tensor_metadata[name] = {
                "data_ptr": tensor.data_ptr(),
                "shape": tuple(tensor.shape),
                "stride": tuple(tensor.stride()),
                "probe_error": "expected a two-dimensional tensor",
            }
            return

        num_rows = min(len(self.req_ids), tensor.shape[0], self.pool.max_num_reqs)
        eos_token_ids = self.pool.eos_token_ids if include_eos else ()
        indices = build_probe_indices(tensor.shape[1], eos_token_ids)
        destination = self.pool.probes_device[self.slot, stage]
        for probe_index, source_index in enumerate(indices):
            destination[:num_rows, probe_index].copy_(tensor[:num_rows, source_index])

        self.probe_indices[stage] = indices
        self.tensor_metadata[name] = {
            "data_ptr": tensor.data_ptr(),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
        }

    def snapshot_token_ids(
        self,
        stage: int,
        name: str,
        token_ids: torch.Tensor,
    ) -> None:
        source = token_ids.unsqueeze(-1) if token_ids.ndim == 1 else token_ids
        if source.ndim != 2:
            self.tensor_metadata[name] = {
                "data_ptr": token_ids.data_ptr(),
                "shape": tuple(token_ids.shape),
                "stride": tuple(token_ids.stride()),
                "snapshot_error": "expected a one- or two-dimensional tensor",
            }
            return

        num_rows = min(len(self.req_ids), source.shape[0], self.pool.max_num_reqs)
        num_tokens = min(source.shape[1], self.pool.max_sample_tokens)
        self.pool.token_ids_device[
            self.slot,
            stage,
            :num_rows,
            :num_tokens,
        ].copy_(source[:num_rows, :num_tokens])

        self.token_shapes[stage] = (num_rows, num_tokens)
        self.tensor_metadata[name] = {
            "data_ptr": token_ids.data_ptr(),
            "shape": tuple(token_ids.shape),
            "stride": tuple(token_ids.stride()),
        }

    def enqueue_to_host(self, copy_stream: Any) -> None:
        """Queue trace D2H before the async output's existing ready event."""
        producer_stream = torch.npu.current_stream()
        with torch.npu.stream(copy_stream):
            copy_stream.wait_stream(producer_stream)
            self.pool.token_ids_host[self.slot].copy_(
                self.pool.token_ids_device[self.slot],
                non_blocking=True,
            )
            self.pool.probes_host[self.slot].copy_(
                self.pool.probes_device[self.slot],
                non_blocking=True,
            )

    def log_after_to_host(self, output: ModelRunnerOutput) -> None:
        """Read only pinned host mirrors after the parent's existing wait."""
        token_snapshots: dict[str, list[list[int]]] = {}
        for stage, name in (
            (TOKEN_STAGE_AFTER_SAMPLE, "after_sample"),
            (TOKEN_STAGE_BEFORE_ASYNC_OUTPUT, "before_async_output"),
        ):
            shape = self.token_shapes.get(stage)
            if shape is None:
                continue
            num_rows, num_tokens = shape
            token_snapshots[name] = self.pool.token_ids_host[
                self.slot,
                stage,
                :num_rows,
                :num_tokens,
            ].tolist()

        probe_snapshots: dict[str, dict[str, Any]] = {}
        for stage, name in (
            (PROBE_STAGE_HIDDEN_STATES, "hidden_states"),
            (PROBE_STAGE_LOGITS, "logits"),
        ):
            indices = self.probe_indices.get(stage)
            if indices is None:
                continue
            num_rows = min(len(self.req_ids), self.pool.max_num_reqs)
            probe_snapshots[name] = {
                "indices": indices,
                "values": self.pool.probes_host[
                    self.slot,
                    stage,
                    :num_rows,
                    : len(indices),
                ].tolist(),
            }

        record = {
            "step": self.step_id,
            "req_ids": self.req_ids,
            "eos_token_ids": self.pool.eos_token_ids,
            "token_snapshots": token_snapshots,
            "host_output_token_ids": output.sampled_token_ids,
            "probes": probe_snapshots,
            "tensor_metadata": self.tensor_metadata,
        }
        logger.warning("SAMPLE_TRACE %s", record)

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.pool.release(self.slot)


class TracedAsyncGPUModelRunnerOutput(AsyncGPUModelRunnerOutput):
    """Async output that logs trace data after the parent's existing D2H wait."""

    def __init__(
        self,
        *,
        sample_trace: SampleTraceHandle,
        **kwargs: Any,
    ) -> None:
        self._sample_trace = sample_trace
        sample_trace.enqueue_to_host(kwargs["async_output_copy_stream"])
        super().__init__(**kwargs)

    def get_output(self) -> ModelRunnerOutput:
        output = super().get_output()
        try:
            self._sample_trace.log_after_to_host(output)
        finally:
            self._sample_trace.release()
        return output
