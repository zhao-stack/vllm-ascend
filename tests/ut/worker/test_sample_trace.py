# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

from vllm_ascend.worker.sample_trace import (
    SAMPLE_TRACE_MAX_PROBES,
    build_probe_indices,
    normalize_eos_token_ids,
)


def test_normalize_eos_token_ids() -> None:
    assert normalize_eos_token_ids(None) == ()
    assert normalize_eos_token_ids(7) == (7,)
    assert normalize_eos_token_ids([7, 8, "invalid"]) == (7, 8)


def test_build_probe_indices_includes_valid_eos_ids() -> None:
    indices = build_probe_indices(12, (7, 11, 12, -1))

    assert indices[:4] == (0, 4, 8, 11)
    assert 7 in indices
    assert 12 not in indices
    assert -1 not in indices
    assert len(indices) <= SAMPLE_TRACE_MAX_PROBES


def test_build_probe_indices_deduplicates_small_widths() -> None:
    assert build_probe_indices(1, (0,)) == (0,)
    assert build_probe_indices(0, (0,)) == ()
