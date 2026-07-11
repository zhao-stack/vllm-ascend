# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.worker.utils as worker_utils
from vllm_ascend.worker.utils import AscendKVBlockZeroer


def test_kv_block_zeroer_requires_positive_concurrency():
    with pytest.raises(ValueError, match="max_concurrency"):
        AscendKVBlockZeroer(torch.device("cpu"), False, max_concurrency=0)


def test_kv_block_zeroer_rotates_id_buffers(monkeypatch):
    zeroer = AscendKVBlockZeroer(
        torch.device("cpu"),
        False,
        max_concurrency=2,
    )
    zeroer._id_cap = 4
    zeroer._allocate_id_buffers()
    zeroer._meta = (
        torch.tensor([0], dtype=torch.uint64),
        1,
        1,
        1,
    )

    kernel = MagicMock()
    monkeypatch.setattr(worker_utils, "_zero_kv_blocks_kernel", kernel)
    monkeypatch.setattr(worker_utils, "get_vectorcore_num", lambda: 1)

    zeroer.zero_block_ids([1])
    zeroer.zero_block_ids([2])

    assert zeroer._ids_pinned[0][0].item() == 1
    assert zeroer._ids_pinned[1][0].item() == 2
    assert zeroer._id_buffer_index == 0
    assert kernel.__getitem__.return_value.call_count == 2
