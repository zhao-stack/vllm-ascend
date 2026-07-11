# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm_ascend.distributed.kv_transfer.ascend_multi_connector import (
    AscendMultiConnector,
)
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_layerwise_connector import (
    MooncakeLayerwiseConnector,
)
from vllm_ascend.utils import vllm_version_is


def test_update_state_after_alloc_matches_installed_multi_connector_contract():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    chosen = MagicMock()
    other = MagicMock()
    layerwise = MagicMock(spec=MooncakeLayerwiseConnector)
    connector._connectors = [chosen, other, layerwise]
    connector._requests_to_connector = {"req": 0}

    request = SimpleNamespace(request_id="req")
    blocks = MagicMock()
    empty_blocks = blocks.new_empty.return_value

    connector.update_state_after_alloc(request, blocks, 32)

    chosen.update_state_after_alloc.assert_called_once_with(request, blocks, 32)
    expected_other_blocks = empty_blocks if vllm_version_is("0.23.0") else blocks
    other.update_state_after_alloc.assert_called_once_with(
        request,
        expected_other_blocks,
        0,
    )
    layerwise.update_state_after_alloc.assert_called_once_with(request, blocks, 32)


def test_update_state_after_alloc_without_a_chosen_connector():
    connector = AscendMultiConnector.__new__(AscendMultiConnector)
    other = MagicMock()
    connector._connectors = [other]
    connector._requests_to_connector = {}

    request = SimpleNamespace(request_id="cold-request")
    blocks = MagicMock()
    empty_blocks = blocks.new_empty.return_value

    connector.update_state_after_alloc(request, blocks, 0)

    expected_blocks = empty_blocks if vllm_version_is("0.23.0") else blocks
    other.update_state_after_alloc.assert_called_once_with(request, expected_blocks, 0)
