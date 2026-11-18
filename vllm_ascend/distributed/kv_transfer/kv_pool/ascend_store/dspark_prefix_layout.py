# SPDX-License-Identifier: Apache-2.0
"""DSpark prefix ownership, available to the planner without store imports."""

from collections.abc import Sequence

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as KVCacheGroup
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
else:
    from vllm.v1.kv_cache_interface import KVCacheGroup


def draft_only_group_ids(groups: Sequence[KVCacheGroup]) -> tuple[int, ...]:
    """Find groups whose prefix objects live entirely in the draft companion.

    K3.1 mixes target MLA/indexer and draft layers, so its groups still need
    ordinary target hit keys. Other layouts can have separate draft groups,
    for which no ordinary target object is saved. ``is_eagle_group`` marks
    groups containing any draft layer and cannot distinguish these cases.

    Run before scheduler conversion selects one representative layer spec.
    Inspect global specs even when this PP rank has no local layer names.
    """
    result = []
    for group_id, group in enumerate(groups):
        if vllm_version_is("0.33.0"):
            spec = group.kv_cache_spec

        specs = (
            (spec.kv_cache_specs.values() if isinstance(spec, UniformTypeKVCacheSpecs) else (spec,))
            if vllm_version_is("0.33.0")
            else group.distinct_specs
        )
        if (
            specs and all(layer_spec.is_draft for layer_spec in specs)
            if vllm_version_is("0.33.0")
            else specs and all(getattr(layer_spec, "is_draft", False) for layer_spec in specs)
        ):
            result.append(group_id)
    return tuple(result)
