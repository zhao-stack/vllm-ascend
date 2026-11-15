# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from vllm.config import CacheConfig, VllmConfig
from vllm.v1.kv_cache_interface import KVCacheSpec, MambaSpec, MLAAttentionSpec

from vllm_ascend.utils import vllm_version_is

if TYPE_CHECKING:
    # Type-check shared Ascend code against the verified-main storage API.
    # Runtime release exports below retain the frozen registry-based classes.
    from vllm_ascend.core.kv_cache_specs_legacy import (
        register_ascend_kv_cache_specs as register_ascend_kv_cache_specs,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendCircularBufferSpec as AscendCircularBufferSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendIndexerKPoolTailSpec as AscendIndexerKPoolTailSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendMLAAttentionSpec as AscendMLAAttentionSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendSFAIndexerCacheSpec as AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendSlidingWindowMLASpec as AscendSlidingWindowMLASpec,
    )

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs
else:
    from vllm.v1.kv_cache_interface import KVCacheGroup, KVCacheLayerConfig


def get_kv_cache_compression_ratio(kv_cache_spec: KVCacheSpec) -> int:
    """Return the MLA compression ratio across vLLM cache-spec APIs."""
    return kv_cache_spec.tokens_per_state


def get_storage_block_size(kv_cache_spec: KVCacheSpec) -> int:
    """Return the physical token rows represented by one scheduler block."""
    if vllm_version_is("0.33.0") and isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        storage_block_sizes = {get_storage_block_size(spec) for spec in kv_cache_spec.kv_cache_specs.values()}
        assert len(storage_block_sizes) == 1, "All specs in one KV cache group must use the same storage block size."
        return storage_block_sizes.pop()
    # vLLM #53906 added an optional MLA storage-view override. It is not
    # Ascend's derived number of physical rows per logical block.
    if isinstance(kv_cache_spec, AscendMLAAttentionSpec):
        return kv_cache_spec.block_size // kv_cache_spec.tokens_per_state
    if isinstance(kv_cache_spec, MLAAttentionSpec):
        return kv_cache_spec.block_size
    return getattr(kv_cache_spec, "storage_block_size", kv_cache_spec.block_size)


def is_circular_kv_cache_spec(kv_cache_spec: KVCacheSpec) -> bool:
    """Resolve the fixed-block addressing capability through grouped specs."""
    if vllm_version_is("0.33.0") and isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = tuple(kv_cache_spec.kv_cache_specs.values())
        return bool(specs) and all(is_circular_kv_cache_spec(spec) for spec in specs)
    if not vllm_version_is("0.33.0"):
        return isinstance(kv_cache_spec, (AscendCircularBufferSpec, AscendIndexerKPoolTailSpec))
    return isinstance(kv_cache_spec, CircularBufferSpec) or getattr(kv_cache_spec, "is_circular", False)


def is_prefix_cacheable(kv_cache_spec: KVCacheSpec) -> bool:
    """Resolve cacheability through uniform wrappers.

    ``prefix_cacheable`` is the upstream API on the lanes that define it; the
    release lane predates it, so an unset attribute means cacheable.
    """
    if not vllm_version_is("0.33.0"):
        # Cacheability belongs to the allocation policy, not a storage spec.
        assert isinstance(kv_cache_spec, (KVCacheGroup, KVCacheLayerConfig))
        if isinstance(kv_cache_spec, KVCacheGroup):
            return kv_cache_spec.prefix_cacheable
        return kv_cache_spec.manager_cls.prefix_cacheable(kv_cache_spec.manager_config)
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        return all(is_prefix_cacheable(spec) for spec in kv_cache_spec.kv_cache_specs.values())
    return bool(getattr(kv_cache_spec, "prefix_cacheable", True)) and bool(
        getattr(kv_cache_spec, "participates_in_prefix_caching", True)
    )


def declared_kwarg(owner: type, name: str, value: Any) -> dict[str, Any]:
    """Spell one optional field out for a constructor that declares it.

    Used for the fields upstream adds to the sliding-window specs and their
    caches over time -- ``extra_retained_tokens`` on the spec, ``bounded_replay``
    on both the spec and the SWA cache (vLLM #56227). Naming such a keyword
    unconditionally is a ``TypeError`` on a lane that does not have it, and this
    repository supports two: main, which grows the fields, and the release tag,
    which never does. Probing the owner keeps that decision in one place.
    """
    return {name: value} if name in inspect.signature(owner).parameters else {}


# The architectures bounded replay was written for. Upstream wires the feature
# into ``vllm/models/deepseek_v41/`` alone -- the DeepSeek-V4 tree never reads
# the switch -- so the architecture name is the honest test for "this model may
# replay". Both of this project's spellings for the target are listed: missing
# the one a checkpoint declares is a switch that can never be turned on.
#
# The DSpark draft is deliberately left out. Its SWA cache is a different class
# (``DeepseekV41DSparkSWACache``) that does not carry the field, so a draft
# checkpoint would pass a gate that then replays nothing.
BOUNDED_REPLAY_ARCHITECTURES = frozenset(
    {
        "DeepseekV41ForCausalLM",  # this repository's registry key
        "DeepseekV41ForConditionalGeneration",  # the VL spelling
    }
)


def supports_bounded_replay(model_config: Any) -> bool:
    """Whether this model is one the feature was written for."""
    architectures = getattr(model_config, "architectures", None) or ()
    return any(architecture in BOUNDED_REPLAY_ARCHITECTURES for architecture in architectures)


def resolve_bounded_replay(vllm_config: VllmConfig, cache_config: CacheConfig | None) -> bool:
    """Whether a sliding-window layer may declare a replay window.

    The switch comes from upstream (vLLM #56227) and is re-decided here
    because the layer upstream applies it in -- its own SWA attention layer --
    is the one Ascend replaces. The Ascend scheduler conditions, the ones that
    cannot rewind a prefix hit at all, are decided in ``platform.py`` before any
    layer exists: they turn the switch itself off, so this reads the result.

    It reads that switch and nothing else, deliberately: a V4.1 model that does
    not turn it off in its cache config replays.

    There is no capability guard beside them. Keeping a replayed window off the
    blocks below ``replay_start`` -- whose KV the sliding-window group does not
    hold, those positions having been retired to the null block -- is the
    operator's job now: band mode reads ``ori_topk_length`` as the per-token
    visible length and floors the window's left edge with it. See
    ``build_replay_swa_visible_lens`` in ``attention/dsa_v41.py`` for the read
    side and ``npu_sparse_flash_mla``'s band path for the clamp.
    """
    if not supports_bounded_replay(getattr(vllm_config, "model_config", None)):
        # Upstream wires the feature into the DeepSeek-V4.1 tree only -- the V4
        # tree's attention layer never reads the switch -- so a V4 model is
        # silent here too: the feature is not part of this model, and no user
        # can have turned it on for it.
        return False
    # The switch is absent from release lanes without bounded replay. PCP and
    # DP share the same cache contract; decoder batch coordination is handled
    # by DecoderReplayState rather than disabling the cache feature here.
    return bool(getattr(cache_config, "swa_bounded_replay", False))


def requires_padded_page_layout(kv_cache_specs: Iterable[KVCacheSpec]) -> bool:
    """Whether state caches advance by a padded page size shared with attention.

    Ascend mixed pools can bind one physical page to both an attention cache and a
    recurrent state cache. The attention side of such a pool is addressed by an explicit
    physical block stride (``indexes_kv_by_block_stride``) and the state caches are
    padded to that same page size, so every view over the shared page has to advance by
    the padded page size: otherwise a state update for one scheduler block ID would land
    in the pages owned by another block ID.

    The capability belongs to the pool rather than to a single spec, because hybrid
    models also pad Mamba pages through ``cache_config.mamba_page_size_padded`` while
    keeping the packed contiguous state layout, so padding alone does not require this
    layout.
    """
    specs = list(kv_cache_specs)
    state_specs = [spec for spec in specs if isinstance(spec, MambaSpec)]
    if not state_specs:
        return False
    if not any(getattr(spec, "page_size_padded", None) is not None for spec in state_specs):
        return False
    return any(getattr(spec, "indexes_kv_by_block_stride", False) for spec in specs)


def supports_component_major_mla_pd(vllm_config: VllmConfig) -> bool:
    """Whether PD transfer can preserve a component-major MLA cache layout.

    A component-major MLA page packs NoPE and RoPE into one physical slot:
    ``[all NoPE][all RoPE][padding]``.  MooncakeV2 pull connectors transfer
    that page as one raw block when the published base address, block length,
    and block stride describe the complete slot.  Other connectors still use
    the legacy separate NoPE/RoPE protocol.

    Without PD, local layout selection is unrestricted. With supported PD,
    both endpoints must agree on the page layout: raw transfers carry no
    metadata distinguishing component-major from token-fused FlashMLA pages.
    This helper identifies connector support, not cross-endpoint agreement.
    """
    kv_transfer_config = vllm_config.kv_transfer_config
    if kv_transfer_config is None:
        return True

    # Connectors loaded through a custom module path are not identified by the
    # registry name, so conservatively fall back to the legacy layout.
    if getattr(kv_transfer_config, "kv_connector_module_path", None) is not None:
        return False

    return getattr(kv_transfer_config, "kv_connector", None) in {
        "MooncakeConnectorV2",
        "MooncakePullConnector",
    }


# The release registry still consumes spec-owned policy. Verified main uses
# explicit KVCacheLayerConfig/ManagerConfig and storage-only specs.
if not TYPE_CHECKING and vllm_version_is("0.33.0"):
    from vllm_ascend.core.kv_cache_specs_legacy import (
        AscendCircularBufferSpec as AscendCircularBufferSpec,
    )
    from vllm_ascend.core.kv_cache_specs_legacy import (
        AscendIndexerKPoolTailSpec as AscendIndexerKPoolTailSpec,
    )
    from vllm_ascend.core.kv_cache_specs_legacy import (
        AscendMLAAttentionSpec as AscendMLAAttentionSpec,
    )
    from vllm_ascend.core.kv_cache_specs_legacy import (
        AscendSFAIndexerCacheSpec as AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.core.kv_cache_specs_legacy import (
        AscendSlidingWindowMLASpec as AscendSlidingWindowMLASpec,
    )
    from vllm_ascend.core.kv_cache_specs_legacy import (
        register_ascend_kv_cache_specs as register_ascend_kv_cache_specs,
    )
elif not TYPE_CHECKING:
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendCircularBufferSpec as AscendCircularBufferSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendIndexerKPoolTailSpec as AscendIndexerKPoolTailSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendMLAAttentionSpec as AscendMLAAttentionSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendSFAIndexerCacheSpec as AscendSFAIndexerCacheSpec,
    )
    from vllm_ascend.core.kv_cache_specs_main import (
        AscendSlidingWindowMLASpec as AscendSlidingWindowMLASpec,
    )
