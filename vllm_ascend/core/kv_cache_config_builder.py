# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
from __future__ import annotations

from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.33.0"):
    """Ascend KV cache planning hooks for vLLM's pluggable config builder.

vLLM plans the KV cache in ``vllm.v1.core.kv_cache_planning`` and resolves the
planner through ``Platform.get_kv_cache_config_builder_cls`` (see
``NPUPlatform.get_kv_cache_config_builder_cls``). Ascend-specific grouping and
pool accounting live here; materialization stays in the patched planning
functions (see ``patch_kv_cache_utils.py``), which
``DefaultKVCacheConfigBuilder.get_kv_cache_config_from_groups`` calls with the
Core-resolved ``num_blocks``.
"""

from collections import defaultdict

if not vllm_version_is("0.33.0"):
    from collections import Counter

if not vllm_version_is("0.33.0"):
    from dataclasses import replace


from vllm.config import VllmConfig
from vllm.utils.math_utils import cdiv

if vllm_version_is("0.33.0"):
    from vllm.utils.math_utils import round_up

from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder, _approximate_gcd

if vllm_version_is("0.33.0"):
    from vllm.v1.core.kv_cache_planning import unify_hybrid_kv_cache_specs

if not vllm_version_is("0.33.0"):
    from vllm.v1.core.single_type_kv_cache_manager import SlidingWindowManager

if vllm_version_is("0.33.0"):
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec as KVCacheGroup
    from vllm.v1.kv_cache_interface import KVCacheSpec, MLAAttentionSpec, SlidingWindowMLASpec, UniformTypeKVCacheSpecs
else:
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroup, KVCacheLayerConfig, KVCacheTensor

from vllm_ascend.models.deepseek_v41.cache_config import is_deepseek_v41_cache

if vllm_version_is("0.33.0"):
    from vllm_ascend.models.deepseek_v41.cache_config import (
        get_deepseek_v41_pool_bytes_per_block,
        group_cache_specs,
        make_cache_groups,
    )
else:
    from vllm_ascend.models.deepseek_v41.cache_config import group_layer_configs, place_deepseek_v41_cache

from vllm_ascend.models.glm5next.cache_config import (
    _get_glm5_next_cache_layout,
    get_glm5_next_kv_cache_groups,
    get_glm5_next_pool_bytes_per_block,
)

if not vllm_version_is("0.33.0"):
    from vllm_ascend.models.glm5next.cache_config import get_glm5_next_kv_cache_config

from vllm_ascend.models.glm5next.kv_cache import is_glm5_next_cache_spec
from vllm_ascend.models.kimi_k3.cache_config import _get_kimi_k3_dspark_mixed_kv_cache_groups
from vllm_ascend.models.kimi_k3.paired_sfa_cache import paired_sfa_cache_groups, paired_sfa_pool_bytes

if not vllm_version_is("0.33.0"):
    from vllm_ascend.models.kimi_k3.paired_sfa_cache import paired_sfa_cache_config

from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_pool_bytes_per_block

if not vllm_version_is("0.33.0"):
    from vllm_ascend.models.qwen4_exp.cache_config import get_qwen4_exp_kv_cache_config, get_qwen4_exp_kv_cache_groups

from vllm_ascend.quantization.methods.kv_cache.turboquant import TURBOQUANT_CACHE_DTYPE
from vllm_ascend.quantization.methods.kv_cache.turboquant import cache as tq_cache

if not vllm_version_is("0.33.0"):

    def _is_deepseek_v4(specs):
        return any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs)


class AscendKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    """Build Ascend cache layouts with vLLM's cross-worker planning flow."""

    if vllm_version_is("0.33.0"):

        def get_kv_cache_groups(
            self,
            vllm_config: VllmConfig,
            kv_cache_spec: dict[str, KVCacheSpec],
        ) -> list[KVCacheGroup]:
            """Group layers, preferring the Ascend layout when one applies.

            Precedence: Kimi-K3.1, GLM-Next, then DSV4.1 / mixed-page-size shared slots, then
            the Kimi-K3 DSpark mixed grouping, then vLLM's generic grouping.
            """
            if vllm_config.model_config.hf_config.model_type == "kimi_k31":
                return paired_sfa_cache_groups(vllm_config, kv_cache_spec)
            if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
                # vLLM's generic grouping promotes local-attention specs before it
                # groups, so the Ascend layouts must see the same promoted specs.
                unify_hybrid_kv_cache_specs(kv_cache_spec)

            if any(is_glm5_next_cache_spec(spec) for spec in kv_cache_spec.values()):
                return get_glm5_next_kv_cache_groups(vllm_config, kv_cache_spec)

            if is_deepseek_v41_cache(kv_cache_spec):
                return make_cache_groups(group_cache_specs(kv_cache_spec))

            grouped_specs = group_and_unify_kv_cache_specs(kv_cache_spec)
            if grouped_specs is not None and len(grouped_specs) >= 2:
                # A uniform spec is not a mixed-page-size layout; grouping it here
                # would drop the C128 bucket the shared-tuple layout depends on.
                return _get_kv_cache_groups_uniform_groups(grouped_specs)

            kimi_k3_groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(kv_cache_spec)
            if kimi_k3_groups is not None:
                return kimi_k3_groups

            return super().get_kv_cache_groups(vllm_config, kv_cache_spec)
    else:

        def get_kv_cache_groups(  # type: ignore[misc]
            self, vllm_config: VllmConfig, layer_configs: dict[str, KVCacheLayerConfig]
        ) -> list[KVCacheGroup]:
            specs = [config.spec for config in layer_configs.values()]
            if is_deepseek_v41_cache(specs):
                return group_layer_configs(layer_configs)
            if _is_deepseek_v4(specs):
                return _group_deepseek_v4_layers(layer_configs)
            if getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31":
                return paired_sfa_cache_groups(vllm_config, layer_configs)
            if any(is_glm5_next_cache_spec(spec) for spec in specs):
                return get_glm5_next_kv_cache_groups(vllm_config, layer_configs)
            groups = get_qwen4_exp_kv_cache_groups(vllm_config, layer_configs, super().get_kv_cache_groups)
            if groups is None:
                groups = _get_kimi_k3_dspark_mixed_kv_cache_groups(layer_configs)
            if groups is None:
                return super().get_kv_cache_groups(vllm_config, layer_configs)
            return groups

    if not vllm_version_is("0.33.0"):

        def get_kv_cache_config_from_groups(
            self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroup], num_blocks: int
        ) -> KVCacheConfig:
            config = self._get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)
            # Scheduler-local PD role metadata survives deepcopy, as before #61452.
            config.kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
            return config

        def _get_kv_cache_config_from_groups(
            self, vllm_config: VllmConfig, kv_cache_groups: list[KVCacheGroup], num_blocks: int
        ) -> KVCacheConfig:
            specs = tuple(spec for group in kv_cache_groups for spec in group.distinct_specs)
            if is_deepseek_v41_cache(specs):
                return place_deepseek_v41_cache(vllm_config, kv_cache_groups, num_blocks)
            if not _is_deepseek_v4(specs):
                if getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31":
                    return paired_sfa_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
                if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
                    return get_glm5_next_kv_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
                config = get_qwen4_exp_kv_cache_config(vllm_config, kv_cache_groups, 0, num_blocks=num_blocks)
                if config is not None:
                    return config
                return super().get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, num_blocks)

            if tq_cache.uses_turboquant_groups(kv_cache_groups):
                tensors = tq_cache.cache_tensors(kv_cache_groups, num_blocks)
            else:
                tensors = _place_deepseek_v4_cache(kv_cache_groups, num_blocks)
            return KVCacheConfig(
                num_blocks=num_blocks,
                kv_cache_tensors=tensors,
                kv_cache_groups=kv_cache_groups,
                prefix_cache_retention_interval=vllm_config.cache_config.prefix_cache_retention_interval,
            )

    def _get_pool_bytes_per_block(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroup],
    ) -> int:
        if vllm_version_is("0.33.0"):
            """One global block ID consumes sum(page sizes) x layer tuples.

        The default implementation materializes a one-block pool and requires a
        single device backing size, which the DSV4 shared-tuple, the TurboQuant
        packed, the DSV4.1, and the GLM-Next layouts all violate.
        """
            if vllm_config.model_config.hf_config.model_type == "kimi_k31":
                return paired_sfa_pool_bytes(kv_cache_groups)
            if is_deepseek_v41_cache(kv_cache_groups):
                return get_deepseek_v41_pool_bytes_per_block(kv_cache_groups)
            if _get_glm5_next_cache_layout(kv_cache_groups) is not None:
                return get_glm5_next_pool_bytes_per_block(kv_cache_groups)
            qwen_bytes = get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        else:
            specs = {name: spec for group in kv_cache_groups for name, spec in group.kv_cache_specs.items()}

        if qwen_bytes is not None if vllm_version_is("0.33.0") else is_deepseek_v41_cache(specs.values()):
            if not vllm_version_is("0.33.0"):
                from vllm_ascend.models.deepseek_v41.cache_config import get_layer_tuples

                page_sizes, _ = get_layer_tuples(specs)

            return qwen_bytes if vllm_version_is("0.33.0") else sum(page_sizes)
        if (
            tq_cache.uses_turboquant_groups(kv_cache_groups)
            if vllm_version_is("0.33.0")
            else getattr(vllm_config.model_config.hf_config, "model_type", None) == "kimi_k31"
        ):
            return (
                tq_cache.pool_bytes_per_block(kv_cache_groups)
                if vllm_version_is("0.33.0")
                else paired_sfa_pool_bytes(kv_cache_groups)
            )
        if (
            _is_deepseek_v4_groups(kv_cache_groups)
            if vllm_version_is("0.33.0")
            else _get_glm5_next_cache_layout(kv_cache_groups) is not None
        ):
            return (
                _deepseek_v4_pool_bytes_per_block(kv_cache_groups)
                if vllm_version_is("0.33.0")
                else get_glm5_next_pool_bytes_per_block(kv_cache_groups)
            )
        return (
            super()._get_pool_bytes_per_block(vllm_config, kv_cache_groups)
            if vllm_version_is("0.33.0")
            else get_qwen4_exp_pool_bytes_per_block(kv_cache_groups)
        )


if vllm_version_is("0.33.0"):

    def _page_sizes(spec: UniformTypeKVCacheSpecs) -> set[int]:
        """Distinct page sizes across a group's cache specs."""
        return {s.page_size_bytes for s in spec.kv_cache_specs.values()}

    def group_and_unify_kv_cache_specs(
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> list[UniformTypeKVCacheSpecs] | None:
        """Group the KV cache specs and unify each group into one
        UniformTypeKVCacheSpecs. Currently, this is only used for DeepseekV4.
        """
        if not any(isinstance(spec, SlidingWindowMLASpec) for spec in kv_cache_spec.values()):
            return None

        logical_block_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
        grouped_swa_mla_specs: dict[int, dict[str, KVCacheSpec]] = defaultdict(dict)
        for name, spec in kv_cache_spec.items():
            if isinstance(spec, SlidingWindowMLASpec):
                grouped_swa_mla_specs[spec.block_size][name] = spec
            elif isinstance(spec, MLAAttentionSpec):
                logical_block_specs[spec.block_size][name] = spec

        mla_uniform_specs = []
        for block_size in sorted(logical_block_specs):
            spec_dict = logical_block_specs[block_size]
            assert len(spec_dict) > 0
            mla_uniform_specs.append(UniformTypeKVCacheSpecs.from_specs(spec_dict))
        assert mla_uniform_specs is not None

        swa_uniform_specs: list[UniformTypeKVCacheSpecs] = []
        for spec_dict in grouped_swa_mla_specs.values():
            uniform_spec = UniformTypeKVCacheSpecs.from_specs(spec_dict)
            assert uniform_spec is not None
            swa_uniform_specs.append(uniform_spec)

        return [*mla_uniform_specs, *swa_uniform_specs]

    def _get_kv_cache_groups_uniform_groups(
        grouped_specs: list[UniformTypeKVCacheSpecs],
    ) -> list[KVCacheGroup]:
        """Generate the KV cache groups from the grouped specs."""
        assert len(grouped_specs) > 0 and all(isinstance(spec, UniformTypeKVCacheSpecs) for spec in grouped_specs)
        if any(
            getattr(spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE
            for group in grouped_specs
            for spec in group.kv_cache_specs.values()
        ):
            return tq_cache.group_specs(grouped_specs)
        # For now, we restrict the first grouped_spec to be UniformTypeKVCacheSpecs
        # containing only MLAAttentionSpec.
        full_mla_spec = grouped_specs[0]
        full_mla_c128_spec = grouped_specs[1]

        assert all(isinstance(spec, MLAAttentionSpec) for spec in full_mla_spec.kv_cache_specs.values())
        full_mla_group = KVCacheGroup(
            layer_names=list(full_mla_spec.kv_cache_specs.keys()),
            kv_cache_spec=full_mla_spec,
        )
        full_mla_c128_group = KVCacheGroup(
            layer_names=list(full_mla_c128_spec.kv_cache_specs.keys()),
            kv_cache_spec=full_mla_c128_spec,
        )

        # We define a layer tuple as a group of layers with different page sizes, and
        # one UniformTypeKVCacheSpecs contains a list of layer tuples.
        # For example, if we have 11 C4 layers and 10 C128 layers, we can define a layer
        # tuple as [C4I, C4A, C128], and the full_mla_group will contain "11" layer tuples.
        # The other uniform KV cache specs will be similarly partitioned into layer tuples.
        # Say we have 21 SWA layers, all with the same page size, then we will have "21"
        # layer tuples.
        num_layer_tuples_per_group: list[int] = [g_spec.get_max_layers_per_page_size() for g_spec in grouped_specs]
        # Choose `num_layer_tuples` to minimize total padding across groups.
        num_layer_tuples = _approximate_gcd(num_layer_tuples_per_group, lower_bound=num_layer_tuples_per_group[0])
        # Round up to the nearest multiple of `num_layer_tuples` (i.e., padding)
        num_layer_tuples_per_group = [round_up(x, num_layer_tuples) for x in num_layer_tuples_per_group]

        # TODO(cmq): this is not general enough
        swa_mla_specs = grouped_specs[2:]

        assert all(
            isinstance(spec, SlidingWindowMLASpec) for group in swa_mla_specs for spec in group.kv_cache_specs.values()
        )

        # Split each SWA UniformKV group into smaller groups to align their #(layer tuples)
        # Possibly padding layer tuples for this.
        # Additionally, we also pad KV blocks in each SWA layer, to align the page size
        # with the corresponding layer in the full-MLA group.
        all_page_sizes = _page_sizes(full_mla_spec)
        swa_mla_groups = []
        for sm_spec in swa_mla_specs:
            sm_page_sizes = _page_sizes(sm_spec)
            layers_per_size: dict[int, list[str]] = defaultdict(list)
            assert max(sm_page_sizes) <= max(all_page_sizes)

            # Unify page size by padding layers' page_size to the nearest larger page_size.
            # Compute candidate (nearest larger page_size) for each unique page size.
            size_to_candidate: dict[int, int] = {}
            for ps in sm_page_sizes:
                size_to_candidate[ps] = min(x for x in all_page_sizes if x >= ps)
            # Pad and collect layer names per page size.
            for layer_name, layer_spec in sm_spec.kv_cache_specs.items():
                current_size = layer_spec.page_size_bytes
                candidate = size_to_candidate[current_size]
                if current_size < candidate:
                    object.__setattr__(layer_spec, "page_size_padded", candidate)
                layers_per_size[candidate].append(layer_name)
            # NOTE(yifan): for now, inside a UniformKV group, each page_size should
            # have the same number of layers. This also means we don't need to pad layers
            # inside a partial-full layer tuple.
            assert len(set(len(layers) for layers in layers_per_size.values())) == 1
            num_layers_per_size = len(next(iter(layers_per_size.values())))

            # Split layers inside each UniformKV group for aligned #(layers).
            num_tuple_groups = cdiv(num_layers_per_size, num_layer_tuples)
            layer_tuples = list(zip(*layers_per_size.values()))
            for i in range(num_tuple_groups):
                group_layer_tuples = layer_tuples[i::num_tuple_groups]
                # Flatten tuples and build dict for from_specs
                group_layer_names = [name for layer_tuple in group_layer_tuples for name in layer_tuple]
                group_layer_specs = {name: sm_spec.kv_cache_specs[name] for name in group_layer_names}
                sub_sm_spec = UniformTypeKVCacheSpecs.from_specs(group_layer_specs)
                assert sub_sm_spec is not None
                swa_mla_groups.append(
                    KVCacheGroup(
                        layer_names=group_layer_names,
                        kv_cache_spec=sub_sm_spec,
                    )
                )

        return [full_mla_group, full_mla_c128_group, *swa_mla_groups]
else:

    def _group_deepseek_v4_layers(layer_configs: dict[str, KVCacheLayerConfig]) -> list[KVCacheGroup]:
        full: defaultdict[tuple, dict[str, KVCacheLayerConfig]] = defaultdict(dict)
        local: defaultdict[tuple, dict[str, KVCacheLayerConfig]] = defaultdict(dict)
        for name, config in layer_configs.items():
            if config.manager_cls is SlidingWindowManager:
                local[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
            else:
                full[config.spec.block_size, config.manager_config, config.spec.dcp_sharded][name] = config
        full_groups = [layers for _, layers in sorted(full.items(), key=lambda item: item[0][0])]
        grouped_layers = [*full_groups, *local.values()]
        if any(
            getattr(config.spec, "cache_dtype_str", None) == TURBOQUANT_CACHE_DTYPE for config in layer_configs.values()
        ):
            return tq_cache.group_layers(grouped_layers)

        if not full_groups:
            return [KVCacheGroup.from_layers(group) for group in grouped_layers]
        page_sizes = {config.spec.page_size_bytes for config in full_groups[0].values()}
        counts = [
            max(Counter(config.spec.page_size_bytes for config in group.values()).values()) for group in grouped_layers
        ]
        num_tuples = _approximate_gcd(counts, lower_bound=counts[0])
        groups = [KVCacheGroup.from_layers(group) for group in full_groups]
        for layers in local.values():
            buckets = defaultdict(list)
            for name, config in layers.items():
                current_size = config.spec.page_size_bytes
                page_size = min(size for size in page_sizes if size >= current_size)
                if page_size != current_size:
                    config = replace(config, spec=replace(config.spec, page_size_padded=page_size))
                buckets[page_size].append((name, config))
            assert len({len(bucket) for bucket in buckets.values()}) == 1
            tuples = list(zip(*buckets.values(), strict=True))
            num_groups = cdiv(len(tuples), num_tuples)
            for i in range(num_groups):
                groups.append(KVCacheGroup.from_layers(dict(pair for slot in tuples[i::num_groups] for pair in slot)))
        return groups


def _get_deepseek_v4_cache_layout(
    kv_cache_groups: list[KVCacheGroup],
) -> tuple[list[int], list[dict[int, list[str]]], list[str], int, int]:
    """Return the geometry shared by DSV4 planning and memory accounting.

    Precondition: kv_cache_groups[0] is the full-MLA group; its page sizes
    define the canonical bucket set. Non-full-MLA groups must have been
    page_size-padded upstream so every layer's page_size matches one of the
    full-MLA bucket sizes.

    For each group, bucket its layers by page_size_bytes and place each layer
    at tuple_idx = position-within-bucket.
    """
    if vllm_version_is("0.33.0"):
        full_mla_spec = kv_cache_groups[0].kv_cache_spec
        assert isinstance(full_mla_spec, UniformTypeKVCacheSpecs)

    page_sizes = sorted(
        _page_sizes(full_mla_spec)
        if vllm_version_is("0.33.0")
        else {spec.page_size_bytes for spec in kv_cache_groups[0].distinct_specs}
    )

    # Pre-bucket each group's layers by page_size (registration order within
    # bucket). bucketed[g_idx][page_size] = [layer_name, ...].
    mtp_layer_names = []
    mtp_page_size = 0
    bucketed: list[dict[int, list[str]]] = []
    for group in kv_cache_groups:
        if vllm_version_is("0.33.0"):
            assert isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)

        specs = (group.kv_cache_spec if vllm_version_is("0.33.0") else group).kv_cache_specs
        b: dict[int, list[str]] = defaultdict(list)
        for name in group.layer_names:
            padded = getattr(specs[name], "page_size_padded", None)
            page_size = padded if padded is not None else specs[name].page_size_bytes
            if "mtp" not in name:
                b[page_size].append(name)
            else:
                mtp_layer_names.append(name)
                mtp_page_size = page_size
        bucketed.append(b)

    # num_layer_tuples = longest bucket list across all groups. For the
    # full-MLA group this equals the count of layers in the largest
    # per-page-size bucket (= get_num_layer_tuples()); for SWA sub-groups
    # this equals the sub-group size (each has a single page_size).
    num_layer_tuples = max(len(layers) for b in bucketed for layers in b.values()) + len(mtp_layer_names)

    return page_sizes, bucketed, mtp_layer_names, mtp_page_size, num_layer_tuples


if vllm_version_is("0.33.0"):

    def _is_deepseek_v4_groups(kv_cache_groups: list[KVCacheGroup]) -> bool:
        if (
            is_deepseek_v41_cache(kv_cache_groups)
            or not kv_cache_groups
            or not all(isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs) for group in kv_cache_groups)
        ):
            return False
        for group in kv_cache_groups:
            group_spec = group.kv_cache_spec
            assert isinstance(group_spec, UniformTypeKVCacheSpecs)
            specs = group_spec.kv_cache_specs.values()
            if any(getattr(spec, "model_version", None) == "deepseek_v4" for spec in specs):
                return True
        return False

    def _deepseek_v4_pool_bytes_per_block(kv_cache_groups: list[KVCacheGroup]) -> int:
        """Bytes one global block ID consumes across the shared tuple slots.

        vLLM re-plans the ranks with more KV memory against this value. The
        standardized per-group layout has a different divisor from Ascend's
        shared-tuple layout, so deriving it from the tuple geometry keeps every
        rank's ``num_blocks`` consistent.
        """
        page_sizes, _, _, _, num_layer_tuples = _get_deepseek_v4_cache_layout(kv_cache_groups)
        return sum(page_sizes) * num_layer_tuples
else:

    def _place_deepseek_v4_cache(kv_cache_groups: list[KVCacheGroup], num_blocks: int) -> list[KVCacheTensor]:
        (
            page_sizes,
            bucketed,
            mtp_layer_names,
            mtp_page_size,
            num_tuple_slots,
        ) = _get_deepseek_v4_cache_layout(kv_cache_groups)

        bytes_per_tuple = sum(page_sizes)
        tuple_stride = bytes_per_tuple * num_blocks
        backing_size = tuple_stride * num_tuple_slots

        # Within every tuple slot, page-size buckets are placed consecutively.
        page_offsets: dict[int, int] = {}
        page_prefix = 0
        for page_size in page_sizes:
            page_offsets[page_size] = page_prefix * num_blocks
            page_prefix += page_size

        tensors: list[KVCacheTensor] = []

        # Keep each descriptor inside one cache group. Descriptors from different
        # groups alias corresponding tuple slots by using identical geometry.
        for group_buckets in bucketed:
            for page_size in page_sizes:
                layer_names = group_buckets.get(page_size)
                if not layer_names:
                    continue

                tensors.append(
                    KVCacheTensor(
                        size=backing_size,
                        layers=list(layer_names),
                        offset=page_offsets[page_size],
                        layer_stride=tuple_stride,
                        block_stride=page_size,
                    )
                )

        # MTP layers receive trailing tuple slots. Unused page buckets remain
        # padding so num_blocks and memory accounting retain the existing contract.
        normal_tuple_slots = num_tuple_slots - len(mtp_layer_names)
        for index, layer_name in enumerate(mtp_layer_names):
            slot = normal_tuple_slots + index
            tensors.append(
                KVCacheTensor(
                    size=backing_size,
                    layers=[layer_name],
                    offset=slot * tuple_stride + page_offsets[mtp_page_size],
                    layer_stride=0,
                    block_stride=mtp_page_size,
                )
            )

        return tensors
