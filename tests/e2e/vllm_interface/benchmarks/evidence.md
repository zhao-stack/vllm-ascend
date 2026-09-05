# Reviewed contract evidence

The JSON manifests contain exact old/new upstream and baseline/adapted Ascend
SHAs. Git contents at those commits take precedence over PR description wording.
Expected decisions below were established through source review and independent
attribute/call-binding checks, not copied wholesale from analyzer output.

## PR13358 development cases

- [Ascend PR13358](https://github.com/vllm-project/vllm-ascend/pull/13358): upstream
  `KVConnectorBase_V1.requires_kv_delivery` newly reads `_kv_transfer_config`.
  Both Mooncake connector constructors override initialization without setting
  that field or calling the upstream initializer. These are two affected consumers
  of one newly introduced upstream state requirement, not two upstream roots.
  The initial development annotation incorrectly split them into independent
  roots. Its first failing score is retained; the corrected case requires both
  consumer findings under one root, without changing detector decisions.
- The upstream DFlash `_prepare_dflash_inputs_kernel` adds temperature/seed input
  and output pointers. The installed Ascend Triton replacement retains the old
  parameter list. Kernel launch signature alignment must remain a P0 patch result.
- `MultiHeadLatentAttentionWrapper.__init__` adds
  `non_causal_multi_token_decode` and `allow_short_prefill_indexer_scoring_skip`.
  `AscendMultiHeadLatentAttention.__init__` must accept the expanded interface.
  The optional nature of these parameters does not waive full override alignment.

## PR13477 development cases

- [vLLM PR44941](https://github.com/vllm-project/vllm/pull/44941) removes the old
  `FusedMoE` factory export. Three Ascend imports, two direct calls, one saved
  module attribute read, and one monkey-patch installation group share this
  upstream removal root. [Ascend PR13477](https://github.com/vllm-project/vllm-ascend/pull/13477)
  changes the affected references to `FusedMoEFactory`.
- The exact upstream interval removes `GPUModelRunner.calculate_kv_scales` and
  the runtime registry keys `Q_SCALE_CONSTANT`, `K_SCALE_CONSTANT`, and
  `V_SCALE_CONSTANT`. Remaining type-only declarations do not make the environment
  attributes available. Ascend reads all three in `DSAAttention.__init__`; the
  adaptation removes those reads and gates the old runner field use.
  The related change is [vLLM PR49389](https://github.com/vllm-project/vllm/pull/49389).
- [vLLM PR50721](https://github.com/vllm-project/vllm/pull/50721) removes
  `RoutedExpertsCapturer.clear_buffer`. The downstream typed capturer call remains
  in the pre-upgrade baseline and is gated by the adaptation.
- [vLLM PR50910](https://github.com/vllm-project/vllm/pull/50910) renames
  `gumbel_sample` keyword arguments to `logits_cache` and `logits_cache_col`.
  The actual new speculator call fails CPython binding against the downstream
  patch. The Ascend PR leaves this implementation unchanged: this is an adaptation
  omission, not an analyzer false positive.
- IPC `trainer_init` makes `source` optional in its signature but still raises
  `ValueError` for `None`. MLA prefill renames `chunk_idx` to `chunk`, while the
  downstream method is a raise-only placeholder. CPU offload adds five optional
  constructor parameters, while the current downstream caller uses only the old
  three. All three are strict interface-alignment cases, not reproduced successful
  upstream execution paths that now fail on Ascend.
- `postprocess_mamba_fused_kernel` has identical old/new upstream signatures, but
  the baseline Ascend signature is already incompatible. The PR fixes that
  historical signature gap; other buffer/algorithm changes in the file have
  separate semantics outside this signature case.

## PR14131 independent acceptance cases

The following expectations are frozen before the first full scan of
[Ascend PR14131](https://github.com/vllm-project/vllm-ascend/pull/14131):

- Upstream `InputBatch.make_dummy` adds optional `max_query_len`. The pinned
  downstream override lacks it; the adapted main-lane override accepts it and
  forwards it to the upstream classmethod. Expect a P1 interface-alignment root.
- Upstream `postprocess_mamba_fused_kernel` adds `TEMPORAL_TILES=1`. The baseline
  patch lacks it; the adapted main-lane replacement accepts it. Expect a P0
  monkey-patch signature root. This check does not validate the new 3-D launch
  grid or tiled-copy algorithm.
- Upstream `BaseRouter.__init__` assigns `self.top_k = top_k` at both endpoint
  commits. The 310P router's PR change concerns tensor dtype. Its unchanged
  inherited `top_k` reads must not become field-presence findings.

The new dataclass default ordering, tensor layouts/dtypes, scheduling behavior,
and other PR changes are outside these three acceptance cases. They must not be
counted as validated merely because these selected cases pass.
