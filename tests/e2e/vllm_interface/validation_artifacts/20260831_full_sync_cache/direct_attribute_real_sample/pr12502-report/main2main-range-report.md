# vLLM main2main Interface Compatibility Report

- vLLM range: `54503ecec0f3ac31e5ecfc5f28652e4cc42307b5` → `fe784ff22e630a31fd798f392b01e0a75c18f047`
- vllm-ascend baseline: `6003e3222b7a6d2f08753e03fe2aa44690da2dcf`
- Required downstream changes: 41
- Strict contract incompatibilities, including review items: 41
- Compatibility warnings: 11
- Preexisting issues: 0
- Statically unresolved: 216

## Required Upgrade Work

### direct_attribute / member attribute: vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:79

- Change: upstream symbol was removed
- Downstream interface: `AscendDFlashSpeculator.self.dflash_causal`
- Suggested action: Update the downstream member read for the new upstream object layout and add an attribute-presence regression test.

### direct_attribute / member attribute: vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:84

- Change: upstream symbol was removed
- Downstream interface: `AscendDSparkSpeculator.self.dflash_causal`
- Suggested action: Update the downstream member read for the new upstream object layout and add an attribute-presence regression test.

### direct_call / call arguments: vllm_ascend/core/recompute_scheduler.py:634

- Change: callable parameter contract changed
- Downstream interface: `RecomputeScheduler.self._mamba_block_aligned_split`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:341

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:342

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:369

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:370

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:425

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/ops/gdn.py:426

- Change: upstream symbol was removed
- Downstream interface: `AscendGatedDeltaNetAttention.l2norm_fwd`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/patch/platform/patch_balance_schedule.py:581

- Change: callable parameter contract changed
- Downstream interface: `BalanceScheduler.self._mamba_block_aligned_split`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:47

- Change: callable parameter contract changed
- Downstream interface: `AscendDFlashSpeculator.super().set_attn`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:59

- Change: callable parameter contract changed
- Downstream interface: `AscendDSparkSpeculator.super().set_attn`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/eagle/speculator.py:146

- Change: callable parameter contract changed
- Downstream interface: `AscendEagleSpeculator.super().set_attn`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_import / imported symbol: vllm_ascend/_310p/ops/fla/idex.py:19

- Change: import module moved: vllm/model_executor/layers/fla/ops/index.py -> vllm/third_party/flash_linear_attention/ops/index.py
- Downstream interface: `.prepare_lens`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/_310p/ops/fla/idex.py:20

- Change: import module moved: vllm/model_executor/layers/fla/ops/utils.py -> vllm/third_party/flash_linear_attention/ops/utils.py
- Downstream interface: `.tensor_cache`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/_310p/ops/fla/l2norm.py:24

- Change: import module moved: vllm/model_executor/layers/fla/ops/utils.py -> vllm/third_party/flash_linear_attention/ops/utils.py
- Downstream interface: `.tensor_cache`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/ops/bailing_moe_linear_attn.py:29

- Change: import module moved: vllm/model_executor/layers/fla/ops/layernorm_guard.py -> vllm/third_party/flash_linear_attention/ops/layernorm_guard.py
- Downstream interface: `.layernorm_fn`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/ops/gdn.py:22

- Change: import module moved: vllm/model_executor/layers/fla/ops/l2norm.py -> vllm/third_party/flash_linear_attention/ops/l2norm.py
- Downstream interface: `.l2norm_fwd`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/ops/triton/fla/chunk.py:17

- Change: import module moved: vllm/model_executor/layers/fla/ops/utils.py -> vllm/third_party/flash_linear_attention/ops/utils.py
- Downstream interface: `.SUPPRESS_LEVEL`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/patch/worker/patch_idex_310.py:14

- Change: import module moved: vllm/model_executor/layers/fla/ops/index.py -> vllm/third_party/flash_linear_attention/ops/index.py
- Downstream interface: `.vllm.model_executor.layers.fla.ops.index.prepare_chunk_indices`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/patch/worker/patch_idex_310.py:16

- Change: import module moved: vllm/model_executor/layers/fla/ops/index.py -> vllm/third_party/flash_linear_attention/ops/index.py
- Downstream interface: `.vllm.model_executor.layers.fla.ops.index.prepare_chunk_offsets`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/patch/worker/patch_triton.py:1

- Change: import module moved: vllm/model_executor/layers/fla/ops/__init__.py -> vllm/third_party/flash_linear_attention/ops/__init__.py
- Downstream interface: `.vllm.model_executor.layers.fla.ops`
- Suggested action: Update the imported module or symbol path and add an import-boundary regression test.

### direct_import / imported symbol: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:13

- Change: import module or symbol was removed
- Downstream interface: `.AttentionStatePair`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

### direct_import / imported symbol: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:16

- Change: import module or symbol was removed
- Downstream interface: `.PrefillSpeculatorCudaGraphManager`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

### direct_import / imported symbol: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:16

- Change: import module or symbol was removed
- Downstream interface: `.DecodeSpeculatorCudaGraphManager`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

### direct_import / imported symbol: vllm_ascend/worker/v2/spec_decode/eagle/speculator.py:31

- Change: import module or symbol was removed
- Downstream interface: `.AttentionStatePair`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

### inheritance / upstream target: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:32

- Change: upstream target existed at old and was removed at new
- Downstream interface: `PrefillEagleAclGraphManager.PrefillEagleAclGraphManager`
- Suggested action: Review the new base-class path and MRO; do not guess a replacement when the inheritance chain is incomplete.

### inheritance / upstream target: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:124

- Change: upstream target existed at old and was removed at new
- Downstream interface: `DecodeEagleAclGraphManager.DecodeEagleAclGraphManager`
- Suggested action: Review the new base-class path and MRO; do not guess a replacement when the inheritance chain is incomplete.

### monkey_patch / upstream target: vllm_ascend/patch/worker/patch_idex_310.py:14

- Change: upstream target existed at old and was removed at new
- Downstream interface: `.prepare_chunk_indices_310`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### monkey_patch / upstream target: vllm_ascend/patch/worker/patch_idex_310.py:16

- Change: upstream target existed at old and was removed at new
- Downstream interface: `.prepare_chunk_offsets_310`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### monkey_patch / upstream target: vllm_ascend/patch/worker/patch_triton.py:15

- Change: upstream target existed at old and was removed at new
- Downstream interface: `.fused_recurrent_gated_delta_rule_fwd_kernel`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### monkey_patch / upstream target: vllm_ascend/patch/worker/patch_triton.py:124

- Change: upstream target existed at old and was removed at new
- Downstream interface: `._fused_recurrent_packed_decode_pytorch`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### override / call arguments: vllm_ascend/core/single_type_kv_cache_manager.py:38

- Change: callable runtime signature contract changed
- Downstream interface: `CompressAttentionManager.get_num_blocks_to_allocate`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / replacement return protocol: vllm_ascend/core/single_type_kv_cache_manager.py:197

- Change: callable return contract changed
- Downstream interface: `CompressAttentionManager.find_longest_cache_hit`
- Suggested action: Update the patch or override return protocol for the new upstream contract and add a return-value regression test.

### override / replacement return protocol: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:255

- Change: callable return contract changed
- Downstream interface: `AscendHybridKVCacheCoordinator.find_longest_cache_hit`
- Suggested action: Update the patch or override return protocol for the new upstream contract and add a return-value regression test.

### override / replacement return protocol: vllm_ascend/patch/platform/patch_mamba_manager.py:26

- Change: callable return contract changed
- Downstream interface: `AscendMambaManager.find_longest_cache_hit`
- Suggested action: Update the patch or override return protocol for the new upstream contract and add a return-value regression test.

### override / call arguments: vllm_ascend/patch/platform/patch_mamba_manager.py:52

- Change: callable runtime signature contract changed
- Downstream interface: `AscendMambaManager.get_num_blocks_to_allocate`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:41

- Change: callable runtime signature contract changed
- Downstream interface: `AscendDFlashSpeculator.set_attn`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:53

- Change: callable runtime signature contract changed
- Downstream interface: `AscendDSparkSpeculator.set_attn`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/eagle/speculator.py:140

- Change: callable runtime signature contract changed
- Downstream interface: `AscendEagleSpeculator.set_attn`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/eagle/speculator.py:166

- Change: callable runtime signature contract changed
- Downstream interface: `AscendEagleSpeculator.capture`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

## Manual Review

No optional-contract delta or masked preexisting incompatibility was proven.
## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
