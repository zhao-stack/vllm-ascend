# vLLM main2main Interface Compatibility Report

- vLLM range: `1f486d96a17303ce8db8e02be39545b2be338446` → `95ed0feaa5cd7fb16d72c53ce04950aaf07c4698`
- vllm-ascend baseline: `3b75c4ecf8ef471fc751ce34af806e1be407f397`
- Required downstream changes: 5
- Strict contract incompatibilities, including review items: 5
- Compatibility warnings: 1
- Preexisting issues: 30
- Statically unresolved: 242

## Required Upgrade Work

### direct_call / call arguments: vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py:91

- Change: callable parameter contract changed
- Downstream interface: `RecomputeCPUOffloadScheduler.get_kv_cache_coordinator`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:461

- Change: upstream symbol was removed
- Downstream interface: `.self._generate_tool_call_id`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call target: vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:757

- Change: upstream symbol was removed
- Downstream interface: `.self._reset_streaming_state`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/block_table.py:160

- Change: callable parameter contract changed
- Downstream interface: `BlockTable._compute_slot_mapping_kernel[num_reqs + 1,]`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_import / imported symbol: vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:38

- Change: import module or symbol was removed
- Downstream interface: `.DeepSeekV4ToolParser`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

## Manual Review

### override / review: vllm_ascend/_310p/kv_block_zeroer.py:32

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/_310p/npu_input_batch.py:10

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/core/single_type_kv_cache_manager.py:31

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:69

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/worker/npu_input_batch.py:34

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/worker/utils.py:61

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
