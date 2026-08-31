# vLLM main2main Interface Compatibility Report

- vLLM range: `1f486d96a17303ce8db8e02be39545b2be338446` → `e5588e49bc2642670116664a7fc4096e27adb179`
- vllm-ascend baseline: `ccc0a3f1c9c6cc36b5ac38274bebf8e82019be05`
- Required downstream changes: 2
- Strict contract incompatibilities, including review items: 2
- Compatibility warnings: 1
- Preexisting issues: 9
- Statically unresolved: 214

## Required Upgrade Work

### direct_call / call arguments: vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py:91

- Change: callable parameter contract changed
- Downstream interface: `RecomputeCPUOffloadScheduler.get_kv_cache_coordinator`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_import / imported symbol: vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:47

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

### override / review: vllm_ascend/core/single_type_kv_cache_manager.py:33

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/patch/platform/patch_kv_cache_coordinator.py:79

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
