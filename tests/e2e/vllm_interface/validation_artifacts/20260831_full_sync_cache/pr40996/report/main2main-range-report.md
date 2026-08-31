# vLLM main2main Interface Compatibility Report

- vLLM range: `2d814a00820daec7082599bea75ae1d0959a346c` → `95ed0feaa5cd7fb16d72c53ce04950aaf07c4698`
- vllm-ascend baseline: `3b75c4ecf8ef471fc751ce34af806e1be407f397`
- Required downstream changes: 1
- Strict contract incompatibilities, including review items: 1
- Compatibility warnings: 0
- Preexisting issues: 2
- Statically unresolved: 229

## Required Upgrade Work

### direct_call / call arguments: vllm_ascend/worker/block_table.py:160

- Change: callable parameter contract changed
- Downstream interface: `BlockTable._compute_slot_mapping_kernel[num_reqs + 1,]`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

## Manual Review

### override / review: vllm_ascend/_310p/npu_input_batch.py:10

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/worker/npu_input_batch.py:34

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
