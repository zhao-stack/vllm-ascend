# vLLM main2main Interface Compatibility Report

- vLLM range: `ce29c26b31d432b1b4bc028c46bb2c3b07a667d8` → `c7560af42487b1570c4e6f4cea5df1605a4d59fc`
- vllm-ascend baseline: `60f0238b0eec4c91fe466497ae8862daf521aecc`
- Required downstream changes: 1
- Strict contract incompatibilities, including review items: 1
- Compatibility warnings: 0
- Preexisting issues: 0
- Statically unresolved: 225

## Required Upgrade Work

### direct_call / call target: vllm_ascend/core/recompute_scheduler.py:907

- Change: upstream symbol was removed
- Downstream interface: `RecomputeScheduler.self._get_routed_experts`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

## Manual Review

No optional-contract delta or masked preexisting incompatibility was proven.
## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
