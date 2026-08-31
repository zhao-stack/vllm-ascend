# vLLM main2main Interface Compatibility Report

- vLLM range: `fe784ff22e630a31fd798f392b01e0a75c18f047` → `d02df748bf9efd99022f1a062597dc3cb3808485`
- vllm-ascend baseline: `7573ea0e6e94e165181423b292da87bfd8d6c10f`
- Required downstream changes: 6
- Strict contract incompatibilities, including review items: 7
- Compatibility warnings: 4
- Preexisting issues: 6
- Statically unresolved: 190

## Required Upgrade Work

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:291

- Change: callable parameter contract changed
- Downstream interface: `AscendAutoRegressiveSpeculator.super()._multi_step_decode`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:83

- Change: callable parameter contract changed
- Downstream interface: `AscendDFlashSpeculator.self._build_draft_attn_metadata`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:88

- Change: callable parameter contract changed
- Downstream interface: `AscendDSparkSpeculator.self._build_draft_attn_metadata`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:272

- Change: callable runtime signature contract changed
- Downstream interface: `AscendAutoRegressiveSpeculator._multi_step_decode`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:293

- Change: callable runtime signature contract changed
- Downstream interface: `AscendAutoRegressiveSpeculator._build_draft_attn_metadata`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/worker/v2/spec_decode/mtp/speculator.py:74

- Change: callable runtime signature contract changed
- Downstream interface: `AscendMTPSpeculator._build_draft_attn_metadata`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

## Manual Review

### override / review: vllm_ascend/worker/v2/block_table.py:65

- Reason: The downstream override does not accept the new optional parameter `out`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/_310p/kv_block_zeroer.py:32

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/ops/linear.py:111

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/ops/linear.py:381

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/worker/utils.py:61

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
