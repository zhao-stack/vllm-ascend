# vLLM main2main Interface Compatibility Report

- vLLM range: `d02df748bf9efd99022f1a062597dc3cb3808485` → `0351e9aa1fdf1a51329d1906881528dfe61fc88e`
- vllm-ascend baseline: `97f72b814140520e7a20622dc76b2d2fcdca0f7a`
- Required downstream changes: 8
- Strict contract incompatibilities, including review items: 15
- Compatibility warnings: 15
- Preexisting issues: 6
- Statically unresolved: 229

## Required Upgrade Work

### direct_call / call arguments: vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:105

- Change: callable parameter contract changed
- Downstream interface: `EagleAclGraphManager.prepare_inputs_to_capture`
- Suggested action: Update downstream arguments or return-value consumption and add an interface regression test for this callsite.

### direct_import / imported symbol: vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py:24

- Change: import module or symbol was removed
- Downstream interface: `.IPCTrainerSendWeightsArgs`
- Suggested action: Update the downstream target. If upstream removed the capability, remove the patch or inheritance edge and add an alternative implementation.

### inheritance / upstream target: vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py:36

- Change: upstream target existed at old and was removed at new
- Downstream interface: `NPUIPCTrainerSendWeightsArgs.NPUIPCTrainerSendWeightsArgs`
- Suggested action: Review the new base-class path and MRO; do not guess a replacement when the inheritance chain is incomplete.

### monkey_patch / call arguments: vllm_ascend/patch/worker/patch_deepseek_v2.py:290

- Change: callable runtime signature contract changed
- Downstream interface: `._deepseek_v2_mla_attention_init`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### monkey_patch / call arguments: vllm_ascend/patch/worker/patch_mamba_utils.py:284

- Change: callable runtime signature contract changed
- Downstream interface: `.preprocess_mamba`
- Suggested action: Update the replacement signature for the new upstream call contract and verify that the patch installation path is still active.

### override / call arguments: vllm_ascend/_310p/ops/vocab_parallel_embedding.py:66

- Change: callable runtime signature contract changed
- Downstream interface: `AscendParallelLMHead310.__init__`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/ops/mla.py:67

- Change: callable runtime signature contract changed
- Downstream interface: `AscendMultiHeadLatentAttention.__init__`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

### override / call arguments: vllm_ascend/ops/vocab_parallel_embedding.py:256

- Change: callable runtime signature contract changed
- Downstream interface: `AscendParallelLMHead.__init__`
- Suggested action: Synchronize the override parameters and check super() calls and keyword forwarding.

## Manual Review

### override / review: vllm_ascend/_310p/ops/vocab_parallel_embedding.py:44

- Reason: The downstream override does not accept the new optional parameter `disable_tp`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/ops/fused_moe/fused_moe.py:118

- Reason: The downstream override does not accept the new optional parameter `fused_output_is_reduced`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/ops/fused_moe/fused_moe.py:128

- Reason: The downstream override does not accept the new optional parameter `output_is_reduced`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/ops/vocab_parallel_embedding.py:52

- Reason: The downstream override does not accept the new optional parameter `disable_tp`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/patch/worker/patch_distributed.py:101

- Reason: The downstream override does not accept the new optional parameter `use_all2all`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:371

- Reason: The downstream override does not accept the new optional parameter `query_start_loc_np`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/worker/v2/states.py:27

- Reason: The downstream override does not accept the new optional parameter `num_prefill_lookahead`, and no evidence proves that runtime dispatch passes that parameter to this implementation.
- Suggested action: Review whether the new optional parameter can reach this downstream override at runtime. If it can, update the override signature and handle the new argument.

### override / review: vllm_ascend/_310p/npu_input_batch.py:10

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/distributed/device_communicators/npu_communicator.py:41

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/ops/dsa.py:62

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

### override / review: vllm_ascend/worker/npu_input_batch.py:34

- Reason: Downstream was already incompatible at old, but this range adds another exact parameter delta.
- Suggested action: Upstream introduced another exact parameter delta while downstream was already incompatible at old. Review this delta separately instead of treating it as a confirmed upgrade regression.

## Notes

`preexisting` means old and new are both incompatible and is not attributed to this upgrade. `analysis_unresolved` means the available static evidence was insufficient and the analyzer did not guess.
