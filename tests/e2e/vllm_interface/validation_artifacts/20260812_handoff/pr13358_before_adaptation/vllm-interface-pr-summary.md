# vLLM Interface Compatibility

**Result: BREAKS FOUND**

- vLLM range: `d02df748bf9efd99022f1a062597dc3cb3808485` -> `0351e9aa1fdf1a51329d1906881528dfe61fc88e`
- vllm-ascend baseline: `97f72b814140520e7a20622dc76b2d2fcdca0f7a`
- Scope: downstream imports, overrides, and direct upstream-call contracts
- Monkey patches, inheritance-only findings, generator reviews, and historical incompatibilities are intentionally excluded.
- Introduced breaks: 10
- Root causes: 10

## Introduced breaks

### 1. P1 direct_call / call_arguments

- Upstream: `vllm/v1/worker/gpu/cudagraph_utils.py:prepare_inputs_to_capture`
- Downstream: `vllm_ascend/worker/v2/spec_decode/eagle/aclgraph.py:105`
- Change: callable parameter contract changed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 2. P1 direct_import / symbol_presence

- Upstream: `vllm/distributed/weight_transfer/ipc_engine.py:IPCTrainerSendWeightsArgs`
- Downstream: `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py:24`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 3. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:MoERunner._maybe_reduce_shared_expert_output`
- Downstream: `vllm_ascend/ops/fused_moe/fused_moe.py:118`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 4. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:MoERunner._maybe_reduce_final_output`
- Downstream: `vllm_ascend/ops/fused_moe/fused_moe.py:128`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 5. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/mla.py:MultiHeadLatentAttentionWrapper.__init__`
- Downstream: `vllm_ascend/ops/mla.py:67`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 6. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/vocab_parallel_embedding.py:VocabParallelEmbedding.__init__`
- Downstream: `vllm_ascend/ops/vocab_parallel_embedding.py:52`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 7. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/vocab_parallel_embedding.py:ParallelLMHead.__init__`
- Downstream: `vllm_ascend/ops/vocab_parallel_embedding.py:256`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 8. P1 override / call_arguments

- Upstream: `vllm/distributed/parallel_state.py:GroupCoordinator.__init__`
- Downstream: `vllm_ascend/patch/worker/patch_distributed.py:101`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 9. P1 override / call_arguments

- Upstream: `vllm/v1/worker/gpu/spec_decode/speculator.py:DraftModelSpeculator._build_draft_attn_metadata`
- Downstream: `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:371`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 10. P1 override / call_arguments

- Upstream: `vllm/v1/worker/gpu/states.py:RequestState.__init__`
- Downstream: `vllm_ascend/worker/v2/states.py:27`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。
