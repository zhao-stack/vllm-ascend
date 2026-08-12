# vLLM Interface Compatibility

**Result: BREAKS FOUND**

- vLLM range: `d02df748bf9efd99022f1a062597dc3cb3808485` -> `0351e9aa1fdf1a51329d1906881528dfe61fc88e`
- vllm-ascend baseline: `1be01b66dd14848cb6c0422381b3263b6343b24b`
- Scope: downstream imports, overrides, and direct upstream-call contracts
- Monkey patches, inheritance-only findings, generator reviews, and historical incompatibilities are intentionally excluded.
- Introduced breaks: 4
- Root causes: 4

## Introduced breaks

### 1. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/mla.py:MultiHeadLatentAttentionWrapper.__init__`
- Downstream: `vllm_ascend/ops/mla.py:67`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 2. P1 override / call_arguments

- Upstream: `vllm/model_executor/layers/vocab_parallel_embedding.py:VocabParallelEmbedding.__init__`
- Downstream: `vllm_ascend/ops/vocab_parallel_embedding.py:52`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 3. P1 override / call_arguments

- Upstream: `vllm/v1/worker/gpu/spec_decode/speculator.py:DraftModelSpeculator._build_draft_attn_metadata`
- Downstream: `vllm_ascend/worker/v2/spec_decode/autoregressive/speculator.py:371`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 4. P1 override / call_arguments

- Upstream: `vllm/v1/worker/gpu/states.py:RequestState.__init__`
- Downstream: `vllm_ascend/worker/v2/states.py:27`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。
