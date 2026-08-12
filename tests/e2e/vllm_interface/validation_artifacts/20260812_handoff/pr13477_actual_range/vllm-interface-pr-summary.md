# vLLM Interface Compatibility

**Result: BREAKS FOUND**

- vLLM range: `0351e9aa1fdf1a51329d1906881528dfe61fc88e` -> `58d3918e3ea0a544ffedadad2ba84559e9c51d8f`
- vllm-ascend baseline: `86db2ed32e714f5395905d144494b78a99964dca`
- Scope: downstream imports, overrides, and direct upstream-call contracts
- Monkey patches, inheritance-only findings, generator reviews, and historical incompatibilities are intentionally excluded.
- Introduced breaks: 8
- Root causes: 4

## Introduced breaks

### 1. P1 direct_call / call_target_presence

- Upstream: `vllm/model_executor/layers/fused_moe/__init__.py:FusedMoE`
- Downstream: `vllm_ascend/models/deepseek_v4.py:436`
- Change: upstream symbol was removed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 2. P1 direct_call / call_target_presence

- Upstream: `vllm/model_executor/layers/fused_moe/__init__.py:FusedMoE`
- Downstream: `vllm_ascend/models/minimax_m3/minimax_m3.py:549`
- Change: upstream symbol was removed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 3. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/models/deepseek_v4.py:47`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 4. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/models/minimax_m3/minimax_m3.py:43`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 5. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/ops/fused_moe/fused_moe.py:21`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 6. P1 override / call_arguments

- Upstream: `vllm/distributed/weight_transfer/ipc_engine.py:IPCTrainerWeightTransferEngine.trainer_init`
- Downstream: `vllm_ascend/distributed/weight_transfer/npu_ipc_engine.py:613`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 7. P1 override / call_arguments

- Upstream: `vllm/v1/attention/backends/mla/prefill/base.py:MLAPrefillBackend.run_prefill_context_chunk`
- Downstream: `vllm_ascend/patch/platform/patch_mla_prefill_backend.py:38`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。

### 8. P1 override / call_arguments

- Upstream: `vllm/v1/simple_kv_offload/worker.py:SimpleCPUOffloadWorker.__init__`
- Downstream: `vllm_ascend/simple_kv_offload/worker.py:62`
- Change: callable runtime signature contract changed
- Suggested action: 同步 override 参数并检查 super() 调用和关键字转发。
