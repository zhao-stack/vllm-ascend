# vLLM Interface Compatibility

**Result: BREAKS FOUND**

- vLLM range: `0351e9aa1fdf1a51329d1906881528dfe61fc88e` -> `beca88e59ea75a7aa1af72a5ae50188fa91d4e3d`
- vllm-ascend baseline: `61cfd1fc6a79ae139a3c5bdb8051ba7edb9c022e`
- Scope: downstream imports, overrides, and direct upstream-call contracts
- Monkey patches, inheritance-only findings, generator reviews, and historical incompatibilities are intentionally excluded.
- Introduced breaks: 4
- Root causes: 1

## Introduced breaks

### 1. P1 direct_call / call_target_presence

- Upstream: `vllm/model_executor/layers/fused_moe/__init__.py:FusedMoE`
- Downstream: `vllm_ascend/models/minimax_m3/minimax_m3.py:549`
- Change: upstream symbol was removed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 2. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/models/deepseek_v4.py:47`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 3. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/models/minimax_m3/minimax_m3.py:43`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。

### 4. P1 direct_import / symbol_presence

- Upstream: `vllm/model_executor/layers/fused_moe/layer.py:FusedMoE`
- Downstream: `vllm_ascend/ops/fused_moe/fused_moe.py:20`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。
