# vLLM Interface Compatibility

**Result: BREAKS FOUND**

- vLLM range: `1f486d96a17303ce8db8e02be39545b2be338446` -> `e5588e49bc2642670116664a7fc4096e27adb179`
- vllm-ascend baseline: `3b75c4ecf8ef471fc751ce34af806e1be407f397`
- Scope: downstream imports, overrides, and direct upstream-call contracts
- Monkey patches, inheritance-only findings, generator reviews, and historical incompatibilities are intentionally excluded.
- Introduced breaks: 5
- Root causes: 5

## Introduced breaks

### 1. P1 direct_call / call_arguments

- Upstream: `vllm/v1/core/kv_cache_coordinator.py:get_kv_cache_coordinator`
- Downstream: `vllm_ascend/distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py:91`
- Change: callable parameter contract changed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 2. P1 direct_call / call_target_presence

- Upstream: `vllm/tool_parsers/__init__.py:deepseekv4_tool_parser._generate_tool_call_id`
- Downstream: `vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:461`
- Change: upstream symbol was removed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 3. P1 direct_call / call_target_presence

- Upstream: `vllm/tool_parsers/__init__.py:deepseekv4_tool_parser._reset_streaming_state`
- Downstream: `vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:757`
- Change: upstream symbol was removed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 4. P1 direct_call / call_arguments

- Upstream: `vllm/v1/worker/block_table.py:_compute_slot_mapping_kernel`
- Downstream: `vllm_ascend/worker/block_table.py:160`
- Change: callable parameter contract changed
- Suggested action: 同步下游调用参数或返回值消费方式，并为该调用点补充接口级回归测试。

### 5. P1 direct_import / symbol_presence

- Upstream: `vllm/tool_parsers/deepseekv4_tool_parser.py:DeepSeekV4ToolParser`
- Downstream: `vllm_ascend/patch/platform/patch_deepseek_v4_tool_call_parser.py:38`
- Change: import module or symbol was removed
- Suggested action: 更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。
