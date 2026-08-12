# vllm-interface 上游 CI 脚本工作交接（2026-08-12）

## 1. 交接入口

- 远程仓库：`https://github.com/zhao-stack/vllm-ascend.git`
- 交接分支：`codex/vllm-interface-handoff-20260812`
- 分支起点：`51ae7395dca6f37e8a0ab437f0d199f089bb9c1f`
- 公共分析引擎：`tools/vllm_interface_contracts`
- CI/测试说明：`tests/e2e/vllm_interface/README.md`
- 演进记录：`tests/e2e/vllm_interface/GENERATOR_CHANGELOG.md`
- 完整验收记录：`tests/e2e/vllm_interface/接口分析引擎与Skill融合重构验收记录.md`
- 本次原始报告：`tests/e2e/vllm_interface/validation_artifacts/20260812_handoff`

本分支用于继续完成 `vllm-interface` 上游 PR 感知能力及其 CI 接入。当前仍是 awareness 模式：发现 break 默认生成报告并返回成功；只有显式传入 `--fail-on introduced` 才作为阻断检查。合入上游 CI 前不建议直接开启阻断。

## 2. 当前脚本状态

| 项目 | 当前值 |
|---|---|
| generator | `0.36.0` |
| generator JSONL schema | `6` |
| range analyzer | `1.3.0` |
| range report schema | `4` |
| fixed analysis plan | `2` |
| `main2main` 场景 | patch、override、inheritance、direct import、exact direct call、return protocol、generator findings |
| `vllm-interface` 场景 | direct import、override、exact direct call；inheritance/MRO 只作前置证据 |
| `vllm-interface` 明确跳过 | monkey patch、inheritance finding、generator finding、普通字段/属性、扩展语义协议 |

当前 `vllm-interface` 的产物为：

- `vllm-interface-pr-summary.md`
- `vllm-interface-pr-report.json`
- `vllm-interface-introduced-breaks.csv`
- `vllm-interface-analysis-metadata.json`

PR 展示层只保留满足四个 action gate 的 `introduced_break/modify`：

```text
relationship_verified
AND contract_changed
AND runtime_reachable
AND version_lane_matches
```

其中 `runtime_reachable` 是“源码已证明下游 callsite 或 override 安装关系”，不是“当前模型/设备测试一定执行到该路径”。因此 override 新增可选参数仍可能被严格接口可替换性检查提升为 P1，人工复核时要继续区分“契约不完整”和“本次调用立即报错”。

## 3. 本分支尚未合入的代码变化

### 3.1 direct import 纳入上游 PR 场景

`vllm-interface` 已从“override + direct call”扩展为“direct import + override + direct call”。上游删除模块或导出符号时，脚本会报告准确的下游 import 位置。

同一个上游 callable 同时引起 import 和 call break 时，PR 摘要会按旧上游 callable 的精确指纹聚合根因；JSON/CSV 仍保留每一个下游受影响位置。

### 3.2 Triton `kernel[grid](...)` direct call

原发现器只识别普通 `func(...)`，无法从 AST 的 `Subscript` 结构中识别 `_compute_slot_mapping_kernel[grid](...)`。当前实现会：

1. 从 `kernel[grid]` 解出唯一上游 callable；
2. 使用外层 `(...)` 的位置参数和关键字参数；
3. 分别绑定 old/new kernel 定义；
4. 只接受唯一解析且仅有一个规范 `vllm.triton_utils.triton.jit` 装饰器的函数；
5. 对普通 `mapping[key](...)`、额外装饰器和无法唯一解析的目标保持 fail closed；
6. 在报告证据中写入 `invocation_kind=triton_kernel_launch`。

这项修改只解决“下游直接调用上游 Triton kernel”的语法识别，不扩大 monkey-patch replacement 的扫描范围。

### 3.3 对应文件

- `tools/vllm_interface_contracts/analysis_plans.py`
- `tools/vllm_interface_contracts/call_contracts.py`
- `tools/vllm_interface_contracts/range_analysis.py`
- `tests/e2e/vllm_interface/generator_tests/test_analysis_plans.py`
- `tests/e2e/vllm_interface/generator_tests/test_call_contracts.py`
- `tests/e2e/vllm_interface/generator_tests/test_range_analysis.py`
- `tests/e2e/vllm_interface/README.md`
- `tests/e2e/vllm_interface/GENERATOR_CHANGELOG.md`
- `tests/e2e/vllm_interface/接口分析引擎与Skill融合重构验收记录.md`

### 3.4 本分支提交前检查

- 生成器完整回归：`295 passed in 59.66s`。
- Ruff lint：通过。
- 本次修改的 Python 文件 Ruff format check：通过。
- `git diff --check`：通过。

本机没有安装上层 E2E fixture 所需的 `huggingface_hub`，直接对 `tests/e2e/vllm_interface/generator_tests` 运行 pytest 会先加载 `tests/e2e/conftest.py` 并在收集前失败。这里的生成器测试是纯源码测试，验收时使用 `--confcutdir=tests/e2e/vllm_interface/generator_tests` 隔离了无关的上层 fixture。安装完整仓库测试依赖后也可以去掉该参数再跑。

## 4. PR11709 验证

精确输入：

- vLLM old：`1f486d96a17303ce8db8e02be39545b2be338446`
- vLLM new：`e5588e49bc2642670116664a7fc4096e27adb179`
- vllm-ascend baseline：`3b75c4ecf8ef471fc751ce34af806e1be407f397`

修复 Triton 语法后得到 5 条 actionable introduced break：4 条 direct call、1 条 direct import。

| 下游位置 | 类型 | 复核结论 |
|---|---|---|
| `distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py:91` | direct call | `get_kv_cache_coordinator` 调用形态确实不再直接满足新接口；PR 的兼容 wrapper 属于 monkey patch，当前场景不会分析 wrapper 的安装和修复覆盖，因此不能仅凭该 finding 认定 PR 漏改。 |
| `patch/platform/patch_deepseek_v4_tool_call_parser.py:461` | direct call target presence | `_generate_tool_call_id` 目标删除，命中准确。 |
| `patch/platform/patch_deepseek_v4_tool_call_parser.py:757` | direct call target presence | `_reset_streaming_state` 目标删除，命中准确。 |
| `worker/block_table.py:160` | Triton direct call | 本次新增命中。old 调用可绑定；new 新增必需的 `KV_CACHE_BLOCK_SIZE`，随后还缺 `BLOCKS_PER_KV_BLOCK`。 |
| `patch/platform/patch_deepseek_v4_tool_call_parser.py:38` | direct import | `DeepSeekV4ToolParser` 旧 import 失效，命中准确。 |

原始报告：[PR11709 after Triton fix](validation_artifacts/20260812_handoff/pr11709_after_triton_fix/vllm-interface-pr-summary.md)。

## 5. PR13358 适配前后双跑

vLLM 区间：

- old：`d02df748bf9efd99022f1a062597dc3cb3808485`
- new：`0351e9aa1fdf1a51329d1906881528dfe61fc88e`

下游节点：

- PR base：`97f72b814140520e7a20622dc76b2d2fcdca0f7a`
- PR head：`1be01b66dd14848cb6c0422381b3263b6343b24b`

适配前报告 10 条：8 override、1 direct call、1 direct import。PR head 再跑后剩 4 条，说明 PR 实际消除了脚本报告中的 6 条：

- Eagle `prepare_inputs_to_capture()` 的 `full_cudagraph`；
- 删除的 `IPCTrainerSendWeightsArgs` import；
- Fused MoE 两个 override 参数；
- `AscendParallelLMHead.disable_tp`；
- `GroupCoordinatorPatch.use_all2all`。

PR head 剩余 4 条的人工结论：

1. `AscendMultiHeadLatentAttention.__init__` 缺少 `non_causal_multi_token_decode` 和 `allow_short_prefill_indexer_scoring_skip`。上游 `deepseek_v2.py` 有具体构造调用传入这两个关键字，这是 PR 的真实硬漏改。
2. `AscendVocabParallelEmbedding.__init__` 缺少可选 `disable_tp`。严格替换契约不完整，但该区间没有找到直接以 `VocabParallelEmbedding(..., disable_tp=...)` 调用的证据，建议作为 P2/review，而不是直接认定当前运行必现。
3. `AscendAutoRegressiveSpeculator._build_draft_attn_metadata` 缺少可选 `query_start_loc_np=None`。当前 AutoRegressive 调用链未传入该参数，属于接口同步风险，不是已证明的立即 break。
4. `AscendRequestState.__init__` 缺少可选 `num_prefill_lookahead=1`。下游自己构造并使用上游默认值，当前调用链不会立即报错。

脚本未体现、但 PR 实际修改的重点：

- `NPUCommunicator.use_all2all` 和 `NPUInputBatch.use_replayssm` 被 old=false/new=false 的历史签名不兼容遮蔽；需要后续增加 `new_delta_on_preexisting_break`。
- `AscendParallelLMHead310` 是下游二级子类；根因命中第一层，但未把影响传播到 `上游 -> Ascend -> 310P`。
- weight-transfer import 根因命中准确，但完整 stateful engine/factory 迁移超出单一 import finding 能给出的修改范围。
- DFlash replacement 是 monkey patch；Mooncake `_kv_transfer_config` 是字段属性；两者均不在当前场景范围内。

原始报告：

- [PR13358 before adaptation](validation_artifacts/20260812_handoff/pr13358_before_adaptation/vllm-interface-pr-summary.md)
- [PR13358 after adaptation](validation_artifacts/20260812_handoff/pr13358_after_adaptation/vllm-interface-pr-summary.md)

## 6. PR13477 验证

实际区间报告的精确输入：

- vLLM old：`0351e9aa1fdf1a51329d1906881528dfe61fc88e`
- vLLM new：`58d3918e3ea0a544ffedadad2ba84559e9c51d8f`
- vllm-ascend baseline：`86db2ed32e714f5395905d144494b78a99964dca`

当前报告为 8 条、4 个根因：2 direct call、3 direct import、3 override。FusedMoE 删除/迁移同时影响多个 import 和构造调用，脚本能够保留具体落点。

人工对照 PR 修改后，没有发现新的“直接接口或 vLLM import”阶段内漏检。`self.routed_experts_capturer.clear_buffer()` 确实会受上游方法删除影响，但 receiver 类型只能通过“上游基类方法给 `self` 普通字段赋值”才能证明；它既不是直接 import/构造，也不是下游 override。该样本已记录为下一阶段字段 receiver 类型传播的验收用例。

原始报告：

- [PR13477 actual range](validation_artifacts/20260812_handoff/pr13477_actual_range/vllm-interface-pr-summary.md)
- [PR13477 direct-import plan replay](validation_artifacts/20260812_handoff/pr13477_direct_import_plan_replay/vllm-interface-pr-summary.md)

第二份报告是用于隔离验收 direct-import 计划和根因聚合的固定回放，其 SHA 与实际 PR 区间不同，不能混作同一轮结果。

## 7. 当前准确性结论

1. direct import：在三组样本中都能准确定位“old 存在、new 不再解析”的 vLLM import；没有发现阶段内漏检。
2. direct call：普通 Python 调用以及标准 Triton `kernel[grid](...)` 必需参数变化能够准确绑定 old/new。
3. override：签名差异识别准确，但“上游新增可选参数”会按严格可替换性报告，不能全部解释为当前调用必现故障。
4. PR 修复评估：脚本适合找根因和受影响位置，但 monkey patch、factory/registration、字段属性和大范围协议迁移不一定能自动归因到完整修复点。
5. PR13358 表明当前报告适合作为“接口修改候选清单”，尚不适合不经人工复核直接阻断 CI。

## 8. 已知缺口与下一步优先级

### P1：准备上游 CI 合入

1. 将 `vllm-interface` 命令接入目标 workflow，确保 checkout 能访问 old/new 两个 vLLM SHA。
2. 上传四类原始产物，先保持非阻断；不要默认启用 `--fail-on introduced`。
3. 缓存键必须包含 engine/schema/plan、old/new vLLM SHA 和 vllm-ascend SHA；不能只按分支名缓存。
4. 在 Linux CI 上回跑本文件中的三个固定区间，确认路径、编码和 Git snapshot 行为与 Windows 一致。

### P1：降低 P1 告警噪声

1. 对 override 新增可选参数增加分级：存在具体新调用传参时保持 P1；只有可替换性差异时输出 P2/review。
2. 保留严格契约 finding，不要直接删掉；调整的是 action/priority 和展示层。

### P2：补全结构性漏检

1. old/new 都不兼容时，额外比较新引入的参数/目标差异，输出 `new_delta_on_preexisting_break`，解决 PR13358 的 `NPUCommunicator`、`NPUInputBatch` 遮蔽问题。
2. 将已证明的上游契约变化传播到下游二级/三级注册替换子类，例如 `AscendParallelLMHead310`。
3. 下一阶段实现普通字段 receiver 类型传播和 attribute presence/read；首个验收样本为 `RoutedExpertsCapturer.clear_buffer()`。
4. Mamba postprocess、buffer/index/kernel 数据协议属于语义协议阶段，不与接口/import 修改混做。

### 保持不变的边界

- 不在 `vllm-interface` 中默认启用 monkey-patch collection；main2main 全量场景继续负责 patch。
- 不把注释、同名函数或模糊 MRO 当作关系证据。
- 动态 `*args/**kwargs`、多重装饰器、无法唯一解析的 receiver 继续 fail closed/unresolved。

## 9. 换设备后的继续方式

```bash
git clone https://github.com/zhao-stack/vllm-ascend.git
cd vllm-ascend
git switch codex/vllm-interface-handoff-20260812

python -m pytest --confcutdir=tests/e2e/vllm_interface/generator_tests \
  tests/e2e/vllm_interface/generator_tests -q

python -m ruff check \
  tools/vllm_interface_contracts/analysis_plans.py \
  tools/vllm_interface_contracts/call_contracts.py \
  tools/vllm_interface_contracts/range_analysis.py \
  tests/e2e/vllm_interface/generator_tests/test_analysis_plans.py \
  tests/e2e/vllm_interface/generator_tests/test_call_contracts.py \
  tests/e2e/vllm_interface/generator_tests/test_range_analysis.py

python -m ruff format --check \
  tools/vllm_interface_contracts/analysis_plans.py \
  tools/vllm_interface_contracts/call_contracts.py \
  tools/vllm_interface_contracts/range_analysis.py \
  tests/e2e/vllm_interface/generator_tests/test_analysis_plans.py \
  tests/e2e/vllm_interface/generator_tests/test_call_contracts.py \
  tests/e2e/vllm_interface/generator_tests/test_range_analysis.py
git diff --check
```

查看 CLI：

```bash
python -m tools.vllm_interface_contracts analyze-range --help
```

典型上游 PR 分析：

```bash
python -m tools.vllm_interface_contracts analyze-range \
  --scenario vllm-interface \
  --vllm-root /path/to/vllm-at-new \
  --ascend-root /path/to/vllm-ascend-baseline \
  --expect-ascend-sha <baseline-sha> \
  --old <old-vllm-sha> \
  --new <new-vllm-sha> \
  --output-dir /path/to/output
```

运行前必须确认 vLLM checkout 的 HEAD 等于 `new`，vllm-ascend baseline 的 HEAD 等于 `--expect-ascend-sha`，且两个工作树不是意外 dirty 状态。

## 10. 原始产物目录说明

每个子目录均保留 Markdown、JSON、CSV 和 metadata 四个未经二次编辑的脚本输出：

```text
validation_artifacts/20260812_handoff/
├── pr11709_after_triton_fix/
├── pr13358_before_adaptation/
├── pr13358_after_adaptation/
├── pr13477_actual_range/
└── pr13477_direct_import_plan_replay/
```

继续修改分类或展示逻辑时，优先用这些 JSON 做结果对比；不要把人工结论反写成生成器关系或静态 mapping。
