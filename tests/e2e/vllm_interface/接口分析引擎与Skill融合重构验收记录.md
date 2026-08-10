# 接口分析引擎与 Skill 融合重构验收记录

## 1. 最终结构

- 唯一分析引擎：`tools/vllm_interface_contracts`。
- 旧生成命令：`tests/e2e/vllm_interface/generate_interface_boundaries.py`，仅作为兼容入口。
- main2main skill：只负责参数和 SHA 校验、选择 `new/legacy/compare`、精确缓存、调用公共引擎和输出中文报告。
- 默认行为：发现风险只预警；只有显式传入 `--fail-on introduced` 才返回阻断错误。
- 分析方式：只读 AST 和 Git 中的源码，不导入 vLLM、torch_npu，不需要 NPU。

## 2. 可逐步回退的提交

| 提交 | 内容 | 为什么拆开 | 当步回归 | 最终 golden 覆盖 |
|---|---|---|---|---|
| `28902b705` | 把原生成器和 schema 移到公共包，旧路径改成兼容入口 | 先只移动代码，不改变分析语义 | 原 213 个测试通过 | 是 |
| `bfe12f176` | 增加 old/new 区间分析、统一模型和 JSON/CSV/Markdown 输出 | 区间分类能力可独立回退 | 新增区间测试通过 | 是 |
| `49dfc7d5f` | 修复 import 记录中 `None` 与字符串混排 | 真实仓库回放发现的确定性排序问题 | 定向测试和全量测试通过 | 是 |
| `7a68ae82d` | 只有 old 目标被源码证明存在时才认定 direct import break | 去掉 package 子模块和旧目标不存在造成的误报 | 区间测试增至 7 个 | 是 |
| `f1363b720` | 升级报告只保留变化项，并识别 old 存在、new 删除的 patch | 避免把未变化关系写成 unresolved | 区间测试增至 9 个 | 是 |
| `c7a1a8660` | 类自身直接声明成员时，不要求外部父类 MRO 完整 | 解决当前 main 的动态 `setattr` 名称误判；父类成员仍不猜 | 223 个测试通过 | 是 |
| `a34984732` | 格式化兼容入口 | 与逻辑修改分离 | 223 个测试通过 | 是 |
| `b9bdfa0d2` | 补充公共引擎和 skill 使用说明 | 文档可独立维护 | 文档和 CLI 参数核对通过 | 是 |

最终固定源码回放：

- vLLM：`88402a41c4ab272ebbbd33f4a77fbbac0431cbb9`
- vllm-ascend：`81d3450128528be2c343232fcc28220814a15fd6`
- PyTorch 快照：`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`
- 结果：972 条关系、173 条 finding，与重构前 972 条关系全部精确匹配。
- JSONL SHA-256：`725504ef474f4cf52f6a1a06a12a0440b56c6d7ff8861a7a39e47cd48a815cc5`，逐字节一致。

## 3. 重构历史基线：PR13477 区间双跑

固定输入：

- vLLM old：`0351e9aa1fdf1a51329d1906881528dfe61fc88e`
- vLLM new：`beca88e59ea75a7aa1af72a5ae50188fa91d4e3d`
- vllm-ascend baseline：`61cfd1fc6a79ae139a3c5bdb8051ba7edb9c022e`

最终新引擎识别出 4 个 `introduced_break` 文件：

- `vllm_ascend/models/deepseek_v4.py`
- `vllm_ascend/models/minimax_m3/minimax_m3.py`
- `vllm_ascend/ops/fused_moe/fused_moe.py`
- `vllm_ascend/patch/platform/patch_fused_moe.py`

这 4 项都能证明 old 目标存在、new 目标不存在，且四个严格门槛全部为 true。JSON 中 4 条记录与 introduced-break CSV 的 4 行一致。

旧分析器独有的 `vllm_ascend/_310p/fused_moe/fused_moe.py` 是候选，不是严格 break：上游删除基类的 `is_monolithic`，但下游仍自行 override；旧结果的运行路径和版本门槛均为 unknown。

## 4. 重构历史基线：其他验收结果

- 生成器测试：223 个通过，其中原有 213 个全部保留。
- skill 测试：50 个通过，其中原有 46 个全部保留。
- skill `quick_validate`：通过。
- `py_compile`、ruff check、ruff format check：通过。
- 独立 coverage auditor：1023 个候选全部归类，missing、conflicting、orphan、generator issue review 均为 0。
- 缓存：同一组精确输入第二次调用耗时 0.388 秒，没有重新执行完整 AST 扫描。
- fresh 子代理：只根据 skill 文档，在固定小样本上成功执行 `validate` 和 `predict`，准确输出 1 条 `introduced_break` 和上下游源码证据；未发现影响使用的文档缺口。

## 5. 当前 main 无 NPU 回放

2026-08-06 从 GitHub 远端实时确认并固定：

- vLLM main：`2e09247c2d7b6b97d13af6e71a85bf8d1271deb6`
- vllm-ascend main：`bc5a79ede16fc4ded332471ae1fbc6303fb000f4`

最终生成 1011 条关系，其中 inheritance 210、monkey patch 121、override 680；generator issue 为 0。整个过程只读取源码，没有导入运行时包，也没有使用 NPU。

## 6. 本次未做的事情

- 没有接入正式 CI。
- 没有创建 PR。
- 旧 analyzer 和旧 method 映射仍保留一个兼容周期，可通过 `--engine-mode legacy` 使用。
- field、legacy static call protocol、registration 等 expanded 表仍是 opt-in，不影响默认动态分析。

## 7. 接口层双向契约补强（2026-08-06）

本轮只扩展公共引擎的区间/当前契约分析，生成器仍为 `0.36.0`、JSONL schema 仍为 6；range analyzer 升级为 `1.1.1`、range schema 升级为 2。公共引擎、兼容入口和 skill 薄适配层的职责边界没有变化。

默认 `exact-contracts` 现在覆盖两种方向、四类契约：

- 下游调用上游：逐调用点解析唯一上游目标，分别对 old/new 绑定具体位置参数和关键字参数；同时检查已证明的解包、固定下标、迭代、上下文管理器和 `await` 返回值消费。
- 上游契约到下游实现：对已证明的 patch/override，继续检查下游安装后的输入签名能否替代上游接口，并新增下游返回协议相对上游返回协议的协变检查。

对 imported、annotated 或 constructed vLLM receiver，分析只沿可唯一证明的单继承链分别解析 old/new 成员。下游 `self`/`super` 先从固定 vllm-ascend 基线的完整 MRO 证明一个有效上游 owner，再在 old/new 验证同一 owner；owner 移动不猜测。动态 `*args`/`**kwargs`、歧义 callee、多继承或外部/不完整继承链在调用发现阶段直接跳过；dependency 已证明但 old/new 端点、运行签名或受约束返回协议不可证明时，输出 `analysis_unresolved`。未使用、直接转发或逃逸的返回值不生成返回消费 finding。接口层的 `runtime_reachable` 只要求源码中存在具体调用点或已证明的 patch/override 安装关系，不要求完整模型、设备和进程运行路径已经实际触发；因此已证明的接口不匹配不会因当前尚未调用而被隐藏。

最终验证证据：

- 公共引擎生成器测试：283 passed；Ruff：通过。
- skill 工作副本和已安装副本：显式设置 `VLLM_INTERFACE_ENGINE_TEST_ROOT` 后均为 50 passed；7 个同步文件 SHA-256 逐一一致。`quick_validate` 和 Ruff 均通过。
- PR13477 固定区间最终回放：1014 条生成器关系、5005 个精确下游调用依赖、229 条生成器 finding；区间 finding 共 230 条，其中 `introduced_break=5`、`compatibility_warning=1`、`preexisting=6`、`analysis_unresolved=218`。
- 5 条 introduced break 覆盖原有 4 个风险文件；新增的第 5 条是 `vllm_ascend/models/minimax_m3/minimax_m3.py:549` 对已删除 `FusedMoE` 的精确构造调用，属于 `direct_call/call_target_presence`，与同文件的 import 删除是不同契约证据。
- 最终报告：`analysis_runs/main2main_pr13477_20260806_interface_contract_v2_final_validation/main2main-range-report.md`；完整 JSON：同目录 `main2main-range-report.json`，SHA-256 为 `fe854374cc8ef41bca52ac6022a53802fe1227f8d7266819e98300fc2513e9c7`。
- 总耗时 1027 秒，其中仓库索引 438.739 秒、关系生成 355.382 秒、区间关系/import 108.976 秒、direct-call 发现 64.350 秒、old/new endpoint 比较 52.382 秒。

因为生成器关系语义和 JSONL schema 均未变化，本轮按约定没有重复执行固定 972 条 golden 全量回放，也没有重复执行当前 main 全量回放；上一轮逐字节 golden 和当前 main 验收结果继续有效。

## 8. 按使用场景选择执行步骤（2026-08-10）

公共引擎新增两套固定执行计划，不开放任意组合的底层开关，也没有复制 AST、MRO、patch、签名或返回值分析逻辑：

- `main2main`：默认计划，执行 patch、override、inheritance、direct import、direct call、返回协议和 generator finding，继续输出原有 `main2main-*` 全量报告。
- `vllm-interface`：上游 PR 预警计划。只分析 override 和下游调用上游的参数/返回值契约；inheritance/MRO 只作为 override 解析前置步骤，不输出 inheritance finding；monkey patch、direct import 和 generator finding 转换在执行前跳过。

range analyzer 升级为 `1.2.0`、range schema 升级为 3、固定计划版本为 1。报告元数据新增场景、能力执行状态和分阶段耗时；被跳过的步骤记为 `null`，可以区分“没有发现问题”和“根本没有执行该分析”。生成器版本仍为 `0.36.0`、JSONL schema 仍为 6。

`vllm-interface` 场景只输出本次上游区间新增且需要修改的 override/direct-call break：

- `vllm-interface-pr-summary.md`：适合 CI 日志和 Job Summary；
- `vllm-interface-pr-report.json`：机器可读的 PR 新增 break；
- `vllm-interface-introduced-breaks.csv`：下载后筛选；
- `vllm-interface-analysis-metadata.json`：输入 SHA、执行计划和耗时。

PR 级产物不展示 `preexisting`、`analysis_unresolved`、patch、direct import 或 inheritance-only 项。命令默认仍不阻塞上游 PR，只有显式设置 `--fail-on` 才改变退出码。main2main skill 固定调用 `--scenario main2main`，并把场景和计划版本纳入缓存身份；skill 仍只负责校验、`new/legacy/compare` 模式、缓存和中文呈现。

本轮验收结果：

- 公共引擎测试：290 passed；场景化 CLI 日志补强后的定向区间测试：47 passed；Ruff 和 `git diff --check` 通过。
- skill 工作副本：51 passed；`quick_validate` 和 Ruff 通过。
- PR13477 `main2main` 冷启动：总耗时 1042.502 秒，1014 条关系、5005 个 direct-call 依赖、229 条 generator finding、230 条区间 finding、5 条 introduced break。与 schema 2 最终报告的 230 个 finding ID 集合逐项比较，差异为 0。
- PR13477 `vllm-interface` 冷启动：总耗时 769.081 秒，只输出 1 条 actionable introduced break：`vllm_ascend/models/minimax_m3/minimax_m3.py:549` 的 `direct_call/call_target_presence`。PR JSON、CSV、Markdown 和 CLI 日志均不展示历史或 unresolved 项。
- 上游场景的 monkey-patch、direct-import、generator-finding-conversion 耗时均为 `null`；全量场景对应耗时分别为 205.512 秒、57.697 秒、4.700 秒，证明这些步骤是在执行前跳过，而不是分析后过滤。
- 两套场景共同保留 5005 个 direct-call 依赖；上游场景保留 683 条 override 分析关系，另收集 210 条 inheritance 关系作为 MRO 前置证据但不输出 finding。
- 因为本轮修改了 `generator.generate()` 的计划调度入口，重新执行固定 golden：耗时 752 秒，972 条关系全部 exact match，173 条 finding 和状态分布不变，old-only/new-only、descriptor/signature change、generator issue 均为 0；SHA-256 仍为 `725504ef474f4cf52f6a1a06a12a0440b56c6d7ff8861a7a39e47cd48a815cc5`。
- 本轮代表性回放保存在 `analysis_runs/interface_engine_scenario_refactor_20260810`，没有覆盖旧验收报告，也没有重复执行当前 main 全量回放。
