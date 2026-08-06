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

## 3. PR13477 区间双跑

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

## 4. 其他验收结果

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
- field、call protocol、registration 等 expanded 表仍是 opt-in，不影响默认动态分析。
