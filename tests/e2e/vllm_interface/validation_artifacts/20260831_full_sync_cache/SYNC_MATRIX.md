# PR #14560 to full-main2main sync matrix

## Classification

| Area | Classification | Result in the full analyzer |
|---|---|---|
| Python 3.10/3.11/3.12 and mypy fixes (`d8880d0d8`, equivalent `313081b69`) | Common | Ported without changing analysis gates or classification semantics. |
| Python 3.10 `ast.TryStar` compatibility | Common | Ported through a typed `getattr(ast, "TryStar", ())` compatibility value. |
| Direct import, exact override, inheritance, direct call, and deleted old-only method resolution | Common | Ported and covered by the shared regression suite. |
| Triton `kernel[grid](...)` binding | Common | Ported with outer-call argument binding and canonical Triton decorator proof. |
| Transitive subclass impact expansion | Common | Preserved and validated with the PR13358 `310` subclasses. |
| Tuple return comparison and transparent `return super().same_method(...)` propagation | Common | Ported; dynamic/local-variable data flow remains fail closed. |
| Non-callable final binding blocks MRO fallback | Common | `_definitely_non_callable()` and its regression test are present. |
| Commit-independent canonical decorator recognition (`torch.inference_mode`, `torch.compiler.disable`, `vllm.tracing.instrument`) | Common | Ported from `759ca98fd` and the final PR lineage. |
| Parallel repository parsing and file-fragment reuse | Common | Ported from the full backup at `e36e516b8`; local persistence is layered on top. |
| CI-only `vllm-interface` analysis plan | PR-only | Not made the default. It remains available only as the explicit `vllm-interface` scenario. |
| CI workflow integration and one-shot container assumptions | PR-only | Not ported to the local full branch. |
| Removal of monkey patch, inheritance findings, generator findings, validate, reports, external roots, and expanded profile | PR-only | Explicitly not ported. |
| Removal of persistent pickle caches (`e3d6b0755`) | PR-only | Explicitly not ported. A safer versioned local cache replaces the historical cache. |
| `main2main`, monkey-patch consumers, validate, JSON/CSV/Markdown/metadata output, external roots | Full-only | Preserved. |
| Persistent local cache and cache-management CLI | Full-only | Preserved and redesigned with safe invalidation and recovery. |
| Shared analyzer refactors versus full reports/consumers | Fused | Common source-analysis logic is shared; full plans and consumers remain authoritative for local main2main. |

## Common commits and capabilities used

- `4db8f82ad`: reusable performance and parallel-analysis changes.
- `ceb734120`: common relation-analysis simplification, selectively fused with
  the full scenario rather than copying the reduced plan.
- `28f8e810a`: common diagnostics and English report presentation.
- `759ca98fd`: removal of commit-pinned decorator rules.
- `b5ccc2f55`: streamlined common analyzer structure and the pre-removal cache
  boundary used for behavior comparison.
- `d8880d0d8`: Python-version and mypy corrections, including the final-binding
  non-callable proof.

The code was ported by functionality. No PR directory was copied over the full
analyzer, and the CI branch/worktree was not modified.

## Full-only consumer check

Historical reports prove monkey-patch findings are consumed, classified, and
prioritized:

- PR11709: 79 monkey-patch findings, including one compatibility warning.
- PR13358: 34 monkey-patch findings, including two `P0/modify` introduced breaks.
- PR12502: 29 monkey-patch findings, including four `P0/modify` introduced breaks.

These findings appear in JSON, all-findings CSV, Markdown, summary counts, and
action/priority decisions. They are not discovery-only records.
