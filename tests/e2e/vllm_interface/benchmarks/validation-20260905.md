# Benchmark workflow validation: 2026-09-05

## Scope and provenance

The local full analyzer is generator 0.46.0 / range analyzer 2.7.0. Work started
at `7bd156ac14fa5d289c942ce88652d7de85307817` on
`codex/main2main-full-sync-cache`, retaining the pending 2.5/2.6 improvements.
No Ascend runtime adaptation, version marker, or PR14560 CI branch was changed.

All three replays used the exact SHAs in the accompanying manifests, the
`main2main` scenario, `exact-contracts`, and no external source overrides.
Validate, predict, and merged-PR evaluation completed for each case.
The engine source was frozen throughout all three runs. Raw logs, reports,
frozen manifests, scores, initial failures, and endpoint evidence are retained
in the local `analysis_runs/main2main_benchmark_20260905` artifact directory.

## Reviewed results

| PR | Reviewed cases passing | Actionable roots / findings in full report | Unresolved findings | Unassessed actionable roots |
| --- | ---: | ---: | ---: | ---: |
| 13358 | 3/3, covering four consumers | 15 / 19 | 424 | 12 |
| 13477 | 11/11 | 10 / 16 | 434 | 0 |
| 14131 | 3/3 independent acceptance cases | 10 / 11 | 475 | 8 |

These are selected-contract results, not global accuracy percentages. The
PR14131 expectations were frozen before its first scan and were not changed.
Both added-parameter cases passed; unchanged inherited `top_k` reads produced
no attribute finding. Independent CPython binding witnesses also confirmed
old-call acceptance, new-keyword rejection by the baseline, and acceptance
after the PR's main-lane adaptation. No GPU/NPU execution is claimed.

### PR13358 development-oracle correction

The original four-case manifest passed every classification, priority, gate,
and consumer-count check, but failed one cross-case root check. It incorrectly
treated the two Mooncake constructors as independent upstream causes. Both
fail because of the same new upstream property/state requirement, so the
established upstream-cause deduplication correctly assigns one root.

The first frozen manifest and its 3/4 score remain preserved. The corrected
manifest combines those consumers into one case with two mandatory evidence
checks and one root. Its score is saved separately. No analyzer decision was
changed to resolve this annotation error. A scorer regression test verifies
that two consumers can share one reviewed upstream-cause case.

All 19 previously actionable finding IDs remain present. The older complete
report used `expanded`, so its total findings must not be compared directly
with this `exact-contracts` run.

### PR13477 field-lookup delta

Compared with the exact-input 2.6.0 report:

- All 10 actionable roots and 16 actionable findings are retained.
- All 29 monkey-patch finding IDs, classifications, priorities, and actions
  are unchanged.
- No finding was added. Exactly 1,046 unresolved attribute reviews disappeared;
  total unresolved findings fell from 1,480 to 434, a 70.68% reduction.
- Every removed review retains its downstream dependency. The completed scan's
  endpoint caches prove the field present and compatible at both SHAs for all
  1,046 entries. Full endpoint/source-line evidence is saved; this is not a claim
  of 1,046 independent runtime executions.
- Another 96 affected entries remain unresolved P2 reviews. Their upstream
  endpoint/root identities are now grounded, but classification, action, and
  priority do not change. Unknown initialization is not silently discarded.
- Triage reduces 1,480 raw unresolved entries to 567 review groups before the
  fix, and 434 entries to 168 groups afterwards, while retaining every location.

## Recorded execution times

Seconds below are measured wrapper wall times. The three workflows ran
concurrently; these are not controlled cold/hot-cache performance measurements.
The range report's internal phase durations can overlap and must not be added
as though they were sequential wall times.

| PR | Validate | Predict | Evaluate | Workflow sum |
| --- | ---: | ---: | ---: | ---: |
| 13358 | 1278.343 | 1174.132 | 0.382 | 2452.857 |
| 13477 | 1394.221 | 1370.775 | 0.916 | 2765.912 |
| 14131 | 1452.915 | 1512.084 | 0.744 | 2965.743 |

## Checks and remaining limits

- Final CPU-only analyzer suite: 437 passed in 191.51 seconds.
- Ruff lint and format checks: passed, including repository-pinned Ruff in an
  isolated full-CI snapshot.
- Mypy Python 3.10, 3.11, and 3.12 targets: passed for all 12 analyzer modules.
- Compileall, changed Markdown checks, and scoped codespell checks: passed.
- Full `bash format.sh ci` was executed, but is **not green**. Historical raw
  logs contain encoding issues; historical IDs/SHAs trigger spelling false
  positives; two historical reports contain repeated headings. Windows lacks
  `/bin/bash` for several hooks and shellcheck. Existing forbidden-import rules
  also flag the full analyzer's trusted-cache pickle imports. These failures
  were recorded, not bypassed or reported as passing.
- CI autoformat changes to historical reports remain only in the isolated
  test snapshot; none were copied into the implementation worktree.

The largest remaining review category is upstream instance-field binding
(338 entries in PR13477). Future iterations should select independently
labelled initialization, external-MRO, descriptor, and dynamic-binding cases
before extending those rules. Do not suppress unknowns merely to lower counts.
PR14131's other eight actionable roots remain outside this acceptance oracle;
use a separate source review before making claims about them.
