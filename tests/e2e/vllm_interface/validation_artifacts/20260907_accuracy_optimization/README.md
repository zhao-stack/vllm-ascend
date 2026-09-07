# Full main2main accuracy optimization

This work extends the local full analyzer on `codex/main2main-full-sync-cache`.
It does not modify the separate PR #14560 upstream-interface CI branch or
historical Ascend runtime adaptations.

## Baseline and audit

The starting engine commit is
`b8176f223b2f730f36b888fd2bf79f33c35cd07d` (generator 0.46.0,
range analyzer 2.7.0). The source-reviewed #14131 and #14746 audit preceded
these changes. Those cases are now development regressions, not new blind
holdouts. Existing frozen benchmark manifests remain unchanged.

## Changes

| Audit issue | Implementation | Safety boundary |
|---|---|---|
| A: New parameters hidden by historical signature debt | Preserve the historical finding and add an independent new-call witness | The old witness must bind to the pinned replacement; adding new keywords must independently fail. Extra local constructor context requires a concrete local construction call and is labeled interface alignment, not proven upstream runtime dispatch. |
| B: Class replaced by a scoped factory | Compare the class constructor protocol with the installed factory | Reuse the existing resolver; custom metaclasses, class decorators and custom `__new__` remain unknown. |
| C: Patched free method reads removed buffers | Bind the replacement receiver through a verified ordinary-method installation; compare literal `nn.Module` registration | Require one exact receiver, a direct supported torch base, normal super initialization, and registrations on every completing constructor branch. Dynamic names, helpers, registries, receiver escape and custom attribute providers do not prove absence. |
| D: Conditional inherited state | Attach explicit constructor activation evidence to supported element-flag paths, including a uniquely resolved zero-argument initialization helper called directly by the constructor | Only one helper level and a restricted positive numeric-input/alignment protocol are supported. Uncalled helpers, early returns, reassigned arguments, arbitrary algebra, dead branches and contradictory flag writes do not qualify. Findings describe a conditional interface contract, not every workload. |
| E: Dataclass definition failure | Compare the upstream field layout composed with the pinned downstream declaration | Respect defaults, keyword-only fields, init exclusions, ClassVar and InitVar. Unknown decorators, dynamic declarations and multiple inheritance are outside this detector's proven subset. |
| F: Duplicate upstream causes | Normalize parameter deltas across call/override detectors and constructor factories; distinguish an unresolved old path from a Git-proven relocation destination | Keep every affected consumer. Do not merge different symbols, return contracts or field requirements. |

## Cache compatibility

Generator 0.47.0 and range analyzer 2.8.0 invalidate prior component identities.
The range report schema is 15, snapshot schema is 6, and direct-attribute
schema is 5. Cached relation graphs are restored on the live generator before
dependent receiver discovery. Source indexes and graph caches retain their
existing commit/configuration/dirty-worktree protections. No final verdict is
reused across engine source fingerprints. Pickle remains private, controlled
tool-cache data and must never be loaded from untrusted files.

## Validation policy

- Run the complete pure-source analyzer suite, including monkey patches and
  persistent-cache tests, without the unrelated model-runtime E2E fixtures.
- Use a new workspace-owned pytest temporary directory for each run.
- Check mypy targets 3.10, 3.11 and 3.12, Ruff lint/format, compileall and diff
  whitespace. Target-version mypy is not a claim of runtime execution on all
  three interpreters.
- Replay exact baseline/old/new pairs through the thin skill's full
  `validate`, `predict`, and `evaluate` workflow.
- Inspect introduced CSVs before the full reports. Match source-reviewed
  contracts and consumer locations, not raw finding IDs or counts alone.
- Preserve failed setup attempts and superseded engine-fingerprint logs.
  A source change requires a fresh workflow; unchanged safe component caches
  can still be reused by that workflow.

Historical source/binding witnesses establish structural compatibility only.
This work does not claim end-to-end NPU execution or global scanner accuracy.
