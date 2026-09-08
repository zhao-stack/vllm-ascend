# PR 14872 first accuracy-fix slice

This is an incremental implementation, not a completed full-accuracy audit.
The original full analyzer was backed up to the user's fork before editing:
`codex/main2main-full-pre-pr14872-20260908`, at
`ea6b6e513b964d77636b6e0455187290625831e9`.
Implementation uses `codex/main2main-pr14872-accuracy`; the upstream-CI branch
for PR 14560 is unchanged.

## Implemented contracts

- Executable-import discovery distinguishes ordinary writes from reads. RHS,
  receiver and augmented-assignment reads remain visible; actual monkey-patch
  target removal findings are retained.
- Generated dataclass constructor arguments are derived without executing source.
  Field inheritance, required/default arguments, keyword-only fields, InitVar,
  ClassVar and default factories have regression coverage. Unknown construction
  transforms are not promoted to exact signatures.
- Range version/schema: 2.9.0 / 16. Affected snapshot/import cache schemas: 7 / 2.
  Full main2main, monkey-patch, validation, cache and report entry points remain.

## Verification

Engine source SHA-256:
`3b6562e5c785da9cc6fac53f2a83bd675b084ee3cd78af1141ed13b8da85f93f`.
The digest covers sorted package Python filenames followed by their raw bytes;
it remained unchanged during the logged checks.

| Check | Result | Wall seconds |
| --- | --- | ---: |
| Complete analyzer pytest suite | 482 passed | 306.245 |
| mypy, Python 3.10 target | Passed | 7.589 |
| mypy, Python 3.11 target | Passed | 9.356 |
| mypy, Python 3.12 target | Passed | 7.534 |
| Ruff lint | Passed | 0.170 |
| Ruff format check | Passed | 0.154 |
| compileall | Passed | 0.189 |
| git diff whitespace check | Passed | 0.261 |

Pytest ran on Python 3.11 and reported 305.53 seconds internally. The mypy
targets do not claim runtime testing on three interpreters. The suite includes
monkey-patch, full-scenario and persistent-cache regressions; 15 tests were added.
Raw commands, stdout/stderr and exit codes are retained locally under
`analysis_runs/main2main_optimization_20260908/quality-first-slice/`.

The repository-wide `bash format.sh ci` was run in a separate worktree and
finished with exit code 1 after 273.447 seconds. Ruff, codespell and clang-format
passed. Remaining failures include spelling checks on historical hexadecimal
finding IDs, duplicate headings in old generated reports, Windows `/bin/bash`
and shellcheck availability, and the blanket pickle import ban rejecting the
existing full-analyzer cache modules. Four old report files were automatically
formatted only in the isolated checkout; those changes were not imported.
This repository-wide gate is not reported as passed. Raw `format-ci.*` logs
and status are retained beside the quality directory.

## Named historical source verification

- Ascend baseline: `203ae3eb0a734c3a7ba638d84e116bcca24b1a09`.
- vLLM old: `ba07e4a48fc951300d97eb506217dd530583dea3`.
- vLLM new: `e6bfe03ad73a3330cb427885aa90d97a12e1c704`.
- The four previously discovered KVCacheTensor constructor calls were checked
  against their exact baseline source positions and re-compared using the shared
  engine. All four change from unresolved to introduced argument breaks.
- The two patch Store sites no longer generate obsolete-target import reads.
- Evidence is saved locally in `pr14872-constructor-targeted.json`, next to the
  quality directory. This targeted check is not a fresh full historical replay.

## Remaining work

Container element field tracing, concrete polymorphic receiver/owner resolution,
downstream class constructors inheriting upstream dataclass fields, patch-consumer
redirection, dynamic-member diagnostics and progress visibility remain pending.
Three separate acceptance probes still reproduce removed-field misses through
loops, aliases and comprehensions; they are not counted as passing tests here.
Full frozen-engine historical replay and timing comparisons remain required.
