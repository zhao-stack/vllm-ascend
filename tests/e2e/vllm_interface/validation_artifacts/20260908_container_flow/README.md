# Container type-flow foundation

This incremental slice adds reusable source-derived receiver paths. It does not
claim to complete the full analyzer accuracy objective or the PR 14872 replay.
The counts below describe the earlier 2.10.0 state; see the sibling
`20260908_annotation_flow` evidence for the subsequent 2.11.0 changes.

## Implementation and checks

- Range analyzer 2.10.0, output schema 17; snapshot/call/attribute cache schemas
  advance to 8/3/6. The full main2main and monkey-patch entry points are retained.
- Supported operations: annotated fields, aliases, loops, homogeneous sequence
  elements, mapping values/items, indices, enumerate and comprehensions.
- New and old snapshots resolve the complete type path independently. An unknown
  old/new path is not promoted to a definite introduced break.
- Thirty-four new regression cases cover detections and counterexamples,
  including mutations, branch joins, same-name comprehension parameters,
  literal-false filters, zero-iteration, non-leaking comprehension targets,
  unreachable reads after exits, short-circuit expressions and cache round trips.
- Complete analyzer suite: 516 passed. Python 3.10/3.11/3.12 mypy targets,
  Ruff lint/format, compileall and whitespace checks passed. Runtime tests used
  Python 3.11, not three separate interpreters.
- Logged pytest wall time: 290.824 seconds. This is not a historical scan timing.
- Engine source fingerprint remained unchanged throughout the checks:
  `aa98fa2ea6bb13468b0a3af014f2f522e392c30ae45f2fb0f363dc2d836d4d20`.

Raw quality evidence is retained locally under
`analysis_runs/main2main_optimization_20260908/quality-container-flow-final2/`.
The previous run exposed Git for Windows' path-length limit in a symlink test.
The final run used a fresh, shorter temporary directory without weakening the
test. Repository-wide formatting CI is a separate gate: its prior attempt
failed on historical generated reports, unavailable Windows shell hooks and
the existing full-cache pickle imports. It is not counted as passed here.

## Real source probe and remaining gaps

The named-field probe uses Ascend baseline
`203ae3eb0a734c3a7ba638d84e116bcca24b1a09` and vLLM
`ba07e4a48fc951300d97eb506217dd530583dea3` to
`e6bfe03ad73a3330cb427885aa90d97a12e1c704`.

Of 37 raw shared_by read sites, this container-flow path resolves three, at
`vllm_ascend/_310p/model_runner_310p.py:736`, `:737` and `:793`. The engine
classifies these three as introduced field breaks. The other 34 sites remain
unresolved by this path; they must not be counted as fixed or silently dropped
from follow-up work. This is not a standard recall denominator or a full scan.

The first cold-index probe was deliberately terminated to switch to the existing
controlled source caches, not because an observation timeout proved it stopped.
The final replacement probe completed in 73.991 seconds, with 2,264 upstream fragment
hits, a downstream index hit and an unchanged engine fingerprint. Its evidence
is saved as `pr14872-field-flow-accepted.json` beside the quality directory.

The immediate next work is annotation-only imports under TYPE_CHECKING,
self-held configuration typing and source-derived effects for helper calls.
Annotation lookup must stay separate from runtime-import discovery.
Current invalidation is too broad when a primitive
field is passed to a helper or a known helper only reads its argument. Later
work still includes concrete polymorphic owners, downstream dataclass calls,
patch-consumer redirection, diagnostics, full historical replay and timings.
