# Annotation namespaces and scalar argument evidence

This is an incremental accuracy improvement, not a completed full PR replay.
The earlier container-flow evidence remains under `20260908_container_flow`;
its version, fingerprint and counts describe that earlier tested state.

## Implementation

- Range analyzer 2.11.0, output schema 18. Snapshot/import/call/attribute cache
  schemas advance to 9/3/4/7 so previous dependencies and endpoints are rebuilt.
- Separate annotation and runtime namespaces use the same source scope
  interpreter. Unique TYPE_CHECKING imports can explain annotated values but
  cannot supply executable import or callable evidence. Conflicting bindings,
  uncertain branches and reassignments remain unknown.
- Snapshot field annotations are resolved at each exact upstream revision.
- Nested local names do not rebind a module typing alias; real writes and
  nested global declarations retain conservative guards.
- Proven immutable scalar arguments do not expose their parent containers for
  mutation. Mutable or unknown arguments still invalidate derived bindings.
- Full main2main, monkey-patch, cache, validate and report entry points remain
  in the full analyzer. The PR 14560 CI branch is not modified.

## Real PR 14872 probes

Exact inputs: Ascend baseline `203ae3eb0a734c3a7ba638d84e116bcca24b1a09`,
vLLM old `ba07e4a48fc951300d97eb506217dd530583dea3`,
vLLM new `e6bfe03ad73a3330cb427885aa90d97a12e1c704`.

The source fingerprint is
`d99123ca73286f23c24255289b02a6c11c3910ce68eb013b836f53a70c8cd141`.

The named-field probe resolves five of 37 raw shared_by reads, compared with
three at range 2.10.0. All five are P1 introduced field breaks. Newly resolved:

- `_310p/model_runner_310p.py:756`: passing `kv_cache_tensor.size` to
  `torch.zeros` no longer invalidates the parent tensor type.
- `distributed/kv_transfer/kv_pool/recompute_cpu_offload/manager.py:156`:
  the parameter's KVCacheConfig annotation is imported only under TYPE_CHECKING.

The other three sites remain `_310p/model_runner_310p.py:736`, `:737`, `:793`.
The probe took 81.147 seconds with unchanged source; upstream file fragments
and the downstream index hit cache. This is not a full-scan timing comparison.
The other 32 raw reads remain unresolved by this path, not silently waived or
counted as repaired. Raw read count is not a standard recall denominator.

The constructor probe also reconfirms the four previously unresolved dataclass
calls as introduced argument breaks, with the two patch Store/import duplicates
still absent. Neither probe replaces full dependency discovery and replay.

Local evidence lives under `analysis_runs/main2main_optimization_20260908/`:
`pr14872-field-flow-annotations.json`, `pr14872-constructor-annotations.json`
and `quality-annotation-flow/`.

## Remaining work

Self-held config, append-built containers and source-derived read-only helper
effects still need tracing. Concrete polymorphic owners, inherited downstream
dataclass constructors and patch-consumer redirection remain separate gaps.
The engine must be frozen and fully replayed against the exact historical
inputs before claiming a new overall accuracy result. No NPU integration
execution or global precision/recall is claimed by this evidence.

## Quality checks

- Complete analyzer suite: 533 passed in 310.39 pytest seconds (311.071 wall
  seconds), using Python 3.11. Mypy targets 3.10/3.11/3.12 all passed.
- The initial Ruff lint run found five long test strings. Splitting adjacent
  literals preserved the test AST fingerprint exactly:
  `3a96554a08c66306c2719c7513ada1fd3838e7b8270a2605b6f571ef13c9b8eb`.
  The engine source also remained unchanged. Ruff lint/format, compileall and
  diff check subsequently passed in `quality-annotation-style-final/`.
- The initial combined status is retained as failed, not rewritten as green.
  Its pytest/mypy passes and the separate final style result together record
  the completed checks. No assertions or test cases were removed.
- Repository-wide `format.sh ci` is a separate, unmet gate. The isolated run
  reports preexisting generated-report spelling/headings, Windows shell-hook
  availability and the existing full-cache pickle import ban. Its original
  raw output is retained; unrelated report rewrites are not copied back.
