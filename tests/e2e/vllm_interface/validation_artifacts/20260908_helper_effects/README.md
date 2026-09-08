# Source-derived helper effects

This incremental slice preserves type evidence across source-proven read-only
helper calls. It does not complete the broader analyzer accuracy objective.

## Implementation and safety boundaries

- Range analyzer 2.12.0, range schema 19, direct-call cache schema 6 and
  direct-attribute cache schema 9 invalidate previous discoveries.
- Helpers must resolve to one ordinary, undecorated downstream function with
  an exact argument binding. Local shadowing, rebinding, ambiguous targets,
  dynamic argument expansion and unsupported signatures remain unknown.
- Effects are parameter-specific. Mutating one argument does not make an
  unrelated read-only argument unsafe. Effects follow local aliases, fresh
  dictionaries/lists, loop joins and source-supported isinstance narrowing.
- Unknown calls/getters, external stores, parameter/element mutations and
  unsupported control flow remain barriers. This is not a function-name
  whitelist or arbitrary execution of repository Python code.
- Borrowed return values retain their original object origins. Returning a
  new container does not hide references to mutable parameter objects inside
  it; a subsequent mutation through that return invalidates source evidence.
  Returning a modified parameter does not restore its invalidated type path.
- Direct return aliases and collected sequences retain a parameter-rooted
  field/iteration path that is replayed independently at old and new. An
  inferred new-only return owner is never used as proof of the old owner.
  Complex returns without such a path retain alias effects but cannot prove
  an exact historical receiver contract.
- Only known container views receive read-only treatment. A method called
  values/items/keys on an unknown object is not automatically safe.
- Isinstance requires a proven ordinary target class and inheritance chain.
  Custom metaclasses, unknown decorators and locally shadowed type names do
  not receive the read-only builtin exemption.
- Full main2main, monkey-patch, persistent-cache, validate and report entry
  points remain intact; the PR 14560 CI branch is not modified.

## Regression evidence

The helper tests cover positive reads, keyword argument binding, separate
parameter effects, aliases returned in containers, caller-side mutation,
unknown/decorated/rebound helpers, implicit protocols, fresh-container alias
escapes, source narrowing and independent old/new return-owner resolution.

The return-owner counterexample first reproduced a false introduced break:
old returned OldTensor (already missing the field), but the prototype compared
the unused old NewTensor class. Parameter-rooted return paths remove that false
attribution while positive helper-return field breaks remain detectable.

Raw local evidence is retained under
`analysis_runs/main2main_optimization_20260908/`, including the initial
`quality-helper-effects/` result and final verification in
`quality-helper-effects-release/`. These represent different source fingerprints;
the earlier 554-test run does not include the three return-path cases, and the
intermediate 557-test run predates the two custom-instance-check safeguards.
The 559-test run predates the mutating-helper-return safeguard.

Final accepted engine fingerprint:
`bbf26b3e1c4f337f89dde042e065503436ca587d695cd7d2d81a79b81c024689`.
The complete suite passed 560 tests in 366.97 pytest seconds. Mypy targets
3.10/3.11/3.12, Ruff lint, compileall and diff check also passed. The release
check retained a Ruff format failure for two files; automatic formatting then
preserved the complete engine AST fingerprint
`13c8feb124a508fa6f05abda978e0f6a2be20299028878e425ae7062bce1930c`.
Final selected style checks in `quality-helper-effects-style-final/` passed
Ruff lint/format, compileall and diff check. This selected check is not another
full suite run; the pre-format byte fingerprint is recorded in the release
status file. No failing result was rewritten as passing. Runtime tests used
Python 3.11; the three mypy targets are not three runtime interpreters.

## Historical scope and remaining work

The named-field probe uses Ascend baseline
`203ae3eb0a734c3a7ba638d84e116bcca24b1a09`, vLLM old
`ba07e4a48fc951300d97eb506217dd530583dea3`, and vLLM new
`e6bfe03ad73a3330cb427885aa90d97a12e1c704`.
It is not a replacement for full dependency discovery or historical replay.
Raw shared_by site count is not a global precision/recall denominator.

The final probe resolves 11 of 37 raw shared_by reads, up from five at 2.11.0.
All 11 are P1 introduced field breaks. The six additional sites are:

- `_310p/worker/v2/model_runner.py:599`: ordinary isinstance checks no longer
  erase the configuration type.
- `worker/v2/attn_utils.py:598`, `:612`, `:616`, `:623`, `:636`: the local
  `_get_layer_kv_cache_specs` helper preserves its read-only configuration input.

The other five sites remain unchanged. The final field probe took 72.881
seconds internally (83.398 seconds including the process lifetime), with
source-index cache hits and an unchanged engine fingerprint. This is a
targeted-probe measurement, not a full-scan speedup claim.

The four dataclass argument breaks and zero patch Store/import duplicates were
also reconfirmed (11.506 process seconds). Raw stdout/stderr, commands, exit
codes and fingerprints are retained as `pr14872-field-flow-helpers-formatted.*`
and `pr14872-constructor-helpers-formatted.*` alongside the local quality
evidence. Both used the final formatted source and retained identical engine
fingerprints before and after execution.
The remaining 26 raw field sites are retained for follow-up, not counted as
repaired or as a reviewed global false-negative denominator.

Repository-wide `format.sh ci` is separate from the passing analyzer checks.
The isolated check still reports historical generated-report spelling/headings,
missing Windows shell tools and the existing full-cache pickle import ban.
No hooks were disabled; unrelated generated-report rewrites were not copied
back to the implementation branch.

Self-held configuration provenance, append-built collections, more complex
helper effects, concrete polymorphic owners, inherited downstream dataclass
constructors and patch-consumer redirection remain explicit follow-up work.
The upgraded engine must be frozen and fully replayed before claiming a new
overall historical accuracy result. No NPU integration result is claimed.
