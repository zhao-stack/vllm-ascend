# Dictionary query effects

Range analyzer 2.14.0 preserves source evidence across proven builtin dict get
calls. This is an incremental full-local-analyzer change, not the PR 14560 CI
variant or a full historical replay result.

## Proof boundaries

Ordinary isinstance conditions can select a dictionary field on a known
subclass. Conditional dictionary alternatives retain both their types and
object origins. The query is exempt from mutation invalidation only when the
receiver is a builtin dict form and both stored keys and the supplied key use
known scalar protocols. Custom mappings/subclasses, unknown keys, dynamic
argument expansion and keyword calls remain barriers.

Both the dictionary element and the default may be returned. Their origins are
retained so later mutation invalidates borrowed configuration objects. Unless
the receiver is a proven empty literal, this slice does not invent one exact
historical result path. General field/call resolution through such ambiguous
get results remains incomplete. Helper summaries use the same query rule.

Literal dictionaries retain borrowed values and evaluate key/value pairs in
source order. Their object aliases cannot hide a later mutation from the
original parameter. The original tests reproduced both lost read-only evidence
and missed mutations through a literal dictionary result.

Range schema 21 and direct-call/direct-attribute cache schemas 9/12 invalidate
affected discoveries. Snapshot/import schemas remain 9/3. No repository Python
code, descriptors or model execution is used to prove these effects.

## Exact historical inputs

- Ascend baseline: `203ae3eb0a734c3a7ba638d84e116bcca24b1a09`.
- Actual adaptation: `fd815467c221ee600137f6bdd53fe354d5e7c999`.
- vLLM old: `ba07e4a48fc951300d97eb506217dd530583dea3`.
- vLLM new: `e6bfe03ad73a3330cb427885aa90d97a12e1c704`.

The named shared_by probe now includes native offloading_connector.py:195,
whose type evidence was previously invalidated by get at line 176. The actual
PR replaces that read with get_kv_cache_tensor_layers, matching the detected
removed-field contract. This does not evaluate memory-layout semantics.

## Verification

The final engine fingerprint is
`c13225c2bccf9231d121b9640e70eba2d1d04d85fe3b7ba8b3527c25a6dabc08`.
The complete suite passed 605 tests in 418.48 pytest seconds, including 17 new
mapping cases. Mypy targets 3.10/3.11/3.12, Ruff lint/format, compileall and diff
check passed in the same unchanged-fingerprint run. Runtime tests used Python
3.11. The earlier preflight retains its local-variable mypy failure; renaming
the AST loop variable resolved the annotation collision without changing
analysis decisions. Raw quality evidence is in `quality-mapping-flow/`.

The final named-field probe resolves 16 of 37 raw shared_by reads, compared
with 15 at 2.13.0. All 16 are P1 introduced breaks; no prior finding ID was lost.
Native offloading's new finding retains its upstream instance-field path at
both endpoints. The four constructor argument breaks and zero Store/import
duplicates were also reconfirmed.

The field probe took 105.233 internal seconds / 118.902 process seconds;
the constructor probe took 14.151 process seconds. These are measured probe
times, not full-scan timings or a controlled speedup benchmark. Both retained
the same source fingerprint before and after execution. Raw JSON/stdout/stderr
and status files are `pr14872-field-flow-mapping-final.*` and
`pr14872-constructor-mapping-final.*` under the local directory
`analysis_runs/main2main_optimization_20260908/`.

Repository-wide format.sh ci is a separate gate with historical report
spelling/headings, Windows shell-tool availability and existing full-cache
pickle-import failures. No hooks are disabled or raw historical evidence
rewritten to hide those failures.

Full historical replay, global precision/recall adjudication, disjoint owned
collection effects, append-built values and other receiver/constructor/patch
consumer gaps remain open. Raw field site counts are not a recall denominator.
