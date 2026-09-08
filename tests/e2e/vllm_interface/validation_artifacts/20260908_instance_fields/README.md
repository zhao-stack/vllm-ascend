# Stored configuration field provenance

Range analyzer 2.13.0 adds source paths for stored constructor parameters and
explicit non-null narrowing. It preserves the full local main2main plan,
monkey-patch reports, validation and persistent caches. The separate PR 14560
CI branch is not changed.

## Contract rules

- A plain stored instance field requires one unconditional constructor store
  from an annotated parameter, with no competing writes, early returns or
  parameter rebinding. Unknown custom allocators remain unresolved.
- Inherited storage retains the upstream declaring owner in an
  `instance_field:<name>` path. Both snapshots resolve that owner's constructor
  independently. The new constructor annotation is not evidence of the old type.
- A downstream-local store can use its pinned annotation as the path root.
  A nullable parameter still needs a source-supported non-null assertion or
  branch before its fields are usable.
- Overridden initializers without a proven store, ambiguous bases, properties,
  custom attribute protocols and other field writes remain unresolved. Plain
  assignment to a different member is distinct from a property setter, which
  may mutate the configuration and must invalidate its evidence.
- Rebinding or escaping self, changing the configuration and mutating an alias
  invalidate source evidence. Fields stored from the same constructor parameter
  share an origin; mutation through either member invalidates both paths.
  No constructor, descriptor or model is executed.
- Range schema 20 and direct-call/direct-attribute cache schemas 8/11 prevent
  reuse of discoveries produced by earlier rules. Snapshot/import schemas
  remain 9/3; only the affected discovery schemas change.

## Historical scope

The exact source inputs remain Ascend baseline
`203ae3eb0a734c3a7ba638d84e116bcca24b1a09`, vLLM old
`ba07e4a48fc951300d97eb506217dd530583dea3`, and vLLM new
`e6bfe03ad73a3330cb427885aa90d97a12e1c704`.
The actual Ascend adaptation is
`fd815467c221ee600137f6bdd53fe354d5e7c999`.

Named-field probes are not full discovery or full historical replay. The raw
shared_by site count is not a reviewed global recall denominator. Remaining
gaps include append-built collections, complex helper effects, polymorphic
receiver owners, inherited downstream dataclass construction and patch
consumer redirection. No NPU integration result is claimed.

## Named-source evidence

The final engine fingerprint is
`86b3ec85bd4e17827a96a8dcb7179713231f4b06a963b832514339c7780396f7`.
The named-field probe resolves 15 of 37 raw shared_by sites, compared with 11
at 2.12.0. All 15 are P1 introduced field breaks; none of the earlier 11 were
lost. The four additional baseline sites are:

- `distributed/kv_transfer/kv_p2p/mooncake_connector.py:2469` and `:2498`.
- `distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py:1376`.
- `distributed/kv_transfer/kv_pool/recompute_cpu_offload/worker.py:70`.

The first three now have a stored KVCacheConfig annotation path. The last has
an optional KVCacheConfig root followed by non-null narrowing. At all four
sites, the actual PR replaces shared_by with get_kv_cache_tensor_layers; that
helper selects the old field for the release lane and layers for main. This
matches the removed-field contract, not a claim about memory-layout semantics.

The field probe took 98.532 internal seconds / 110.773 process seconds. The
constructor probe took 12.758 process seconds and reconfirmed four introduced
argument breaks with zero Store/import duplicates. Both retained identical
source fingerprints before and after execution. These measured process times
are not a controlled benchmark or a full-scan speedup claim.

Raw stdout/stderr, commands, exit codes, fingerprints and findings are retained
under `analysis_runs/main2main_optimization_20260908/` as
`pr14872-field-flow-instance-final.*` and
`pr14872-constructor-instance-final.*`. The remaining 22 raw sites are neither
waived nor automatically classified as confirmed false negatives.

An observer around the existing engine's invalidate method retained the first
evidence-loss locations for two remaining cases, without changing decisions:

- Native offloading_connector.py:195 now has the correct inherited field seed,
  but the get call at line 176 cannot prove a dictionary receiver. Passing
  group_spec as its default invalidates the configuration origin. This still
  needs mapping/conditional-value analysis, not another initializer heuristic.
- Mooncake layerwise line 1448 has the correct local seed, but line 1410 mutates
  self.attn_resharding_group_idx through add. That member's separate fresh-set
  identity is not proven; the conservative self-origin barrier discards the
  configuration path. Proving disjoint owned containers remains follow-up work.

These observations and raw output are saved as
`pr14872-instance-diagnostics-final.*`. They explain remaining conservative
analysis gaps; they do not imply that the PR's corresponding adaptations are
wrong or that either case has been fixed in this slice.

## Quality verification

The 28 instance-field regression cases passed, including direct-call argument
comparison, old/new owner changes, early returns, shared-member aliases and
property setters. The earlier complete 584-test run passed before the final
four cases and safety adjustments. The final complete suite passed 588 tests
in 391.62 pytest seconds with an unchanged source fingerprint. Mypy targets
3.10/3.11/3.12, Ruff lint/format, compileall and diff check all passed. Runtime
tests used Python 3.11; mypy target versions are not runtime interpreter tests.
See `quality-instance-flow-final/status.json` and its raw output files.

Repository-wide format.sh ci is a separate gate. Its earlier failures include
historical generated-report spelling/headings, missing Windows shell tools and
the existing full-cache pickle import ban. No hooks are disabled and unrelated
isolated-checkout rewrites must not be copied into this branch.
