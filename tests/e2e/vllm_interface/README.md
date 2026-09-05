# vLLM interface boundary tests

Current implementation and validation handoff: [`VLLM_INTERFACE_HANDOFF_20260812.md`](VLLM_INTERFACE_HANDOFF_20260812.md).

Reviewed full-analyzer benchmarks, scoring, and unresolved triage:
[`benchmarks/README.md`](benchmarks/README.md).

`singlecard/test_interface_boundaries.py` provides a CPU-only boundary check for vLLM callables coupled to
vllm-ascend. The existing upstream `vllm-interface` job collects it because that job runs the complete
`tests/e2e/vllm_interface` directory.

The compact `interface_boundaries.jsonl` file stores one upstream callable per line. Each record contains the upstream
signature boundary and all related vllm-ascend patch, override, direct-call, or inheritance endpoints.
Its historical `direct_callable` consumer record is distinct from the range analyzer's concrete `direct_call`
dependencies; the latter are contract-analysis-only and do not enter the schema-6 relation golden.

The test checks:

- upstream files, classes, callables, and parameter boundaries;
- downstream patch and override endpoint boundaries;
- direct calls for missing/extra positional parameters and unsupported/missing keywords, including exact Triton
  `kernel[grid](...)` launches;
- full-main2main member reads whose exact upstream module, class, property, dataclass, slot, or initialized instance
  attribute is removed;
- direct inheritance edges.

For monkey-patched callables, direct calls are checked against the replacement signature. The test parses Python source
with `ast`; it does not import `torch_npu`, initialize an NPU, download a model, or execute inference.

## Source-based mapping generator (POC)

The implementation lives in the reusable `tools.vllm_interface_contracts` package.  The historical
`generate_interface_boundaries.py` path remains a compatibility entry point, so existing local commands and CI jobs do
not need to change.  New callers should use the package CLI:

```bash
python -m tools.vllm_interface_contracts generate --help
python -m tools.vllm_interface_contracts analyze-range --help
python -m tools.vllm_interface_contracts validate --help
```

The main2main skill is intentionally only an adapter around this package. It validates the requested SHAs, selects
`new`, `legacy`, or `compare` mode, passes cache controls to the engine, and renders the requested report. AST indexing, MRO,
override, monkey-patch, import, callsite, signature, and return-contract decisions are owned by this package alone.

Local handoff notes retain the historical design discussion and validation evidence; this README documents the public
CLI and analyzer behavior in English.

`generate_interface_boundaries.py` rebuilds the low-noise subset of the mapping directly from a checked-out vLLM and
vllm-ascend source pair. It currently discovers:

- explicit monkey patches made with assignment or `setattr`;
- patch targets imported at module or function scope;
- simple target aliases such as `PATCH_TARGET = ImportedVllmClass`;
- `setattr` names resolved from string constants, string collections, or one live candidate;
- lambda, direct `property(...)`, class-body callable aliases, and statically provable wrapper factories;
- direct inheritance from a statically resolved vLLM class;
- verified overrides whose effective parent implementation is resolved through the combined MRO;
- generated dataclass constructors, typed lazy exports, patch save/restore lifecycle, and field-mutation findings;
- exact Triton `kernel[grid](...)` launch signatures recognized by canonical decorator symbols, including literal
  heuristic-generated parameters;
- exact local helpers that select one literal-named class from a complete MRO, return its module, and patch that module
  through `sys.modules[name]` or `sys.modules.get(name)`;
- direct `contextlib.contextmanager` wrappers, keeping the source/reporting contract separate from the wrapper's broad
  runtime entry;
- optional exact external source indexes for methods inherited by a vLLM class, without treating external-only overrides as vLLM edges.

The POC targets vLLM main. Branches guarded by an exact `vllm_version_is("<tag>")` check are treated as release-only;
the opposite branch is indexed for main. Top-level imports under the selected branch and `try` blocks are included.
An incomplete or ambiguous vLLM/vllm-ascend MRO is reported as unresolved instead of choosing a likely parent.
The MRO-selected-module rule is fail-closed as well: the receiver class, complete MRO, literal selected class name, and
single vLLM owner must all be proven. Ordinary callables assigned to a proven instance are checked as instance
attributes; class descriptor rules are applied only when the target is a class namespace.

The generator is consumer-first. A downstream patch or inheritance declaration whose upstream target cannot be resolved
is kept in the main output as a finding instead of being silently dropped. Findings distinguish an upstream risk, an
expected injection, an excluded inactive branch, and a static-analysis review. The optional unresolved output mirrors
these findings for convenient review. It is AST-only and requires neither an NPU nor package imports.

Schema version 6 stores verified relations under `u`/`c`, candidate findings under `f`, the definition source package
under `p`, the replacement definition file
in each consumer, and patch evidence separately under `e`. Each finding includes `status`, `reason_code`, and whether it
represents a generator limitation. Evidence includes the assignment file and line, lexical scope, guards, patch kind,
and every statically discovered assignment occurrence. Signature contracts separately record source definitions,
runtime entries, reported signatures, receiver-bound calls, and access protocols. Python parse failures stop generation
instead of silently reducing coverage.

An external source root must be reproducible. The generator accepts either a Git checkout whose HEAD equals the expected
SHA or a `.interface-source.json` snapshot manifest that records the exact upstream commit and SHA-256 of every included
Python file. An unknown external parent keeps the MRO in review; the generator never chooses a later vLLM method through
an incomplete chain.

Example:

```bash
python tests/e2e/vllm_interface/generate_interface_boundaries.py \
  --vllm-root /path/to/vllm \
  --ascend-root /path/to/vllm-ascend \
  --expect-vllm-sha <vllm-sha> \
  --expect-ascend-sha <vllm-ascend-sha> \
  --external-root torch=/path/to/pytorch-source \
  --expect-external-sha torch=<pytorch-sha> \
  --output generated_boundaries.jsonl \
  --unresolved-output unresolved_relations.jsonl \
  --compare-with tests/e2e/vllm_interface/interface_boundaries.jsonl \
  --report comparison_report.json
```

The expected SHA options are recommended for reproducible local generation so that a comparison cannot accidentally use
a different source pair.
The comparison report separates exact edge matches from downstream endpoint coverage; this prevents a re-export path
change from being counted as a missing downstream dependency.

`audit_interface_boundary_coverage.py` independently enumerates source candidate sites and checks that each has exactly
one disposition in the generated mapping. It can follow direct downstream helpers with call-site module arguments, but
does not enter vLLM or external helper bodies and reinterpret their normal field assignments as downstream patches.

## Exact upstream range analysis

`analyze-range` generates downstream dependencies from the requested vllm-ascend baseline, checks each dependency
against exact old and new vLLM commits, and writes JSON, JSONL-derived CSV, and Markdown reports.  The vLLM checkout must
be at the requested new SHA; old files are read directly from Git, so the command never imports vLLM or requires an NPU.

```bash
python -m tools.vllm_interface_contracts analyze-range \
  --vllm-root /path/to/vllm-at-new-sha \
  --ascend-root /path/to/vllm-ascend-baseline \
  --old <old-sha> \
  --new <new-sha> \
  --expect-ascend-sha <ascend-sha> \
  --scenario main2main \
  --cache-dir /path/to/private-cache-parent \
  --output-dir /path/to/report-dir
```

## Local persistent cache

The local CLI enables a private persistent cache by default. `--cache-dir` selects its parent directory, and the tool
creates and reads only the `vllm-interface-contracts` child. `--no-cache` disables every persistent read and write.
`python -m tools.vllm_interface_contracts cache clear --cache-dir <parent>` removes only that tool-owned child.

The cache covers the downstream AST/symbol index, content-addressed upstream file fragments, generated inheritance,
override, monkey-patch, direct-import, direct-call, direct-attribute, and Triton-call dependencies, plus the lazily
populated old/new Git snapshot indexes. Keys include normalized repository paths, exact commit SHAs, component and
global cache schema versions, analyzer versions, the Python implementation/version, and analysis configuration.
Committed-source caches are bypassed when the relevant package tree has uncommitted or untracked source changes.

Cache entries use pickle because they contain Python AST objects. Pickle is unsafe for untrusted data. The analyzer does
not accept an arbitrary pickle path and must read only entries it created under its private cache directory. Corrupt
entries are deleted and rebuilt. Writes use a temporary file, `fsync`, an atomic replace, and a per-entry lock so an
interrupted or concurrent writer cannot leave the authoritative entry partially written. Cache status, commit, hit,
miss, bypass, corruption, load/build/write timings, and an estimated saved duration are recorded in report metadata and
printed by the CLI.

`--scenario` selects one of two fixed execution plans; it is not a collection of independent low-level switches:

- `main2main` (default) runs the full exact-contract analysis: patch, override, inheritance, direct import, direct call,
  direct attribute, inherited instance state, return protocol, and generator findings. It preserves the existing report
  names and behavior.
- `vllm-interface` is the upstream PR awareness plan. It analyzes direct imports, overrides, and exact downstream-call
  contracts. Inheritance/MRO discovery still runs as an override prerequisite, but it does not emit inheritance
  findings. Monkey-patch collection and patch-oriented generator-finding conversion are skipped before execution. This
  plan accepts only `exact-contracts`.

The upstream plan writes `vllm-interface-pr-summary.md`, `vllm-interface-pr-report.json`,
`vllm-interface-introduced-breaks.csv`, and `vllm-interface-analysis-metadata.json`. Its PR-facing Markdown, JSON, and
CSV contain only actionable `introduced_break` findings for direct-import, override, or direct-call contracts; historical and
unresolved items are intentionally absent. The metadata records every capability as `analyzed`, `prerequisite`, or
`skipped`, plus per-phase elapsed time. The command remains awareness-only unless `--fail-on` is explicitly selected.

Findings are separated into `introduced_break`, `compatibility_warning`, `preexisting`, `fixed`, and
`analysis_unresolved`.  Unchanged verified relationships are omitted from the upgrade report.  By default the command
only warns and exits successfully; `--fail-on introduced` is available for a future blocking CI, but is not the default.

Range schema version 14 gives every finding a stable `root_cause_id`. Multiple relation findings remain in JSON and CSV
as independent evidence, but summary action counts are based on the unique upstream changes that caused them. The
summary exposes `actionable_introduced_findings` for the raw relation count and `actionable_introduced_break` for the
deduplicated root-cause count. Presence changes are grouped by the canonical upstream symbol, exact parameter changes
by the canonical callable and serialized delta, and inherited-state changes by the property or method plus its required
attribute.

The full `main2main` scenario treats an exact optional-only upstream signature addition as a P1 required alignment even
when no current upstream dispatch site can be proven. This preserves the downstream implementation contract for future
callers. The reduced `vllm-interface` CI scenario retains its narrower P2 review policy unless exact call and dispatch
evidence proves runtime reachability, so this local full-analysis policy does not expand CI scope.

Range schema version 12 adds full-main2main `inherited_state / required_instance_attribute` findings. When an effective
upstream method or property newly reads an instance attribute, the analyzer checks whether the pinned downstream
subclass overrides `__init__` and still establishes that state. A direct main-path assignment or a provable
`super().__init__()` chain satisfies the contract. Conditional initialization, dynamic providers, incomplete MROs, and
explicit unresolved initializer calls fail closed as review findings. Reads protected by `hasattr` or
`AttributeError` are not hard requirements. This phase is intentionally skipped by the `vllm-interface` scenario.

Schema version 11 introduced full-main2main `direct_attribute / attribute_presence` findings and their cache/timing
metadata. Version 12 retains those exact call and return contract families without changing the generator JSONL schema
or its fixed relation golden:

- downstream-to-upstream calls: uniquely resolved module functions, constructors, class/static methods, annotated or
  provably constructed instances are checked by binding each concrete `args`/`kwargs` shape independently at old and
  new. For downstream `self`/`super`, the pinned vllm-ascend MRO must first prove one effective upstream owner, and each
  snapshot then validates that exact owner. Immediate or uniquely aliased tuple/list unpacking, literal subscripts,
  iteration, context-manager use, and `await` are checked separately against the old/new return protocols;
- upstream-to-downstream implementations: each generator-proven patch or override keeps the existing installed-input
  substitutability check, while exact return annotations or statically proven return paths are checked covariantly
  against the old and new upstream return protocols;
- downstream member reads: direct module/class values, properties, dataclass fields, literal slots, annotated or
  constructed instances, and `self`/`super` reads through a complete pinned MRO are compared for old/new presence.
  Unresolved multiple/external inheritance, dynamic attribute providers, dynamic `setattr`, and conditional field
  initialization fail closed. Reads protected by `hasattr`, `AttributeError`, or `vllm_version_is` are excluded.

Triton subscript launches are handled as a direct-call protocol, not as an ordinary Python subscript. The detector
resolves the callable inside `kernel[grid]`, keeps the arguments from the outer `(...)`, and binds them independently
against old and new. This path currently requires one uniquely resolved upstream function with exactly one canonical
`vllm.triton_utils.triton.jit` decorator. Ordinary `mapping[key](...)` calls and additional or unresolved decorator
stacks remain fail-closed. This range capability is independent of monkey-patch collection, which remains skipped by
the `vllm-interface` scenario.

Imported, annotated, or constructed vLLM receiver members are re-resolved at old and new only through a unique,
statically provable single-inheritance vLLM chain. Multiple, external, incomplete, or otherwise ambiguous receiver
hierarchies are not selected; a moved or missing `self`/`super` owner is unknown rather than guessed. Constructor calls
are exact only when class decorators/keywords, custom or inherited `__new__`, metaclass behavior, and the effective
`__init__` are all proven safe. Descriptor and overload handling likewise requires a unique runtime implementation and
origin-proven builtins; shadowed decorators or overload-only/conditional bindings are unknown. Return covariance is a
conservative structural check over the protocols and shapes that the analyzer can prove, not arbitrary nominal subtype
inference. For this interface-only pass, a concrete downstream callsite or a generator-proven patch/override
installation is sufficient source reachability evidence; full model/device/runtime-path reachability is intentionally
not modeled.

The detector omits a dependency when the callsite itself cannot be uniquely resolved, including dynamic
`*args`/`**kwargs`, ambiguous callees, and unsupported receiver hierarchies. Once a dependency is proven, an ambiguous
old/new endpoint, runtime signature, or constrained return shape becomes `analysis_unresolved`; neither path becomes
`modify`. Unused, forwarded, escaping, or otherwise unconstrained return values do not produce a return-use finding.
Calls nested under any `if` condition containing a call named `vllm_version_is` are excluded from this default detector
on both branches. These dynamic exact call contracts are different from the historical static `call_protocol` mapping
table, which remains an opt-in `expanded`/legacy review asset.

`validate` applies the same enabled detectors to the checked-out source pair and includes current direct-call,
direct-attribute, and patch/override-return risks in `contract_findings`. `analyze-range` remains the command that
attributes a compatibility transition specifically to `old -> new`.
