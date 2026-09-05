# Full main2main analyzer benchmarks

These benchmarks evaluate independently reviewed source contracts against pinned
upstream old/new commits and a pre-upgrade Ascend baseline. They complement the
small repository fixtures in `generator_tests`; they do not execute model inference.
See the [2026-09-05 validation record](validation-20260905.md) for actual results,
the retained development-oracle correction, execution times, and CI limitations.

## Benchmark layers

1. Run focused rule tests after each analyzer change. Include a failing dependency,
   an unchanged/compatible dependency, ambiguous evidence, and a repaired consumer.
2. Run all CPU-only analyzer tests before accepting a change:

   ```bash
   python -m pytest --confcutdir=tests/e2e/vllm_interface/generator_tests tests/e2e/vllm_interface/generator_tests -q
   ```

3. Replay the pinned development cases when analysis semantics change. Use the
   existing main2main workflow to validate, predict, and evaluate the exact inputs.
   Reuse an existing result only if all source/configuration and engine identities match.
4. Freeze independent-case expectations before their first scan. Score them only
   after implementation has passed the development tests. If a holdout failure is
   used to change a rule, retain the initial failure and move that case into the
   development set for the next iteration; select fresh acceptance cases.

`pr13358.json` and `pr13477.json` are development cases. `pr14131-holdout.json`
contains three source-reviewed acceptance cases frozen before scanning that PR.
The manifests deliberately cover selected contracts, not every changed file.
The corrected PR13358 manifest has three upstream-cause cases and four consumer
checks. Its initial annotation wrongly split the shared Mooncake state requirement
into independent roots; the validation record retains that first scoring failure.
The old PR13358 report used `expanded`; it cannot be scored against the new
`exact-contracts` manifest. Regenerate the intended configuration instead of
rewriting report metadata or relaxing identity checks.

## Score a report

From the repository root:

```bash
python -m tools.vllm_interface_contracts.benchmark score \
  --manifest tests/e2e/vllm_interface/benchmarks/pr13477.json \
  --report /path/to/main2main-range-report.json \
  --output /path/to/pr13477-score.json

python -m tools.vllm_interface_contracts.benchmark triage \
  --report /path/to/main2main-range-report.json \
  --output /path/to/pr13477-unresolved.json
```

`score` returns 0 when all reviewed checks pass, 1 for a benchmark mismatch, and
2 for invalid input. `triage` groups missing evidence and retains all original
finding IDs and downstream locations. Neither command reruns source analysis.

The scorer checks exact SHAs, external sources, profile, and the full `main2main`
capability plan. It refuses the reduced upstream-CI scenario. Results include
manifest/report digests and engine metadata, so a scored report cannot be
silently confused with a different input or analyzer version.

## Independent expectations

See [evidence.md](evidence.md) for the reviewed causal claims. Each case records
its rationale, evidence, evidence kind, exact source selectors, decision checks,
and expected root count. Selectors identify source symbols and consumers, never
finding IDs, classifications, or priorities. A changed decision must fail the
case instead of making its finding disappear from the comparison.

Use separate cases for independent contracts. Group import/call/patch evidence
for a single removed upstream symbol into one case. `root_count` checks catch
both duplicate roots and accidental merging of different reviewed problems.

The evidence kinds are:

- `runtime_contract`: a concrete dependency and independently established failure
  of its import, attribute access, call binding, or required instance state;
- `interface_alignment`: an override/patch must accept the new upstream interface,
  including optional arguments, without claiming the current workload exercises it;
- `historical`: the same incompatibility predates the selected upstream range;
- `negative_control`: a compatible dependency must not produce a finding.

For each repaired contract, verify baseline + old, baseline + new, and adapted +
new using an isolated executable witness or a focused source fixture. Preserve
runtime/decorator constraints in the witness. For Triton, signature binding
tests launch arguments only; they do not validate GPU compilation or kernel math.
An entire post-adaptation report need not be empty: unrelated preexisting issues
or PR adaptation omissions may remain.

The PR14131 witness is runnable against repositories containing the pinned commits:

```bash
python tests/e2e/vllm_interface/benchmarks/verify_pr14131.py \
  --vllm-root /path/to/vllm \
  --ascend-root /path/to/vllm-ascend \
  --output /path/to/pr14131-witnesses.json
```

It extracts the exact old/new/baseline/adapted function parameter lists, proves
that the selected main-lane signatures accept or reject the added keywords,
and checks the unchanged `BaseRouter.top_k` initialization. Production function
bodies, imports, tensor operations, and Triton kernels are not executed.

## Metrics and review workload

- `known_case_recall` counts reviewed actionable cases with a detected actionable
  finding. `cases_passed` additionally requires all evidence, decisions, priority,
  and root-count checks; one detected callsite does not pass a seven-site case.
- `reviewed_actionable_root_precision` includes only labelled source contracts.
  Unlabelled actionable findings are reported as unassessed, not false positives.
- Inspect unresolved categories and grouped locations separately. A category
  describes why evidence is missing; it does not prove compatibility or a break.
- Preserve raw reports and logs. Never reduce unresolved counts by discarding
  unknown dependencies or weakening action gates.

Select future cases by mechanism and unique upstream cause, including no-break
upgrades and older rare cases. A recent-month window is a useful source of cases,
not proof of mechanism coverage. Do not count overlapping upstream changes in
several PRs as independent evidence of generalization.

Cache checks remain in `test_range_analysis.py`: identity changes, dirty inputs,
corruption, disabled caches, and concurrent/interrupted writes. Compare semantic
findings across cache modes; exclude timings and cache-hit metadata from equality.
Performance measurements require separately controlled cold/hot runs, not the
wall times of concurrent accuracy replays.
