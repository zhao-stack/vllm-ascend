# Main2Main interface gray test

The fork-only manual workflow resolves an upgrade range, runs the pinned complete
source analyzer and publishes one `qa-review.md` for read-only QA. QA is disabled
by default and can be explicitly enabled as described below. This workflow does
not perform adaptation, NPU jobs, marker updates or upstream PRs.

## Inputs and version ownership

Run `Main2Main Interface Gray (CPU, read-only QA)` on branch
`codex/main2main-interface-gray` in `zhao-stack/vllm-ascend`.

- `target_commit`: optional full vLLM SHA. Empty checks out main once and freezes HEAD.
- `ascend_source_sha`: optional historical Ascend SHA. Empty starts from upstream
  main and checks this personal fork's `main2main_baseline` ref.
- `engine_sha`: exact full-engine snapshot commit, currently v2.34.0.
- `force_rescan`: bypass verified report reuse.
- `cache_generation`: increment to replace an immutable bad cache.

There is no independently supplied old SHA. `resolve.py` reads old from the selected
Ascend snapshot's `.github/vllm-main-verified.commit`; new is the frozen vLLM HEAD.
The resolved SHAs are workflow outputs and are cross-checked by `run.py prepare`.
Missing markers, inconsistent inputs and non-ancestor ranges fail. Equal endpoints
produce a no-upgrade Markdown without invoking the analyzer or caching a report.

With no historical Ascend override, the resolver follows the production source
selection pattern: try the personal fork's accumulated baseline, rebase it onto
frozen upstream main, and use fresh main if the baseline is missing, an explicit
target is at/before the baseline, or the rebase conflicts. Rebase is confined to
the disposable runner checkout. Its committer dates are deterministic for reuse.
Network/history errors fail rather than silently discarding adaptation state.
The production flow also supports legacy conf.py markers; this gray wrapper
requires the current marker file. It does not invoke or modify production kickoff.

For the existing PR13477 replay, specify only:

- `target_commit=beca88e59ea75a7aa1af72a5ae50188fa91d4e3d`
- `ascend_source_sha=61cfd1fc6a79ae139a3c5bdb8051ba7edb9c022e`

The workflow derives old `0351e9aa1fdf1a51329d1906881528dfe61fc88e` from that marker.

## QA and internal outputs

The `main2main-qa-review-<run>-<attempt>` artifact contains exactly `qa-review.md`.
It groups introduced breaks by root cause, includes pinned source locations and
short source excerpts, and separately includes unresolved findings whose
`contract_changed` gate is true. Other unresolved findings, warnings and preexisting
issues are explicitly counted but not expanded. This is a focused upgrade review,
not a claim that all unresolved findings have been cleared. QA should return
confirm/reject/insufficient-evidence per root without adapting code.

For rebased local Ascend commits, the Markdown includes source excerpts and commit
identities instead of inaccessible GitHub links. Original reports and run metadata
remain in the separate internal diagnostic artifact for auditing and cache checks.
No `qa-input.json` is produced. Only the optional QA step calls a model.

## Scan and cache

The engine runs full main2main / exact-contracts, workers 1/1, with `--no-cache`.
Only four JSON/Markdown/CSV report files are cached. Reuse requires identical
source SHAs, engine commit/tree, wrapper hash, Python minor version and settings;
file checksums and capability coverage are verified before reuse. `qa-review.md`
is regenerated from the verified report and pinned source on every run.

The GitHub-hosted Ubuntu CPU job uploads artifacts after execution. Incompatible
findings are report content, not workflow failure; execution and validation
failures fail the job. This temporary engine snapshot is not the final upstream
shared-engine integration. Commits use `[skip ci]` and only this workflow is
manually dispatched in the fork.

## Local verification

```bash
python -B tools/main2main_gray/test_run.py --engine-root /path/to/pinned-engine
```

Tests cover actual contract changes, cache reuse/corruption/force, changed Ascend
snapshots, dirty/invalid sources, marker-derived ranges, frozen target validation,
empty/non-ancestor ranges, accumulated baseline selection and deterministic rebase.

## Optional read-only QA

Set the repository Actions secret `MAIN2MAIN_API_KEY` to a DeepSeek API key, then
manually dispatch with `qa_enabled=true`. Default remains false. Credential
preflight fails before scanning if the secret is missing. The API key is passed
only to the QA step; provider errors and headers are never printed.

The QA wrapper calls the official DeepSeek Chat Completions endpoint once with
`deepseek-flash`, thinking disabled, maximum 8,192 output tokens, maximum 80,000
input bytes and a 180-second socket timeout. The Actions step has a five-minute
cap. There are no retries or model tools. It reads only `qa-review.md`; it cannot
browse source URLs, execute commands or modify adaptation code. This validates
report usability, not the production adapter-qa review of a completed code diff.

Each Markdown root must appear exactly once in the model result with a verdict
(confirm, reject or insufficient_evidence), reason and input source citations.
Truncated output, missing roots and invented citation URLs fail validation.
Execution success does not mean all candidates are confirmed or compatibility
has passed. The independent `qa-status.json` records attempted call count, input
hash, supplied usage and verdict counts. The scan's run-status remains the
scan-stage record; the Actions summary adds the actual QA execution separately.
`qa-verdict.md` is published as its own result artifact. No candidates means no
model request. The original production workflow remains unchanged.

API reference: <https://api-docs.deepseek.com/api/create-chat-completion/>.
Network-free QA tests: `python -B tools/main2main_gray/test_qa.py`.
