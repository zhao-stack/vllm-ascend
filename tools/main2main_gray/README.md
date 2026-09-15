# Main2Main interface gray test

This is the first, **QA-disabled** stage of the fork-only experiment. It runs the
complete source analyzer and prepares a QA handoff file, but makes zero model calls.
It does not run kickoff, adapt code, update markers, execute NPU tests, create PRs,
or write the adaptation baseline. The production workflows are unchanged.

## Run manually

In `zhao-stack/vllm-ascend`, open **Actions → Main2Main Interface Gray (CPU, QA off)**
and choose branch `codex/main2main-interface-gray`. The default branch only contains
a launcher notice so GitHub exposes the dispatch button.

The defaults replay the PR13477 historical range against its pre-adaptation Ascend
baseline. Explicit old/new values are historical replay inputs, not a request to
advance the verified marker. All commit inputs must be full lowercase SHAs.

The complete engine is pinned in the separate `codex/main2main-gray-engine` branch.
That branch snapshots the existing clean local engine for reproducible testing;
it is **not** the final upstream shared-engine integration. QA wiring and the
reconciliation with the current upstream engine remain later implementation steps.

The runner is GitHub-hosted `ubuntu-24.04`. No upstream/self-hosted runner or model
secret is used. Fork-only commits include `[skip ci]` to avoid triggering inherited
push CI; this does not suppress the explicit manual dispatch.

## Cache and outputs

Report reuse requires exact old/new, Ascend baseline, engine commit/tree, wrapper
content, Python minor version, scenario/profile and worker configuration. Each
cached file has a checksum and report metadata is checked again before reuse.
External roots are not supported by this first runner; their explicit empty set is
recorded. Dependencies requiring unavailable external source retain the engine's
unresolved semantics.

Only JSON, Markdown and CSV reports are cached. The engine runs with `--no-cache`,
so no persistent pickle/index data is restored. A successful second identical run
must report `cache_status=hit` and must not create a new `scan.log`.

`force_rescan` bypasses reuse. GitHub caches are immutable: increment
`cache_generation` to replace a bad or intentionally refreshed entry. A corrupt
cache is rejected and scanned again; it never yields a successful cache hit.
Concurrent dispatches on the same branch are serialized by Actions; queued runs
follow GitHub's normal concurrency behavior, not an exactly-once guarantee.

Artifacts include run metadata/status, four engine reports, scan logs when the
engine ran, and `qa-input.json` with `status=prepared_not_executed`. A successful scan
can contain introduced breaks; these are findings, not execution failures. Invalid
inputs, dirty sources, incomplete output or analyzer failures fail the job.

## Local integration checks

```bash
python -B tools/main2main_gray/test_run.py --engine-root /path/to/pinned-engine -v
```

These use tiny real Git repositories and the real analyzer to check an introduced
call break, a fixed downstream snapshot, verified report reuse, corrupt cache
recovery, force rescan, input invalidation and invalid/dirty checkout failures.
