# Gray-test engine snapshot

This branch is an isolated test dependency, not the proposed upstream engine integration.

The Python package is copied without semantic changes from the clean local commit
`ae298d9dfc048658ae4e7b7c770ed39da630fcc7`, path `tools/vllm_interface_contracts`.
The existing `tests/e2e/vllm_interface` package and all production workflows remain unchanged.

The gray runner fixes this branch to an exact commit and uses `--scenario main2main`
with `--no-cache`. Only JSON/Markdown/CSV reports are reused across workflow runs.
No serialized executable analyzer indexes are restored.

Before an upstream PR, reconcile this engine with the current upstream analyzer and
validate its existing correctness and performance changes. Do not submit this
temporary duplicate package as the final shared-core architecture.
