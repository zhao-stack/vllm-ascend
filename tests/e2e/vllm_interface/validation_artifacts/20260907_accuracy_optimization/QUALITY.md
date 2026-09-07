# Analyzer quality checks

The final checks used generator 0.47.0 / range analyzer 2.8.0, with source
SHA-256 `05f1290bb81e08d34e94f55f54a524e67430dc9432c60df0b7fe972cba70c142`.
The digest covers each sorted package Python filename followed by its raw
file bytes. It was unchanged throughout the checks.

| Check | Result | Wall time (seconds) |
|---|---|---:|
| Complete analyzer pytest suite | 467 passed | 263.430 |
| mypy, Python 3.10 target | Passed | 0.966 |
| mypy, Python 3.11 target | Passed | 0.838 |
| mypy, Python 3.12 target | Passed | 0.812 |
| Ruff lint | Passed | 0.186 |
| Ruff format check | Passed | 0.157 |
| compileall | Passed | 0.166 |
| git diff whitespace check | Passed | 0.214 |

Pytest reported 262.47 seconds internally; the table includes subprocess
startup and shutdown. Tests ran on Python 3.11. The mypy target checks are
not claims of runtime execution on three installed interpreters.

The suite includes monkey-patch, persistent-cache and audit regression tests.
The command was `python -m pytest -q` with
`--confcutdir=tests/e2e/vllm_interface/generator_tests`, a fresh
workspace-owned `--basetemp`, and the complete
`tests/e2e/vllm_interface/generator_tests` directory. This excludes unrelated
model-runtime E2E setup, not analyzer tests.

Ruff 0.12.12 and mypy 1.17.1 were installed into a workspace-only tools
directory. No global Python environment was modified.

Raw commands, stdout, stderr and exit codes are preserved in the local
`analysis_runs/main2main_audit_optimization_20260907/quality-final2/` folder.
Earlier attempts are retained separately: an inaccessible system pytest
temporary directory, two report-schema expectation updates, and an engine
fingerprint change during an earlier quality run. They are not counted as
successful final acceptance runs.

Historical replay and source-adjudication results are separate from these
quality checks. Passing unit tests alone does not establish global scanner
accuracy or successful NPU execution.

The repository-wide `bash format.sh ci` was attempted after replay. It could
not initialize pre-commit: the default cache database was read-only, and a
retry with a workspace-owned `PRE_COMMIT_HOME` failed to fetch the first
GitHub hook repository because network access was restricted. No hooks ran
and no source was modified by these attempts. This repository-wide hook suite
is not reported as passed; the independently executed checks above did pass.
