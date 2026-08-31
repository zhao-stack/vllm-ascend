# Test and quality results

## Automated tests

```text
python -m pytest -q --confcutdir=tests/e2e/vllm_interface \
  tests/e2e/vllm_interface/generator_tests

327 passed in 120.89s
```

The suite includes the PR-common analyzer tests, existing monkey-patch tests,
and persistent-cache regressions for:

- first-run miss and second-run hit;
- upstream and downstream SHA invalidation;
- global schema and analysis-configuration invalidation;
- dirty-worktree bypass;
- corrupt-file deletion and rebuild;
- complete `--no-cache` read/write disablement;
- namespace-safe cache clearing;
- concurrent and interrupted atomic writes;
- non-callable MRO blocking;
- Python Git-symlink stubs and target-blob invalidation.

## Static checks

| Check | Result |
|---|---|
| mypy with `--python-version 3.10` | Passed, 10 source files |
| mypy with `--python-version 3.11` | Passed, 10 source files |
| mypy with `--python-version 3.12` | Passed, 10 source files |
| Ruff lint | Passed |
| Ruff format check | Passed, 16 files already formatted |
| `compileall` | Passed |
| Skill adapter Ruff/compile smoke | Passed |
| `git diff --check` | Passed |
| Han-character scan of public analyzer code, tests, help, and README | No matches |

The top-level E2E `conftest.py` imports unrelated runtime dependencies that are
not installed on this host. The pure source-analysis tests therefore use the
documented `--confcutdir=tests/e2e/vllm_interface` boundary.
