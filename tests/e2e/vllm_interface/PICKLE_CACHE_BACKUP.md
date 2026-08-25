# Local pickle-cache backup

This branch preserves the persistent repository-index cache implementation that
was present in vllm-ascend PR #14560 at commit
`b5ccc2f55b323c84b63f0f723d0f0ea184449505`.

The cache implementation is retained for trusted local analysis and performance
experiments. It is intentionally not intended for the upstream vLLM PR CI path:
that job creates an ephemeral container without mounting
`~/.cache/vllm-interface`, so the generated cache is discarded after each run.
A shared writable pickle cache would also allow one PR job to affect later jobs.

The preserved implementation contains two cache layers:

- a complete vllm-ascend `RepositoryIndex` pickle keyed by the downstream Git
  version and package tree;
- per-file vLLM index fragments stored as pickle payloads in SQLite and keyed by
  Git blob identity.

Only use these caches in a directory writable by the same trusted local user
that runs the analyzer. Do not restore cache files from untrusted jobs or share
a writable cache directory across pull requests.

Example trusted local invocation:

```bash
python -m tests.e2e.vllm_interface.vllm_interface_contracts analyze-range \
  --vllm-root /path/to/vllm \
  --ascend-root /path/to/vllm-ascend \
  --old <old-sha> \
  --new <new-sha> \
  --expect-ascend-sha <vllm-ascend-sha> \
  --analysis-workers 3 \
  --index-workers 4 \
  --upstream-file-index-cache-dir /trusted/local/cache/file-fragments \
  --downstream-index-cache-dir /trusted/local/cache/repository-index
```

The active PR branch removes this persistent cache and its dedicated tests while
retaining process-parallel source indexing and parallel compatibility-analysis
branches.
