# Full main2main analyzer sync and cache validation

This directory contains the reproducible evidence for the full local
`main2main` analyzer. It is intentionally separate from the reduced
`vllm-interface` CI implementation used by vllm-ascend PR #14560.

## Baselines

- Original full analyzer: `b68437542d89c58ab16ad0f02392f8b6790298f5`
- Full analyzer with the reusable parallel-index implementation:
  `e36e516b8d0e0ffe62f54120938ef7dfb29b97c0`
- PR #14560 reference requested for the sync:
  `d8880d0d8a7e0beb442a0d39cfe524b0d5a6dce2`
- Last preserved local-cache implementation before CI cache removal:
  `b5ccc2f55b323c84b63f0f723d0f0ea184449505`
- Cache-removal commit, used only as a boundary:
  `e3d6b0755` (equivalent final lineage: `d666fbbea`)

The implementation branch starts from the original full analyzer and ports
the reusable implementation from the full backup plus the common fixes from
the PR lineage. It does not merge or modify the CI worktree.

## Evidence

- [Sync classification](SYNC_MATRIX.md)
- [Persistent cache design](CACHE_DESIGN.md)
- [Historical regression results](REGRESSION_RESULTS.md)
- [Measured cache performance](PERFORMANCE.md)
- [Test and quality results](TEST_RESULTS.md)

Each sample directory retains raw `stdout.log`, raw `stderr.log`, JSON, CSV,
Markdown, and metadata-bearing report output. Pickle and SQLite cache files
are deliberately excluded from this directory and are not Git artifacts.
