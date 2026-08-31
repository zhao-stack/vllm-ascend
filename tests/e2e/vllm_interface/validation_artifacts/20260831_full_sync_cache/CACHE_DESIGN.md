# Persistent cache design

## Ownership and trust boundary

The CLI accepts a cache parent through `--cache-dir`, but reads and writes only
the derived `vllm-interface-contracts` child namespace. `cache clear` removes
only that namespace. Arbitrary pickle paths are never accepted.

Pickle is unsafe for untrusted data. Only cache files created by this analyzer
inside its private namespace may be used. Copying untrusted pickle files into
that directory is explicitly unsupported.

## Cached components

1. Downstream `RepositoryIndex`: parsed ASTs, symbols, final bindings, aliases,
   MRO inputs, descriptors, and callable contracts.
2. Downstream relations: inheritance, override, monkey patch, historical
   override candidates, and generator findings.
3. Downstream direct imports.
4. Downstream direct calls, including Triton launch calls and historical
   old-only candidates.
5. Upstream old and new `GitSnapshot` state: source, AST, module bindings,
   endpoint resolution, and call-resolution memoization.
6. Content-addressed upstream new file fragments used to assemble the current
   `RepositoryIndex` efficiently across nearby commits.

Final finding classification, action, priority, and report rendering are not
cached. They are recomputed for every run.

## Identity and invalidation

Every high-level identity contains:

- normalized absolute repository path;
- exact repository commit SHA;
- analyzer/generator version and component schema;
- global cache schema (`4`);
- Python implementation, version, and cache tag;
- analysis scenario, plan, profile, decorator configuration, external roots,
  and other component-specific configuration;
- clean/dirty source state.

Clean Git repositories use tree/blob identities. A dirty tracked or untracked
source tree bypasses persistent caching, preventing reuse of clean results.
Tests additionally prove upstream SHA, downstream SHA, schema, and analysis
configuration changes produce misses.

Git mode-120000 Python symlinks are handled consistently on Windows checkouts:
only Git-declared symlinks are followed, only to a normalized path inside the
same repository, and the target blob chain is included in fragment identities.
Absolute, escaping, missing, and cyclic targets fail closed.

## Corruption, interruption, and concurrency

- Pickle entries use a private envelope containing a magic marker and the full
  serialized identity.
- Invalid type, magic, identity, or payload is treated as invalid/corrupt and
  rebuilt.
- Corrupt files are removed when possible; corruption never aborts analysis.
- Writes use a temporary file, flush, `fsync`, and atomic `os.replace`.
- Per-entry lock files serialize cross-process writers. Stale locks are bounded
  and recoverable.
- SQLite fragment databases are private, versioned, checked, and recreated if
  corrupt.
- Interrupted or concurrent writes are covered by regression tests.

## CLI and diagnostics

```text
python -m tools.vllm_interface_contracts analyze-range ... --cache-dir <parent>
python -m tools.vllm_interface_contracts analyze-range ... --no-cache
python -m tools.vllm_interface_contracts cache clear --cache-dir <parent>
```

Logs report cache enabled/disabled, component, commit, hit/miss/invalid/corrupt
state, load time, write time, and comparable saved work. Reports retain the same
events under `metadata.persistent_cache` and stage timings under
`metadata.stage_timings_seconds`.
