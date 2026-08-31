# Persistent cache performance

## Method

- vLLM worktree path: `analysis_repos/perf-cache-vllm`
- vllm-ascend worktree path: `analysis_repos/perf-cache-ascend`
- Workers: `index-workers=2`, `analysis-workers=2`
- Base range for no-cache/cold/hot:
  `1f486d96a17303ce8db8e02be39545b2be338446` ->
  `95ed0feaa5cd7fb16d72c53ce04950aaf07c4698`
- Base downstream:
  `3b75c4ecf8ef471fc751ce34af806e1be407f397`
- Upstream-change range end:
  `e5588e49bc2642670116664a7fc4096e27adb179`
- Downstream-change commit:
  `ccc0a3f1c9c6cc36b5ac38274bebf8e82019be05`

All runs were sequential. The cold run started after `cache clear`. The no-cache
run did not create the analyzer cache namespace.

## Measured work and wall time

Times are seconds. Repository indexing and comparison branches overlap under
parallel execution, so stage work times must not be summed to reconstruct wall
time. `Total` is the measured end-to-end value including report generation.

| Mode | Downstream scan | Relation generation | Direct-call discovery | Monkey patch | Upstream new index | Old/new snapshots | Contract comparison | Report | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No cache | 15.664 | 128.521 | 69.424 | 210.310 | 326.595 | 86.395 | 208.282 | 0.067 | 895.309 |
| Cold cache | 17.011 | 128.579 | 71.596 | 200.476 | 334.032 | 89.606 | 215.170 | 0.068 | 901.508 |
| Hot cache | 3.530 | 0.022 | 4.362 | 0.000 | 31.170 | 8.449 | 139.899 | 0.066 | 167.324 |
| Upstream new changed | 3.547 | 126.200 | 64.223 | 208.133 | 32.175 | 38.499 | 174.614 | 0.066 | 577.025 |
| Downstream changed | 17.589 | 130.168 | 91.012 | 195.964 | 33.873 | 5.113 | 178.291 | 0.058 | 577.674 |

## Savings

- Cold-cache maintenance cost versus no-cache: **6.199 s / 0.69%**.
- Hot versus no-cache: **727.985 s / 81.31% saved**.
- Hot versus cold: **734.184 s / 81.44% saved**.
- Upstream-new change versus cold: **324.483 s / 35.99% saved**.
- Downstream change versus cold: **323.834 s / 35.92% saved**.
- Cold cache footprint: 7 files, 214,661,395 bytes (about 204.7 MiB).

## Hit and rebuild matrix

| Mode | Downstream index | Relations | Direct imports | Direct calls | Old snapshot | New snapshot | Upstream file fragments |
|---|---|---|---|---|---|---|---|
| No cache | disabled | disabled | disabled | disabled | disabled | disabled | disabled |
| Cold | miss | miss | miss | miss | miss | miss | miss |
| Hot | hit | hit | hit | hit | hit | hit | hit |
| Upstream new changed | hit | miss | hit | miss | hit | miss | 1960/1964 hit |
| Downstream changed | miss | miss | miss | miss | hit | hit | 1964/1964 hit |

## Cost/benefit observations

- The high-level relation cache has the clearest payoff: hot relation and
  monkey-patch generation fall from roughly 329 seconds of cold work to 0.022
  seconds of relation load plus zero monkey generation.
- Snapshot pickle loads remain worthwhile: approximately 8.45 seconds hot
  versus about 81 seconds of recorded snapshot work.
- Content-addressed upstream fragments are useful for nearby commits: the
  upstream-new run rebuilt only 4 of 1964 files. Loading and final merging still
  costs about 32 seconds, so fragment caching has more maintenance overhead than
  relation caching but remains materially below the 334-second cold upstream
  index.
- Cold writes cost about 0.69%, acceptable for repeated local analysis. For a
  one-shot container, `--no-cache` remains preferable, matching the PR #14560
  CI design.
