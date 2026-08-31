# Historical regression results

All runs used `scenario=main2main`, `profile=exact-contracts`, and enabled
monkey-patch analysis. Input repositories were clean and fixed to the recorded
full SHAs. Raw stdout/stderr and complete report files are stored beside this
document.

| Sample | Previous conclusion | Final full result | Conclusion |
|---|---|---|---|
| vLLM PR #39568 | 1 deleted-method break | 1 actionable introduced break | Exact match: `Scheduler._get_routed_experts` -> `vllm_ascend/core/recompute_scheduler.py:907`, `P1/modify direct_call/call_target_presence`. |
| vLLM PR #40996 | Triton launch break known from the PR11709 range | 1 actionable introduced break | Exact PR isolation: `_compute_slot_mapping_kernel` -> `vllm_ascend/worker/block_table.py:160`, `P1/modify`, `invocation_kind=triton_kernel_launch`. |
| vLLM PR #40172 | No break | 0 actionable introduced breaks | Exact match. Full audit items remain review-only. |
| vllm-ascend PR #11709 range | 5 actionable: 4 direct call, 1 direct import | 5 actionable: 4 direct call, 1 direct import | Exact count, relation, file, symbol, and priority match. Includes the Triton call. |
| vllm-ascend PR #12648 range | 7 introduced items before optional-parameter grading | 7 introduced, 6 actionable, 1 `P2/review` | Expected improvement: optional-only `compute_slot_mappings` remains visible but is no longer presented as an immediate modify item. |
| vllm-ascend PR #13358 range | CI scope: 5 modify and 11 override review items | CI-scope core remains 5 modify; full-only adds 1 inheritance and 2 monkey-patch modify items | Expected scenario expansion. Full actionable count is 8. The 11 override review items remain 7 introduced optional differences plus 4 preexisting deltas. |
| vllm-ascend PR #12502 range | CI scope: 33 introduced items | Same 33 core items; full-only adds 2 inheritance and 4 monkey-patch items | Exact core match. Full actionable count is 39. |

## Return-protocol checks

PR12502 retains three `P1/modify replacement_return` findings for
`find_longest_cache_hit` implementations. It also keeps a direct-call
`return_usage` item unresolved when the value escapes through unsupported local
data flow. This demonstrates that tuple/super-return improvements did not turn
the known local-variable/loop data-flow gap into an unsafe guess.

## Monkey-patch preservation

| Sample | Monkey findings | Actionable monkey findings |
|---|---:|---:|
| PR39568 | 91 | 0 |
| PR40996 | 76 | 0 |
| PR40172 | 90 | 0 |
| PR11709 | 79 | 0 |
| PR12648 | 28 | 0 |
| PR13358 | 34 | 2 (`P0/modify`) |
| PR12502 | 29 | 4 (`P0/modify`) |

The PR13358 and PR12502 results prove that monkey-patch relations remain report
consumers and priority inputs after the PR-common fixes were synchronized.

## Additional real-repository regression

PR40172 initially exposed Windows checkouts materializing Git Python symlinks as
text stubs. The analyzer now resolves only Git mode-120000, repository-internal
targets and includes the target blob chain in cache identities. The fixed sample
completed with its original no-break conclusion.
