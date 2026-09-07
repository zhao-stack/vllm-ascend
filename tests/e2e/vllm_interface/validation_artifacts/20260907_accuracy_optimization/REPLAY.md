# Historical accuracy replay

All three exact-pair workflows completed with the unchanged source digest
listed in `QUALITY.md`. They used full `main2main / exact-contracts`, including
monkey patches, with pinned pre-upgrade Ascend baselines and no external roots.
No historical runtime adaptation was edited.

| PR | Actionable roots | Actionable consumers | Reviewed outcome |
|---|---:|---:|---|
| #13477 | 10 | 16 | All 11 frozen cases passed; every prior actionable consumer retained; no additional actionable output |
| #14131 | 20 | 24 | All 3 original frozen cases passed; all 11 prior consumers retained; all 8 planned audit consumers detected; 5 additional consumers independently checked |
| #14746 | 5 | 7 | All 3 original frozen cases passed; all 7 prior consumers retained; 7 inflated roots correctly reduced to 5 |

Under the explicitly selected strict interface-alignment policy, no false
positive was confirmed among these 47 actionable consumers (35 sample-local
roots). This is an output-adjudication result, not global recall or accuracy.
The cases used for implementation are development regressions, not fresh blind
holdouts. The original manifests were not rewritten to match new results.

## PR #14131 audit acceptance

All eight planned consumer checks succeeded:

- New `num_prefill_lookahead` debt is actionable for the coordinator patch
  and hybrid constructor while historical whole-signature findings remain.
- `varlen_decode` is actionable for the model graph constructor and its
  scoped class-to-factory patch. The two consumers share one constructor root.
- The patched vision method's `running_mean` and `running_var` reads produce
  two removed-buffer findings.
- The inherited `cache_blocks` read of `num_reprefillable_tokens` includes
  constructor activation evidence: direct helper call at line 166, flag
  assignment at line 251, and the supported positive-token EAGLE condition.
- The upstream dataclass default causes a separate downstream class-definition
  error, matching the PR's `seq_lens_np` default adaptation.

Five additional consumer findings were independently checked using exact Git
source signatures and standard-library parameter binding:

| Downstream consumer | New keyword | Adjudication |
|---|---|---|
| `NPUModelRunner._dummy_run` | `randomize_inputs` | Strict alignment omission |
| `NPUModelRunner310._dummy_run` | `randomize_inputs` | Same upstream root, second consumer |
| `AscendMLAPrefillBackend.run_prefill_context_chunk` | `out` | Strict alignment omission |
| `DFlashAclGraphManager.__init__` | `varlen_decode` | Shares the already reported upstream root with Eagle |
| `NPUOffloadingWorker.__init__` | `canonical_layout` | Local constructor alignment omission; not upstream runtime-dispatch proof |

All five downstream files are unchanged by the actual merged PR, and the
merged signatures still reject the new keywords. These are not claims of
observed NPU failures. The distinction between whole-signature historical
debt and an independently rejected new call shape remains explicit.

The eight planned plus five additional consumers explain the entire increase
from 11 to 24 actionable findings. Root sharing explains the increase from
10 to 20 independent roots. No previous actionable consumer disappeared.

## PR #14746 root normalization

- PCP import and call share the removed old-path root; the Git-proven
  destination remains separate diagnostic evidence.
- `EplbState.from_mapping` call and override share the removed-parameter root.
- AutoWeightsLoader, DFlash Triton, and EPLBController remain separate roots.

The PCP, AutoWeightsLoader and DFlash adaptations match the actual PR. The
two EPLB method contracts remain unadapted in that PR; they are not false
positives merely because the PR did not edit them.

## Measurements and limits

| PR | Validate seconds | Predict seconds | Evaluate seconds |
|---|---:|---:|---:|
| #13477 | 911.365 | 1464.000 | 0.858 |
| #14131 | 973.885 | 1620.609 | 0.811 |
| #14746 | 985.293 | 1611.498 | 0.870 |

The three workflows ran concurrently. Repository indexes were reused where
valid; range relation/snapshot components were cold. These are not controlled
cold-versus-hot performance comparisons. No savings percentage is inferred.
Parallel phase timings must not be summed as wall time.

Unresolved findings remain: 469, 505 and 501 respectively. They are not counted
as confirmed breaks. Dynamic registration, arbitrary conditions, custom
metaclasses and unproven external layouts remain outside the proven subset.
No GPU/NPU workload or numerical-kernel execution was performed.

`RESULTS.json` records exact inputs, timings, summaries and actionable source
locations. Local raw stdout/stderr, full reports, frozen manifests and detailed
source witnesses remain under
`analysis_runs/main2main_audit_optimization_20260907/` in the review workspace.
The Chinese handoff is `ACCURACY_REVIEW.md` in that local output directory,
not part of the public analyzer package.
