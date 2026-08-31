# Direct upstream member-removal validation

## Exact inputs

- vLLM old: `54503ecec0f3ac31e5ecfc5f28652e4cc42307b5`
- vLLM new: `fe784ff22e630a31fd798f392b01e0a75c18f047`
- vllm-ascend baseline: `6003e3222b7a6d2f08753e03fe2aa44690da2dcf`
- Historical adaptation: vllm-ascend PR #12502, head `e1d4dd608934ba688f984a7cf606d9d530cfafc4`
- Scenario: `main2main`

Both repositories were clean, their checked-out HEADs matched the expected new/baseline SHAs, and the old vLLM SHA was verified as an ancestor of the new SHA.

## Proven field-removal breaks

The old vLLM classes initialized `self.dflash_causal`. The new hierarchy consolidates the common implementation in `DraftModelSpeculator`, which has `_group_causal` but no `dflash_causal`. The pinned downstream baseline still reads the removed member.

| Upstream old endpoint | Upstream new endpoint | Downstream read | Result |
| --- | --- | --- | --- |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:DFlashSpeculator.dflash_causal` | `vllm/v1/worker/gpu/spec_decode/speculator.py:DraftModelSpeculator.dflash_causal` (`missing`) | `vllm_ascend/worker/v2/spec_decode/dflash/speculator.py:79`, `self.dflash_causal` | `introduced_break`, `direct_attribute / attribute_presence`, P1 `modify` |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py:DSparkSpeculator.dflash_causal` | `vllm/v1/worker/gpu/spec_decode/speculator.py:DraftModelSpeculator.dflash_causal` (`missing`) | `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py:84`, `self.dflash_causal` | `introduced_break`, `direct_attribute / attribute_presence`, P1 `modify` |

All four action gates are `true` for both findings. PR #12502 independently confirms the diagnosis by changing both reads from `self.dflash_causal` to `self._group_causal`.

## Full-report regression

The prior report contained 39 actionable introduced breaks. The final report contains 41: the same 39 existing findings plus the two member-removal findings above.

- relations: 1,013
- direct-call dependencies: 4,566
- direct-attribute dependencies: 2,695
- monkey-patch findings: 29
- actionable introduced breaks: 41
- direct-attribute findings: 2, both actionable

No existing actionable finding disappeared. In particular, monkey-patch discovery, reporting, and priority consumption remain enabled in the same full report. The `vllm-interface` execution-plan regression separately proves that direct-attribute analysis is skipped in the PR-CI scenario.

## Cache measurements

| Run | Direct-attribute cache | Direct-attribute discovery | Direct-attribute comparison | Total |
| --- | --- | ---: | ---: | ---: |
| Final `--no-cache` run | disabled | 153.960 s | 102.782 s | 1,019.187 s |
| Schema refresh | miss; every other persisted component hit | 158.819 s | 0.076 s | 239.472 s |
| Fully hot run | hit | 1.907 s | 0.500 s | 171.682 s |

The fully hot run saved 847.505 seconds (83.16%) versus the final no-cache total. For the new component alone, hot discovery saved 152.054 seconds (98.76%) and memoized endpoint comparison saved 102.282 seconds (99.51%). The hot report recorded hits for downstream relations, monkey-patch-containing relation data, direct imports, direct calls, direct attributes, and both upstream snapshots.

## Validation

- Full generator suite: 348 passed.
- mypy targets: Python 3.10, 3.11, and 3.12 all passed.
- Ruff lint and format check: passed.
- `compileall`: passed.
- Skill adapter tests: 47 passed, 4 skipped.
- Final current-pair `validate`: 1,013 relations, 4,566 direct-call dependencies, 2,693 current direct-attribute dependencies; completed successfully.

The `pr12502-report` directory contains the final fully hot JSON, CSV, and Markdown outputs. Raw no-cache, schema-refresh, and hot stdout/stderr is retained beside this file.
