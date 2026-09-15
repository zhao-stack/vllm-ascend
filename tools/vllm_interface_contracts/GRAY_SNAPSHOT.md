# Gray-test engine snapshot

This isolated fork dependency contains the full v2.34.0 Python package copied from
`worktrees/main2main-dataclass-replace` on 2026-09-15, including uncommitted modules.
Source HEAD: `ec88dddfb7c89b494a61cf1a945dc444fff7c3b3` plus working-tree changes.
HEAD alone does not reproduce this snapshot. GRAY_MANIFEST.json records every
Python file SHA-256; all 23 files were checked byte-for-byte against the source.

The September 14 full, no-cache benchmark improved from 536.640 to 401.283 seconds
on PR14872 with analysis_workers=3 and index_workers=1. This gray replay uses
PR13477 and retains workers=1/1 to isolate the engine update against the prior run.
Those timings are not directly comparable across inputs and machines.

The gray runner pins an exact commit, uses main2main / exact-contracts and
--no-cache, and reuses only verified text reports. QA remains disabled.
This temporary duplicate package is not the final upstream integration architecture.
