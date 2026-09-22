# SPDX-License-Identifier: Apache-2.0
"""Live fetch of an isolated fork baseline, against fixed historical sources."""

import json
from pathlib import Path

from resolve import resolve_range
from run import git

root = Path.cwd()
up, down = root / "vllm-source", root / "ascend-source"
source = "6556ad5af1baaf481e996fc2788e233ce672f98e"
old = "e5588e49bc2642670116664a7fc4096e27adb179"
new = "85c09e9885e346ea1612da30ebff5a75f67d2350"
assert git(down, "rev-parse", "HEAD") == source
assert git(up, "rev-parse", "HEAD") == new
baseline = resolve_range(
    up,
    down,
    baseline_url="https://github.com/zhao-stack/vllm-ascend.git",
    baseline_ref="refs/heads/codex/main2main-baseline-gray",
)
assert baseline["source_mode"] == "incremental_rebased"
assert baseline["baseline_sha"] == source
assert baseline["vllm_old_sha"] == old and baseline["vllm_new_sha"] == new
git(down, "checkout", "--detach", source)
fresh = resolve_range(
    up,
    down,
    baseline_url="https://github.com/zhao-stack/vllm-ascend.git",
    baseline_ref="refs/heads/codex/main2main-absent-gray-20260922",
)
assert fresh["fallback_reason"] == "no_baseline_ref"
assert fresh["vllm_old_sha"] == old and fresh["vllm_new_sha"] == new
Path("output").mkdir(exist_ok=True)
Path("output/resolver-live.json").write_text(
    json.dumps(
        {
            "baseline": baseline,
            "fresh": fresh,
            "limitation": "isolated historical baseline ref; nontrivial rebase and conflict exercised by fixtures",
        },
        indent=2,
    )
)
