# SPDX-License-Identifier: Apache-2.0
"""Produce this run's report before the real adapter QA smoke consumes it."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from resolve import resolve_range

root = Path.cwd().resolve()
out = root / "scan-output"
out.mkdir(exist_ok=True)
resolved = resolve_range(
    root / "vllm-source", root / "ascend-source", target="85c09e9885e346ea1612da30ebff5a75f67d2350"
)
assert resolved["vllm_old_sha"] == "e5588e49bc2642670116664a7fc4096e27adb179"
assert resolved["vllm_ascend_sha"] == "6556ad5af1baaf481e996fc2788e233ce672f98e"
(out / "resolved-range.json").write_text(json.dumps(resolved), encoding="utf-8")
runner = root / "engine/tools/main2main_interface/run.py"
subprocess.run(
    [
        sys.executable,
        "-B",
        str(runner),
        "prepare",
        "--engine-root",
        str(root / "engine"),
        "--engine-sha",
        "b73ab761b0e265c9fc2ea2192c624a8c499ac25a",
        "--vllm-root",
        str(root / "vllm-source"),
        "--old",
        resolved["vllm_old_sha"],
        "--new",
        resolved["vllm_new_sha"],
        "--ascend-root",
        str(root / "ascend-source"),
        "--ascend-sha",
        resolved["vllm_ascend_sha"],
        "--resolved-range",
        str(out / "resolved-range.json"),
        "--output",
        str(out),
    ],
    check=True,
)
subprocess.run(
    [
        sys.executable,
        "-B",
        str(runner),
        "scan",
        "--output",
        str(out),
        "--cache-dir",
        str(root / "combined-report-cache"),
        "--force-rescan",
    ],
    check=True,
)
status = json.loads((out / "run-status.json").read_text())
assert status["scan"] == "completed" and status["source_unchanged"]
handoff = root / "report-input"
handoff.mkdir(exist_ok=True)
shutil.copyfile(out / "qa-review.md", handoff / "qa-review.md")
assert (out / "qa-review.md").read_bytes() == (handoff / "qa-review.md").read_bytes()
(handoff / "handoff.json").write_text(
    json.dumps(
        {
            "resolved_range": resolved,
            "report_sha256": hashlib.sha256((handoff / "qa-review.md").read_bytes()).hexdigest(),
            "fresh_scan": True,
        }
    ),
    encoding="utf-8",
)
