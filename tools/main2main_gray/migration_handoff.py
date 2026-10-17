"""Bind the uploaded report to the historical source checkouts before real QA."""

import hashlib
import json
import subprocess
from pathlib import Path

root = Path.cwd()


def sha(folder):
    return subprocess.check_output(["git", "-C", str(root / folder), "rev-parse", "HEAD"], text=True).strip()


old = (root / "ascend-source/.github/vllm-main-verified.commit").read_text().strip()
assert old == "e5588e49bc2642670116664a7fc4096e27adb179"
report = root / "report-input/qa-review.md"
assert old in report.read_text(encoding="utf-8")
assert sha("vllm-source") in report.read_text(encoding="utf-8")
assert sha("flow-source") in report.read_text(encoding="utf-8")
(root / "report-input/handoff.json").write_text(
    json.dumps(
        {
            "fresh_scan": True,
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "resolved_range": {
                "vllm_old_sha": old,
                "vllm_new_sha": sha("vllm-source"),
                "vllm_ascend_sha": sha("ascend-source"),
            },
            "flow_sha": sha("flow-source"),
        }
    ),
    encoding="utf-8",
)
