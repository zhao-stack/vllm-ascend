# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual adapter-qa entrypoint without running the adaptation flow."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    root = Path.cwd().resolve()
    ascend, upstream, output = root / "ascend-source", root / "vllm-source", root / "output"
    output.mkdir(exist_ok=True)
    report = root / "report-input" / "qa-review.md"
    report_hash = hashlib.sha256(report.read_bytes()).hexdigest()
    handoff = root / "report-input/handoff.json"
    if handoff.exists():
        data = json.loads(handoff.read_text(encoding="utf-8"))
        assert data["fresh_scan"] and data["report_sha256"] == report_hash
        resolved = data["resolved_range"]
        assert git(ascend, "rev-parse", "HEAD").decode().strip() == resolved["vllm_ascend_sha"]
        assert git(upstream, "rev-parse", "HEAD").decode().strip() == resolved["vllm_new_sha"]
        assert (ascend / ".github/vllm-main-verified.commit").read_text().strip() == resolved["vllm_old_sha"]
    review_path = output / "review.json"
    # Without a git checkout as cwd, OpenCode uses '/' as its global worktree.
    agent_cwd = root / "gray-code"
    assert Path(git(agent_cwd, "rev-parse", "--show-toplevel").decode().strip()) == agent_cwd
    allowed_reads = {"*": "deny"}
    for folder in (ascend, upstream, root / "flow-source", root / "report-input", output):
        allowed_reads[str(folder) + "/**"] = "allow"
        # CLI sessions can resolve to the global '/' worktree even after bootstrap.
        allowed_reads[str(folder).lstrip("/") + "/**"] = "allow"
        allowed_reads[os.path.relpath(folder, agent_cwd).replace(os.sep, "/") + "/**"] = "allow"
    config = {
        "$schema": "https://opencode.ai/config.json",
        "lsp": False,
        "provider": {"deepseek": {"options": {"apiKey": "{env:MAIN2MAIN_API_KEY}"}, "models": {"deepseek-flash": {}}}},
        "permission": {
            "*": "deny",
            "read": allowed_reads,
            "edit": {
                "*": "deny",
                str(review_path): "allow",
                "../output/review.json": "allow",
                str(review_path).lstrip("/"): "allow",
            },
            "external_directory": {"*": "deny", str(root) + "/**": "allow"},
        },
    }
    config_path = output / "opencode-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    os.environ["OPENCODE_CONFIG"] = str(config_path)
    os.environ["MAIN2MAIN_ADAPTER_TIMEOUT_MINUTES"] = "8"
    os.environ["MAIN2MAIN_MODEL_REVIEW"] = "deepseek/deepseek-flash"
    os.environ["MAIN2MAIN_INTERFACE_REPORT"] = str(report)
    os.chdir(agent_cwd)
    from probe_permissions import probe

    probe(config, report, review_path, output)
    from main2main_flow.flow import Main2MainFlow

    flow = Main2MainFlow()
    args = dict(
        ascend_path=str(ascend),
        vllm_path=str(upstream),
        step_id="historical-pr12020",
        step_dir=str(output),
        release_tag=(ascend / ".github/vllm-release-tag.commit").read_text().strip().lstrip("v"),
    )
    # Real entrypoint, before replaying the historical diff: no diff means no model call.
    with patch("main2main_flow.flow.run_opencode_review") as transport:
        assert flow._run_adapter_qa(**args) == ([], "")
        transport.assert_not_called()
    # Fixed, already-existing adaptation file, not a generated adaptation.
    git(ascend, "fetch", "--depth=1", "origin", "2b4fb0bd416d5589a8fd579f32c22848d9e1564f")
    git(
        ascend,
        "restore",
        "--source=2b4fb0bd416d5589a8fd579f32c22848d9e1564f",
        "--worktree",
        "vllm_ascend/patch/worker/patch_qwen3_5.py",
    )
    before = git(ascend, "diff", "HEAD")
    assert before
    checks = ["no_diff_skips_model"]
    with patch("main2main_flow.flow.run_opencode_review") as transport:
        for name, value in (("missing", str(output / "absent.md")), ("oversized", str(output / "large.md"))):
            if name == "oversized":
                (output / "large.md").write_text("x" * 1_000_001)
            with patch.dict(os.environ, {"MAIN2MAIN_INTERFACE_REPORT": value}):
                issues, _ = flow._run_adapter_qa(**args)
                assert issues and "report" in issues[0]
            transport.assert_not_called()
            checks.append(name + "_report_stops_before_model")
        transport.return_value = ("", "")
        with patch.dict(os.environ, {"MAIN2MAIN_INTERFACE_REPORT": ""}):
            flow._run_adapter_qa(**args)
            assert "Interface detection reference" not in transport.call_args.args[0]
        checks.append("disabled_preserves_original_prompt")
    (output / "large.md").unlink()
    status = {
        "entrypoint": "Main2MainFlow._run_adapter_qa",
        "checks": checks,
        "input_sha256": report_hash,
        "scope": "one_historical_file_diff",
        "adaptation_executed": False,
        "npu_tests_executed": False,
    }
    try:
        issues, session = flow._run_adapter_qa(**args)
        status.update(issues=issues, session_id=session)
        result = json.loads(review_path.read_text(encoding="utf-8"))
        assert result["verdict"] in ("pass", "fail") and isinstance(result["issues"], list)
        usages = result.get("interface_report_usage", [])
        assert usages and all(
            u.get("assessment") and "## " + u["root_cause_id"] in report.read_text(encoding="utf-8") for u in usages
        )
        # A model claiming it read the report is insufficient: require a completed read event.
        events = [json.loads(line) for line in (output / "opencode_qa_raw.jsonl").read_text().splitlines()]
        reads = [
            e
            for e in events
            if e.get("type") == "tool_use"
            and e.get("part", {}).get("tool") == "read"
            and e["part"].get("state", {}).get("status") == "completed"
            and e["part"]["state"].get("input", {}).get("filePath") == str(report)
        ]
        assert reads, "No completed report read event"
        status.update(report_consumed=True, report_roots_used=usages, review_verdict=result["verdict"])
    finally:
        status["source_unchanged"] = (
            git(ascend, "diff", "HEAD") == before and not git(upstream, "status", "--porcelain").strip()
        )
        status["report_unchanged"] = hashlib.sha256(report.read_bytes()).hexdigest() == report_hash
        (output / "acceptance.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    assert status["source_unchanged"] and status["report_unchanged"]


if __name__ == "__main__":
    main()
