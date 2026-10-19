# SPDX-License-Identifier: Apache-2.0
"""Fork-only CPU smoke of the PR's production scan and original QA call sites."""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

from main2main_flow.flow import Main2MainFlow
from main2main_flow.interface_report.handoff import load_assessment, load_report
from main2main_flow.interface_report.run import git
from main2main_flow.interface_report.runtime import scan_for_flow
from main2main_flow.scripts.agent import opencode_adapter
from probe_permissions import probe

FLOW_SHA = "ae112de441995cf8864c5575943f27b13f4ea333"
OLD = "e5588e49bc2642670116664a7fc4096e27adb179"
NEW = "85c09e9885e346ea1612da30ebff5a75f67d2350"
ASCEND = "6556ad5af1baaf481e996fc2788e233ce672f98e"
HISTORICAL_FIX = "2b4fb0bd416d5589a8fd579f32c22848d9e1564f"
FIX_FILE = "vllm_ascend/patch/worker/patch_qwen3_5.py"


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_state(root):
    return {
        str(folder): {
            "head": git(folder, "rev-parse", "HEAD"),
            "status": git(folder, "status", "--porcelain", "--untracked-files=all"),
            "diff": git(folder, "diff", "HEAD"),
        }
        for folder in (root / "ascend-source", root / "vllm-source", root / "flow-source")
    }


class StopBeforePlan(Exception):
    """Test boundary: real detection finished, no planning or adaptation starts."""


def prepare_flow(root, workspace):
    flow = Main2MainFlow(
        vllm_path=str(root / "vllm-source"),
        vllm_ascend_path=str(root / "ascend-source"),
    )
    workspace.mkdir(parents=True)

    def stop_plan(upstream, old, new, **kwargs):
        assert (old, new) == (OLD, NEW)
        raise StopBeforePlan

    with (
        patch("main2main_flow.flow.WORKSPACE_DIR", workspace),
        patch("main2main_flow.flow.scan_for_flow", wraps=scan_for_flow) as scanner,
        patch("main2main_flow.flow.run_plan", side_effect=stop_plan) as planner,
    ):
        try:
            flow.analyze_commit_and_plan_step()
        except StopBeforePlan:
            pass
        else:
            raise AssertionError("Did not stop at the pre-plan boundary")
        scanner.assert_called_once()
        planner.assert_called_once()
    assert flow.state.base_commit == OLD and flow.state.target_commit == NEW
    report = Path(flow.state.interface_report)
    metadata = json.loads((report.parent / "run-metadata.json").read_text())
    assert metadata["inputs"]["vllm_ascend_sha"] == ASCEND
    status = json.loads((report.parent / "run-status.json").read_text())
    assert status["scan"] == "completed" and status["model_calls"] == 0
    return flow, report, metadata


def scan(root):
    output = root / "output"
    assert git(root / "flow-source", "rev-parse", "HEAD") == FLOW_SHA
    assert git(root / "vllm-source", "rev-parse", "HEAD") == NEW
    assert git(root / "ascend-source", "rev-parse", "HEAD") == ASCEND
    before = source_state(root)
    assert all(not value["status"] for value in before.values())
    # No restored cache: the first production call must execute the analyzer.
    cache_home = output / "cache-home"
    assert not cache_home.exists()
    os.environ["XDG_CACHE_HOME"] = str(cache_home)
    first, report, metadata = prepare_flow(root, output / "first")
    assert metadata["cache_status"] == "miss"
    repeated, repeat_report, repeated_metadata = prepare_flow(root, output / "repeated")
    assert repeated_metadata["cache_status"] == "hit"
    assert metadata["cache_key"] == repeated_metadata["cache_key"]
    assert digest(report) == digest(repeat_report)
    assert not (repeat_report.parent / "scan.log").exists()

    # Exercise the real equal-endpoint resolution without changing any marker.
    unchanged = Main2MainFlow(
        vllm_path=first.state.vllm_path,
        vllm_ascend_path=first.state.vllm_ascend_path,
        target_commit=OLD,
    )
    no_change = output / "no-change"
    no_change.mkdir()
    with (
        patch("main2main_flow.flow.WORKSPACE_DIR", no_change),
        patch("main2main_flow.flow.scan_for_flow", wraps=scan_for_flow) as scanner,
        patch("main2main_flow.flow.run_plan") as planner,
        patch("main2main_flow.flow.run_opencode_review") as model,
    ):
        assert unchanged.analyze_commit_and_plan_step() == "HasNoCommit"
        assert not unchanged.state.interface_report
        assert unchanged._run_adapter_qa(
            first.state.vllm_ascend_path,
            first.state.vllm_path,
            "no-change",
            str(no_change),
            "0.15.0",
        ) == ([], "")
        scanner.assert_not_called()
        planner.assert_not_called()
        model.assert_not_called()
    assert source_state(root) == before
    save(
        output / "scan-acceptance.json",
        {
            "flow_sha": FLOW_SHA,
            "range": {"old": OLD, "new": NEW, "ascend": ASCEND},
            "automatic_range": True,
            "fresh_scan": True,
            "cache_reuse": True,
            "no_change_skips_scan_and_model": True,
            "scan_calls_per_upgrade": 1,
            "plan_intercepted": True,
            "sources_unchanged": True,
            "qa_report": str(report),
            "report_sha256": digest(report),
            "report_bytes": report.stat().st_size,
            "state": first.state.model_dump(),
            "adaptation_executed": False,
            "npu_tests_executed": False,
        },
    )


def read_only_config(root, report, review_path):
    agent_cwd = root / "gray-code"
    reads = {"*": "deny"}
    for folder in (root / "qa-ascend", root / "vllm-source", root / "flow-source", root / "output"):
        for spelling in (str(folder), str(folder).lstrip("/"), os.path.relpath(folder, agent_cwd)):
            reads[spelling.replace(os.sep, "/") + "/**"] = "allow"
    return {
        "$schema": "https://opencode.ai/config.json",
        "lsp": False,
        "provider": {"deepseek": {"options": {"apiKey": "{env:MAIN2MAIN_API_KEY}"}, "models": {"deepseek-flash": {}}}},
        "permission": {
            "*": "deny",
            "read": reads,
            "edit": {
                "*": "deny",
                str(review_path): "allow",
                os.path.relpath(review_path, agent_cwd).replace(os.sep, "/"): "allow",
                str(review_path).lstrip("/"): "allow",
            },
            "external_directory": {"*": "deny", str(root) + "/**": "allow"},
        },
    }


def completed_reads(path):
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [
        e["part"]["state"]["input"].get("filePath", "")
        for e in events
        if e.get("type") == "tool_use"
        and e.get("part", {}).get("tool") == "read"
        and e["part"].get("state", {}).get("status") == "completed"
    ]


def qa(root):
    assert os.environ.get("MAIN2MAIN_API_KEY"), "Missing MAIN2MAIN_API_KEY"
    output = root / "output"
    acceptance = json.loads((output / "scan-acceptance.json").read_text())
    assert acceptance["flow_sha"] == FLOW_SHA
    flow = Main2MainFlow(**acceptance["state"])
    report = Path(flow.state.interface_report)
    assert digest(report) == acceptance["report_sha256"]
    before = source_state(root)
    qa_source = root / "qa-ascend"
    subprocess.run(
        ["git", "clone", "--shared", "--no-checkout", str(root / "ascend-source"), str(qa_source)], check=True
    )
    git(qa_source, "checkout", "--detach", ASCEND)
    try:
        git(qa_source, "cat-file", "-e", HISTORICAL_FIX + "^{commit}")
    except subprocess.CalledProcessError:
        git(qa_source, "fetch", "--depth=1", "https://github.com/vllm-project/vllm-ascend.git", HISTORICAL_FIX)
    # Replay a single existing historical file ONLY in a separate review copy.
    git(qa_source, "restore", "--source=" + HISTORICAL_FIX, "--worktree", FIX_FILE)
    assert git(qa_source, "diff", "--name-only", "HEAD") == FIX_FILE
    # Commit only the existing fixture locally, so QA must run with ZERO diff.
    # This partial historical replay is not a complete historical adaptation.
    git(qa_source, "add", FIX_FILE)
    git(qa_source, "-c", "user.name=Gray Fixture", "-c", "user.email=gray@example.invalid",
        "commit", "-m", "Local read-only historical review fixture")
    fixture_head = git(qa_source, "rev-parse", "HEAD")
    assert not git(qa_source, "status", "--porcelain")
    flow.state.steps = [dict(id="historical-pr12020", start_commit=OLD, end_commit=NEW)]
    flow.state.total_steps = 1
    adapter_output = output / "readonly-adapter"
    adapter_output.mkdir()
    summary_path = adapter_output / "step_summary.md"
    qa_output = output / "original-qa"
    qa_output.mkdir()
    review_path = qa_output / "review.json"
    config_path = output / "opencode-config.json"
    os.environ.update(
        OPENCODE_CONFIG=str(config_path),
        MAIN2MAIN_ADAPTER_TIMEOUT_MINUTES="8",
        MAIN2MAIN_MODEL_REVIEW="deepseek/deepseek-flash",
        MAIN2MAIN_MODEL_ADAPT="deepseek/deepseek-flash",
        MAIN2MAIN_INTERFACE_REPORT=str(output / "must-not-override-auto-report.md"),
    )
    os.chdir(root / "gray-code")
    adapter_config = read_only_config(root, report, summary_path)
    save(config_path, adapter_config)
    probe(adapter_config, report, summary_path, adapter_output)
    inputs = dict(
        step_id="historical-pr12020", previous_step_id="", previous_step_summary_path="",
        is_last_step=False, step_dir=str(adapter_output), patch_path="", changed_files_path="",
        ascend_path=str(qa_source), vllm_path=str(root / "vllm-source"),
        release_tag=(qa_source / ".github/vllm-release-tag.commit").read_text().strip().lstrip("v"),
        role="adapter", mode="adapter", error_logs="[]", code_structure_guide_file="code-structure-guide.md",
        **flow._interface_adapter_inputs(flow.state.steps[0]),
    )
    build_prompt = opencode_adapter._build_prompt

    def gray_prompt(values):
        prompt, refs = build_prompt(values)
        prompt += (
            "\n\n## Fork gray boundary (overrides adaptation/output actions above)\n"
            "This is a READ-ONLY source confirmation exercise, not an adaptation run. "
            "Do not edit business code, run bash, tests, imports, builds or MCP tools. "
            "Use only read for the report and exact upstream/downstream source files. "
            "The separate qa-ascend tree contains a SINGLE historical file replay; "
            "judge this actual tree, not assumptions about the complete historical commit. "
            f"Write ONLY {summary_path}; do not write analysis.md or result.json. "
            "Include the required interface-assessment block for EVERY root. "
            "For already compatible code use no_change with source proof. For a confirmed "
            "need to adapt, use insufficient_evidence with a reason explicitly saying "
            "adaptation is needed but intentionally not executed in this read-only gray. "
            "Do not claim adapted. No code edits are permitted. Use bounded targeted reads, "
            "then write the summary and finish."
        )
        (adapter_output / "input-prompt.txt").write_text(prompt, encoding="utf-8")
        return prompt, refs

    status = {
        "flow_sha": FLOW_SHA,
        "entrypoint": "Main2MainFlow._run_adapter_qa",
        "scope": "read_only_adapter_then_independent_qa_on_committed_single_file_fixture",
        "qa_business_diff_bytes": 0,
        "fixture_commit_local_only": fixture_head,
        "adaptation_executed": False,
        "npu_tests_executed": False,
        "report_consumed": False,
    }
    try:
        with (
            patch.object(opencode_adapter, "_build_prompt", side_effect=gray_prompt),
            patch.object(opencode_adapter, "_MAX_STALE_RETRIES", 0),
        ):
            result = opencode_adapter.run_opencode_adapter(inputs)
        contract = load_report(str(report))
        decisions = load_assessment(contract, adapter_output, "historical-pr12020", OLD, NEW)
        assert result.is_noop and all(i["decision"] != "adapted" for i in decisions["items"])
        adapter_reads = completed_reads(adapter_output / "opencode_raw.jsonl")
        assert str(report) in adapter_reads
        assert any(str(qa_source) in p for p in adapter_reads)
        assert any(str(root / "vllm-source") in p for p in adapter_reads)
        assert not git(qa_source, "status", "--porcelain"), "Adapter changed review source"
        # Reuse the EXISTING summary artifact at QA's step path; no new report.
        qa_summary = qa_output / "step_summary.md"
        qa_summary.write_bytes(summary_path.read_bytes())
        config = read_only_config(root, report, review_path)
        save(config_path, config)
        probe(config, report, review_path, qa_output)
        status.update(adapter_assessment=decisions, adapter_report_consumed=True,
                      real_adapter_invocations=1, adapter_read_only_override=True)
        issues, session = flow._run_adapter_qa(
            str(qa_source),
            str(root / "vllm-source"),
            "historical-pr12020",
            str(qa_output),
            (qa_source / ".github/vllm-release-tag.commit").read_text().strip().lstrip("v"),
        )
        review = json.loads(review_path.read_text())
        assert review["verdict"] in ("pass", "fail") and isinstance(review["issues"], list)
        usages = review.get("interface_report_usage", [])
        roots = set(re.findall(r"^## ([0-9a-f]{16})", report.read_text(encoding="utf-8"), re.MULTILINE))
        assert {u["root_cause_id"] for u in usages} == roots
        assert not any(i.startswith("interface-handoff:") for i in issues), issues
        reads = completed_reads(qa_output / "opencode_qa_raw.jsonl")
        assert str(report) in reads, "No completed report read event"
        assert str(qa_summary) in reads, "QA did not read adapter summary"
        assert qa_summary.read_bytes() == summary_path.read_bytes(), "QA changed adapter decisions"
        assert any(str(root / "vllm-source") in p for p in reads), "No upstream source read"
        assert any(str(qa_source) in p for p in reads), "No downstream source read"
        rejected = {i["root_cause_id"] for i in decisions["items"] if i["decision"] == "no_change"}
        assert rejected, "No rejected/already-compatible candidate was exercised"
        assert any(u["root_cause_id"] in rejected and u["judgment"] == "agree" for u in usages), (
            "No independent QA confirmation of adapter's no-change assessment"
        )
        status.update(
            report_consumed=True,
            report_roots_used=usages,
            review_verdict=review["verdict"],
            issues=issues,
            session_id=session,
            real_qa_invocations=1,
            qa_reviewed_no_change_with_empty_diff=True,
        )
    finally:
        status.update(
            sources_unchanged=source_state(root) == before,
            review_copy_unchanged=git(qa_source, "rev-parse", "HEAD") == fixture_head
            and not git(qa_source, "status", "--porcelain"),
            report_unchanged=digest(report) == acceptance["report_sha256"],
        )
        save(output / "qa-acceptance.json", status)
    assert status["sources_unchanged"] and status["review_copy_unchanged"] and status["report_unchanged"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("scan", "qa"))
    args = parser.parse_args()
    (scan if args.phase == "scan" else qa)(Path.cwd().resolve())


if __name__ == "__main__":
    main()
