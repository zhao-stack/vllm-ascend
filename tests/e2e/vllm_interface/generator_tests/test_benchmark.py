# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure benchmark failures cannot be hidden by changed classifications."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.vllm_interface_contracts.benchmark import main, score_report, triage_report


def _inputs() -> dict:
    return {
        "vllm_old_sha": "a" * 40,
        "vllm_new_sha": "b" * 40,
        "vllm_ascend_sha": "c" * 40,
        "external_sources": {},
        "scenario": "main2main",
        "profile": "exact-contracts",
    }


def _finding(name: str = "field", classification: str = "introduced_break", action: str = "modify") -> dict:
    return {
        "id": name,
        "root_cause_id": name,
        "classification": classification,
        "action": action,
        "priority": "P1" if action == "modify" else "P2",
        "relation": "direct_attribute",
        "contract_kind": "attribute_presence",
        "upstream": {"old": {"file": "vllm/api.py", "name": name}},
        "downstream": {"file": "vllm_ascend/consumer.py", "name": name},
        "details": {
            "target": "vllm.api.Base.field",
            "receiver_type": "vllm_ascend.consumer.Child",
            "access_kind": "instance",
        },
    }


def _report() -> dict:
    capabilities = {
        name: {"state": "analyzed"}
        for name in ("monkey_patch", "direct_attribute", "override", "direct_import", "direct_call", "inherited_state")
    }
    return {
        "metadata": {**_inputs(), "range_analyzer_version": "2.7.0", "analysis_plan": {"capabilities": capabilities}},
        "findings": [_finding()],
    }


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "id": "reviewed-example",
        "split": "development",
        "inputs": _inputs(),
        "cases": [
            {
                "id": "removed-field",
                "rationale": "Old initializes the field, new removes it; the consumer still reads it.",
                "evidence": ["Independent attribute-access witness"],
                "evidence_kind": "runtime_contract",
                "root_count": 1,
                "checks": [
                    {
                        "match": {"upstream_symbol.name": "field", "relation": "direct_attribute"},
                        "count": 1,
                        "expected": {"classification": "introduced_break", "action": "modify", "priority": "P1"},
                    }
                ],
            }
        ],
    }


def test_score_excludes_unlabelled_findings_from_precision() -> None:
    report = _report()
    report["findings"].append(_finding("unreviewed"))
    score = score_report(_manifest(), report)
    assert score["passed"]
    assert score["metrics"]["unassessed_actionable_roots"] == 1
    assert score["metrics"]["reviewed_actionable_root_precision"] == 1


@pytest.mark.parametrize("decision", ["classification", "action", "priority"])
def test_wrong_decision_fails_without_disappearing_from_case(decision: str) -> None:
    report = _report()
    report["findings"][0][decision] = {"classification": "analysis_unresolved", "action": "review", "priority": "P2"}[
        decision
    ]
    score = score_report(_manifest(), report)
    assert not score["passed"]
    assert score["cases"][0]["checks"][0]["finding_ids"] == ["field"]
    assert any(decision in error for error in score["cases"][0]["checks"][0]["errors"])


def test_missing_case_reduces_recall() -> None:
    report = _report()
    report["findings"] = []
    score = score_report(_manifest(), report)
    assert not score["passed"]
    assert score["metrics"]["known_case_recall"] == 0
    assert score["metrics"]["reviewed_actionable_root_precision"] is None


def test_negative_control_detects_false_positive() -> None:
    manifest = _manifest()
    case = manifest["cases"][0]
    case["evidence_kind"] = "negative_control"
    case["root_count"] = 0
    case["checks"][0].update(count=0, expected={})
    score = score_report(manifest, _report())
    assert not score["passed"]
    assert score["metrics"]["reviewed_actionable_root_precision"] == 0


@pytest.mark.parametrize(
    "key", ["vllm_old_sha", "vllm_new_sha", "vllm_ascend_sha", "external_sources", "profile", "scenario"]
)
def test_wrong_input_identity_is_rejected(key: str) -> None:
    manifest = _manifest()
    manifest["inputs"][key] = "wrong"
    with pytest.raises(ValueError, match="input mismatch"):
        score_report(manifest, _report())


def test_ci_scope_cannot_be_scored_as_full_analysis() -> None:
    report = _report()
    report["metadata"]["analysis_plan"]["capabilities"]["monkey_patch"]["state"] = "skipped"
    with pytest.raises(ValueError, match="monkey_patch"):
        score_report(_manifest(), report)


def test_decision_based_selector_and_overlapping_cases_are_rejected() -> None:
    manifest = _manifest()
    manifest["cases"][0]["checks"][0]["match"]["classification"] = "introduced_break"
    with pytest.raises(ValueError, match="source contracts"):
        score_report(manifest, _report())
    manifest = _manifest()
    second = copy.deepcopy(manifest["cases"][0])
    second["id"] = "duplicate-contract"
    manifest["cases"].append(second)
    with pytest.raises(ValueError, match="Overlapping"):
        score_report(manifest, _report())


def test_root_duplication_is_not_hidden_by_correct_finding_count() -> None:
    report = _report()
    other = copy.deepcopy(report["findings"][0])
    other.update(id="second-callsite", root_cause_id="incorrect-second-root")
    report["findings"].append(other)
    manifest = _manifest()
    manifest["cases"][0]["checks"][0]["count"] = 2
    score = score_report(manifest, report)
    assert not score["passed"]
    assert score["cases"][0]["observed_root_count"] == 2
    assert not score["cases"][0]["checks"][0]["errors"]


def test_two_consumers_of_one_upstream_cause_share_one_reviewed_case() -> None:
    report = _report()
    second = copy.deepcopy(report["findings"][0])
    second["id"] = "second-consumer"
    second["downstream"]["file"] = "vllm_ascend/other.py"
    report["findings"].append(second)
    manifest = _manifest()
    manifest["cases"][0]["checks"][0]["count"] = 2
    score = score_report(manifest, report)
    assert score["passed"]
    assert score["cases"][0]["observed_root_count"] == 1
    assert score["cases"][0]["checks"][0]["finding_ids"] == ["field", "second-consumer"]


def test_triage_retains_every_unresolved_location_and_is_deterministic() -> None:
    report = _report()
    report["findings"] = [
        _finding("one", "analysis_unresolved", "review"),
        _finding("two", "analysis_unresolved", "review"),
    ]
    before = copy.deepcopy(report)
    result = triage_report(report)
    assert report == before
    assert result == triage_report(report)
    assert result["summary"]["unresolved_findings"] == 2
    assert result["summary"]["review_groups"] == 1
    assert result["groups"][0]["finding_ids"] == ["one", "two"]
    assert result["groups"][0]["category"] == "downstream_receiver_without_upstream_lookup_root"


def test_cli_records_failure_and_refuses_input_overwrite(tmp_path: Path) -> None:
    report_path, manifest_path, output = (tmp_path / name for name in ("report.json", "manifest.json", "score.json"))
    report = _report()
    report["findings"] = []
    report_path.write_text(json.dumps(report), encoding="utf-8")
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    args = ["score", "--report", str(report_path), "--manifest", str(manifest_path)]
    assert main([*args, "--output", str(output)]) == 1
    assert json.loads(output.read_text())["metrics"]["known_case_recall"] == 0
    with pytest.raises(SystemExit):
        main([*args, "--output", str(manifest_path)])
    assert json.loads(manifest_path.read_text()) == _manifest()
