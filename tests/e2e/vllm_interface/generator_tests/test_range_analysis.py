from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tools.vllm_interface_contracts import cache as cache_module
from tools.vllm_interface_contracts import generator as generator_module
from tools.vllm_interface_contracts.cli import main as cli_main
from tools.vllm_interface_contracts.generator import InterfaceBoundaryGenerator, RepositoryIndex
from tools.vllm_interface_contracts.range_analysis import (
    GitSnapshot,
    analyze_range,
    discover_imports,
    validate_current_contracts,
    write_reports,
)


def _write(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", ".")
    _git(root, "-c", "user.name=Interface Test", "-c", "user.email=test@example.com", "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD")


def _commit_with_git_symlink(root: Path, link: str, message: str) -> str:
    """Commit a mode-120000 entry while retaining its Windows text stub."""

    _git(root, "add", ".")
    blob = _git(root, "hash-object", "-w", link)
    _git(root, "update-index", "--add", "--cacheinfo", "120000", blob, link)
    _git(
        root,
        "-c",
        "user.name=Interface Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _git(root, "rev-parse", "HEAD")


def test_git_symlink_stub_resolves_and_tracks_target_blob(tmp_path: Path) -> None:
    root = tmp_path / "vllm"
    root.mkdir()
    _git(root, "init")
    for package in (
        "vllm",
        "vllm/models",
        "vllm/models/deepseek_v4",
        "vllm/models/deepseek_v4/amd",
        "vllm/models/deepseek_v4/nvidia",
    ):
        _write(root, f"{package}/__init__.py", "")
    target = "vllm/models/deepseek_v4/nvidia/model.py"
    link = "vllm/models/deepseek_v4/amd/model.py"
    _write(root, target, "class Base:\n    def run(self, value):\n        return value\n")
    _write(root, link, "../nvidia/model.py\n")
    old_sha = _commit_with_git_symlink(root, link, "old symlink target")

    index = RepositoryIndex(root, "vllm")
    assert not index.parse_errors
    assert "vllm.models.deepseek_v4.amd.model.Base.run" in index.callables
    endpoint = GitSnapshot(root, old_sha).endpoint(link, "Base", "run")
    assert endpoint.symbol_kind == "callable"
    assert endpoint.signature_status == "exact"

    relative_files = tuple(path.relative_to(root).as_posix() for path in sorted((root / "vllm").rglob("*.py")))
    old_identities, old_reason = generator_module._repository_file_cache_identities(
        root,
        "vllm",
        old_sha,
        relative_files,
        frozenset(),
    )
    assert old_reason is None
    assert old_identities is not None

    _write(root, target, "class Base:\n    def run(self, value, required):\n        return value\n")
    _git(root, "add", target)
    _git(
        root,
        "-c",
        "user.name=Interface Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        "new symlink target",
    )
    new_sha = _git(root, "rev-parse", "HEAD")
    new_identities, new_reason = generator_module._repository_file_cache_identities(
        root,
        "vllm",
        new_sha,
        relative_files,
        frozenset(),
    )
    assert new_reason is None
    assert new_identities is not None
    assert old_identities[link][0] != new_identities[link][0]


def _repositories(tmp_path: Path, *, old_method: str, new_method: str) -> tuple[Path, Path, str, str, str]:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", f"class Base:\n    {old_method}\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/base.py", f"class Base:\n    {new_method}\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/child.py",
        "from vllm.base import Base\n\nclass Child(Base):\n    def run(self, value):\n        return value\n",
    )
    ascend_sha = _commit(ascend_root, "baseline")
    return vllm_root, ascend_root, old_sha, new_sha, ascend_sha


def _run(
    vllm_root: Path,
    ascend_root: Path,
    old_sha: str,
    new_sha: str,
    ascend_sha: str,
) -> dict[str, object]:
    return analyze_range(
        vllm_root=vllm_root,
        ascend_root=ascend_root,
        old=old_sha,
        new=new_sha,
        expect_ascend_sha=ascend_sha,
    )


def _call_repositories(
    tmp_path: Path,
    *,
    old_source: str,
    new_source: str,
    consumer_source: str,
) -> tuple[Path, Path, str, str, str]:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/api.py", old_source)
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/api.py", new_source)
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", consumer_source)
    ascend_sha = _commit(ascend_root, "baseline")
    return vllm_root, ascend_root, old_sha, new_sha, ascend_sha


def test_range_classifies_compatible_to_incompatible_override(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = _run(*roots)
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert any(item["relation"] == "override" for item in introduced)
    assert report["summary"]["introduced_break"] >= 1


def test_range_separates_preexisting_incompatibility(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    report = _run(*roots)
    overrides = [item for item in report["findings"] if item["relation"] == "override"]
    assert overrides
    assert overrides[0]["classification"] == "preexisting"
    assert overrides[0]["action"] == "review"
    assert overrides[0]["priority"] == "P2"
    assert overrides[0]["details"]["new_delta_on_preexisting_break"] is True
    assert [item["name"] for item in overrides[0]["details"]["parameter_delta"]["added"]] == ["new_required"]


def test_optional_only_override_is_review_without_exact_upstream_call(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=("class Base:\n    def __init__(self, value, optional=None):\n        self.value = value\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["action"] == "review"
    assert overrides[0]["priority"] == "P2"
    assert overrides[0]["details"]["new_optional_parameters"] == ["optional"]
    assert overrides[0]["details"]["upstream_call_evidence"] == []
    assert overrides[0]["suggestion"] == (
        "Review whether the new optional parameter can reach this downstream override at runtime. "
        "If it can, update the override signature and handle the new argument."
    )

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 0
    assert payload["summary"]["review_findings"] == 1
    assert payload["findings"] == []
    assert len(payload["review_findings"]) == 1
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "**Result: REVIEW**" in markdown
    assert (
        "The downstream override does not accept the new optional parameter `optional`, and the analyzer found no "
        "upstream call in this range that passes it to that override."
    ) in markdown
    assert "strict replacement contract" not in markdown


def test_optional_constructor_override_is_review_when_dispatch_is_not_proven(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self, value, optional=None):\n"
            "        self.value = value\n\n"
            "def build():\n"
            "    return Base(1, optional=True)\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "review"
    assert override["details"]["upstream_call_evidence"] == []
    candidates = override["details"]["candidate_upstream_call_evidence"]
    assert len(candidates) == 1
    assert candidates[0]["dispatch_kind"] == "direct_constructor"


def test_optional_override_stays_actionable_when_call_and_registration_prove_dispatch(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, value):\n        self.value = value\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self, value, optional=None):\n"
            "        self.value = value\n\n"
            "def build():\n"
            "    return Base(1, optional=True)\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n\n"
            "def register():\n"
            "    from vllm.model_executor.custom_op import CustomOp\n"
            "    registry = {'Unrelated': Child}\n"
            "    registry.update({'Base': Child})\n"
            "    for name, op_cls in registry.items():\n"
            "        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["action"] == "modify"
    assert overrides[0]["priority"] == "P1"
    evidence = overrides[0]["details"]["upstream_call_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["matched_parameters"] == ["optional"]
    assert evidence[0]["file"] == "vllm/api.py"
    assert evidence[0]["dispatch_kind"] == "direct_constructor"
    assert evidence[0]["dispatch_proof"][0]["downstream_target"] == "vllm_ascend.consumer.Child"

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 1
    assert payload["summary"]["review_findings"] == 0
    assert len(payload["findings"]) == 1
    assert payload["review_findings"] == []


def test_optional_method_override_ignores_sibling_super_call(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source=(
            "class Base:\n"
            "    def run(self, value, optional=None): return value\n\n"
            "class Sibling(Base):\n"
            "    def invoke(self, value):\n"
            "        return super().run(value, optional=True)\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "review"
    assert override["details"]["upstream_call_evidence"] == []
    candidates = override["details"]["candidate_upstream_call_evidence"]
    assert len(candidates) == 1
    assert candidates[0]["dispatch_kind"] == "super_member"


def test_optional_method_override_is_actionable_for_defining_class_self_dispatch(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n    def run(self, value): return value\n    def invoke(self, value): return self.run(value)\n"
        ),
        new_source=(
            "class Base:\n"
            "    def run(self, value, optional=None): return value\n"
            "    def invoke(self, value): return self.run(value, optional=True)\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    override = next(
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    )
    assert override["action"] == "modify"
    evidence = override["details"]["upstream_call_evidence"]
    assert len(evidence) == 1
    assert evidence[0]["dispatch_kind"] == "self_member"
    assert evidence[0]["receiver_class"] == "vllm.api.Base"


def test_new_upstream_override_contract_conflict_is_introduced(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    pass\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = _run(*roots)
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["compatibility"]["old"]["exists"] is False
    assert overrides[0]["compatibility"]["new"]["exists"] is True


def test_new_upstream_patch_contract_conflict_is_introduced(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    pass\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\ndef replacement(self, value): return value\n\nBase.run = replacement\n"
        ),
    )
    report = _run(*roots)
    patches = [
        item
        for item in report["findings"]
        if item["relation"] == "monkey_patch" and item["contract_kind"] == "call_arguments"
    ]
    assert len(patches) == 1
    assert patches[0]["classification"] == "introduced_break"
    assert patches[0]["compatibility"]["old"]["exists"] is False
    assert patches[0]["compatibility"]["new"]["exists"] is True


def test_signature_status_change_emits_fail_closed_finding(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source=(
            "def wrapper(fn): return fn\n\nclass Base:\n    @wrapper\n    def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value): return value\n"),
    )
    report = _run(*roots)
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "analysis_unresolved"
    assert overrides[0]["upstream"]["old"]["signature_status"] == "exact"
    assert overrides[0]["upstream"]["new"]["signature_status"] == "unknown"
    assert overrides[0]["change"] == "callable runtime signature contract changed"


def test_local_runtime_signature_wrapper_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def target(value): return value\n",
        new_source=(
            "def override(fn):\n"
            "    def wrapped(*args, **kwargs):\n"
            "        return fn(*args, **kwargs)\n"
            "    return wrapped\n\n"
            "@override\n"
            "def target(value): return value\n"
        ),
        consumer_source=(
            "import vllm.api as api\n\ndef replacement(value): return value\n\napi.target = replacement\n"
        ),
    )
    report = _run(*roots)
    patches = [
        item
        for item in report["findings"]
        if item["relation"] == "monkey_patch" and item["contract_kind"] == "call_arguments"
    ]
    assert len(patches) == 1
    assert patches[0]["classification"] == "analysis_unresolved"
    assert patches[0]["upstream"]["old"]["signature"] == patches[0]["upstream"]["new"]["signature"]
    assert patches[0]["upstream"]["old"]["signature_status"] == "exact"
    assert patches[0]["upstream"]["new"]["signature_status"] == "unknown"
    assert patches[0]["change"] == "callable runtime signature contract changed"


def test_direct_import_relocation_is_an_introduced_break(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/old_location.py", "def helper(value): return value\n")
    old_sha = _commit(vllm_root, "old")
    _git(vllm_root, "mv", "vllm/old_location.py", "vllm/new_location.py")
    new_sha = _commit(vllm_root, "move")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm.old_location import helper\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    imports = [item for item in report["findings"] if item["relation"] == "direct_import"]
    assert len(imports) == 1
    assert imports[0]["classification"] == "introduced_break"
    assert "new_location.py" in imports[0]["change"]


def test_report_writer_separates_introduced_csv(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = _run(*roots)
    outputs = write_reports(report, tmp_path / "reports")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    introduced_csv = Path(outputs["introduced_csv"]).read_text(encoding="utf-8-sig")
    assert payload["schema_version"] == 12
    assert payload["metadata"]["stage_timings_seconds"]["report_generation"] >= 0
    assert (
        payload["metadata"]["stage_timings_seconds"]["total_with_report"]
        >= payload["metadata"]["stage_timings_seconds"]["total"]
    )
    assert payload["metadata"]["vllm_old_sha"] == roots[2]
    assert "introduced_break" in introduced_csv
    assert "contract_kind" in introduced_csv.splitlines()[0]
    assert "preexisting" not in introduced_csv


def test_import_discovery_orders_module_and_symbol_references(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "vllm_ascend/imports.py",
        "import vllm\nfrom vllm import config\nvalue = vllm.config.ModelConfig\n",
    )
    references = discover_imports(tmp_path)
    assert [(item.module, item.symbol) for item in references] == [
        ("vllm", None),
        ("vllm", "config"),
        ("vllm.config.ModelConfig", None),
    ]


def test_import_detector_ignores_symbol_missing_at_both_endpoints(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/marker.py", "VALUE = 1\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm import never_existed\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    assert not [item for item in report["findings"] if item["relation"] == "direct_import"]


def test_import_detector_resolves_from_package_import_as_submodule(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/config.py", "VALUE = 1\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/config.py", "VALUE = 2\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(ascend_root, "vllm_ascend/consumer.py", "from vllm import config\n")
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    assert not [item for item in report["findings"] if item["relation"] == "direct_import"]


def test_unchanged_verified_relationship_is_not_a_range_finding(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value): return value + 1",
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["source"] == "dynamic_relation_graph"]


def test_removed_verified_patch_target_is_introduced_break(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Base:\n    def run(self, value): return value\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/base.py", "class Base:\n    pass\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/patch.py",
        "from vllm.base import Base\n\ndef run(self, value): return value\n\nBase.run = run\n",
    )
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    patches = [item for item in report["findings"] if item["relation"] == "monkey_patch"]
    assert len(patches) == 1
    assert patches[0]["classification"] == "introduced_break"
    assert all(patches[0]["gates"].values())


def test_self_call_to_deleted_upstream_method_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["compatibility"]["old"]["compatible"] is True
    assert calls[0]["compatibility"]["new"]["exists"] is False
    assert calls[0]["details"]["lookup_root"] == "vllm.api.Base"
    assert calls[0]["details"]["resolution_basis"] == "old_fallback_self"


def test_deleted_self_call_through_stdlib_structural_base_is_detected(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("from abc import ABC\n\nclass Base(ABC):\n    def removed(self, value): return value\n"),
        new_source=("from abc import ABC\n\nclass Base(ABC):\n    pass\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["owner"] == "Base"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "missing"


def test_super_call_to_deleted_upstream_method_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def run(self, value):\n"
            "        return super().run(value)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["resolution_basis"] == "old_fallback_super"


def test_missing_dynamic_self_call_is_not_guessed_as_an_upstream_dependency(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    pass\n",
        new_source="class Base:\n    CHANGED = True\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.runtime_hook(value)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.runtime_hook"
    ]


def test_deleted_self_call_with_incomplete_mro_is_not_guessed(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from external_package import UnknownMixin\n"
            "from vllm.api import Base\n\n"
            "class Child(UnknownMixin, Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.removed"
    ]


def test_deleted_class_attribute_read_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    VALUE = 1\n",
        new_source="class Base:\n    pass\n",
        consumer_source="from vllm.api import Base\n\ndef use():\n    return Base.VALUE\n",
    )
    report = _run(*roots)
    findings = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_attribute" and item["contract_kind"] == "attribute_presence"
    ]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["action"] == "modify"
    assert findings[0]["priority"] == "P1"
    assert findings[0]["details"]["target"] == "vllm.api.Base.VALUE"
    assert all(findings[0]["gates"].values())
    assert report["summary"]["direct_attribute_dependencies"] == 1


def _inherited_state_repositories(
    tmp_path: Path,
    downstream_initializer: str,
    *,
    new_property_body: str = "return self._state.enabled",
) -> tuple[Path, Path, str, str, str]:
    return _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self, config):\n        self._state = config\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self, config):\n"
            "        self._state = config\n\n"
            "    @property\n"
            "    def requires_state(self):\n"
            f"        {new_property_body}\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, config):\n"
            f"{downstream_initializer}"
        ),
    )


def test_new_inherited_property_read_missing_downstream_state_is_a_break(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        self.local = config\n",
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    finding = findings[0]
    assert finding["classification"] == "introduced_break"
    assert finding["contract_kind"] == "required_instance_attribute"
    assert finding["direction"] == "upstream_inherited_read_to_downstream_state"
    assert finding["action"] == "modify"
    assert finding["priority"] == "P1"
    assert finding["upstream"]["old"]["symbol_kind"] == "missing"
    assert finding["upstream"]["new"]["descriptor"] == "property"
    assert finding["downstream"]["name"] == "__init__"
    assert finding["details"]["required_attribute"] == "_state"
    assert finding["details"]["initialization_status"] == "missing"
    assert report["summary"]["inherited_state_dependencies"] == 1


def test_inherited_state_is_compatible_when_downstream_assigns_the_field(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        self._state = config\n",
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "compatibility_warning"
    assert findings[0]["compatibility"]["new"]["compatible"] is True
    assert findings[0]["action"] == "review"


def test_inherited_state_is_compatible_when_downstream_calls_super_init(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        super().__init__(config)\n",
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "compatibility_warning"
    assert findings[0]["compatibility"]["new"]["compatible"] is True


def test_inherited_state_is_compatible_when_super_init_is_in_required_with_block(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        with context():\n            super().__init__(config)\n",
    )
    _write(
        roots[1],
        "vllm_ascend/context.py",
        "from contextlib import nullcontext\n\ncontext = nullcontext\n",
    )
    _write(
        roots[1],
        "vllm_ascend/consumer.py",
        "from vllm.api import Base\n"
        "from vllm_ascend.context import context\n\n"
        "class Child(Base):\n"
        "    def __init__(self, config):\n"
        "        with context():\n"
        "            super().__init__(config)\n",
    )
    ascend_sha = _commit(roots[1], "with-wrapped initializer")
    report = _run(roots[0], roots[1], roots[2], roots[3], ascend_sha)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "compatibility_warning"
    assert findings[0]["compatibility"]["new"]["compatible"] is True


def test_conditional_downstream_state_initialization_is_unresolved(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        if config:\n            self._state = config\n",
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "analysis_unresolved"
    assert findings[0]["details"]["initialization_status"] == "unknown"


def test_guarded_inherited_attribute_read_is_not_a_hard_requirement(
    tmp_path: Path,
) -> None:
    roots = _inherited_state_repositories(
        tmp_path,
        "        self.local = config\n",
        new_property_body=("if hasattr(self, '_state'):\n            return self._state.enabled\n        return False"),
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert report["summary"]["inherited_state_dependencies"] == 0


def test_terminating_initializer_branch_proves_upstream_required_state(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n"
            "    def __init__(self, config):\n"
            "        if config is not None:\n"
            "            self._state = config\n"
            "        else:\n"
            "            raise ValueError('config is required')\n"
        ),
        new_source=(
            "class Base:\n"
            "    def __init__(self, config):\n"
            "        if config is not None:\n"
            "            self._state = config\n"
            "        else:\n"
            "            raise ValueError('config is required')\n\n"
            "    @property\n"
            "    def requires_state(self):\n"
            "        return self._state.enabled\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self, config):\n"
            "        self.local = config\n"
        ),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["details"]["required_attribute"] == "_state"


def test_inherited_state_groups_members_and_transitive_impacts_by_constructor_field(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self):\n        self._state = 1\n",
        new_source=(
            "class Base:\n"
            "    def __init__(self):\n"
            "        self._state = 1\n\n"
            "    def first(self):\n"
            "        return self._state\n\n"
            "    def second(self):\n"
            "        return self._state\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def __init__(self):\n"
            "        self.local = 1\n\n"
            "class GrandChild(Child):\n"
            "    pass\n"
        ),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "inherited_state"]
    assert len(findings) == 1
    assert findings[0]["details"]["inherited_members"] == [
        "vllm.api.Base.first",
        "vllm.api.Base.second",
    ]
    assert findings[0]["details"]["impacted_downstream_classes"] == [
        "vllm_ascend.consumer.Child",
        "vllm_ascend.consumer.GrandChild",
    ]
    assert report["summary"]["inherited_state_dependencies"] == 4


def test_conditional_inherited_state_read_is_review_not_a_proven_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def __init__(self):\n        self.enabled = False\n        self.payload = 1\n"),
        new_source=(
            "class Base:\n"
            "    def __init__(self):\n"
            "        self.enabled = False\n"
            "        self.payload = 1\n\n"
            "    def use(self):\n"
            "        if self.enabled:\n"
            "            return self.payload\n"
            "        return None\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\nclass Child(Base):\n    def __init__(self):\n        self.local = 1\n"
        ),
    )
    report = _run(*roots)
    by_field = {
        item["details"]["required_attribute"]: item
        for item in report["findings"]
        if item["relation"] == "inherited_state"
    }
    assert by_field["enabled"]["classification"] == "introduced_break"
    assert by_field["enabled"]["details"]["read_condition"] == "unconditional"
    assert by_field["payload"]["classification"] == "analysis_unresolved"
    assert by_field["payload"]["details"]["read_condition"] == "conditional"
    assert by_field["payload"]["action"] == "review"


def test_deleted_annotated_instance_field_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self):\n        self.payload = 1\n",
        new_source="class Base:\n    def __init__(self):\n        pass\n",
        consumer_source=("from vllm.api import Base\n\ndef use(value: Base):\n    return value.payload\n"),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["upstream"]["old"]["descriptor"] == "instance_attribute"
    assert findings[0]["upstream"]["new"]["symbol_kind"] == "missing"


def test_deleted_dataclass_field_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("from dataclasses import dataclass\n\n@dataclass\nclass Base:\n    payload: int\n"),
        new_source=("from dataclasses import dataclass\n\n@dataclass\nclass Base:\n    replacement: int\n"),
        consumer_source=("from vllm.api import Base\n\ndef use(value: Base):\n    return value.payload\n"),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["upstream"]["old"]["descriptor"] == "instance_attribute"
    assert findings[0]["upstream"]["new"]["symbol_kind"] == "missing"


def test_deleted_inherited_self_field_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self):\n        self.payload = 1\n",
        new_source="class Base:\n    def __init__(self):\n        pass\n",
        consumer_source=(
            "from vllm.api import Base\n\nclass Child(Base):\n    def use(self):\n        return self.payload\n"
        ),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["details"]["lookup_root"] == "vllm.api.Base"
    assert findings[0]["details"]["resolution_basis"] == "old_fallback_self"


def test_deleted_self_field_after_upstream_class_consolidation_is_a_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class DFlashSpeculator:\n    def __init__(self):\n        self.dflash_causal = True\n"),
        new_source=(
            "class DraftModelSpeculator:\n"
            "    def __init__(self):\n"
            "        self._group_causal = True\n\n"
            "class DFlashSpeculator(DraftModelSpeculator):\n"
            "    pass\n"
        ),
        consumer_source=(
            "from vllm.api import DFlashSpeculator\n\n"
            "class AscendDFlashSpeculator(DFlashSpeculator):\n"
            "    def build(self):\n"
            "        return self.dflash_causal\n"
        ),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["upstream"]["old"]["owner"] == "DFlashSpeculator"
    assert findings[0]["upstream"]["new"]["owner"] == "DraftModelSpeculator"
    assert findings[0]["downstream"]["name"] == "self.dflash_causal"


def test_instance_field_moved_to_upstream_base_remains_compatible(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Parent:\n    pass\n\nclass Base(Parent):\n    def __init__(self):\n        self.payload = 1\n"
        ),
        new_source=(
            "class Parent:\n    def __init__(self):\n        self.payload = 1\n\nclass Base(Parent):\n    pass\n"
        ),
        consumer_source=("from vllm.api import Base\n\ndef use(value: Base):\n    return value.payload\n"),
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["relation"] == "direct_attribute"]


def test_dynamic_attribute_provider_is_unresolved_not_a_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self):\n        self.payload = 1\n",
        new_source=("class Base:\n    def __getattr__(self, name):\n        return provide(name)\n"),
        consumer_source=("from vllm.api import Base\n\ndef use(value: Base):\n    return value.payload\n"),
    )
    report = _run(*roots)
    findings = [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "analysis_unresolved"
    assert findings[0]["action"] == "review"
    assert findings[0]["priority"] == "P2"


def test_deleted_verified_override_target_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="pass",
    )
    report = _run(*roots)
    overrides = [item for item in report["findings"] if item["relation"] == "override"]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["compatibility"]["old"]["compatible"] is True
    assert overrides[0]["compatibility"]["new"]["exists"] is False
    assert overrides[0]["details"]["override_paths"] == [["vllm_ascend.child.Child.run", "vllm.base.Base.run"]]


def test_direct_call_required_parameter_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["direction"] == "downstream_call_to_upstream"
    assert calls[0]["details"]["call_shape"]["positional_count"] == 1


def test_triton_kernel_launch_required_parameter_is_an_introduced_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "from vllm.triton_utils import triton\n\n"
            "@triton.jit(do_not_specialize=['value'])\n"
            "def kernel(value, BLOCK_SIZE): pass\n"
        ),
        new_source=(
            "from vllm.triton_utils import triton\n\n"
            "@triton.jit(do_not_specialize=['value'])\n"
            "def kernel(value, required, BLOCK_SIZE): pass\n"
        ),
        consumer_source=("from vllm.api import kernel\n\ndef use():\n    kernel[(2,)](1, BLOCK_SIZE=16)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["compatibility"]["old"]["compatible"] is True
    assert calls[0]["compatibility"]["new"]["compatible"] is False
    assert calls[0]["details"]["invocation_kind"] == "triton_kernel_launch"
    assert calls[0]["evidence"][0]["callee"] == "kernel[2,]"


def test_direct_call_keyword_rename_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(*, value): return value\n",
        new_source="def helper(*, item): return item\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(value=1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"


def test_direct_constructor_call_uses_bound_init_signature(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Config:\n    def __init__(self, value): self.value = value\n",
        new_source="class Config:\n    def __init__(self, value, required): self.value = value\n",
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "direct"
    assert calls[0]["upstream"]["old"]["symbol_kind"] == "constructor"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "constructor"


def test_annotated_instance_method_call_is_checked(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Service:\n    def run(self, value): return value\n",
        new_source="class Service:\n    def run(self, value, required): return value\n",
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "instance"


def test_constructed_instance_method_call_is_checked(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    def __init__(self): pass\n    def run(self, value): return value\n"),
        new_source=("class Service:\n    def __init__(self): pass\n    def run(self, value, required): return value\n"),
        consumer_source=(
            "from vllm.api import Service\n\ndef use():\n    service = Service()\n    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call"
        and item["contract_kind"] == "call_arguments"
        and item["details"]["target"].endswith("Service.run")
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["details"]["access_kind"] == "instance"


def test_inherited_instance_method_is_resolved_in_each_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def run(self, value): return value\n\nclass Service(Base):\n    pass\n"),
        new_source=(
            "class Base:\n    def run(self, value, required): return value\n\nclass Service(Base):\n    pass\n"
        ),
        consumer_source=(
            "from vllm.api import Service\n\ndef use():\n    service = Service()\n    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call"
        and item["contract_kind"] == "call_arguments"
        and item["details"]["target"].endswith("Service.run")
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["owner"] == "Base"
    assert calls[0]["upstream"]["new"]["owner"] == "Base"


def test_inherited_constructor_is_resolved_in_each_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n    def __init__(self, value): self.value = value\n\nclass Config(Base):\n    pass\n"
        ),
        new_source=(
            "class Base:\n    def __init__(self, value, required): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["symbol_kind"] == "constructor"


def test_inherited_custom_new_keeps_constructor_analysis_unresolved(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n"
            "    def __new__(cls, *args): return super().__new__(cls)\n\n"
            "class Config(Base):\n"
            "    def __init__(self, value): self.value = value\n"
        ),
        new_source=(
            "class Base:\n"
            "    def __new__(cls, *args): return super().__new__(cls)\n\n"
            "class Config(Base):\n"
            "    def __init__(self, value, required): self.value = value\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["compatibility"]["old"]["exists"] is True
    assert calls[0]["compatibility"]["old"]["compatible"] is None


def test_inherited_metaclass_keeps_constructor_analysis_unresolved(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Meta(type):\n"
            "    def __call__(cls): return super().__call__()\n\n"
            "class Base(metaclass=Meta):\n"
            "    def __init__(self, value): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        new_source=(
            "class Meta(type):\n"
            "    def __call__(cls): return super().__call__()\n\n"
            "class Base(metaclass=Meta):\n"
            "    def __init__(self, value, required): self.value = value\n\n"
            "class Config(Base):\n    pass\n"
        ),
        consumer_source="from vllm.api import Config\n\ndef use():\n    return Config(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_shadowed_classmethod_decorator_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "def classmethod(fn): return fn\n\n"
            "class Service:\n"
            "    @classmethod\n"
            "    def build(cls, value): return cls()\n"
        ),
        new_source=(
            "def classmethod(fn): return fn\n\n"
            "class Service:\n"
            "    @classmethod\n"
            "    def build(cls, value, required): return cls()\n"
        ),
        consumer_source="from vllm.api import Service\n\ndef use():\n    return Service.build(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["upstream"]["old"]["descriptor"] == "unknown"


def test_overload_stubs_use_the_final_runtime_implementation(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "from typing import overload\n\n"
            "@overload\ndef helper(value: int) -> int: ...\n\n"
            "def helper(value): return value\n"
        ),
        new_source=(
            "from typing import overload\n\n"
            "@overload\ndef helper(value: int, required: int) -> int: ...\n\n"
            "def helper(value, required): return value\n"
        ),
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["signature_status"] == "exact"


def test_overload_only_target_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="from typing import overload\n\n@overload\ndef helper(value: int) -> int: ...\n",
        new_source=("from typing import overload\n\n@overload\ndef helper(value: int, required: int) -> int: ...\n"),
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_conditional_method_binding_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Service:\n    def run(self, value): return value\n",
        new_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value, required): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["upstream"]["new"]["symbol_kind"] == "unknown"


def test_changed_ambiguous_method_binding_emits_unresolved_delta(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        new_source=(
            "class Service:\n"
            "    if enabled:\n"
            "        def run(self, value, required): return value\n"
            "    else:\n"
            "        def run(self, value): return value\n"
        ),
        consumer_source=("from vllm.api import Service\n\ndef use(service: Service):\n    return service.run(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_target_presence"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["compatibility"]["old"]["exists"] is None
    assert calls[0]["compatibility"]["new"]["exists"] is None


def test_unchanged_contextmanager_patch_has_no_parameter_delta(tmp_path: Path) -> None:
    source = (
        "from contextlib import contextmanager\n\n"
        "class Base:\n"
        "    @contextmanager\n"
        "    def run(self, value):\n"
        "        yield value\n"
    )
    roots = _call_repositories(
        tmp_path,
        old_source=source,
        new_source=f"UNRELATED = 1\n\n{source}",
        consumer_source=(
            "from contextlib import contextmanager\n"
            "from vllm.api import Base\n\n"
            "@contextmanager\n"
            "def replacement(self, value):\n"
            "    yield value\n\n"
            "Base.run = replacement\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "monkey_patch" and item["contract_kind"] == "call_arguments"
    ]


def test_classmethod_call_uses_bound_descriptor_signature(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    @classmethod\n    def build(cls, value): return cls()\n"),
        new_source=("class Service:\n    @classmethod\n    def build(cls, value, required): return cls()\n"),
        consumer_source=("from vllm.api import Service\n\ndef use():\n    return Service.build(1)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["descriptor"] == "classmethod"


def test_local_parameter_shadowing_does_not_create_a_direct_call(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source=("from vllm.api import helper\n\ndef use(helper):\n    return helper(1)\n"),
    )
    report = _run(*roots)
    assert report["summary"]["direct_call_dependencies"] == 0
    assert not [item for item in report["findings"] if item["relation"] == "direct_call"]


def test_constructed_instance_rebinding_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Service:\n    def run(self, value): return value\n"),
        new_source=("class Service:\n    def run(self, value, required): return value\n"),
        consumer_source=(
            "from vllm.api import Service\n\n"
            "def use(other):\n"
            "    service = Service()\n"
            "    service = other\n"
            "    return service.run(1)\n"
        ),
    )
    report = _run(*roots)
    assert not [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item.get("details", {}).get("target", "").endswith("Service.run")
    ]


def test_dynamic_call_arguments_fail_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source=("from vllm.api import helper\n\ndef use(payload):\n    return helper(**payload)\n"),
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_unknown_call_decorator_transform_fails_closed(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("def wrapper(fn): return fn\n\n@wrapper\ndef helper(value): return value\n"),
        new_source=("def wrapper(fn): return fn\n\n@wrapper\ndef helper(value, required): return value\n"),
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "analysis_unresolved"
    assert calls[0]["action"] == "review"


def test_direct_call_tuple_unpack_return_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return 1, 2\n",
        new_source="def helper(): return 1, 2, 3\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    left, right = helper()\n",
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"
    assert returns[0]["details"]["return_use"]["arity"] == 2


def test_direct_call_mapping_key_return_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return {'value': 1}\n",
        new_source="def helper(): return {'item': 1}\n",
        consumer_source=("from vllm.api import helper\n\ndef use():\n    return helper()['value']\n"),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_await_protocol_change_is_an_introduced_return_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="async def helper(): return 1\n",
        new_source="def helper(): return 1\n",
        consumer_source=("from vllm.api import helper\n\nasync def use():\n    return await helper()\n"),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "return_usage"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_unused_direct_call_return_change_is_not_reported(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(): return 1, 2\n",
        new_source="def helper(): return 1, 2, 3\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper()\n",
    )
    report = _run(*roots)
    assert not [item for item in report["findings"] if item["contract_kind"] == "return_usage"]


def test_override_return_contract_change_is_an_introduced_break(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value, value\n",
        new_source="class Base:\n    def run(self, value): return value, value, value\n",
        consumer_source=(
            "from vllm.api import Base\n\nclass Child(Base):\n    def run(self, value):\n        return value, value\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"


def test_transparent_super_return_follows_each_upstream_snapshot(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("class Base:\n    def run(self) -> dict[str, int] | None:\n        return None\n"),
        new_source=(
            "class Base:\n"
            "    def run(self) -> tuple[dict[str, int] | None, dict[str, int] | None]:\n"
            "        return None, None\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def run(self) -> dict[str, int] | None:\n"
            "        return super().run()\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "compatibility_warning"
    assert returns[0]["compatibility"]["old"]["compatible"] is True
    assert returns[0]["compatibility"]["new"]["compatible"] is True
    downstream_return = returns[0]["details"]["downstream_return"]
    assert downstream_return["variants"] == [
        {
            "kind": "forward",
            "arity": None,
            "keys": (),
            "type_ref": "run",
        }
    ]
    assert "transparent_super_forward_precedes_annotation" in downstream_return["provenance"]


def test_classmethod_variadic_tuple_override_becomes_introduced_return_break(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "class Base:\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[list[int], ...]:\n"
            "        return tuple([] for _ in range(2))\n"
        ),
        new_source=(
            "class Base:\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[tuple[list[int], ...], int]:\n"
            "        blocks = tuple([] for _ in range(2))\n"
            "        return blocks, 0\n"
        ),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    @classmethod\n"
            "    def find(cls) -> tuple[list[int], ...]:\n"
            "        return tuple([] for _ in range(2))\n"
        ),
    )
    report = _run(*roots)
    returns = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "replacement_return"
    ]
    assert len(returns) == 1
    assert returns[0]["classification"] == "introduced_break"
    assert returns[0]["compatibility"]["old"]["compatible"] is True
    assert returns[0]["compatibility"]["new"]["compatible"] is False
    downstream_return = returns[0]["details"]["downstream_return"]
    assert downstream_return["status"] == "exact"
    assert downstream_return["variants"] == [
        {
            "kind": "tuple_variadic",
            "arity": None,
            "keys": (),
            "type_ref": None,
        }
    ]
    assert "unknown_return_transform" not in downstream_return["provenance"]


def test_super_call_is_a_direct_upstream_dependency(tmp_path: Path) -> None:
    vllm_root = tmp_path / "vllm"
    ascend_root = tmp_path / "vllm-ascend"
    vllm_root.mkdir()
    ascend_root.mkdir()
    _git(vllm_root, "init")
    _git(ascend_root, "init")
    _write(vllm_root, "vllm/__init__.py", "")
    _write(vllm_root, "vllm/base.py", "class Base:\n    def run(self, value): return value\n")
    old_sha = _commit(vllm_root, "old")
    _write(vllm_root, "vllm/base.py", "class Base:\n    def run(self, value, required): return value\n")
    new_sha = _commit(vllm_root, "new")
    _write(ascend_root, "vllm_ascend/__init__.py", "")
    _write(
        ascend_root,
        "vllm_ascend/child.py",
        (
            "from vllm.base import Base\n\n"
            "class Child(Base):\n"
            "    def run(self, value):\n"
            "        return super().run(value)\n"
        ),
    )
    ascend_sha = _commit(ascend_root, "baseline")
    report = _run(vllm_root, ascend_root, old_sha, new_sha, ascend_sha)
    direct = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(direct) == 1
    assert direct[0]["classification"] == "introduced_break"


def test_patch_parameter_and_return_breaks_have_distinct_findings(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value, value\n",
        new_source=("class Base:\n    def run(self, value, required): return value, value, value\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "def replacement(self, value):\n"
            "    return value, value\n\n"
            "Base.run = replacement\n"
        ),
    )
    report = _run(*roots)
    patches = [
        item
        for item in report["findings"]
        if item["relation"] == "monkey_patch" and item["classification"] == "introduced_break"
    ]
    assert {item["contract_kind"] for item in patches} == {
        "call_arguments",
        "replacement_return",
    }
    assert len({item["id"] for item in patches}) == 2


def test_current_pair_validation_includes_direct_call_and_replacement_return(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value, value\n",
        new_source=("class Base:\n    def run(self, value, required): return value, value, value\n"),
        consumer_source=(
            "from vllm.api import Base\n\n"
            "def replacement(self, value):\n"
            "    return value, value\n\n"
            "Base.run = replacement\n\n"
            "def use(base: Base):\n"
            "    return base.run(1)\n"
        ),
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    dependencies, attributes, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert dependencies
    assert not attributes
    assert {(item["relation"], item["contract_kind"], item["status"]) for item in findings} >= {
        ("direct_call", "call_arguments", "risk"),
        ("monkey_patch", "replacement_return", "risk"),
    }


def test_current_pair_validation_reports_missing_call_target(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="OTHER = 1\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    dependencies, attributes, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert len(dependencies) == 1
    assert not attributes
    direct = [item for item in findings if item["relation"] == "direct_call"]
    assert len(direct) == 1
    assert direct[0]["status"] == "risk"
    assert "does not exist" in direct[0]["reason"]


def test_current_pair_validation_ignores_unverified_historical_self_candidate(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def removed(self, value): return value\n",
        new_source="class Base:\n    pass\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    def use(self, value):\n"
            "        return self.removed(value)\n"
        ),
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    dependencies, attributes, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert not attributes
    assert not [item for item in dependencies if item.callee == "self.removed"]
    assert not [
        item for item in findings if item["relation"] == "direct_call" and item["downstream"]["name"] == "self.removed"
    ]


def test_validate_cli_reports_current_contract_counts(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    vllm_root, ascend_root, _, _, ascend_sha = roots
    output = tmp_path / "validation.json"
    assert (
        cli_main(
            [
                "validate",
                "--no-cache",
                "--vllm-root",
                str(vllm_root),
                "--ascend-root",
                str(ascend_root),
                "--expect-ascend-sha",
                ascend_sha,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["direct_call_dependencies"] == 1
    assert payload["summary"]["direct_attribute_dependencies"] == 0
    assert payload["summary"]["contract_risks"] == 1
    assert payload["contract_findings"][0]["contract_kind"] == "call_arguments"


def test_validate_reports_a_missing_direct_attribute(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    VALUE = 1\n",
        new_source="class Base:\n    pass\n",
        consumer_source="from vllm.api import Base\n\ndef use():\n    return Base.VALUE\n",
    )
    vllm_root, ascend_root, _, new_sha, ascend_sha = roots
    engine = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha},
    )
    relations, _ = engine.generate()
    direct_calls, attributes, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert not direct_calls
    assert len(attributes) == 1
    assert [(item["relation"], item["contract_kind"], item["status"]) for item in findings] == [
        ("direct_attribute", "attribute_presence", "risk")
    ]


def test_vllm_interface_scenario_runs_only_upstream_pr_capabilities(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    VALUE = 1\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "def replacement(self, value):\n"
            "    return value\n\n"
            "Base.run = replacement\n\n"
            "def read_value():\n"
            "    return Base.VALUE\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    assert report["metadata"]["scenario"] == "vllm-interface"
    capabilities = report["metadata"]["analysis_plan"]["capabilities"]
    assert capabilities["inheritance_mro"]["state"] == "prerequisite"
    assert capabilities["monkey_patch"]["state"] == "skipped"
    assert capabilities["direct_attribute"]["state"] == "skipped"
    assert capabilities["inherited_state"]["state"] == "skipped"
    assert report["summary"]["direct_attribute_dependencies"] == 0
    assert report["summary"]["inherited_state_dependencies"] == 0
    assert not [item for item in report["findings"] if item["relation"] == "direct_attribute"]
    assert capabilities["direct_import"]["state"] == "analyzed"
    assert report["metadata"]["timings_seconds"]["relation_generation.monkey_patch"] is None
    assert report["metadata"]["timings_seconds"]["direct_import_analysis"] is not None
    assert not any(item["relation"] == "monkey_patch" for item in report["findings"])


def test_vllm_interface_scenario_keeps_override_breaks(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value): return value",
        new_method="def run(self, value, required): return value",
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert introduced
    assert {item["relation"] for item in introduced} == {"override"}


def test_vllm_interface_expands_transitive_override_impacts_under_one_root_cause(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "class Parent(Base):\n"
            "    def run(self, value): return value\n\n"
            "class GrandChild(Parent):\n"
            "    def run(self, value): return value\n"
        ),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    introduced = [
        item
        for item in report["findings"]
        if item["classification"] == "introduced_break" and item["relation"] == "override"
    ]
    assert {item["downstream"]["owner"] for item in introduced} == {
        "Parent",
        "GrandChild",
    }
    by_owner = {item["downstream"]["owner"]: item for item in introduced}
    assert by_owner["Parent"]["details"]["impact_kind"] == "direct_override"
    assert by_owner["Parent"]["details"]["override_depth"] == 1
    assert by_owner["GrandChild"]["details"]["impact_kind"] == ("transitive_subclass_override")
    assert by_owner["GrandChild"]["details"]["override_depth"] == 2
    assert by_owner["GrandChild"]["details"]["override_paths"] == [
        [
            "vllm_ascend.consumer.GrandChild.run",
            "vllm_ascend.consumer.Parent.run",
            "vllm.api.Base.run",
        ]
    ]

    outputs = write_reports(report, tmp_path / "upstream-report")
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["schema_version"] == 12
    assert payload["summary"]["introduced_breaks"] == 2
    assert payload["summary"]["root_causes"] == 1
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert (
        "Override path: `vllm_ascend.consumer.GrandChild.run -> vllm_ascend.consumer.Parent.run -> vllm.api.Base.run`"
    ) in markdown


def test_vllm_interface_reports_import_and_call_breaks_as_one_root_cause(
    tmp_path: Path,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="OTHER = 1\n",
        consumer_source=("from vllm.api import helper\n\ndef use():\n    return helper(1)\n"),
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    assert {item["relation"] for item in introduced} == {
        "direct_call",
        "direct_import",
    }
    output_dir = tmp_path / "upstream-report"
    outputs = write_reports(report, output_dir)
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 2
    assert payload["summary"]["root_causes"] == 1
    assert {item["relation"] for item in payload["findings"]} == {
        "direct_call",
        "direct_import",
    }
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "downstream imports, overrides" in markdown
    assert "direct imports" not in markdown
    assert markdown.count("- Upstream: `vllm/api.py:helper`") == 2


def test_vllm_interface_reports_masked_preexisting_delta_as_review(tmp_path: Path) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    report = analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=roots[3],
        expect_ascend_sha=roots[4],
        scenario="vllm-interface",
    )
    assert report["summary"]["preexisting"] >= 1
    outputs = write_reports(report, tmp_path / "upstream-reports")
    assert Path(outputs["markdown"]).name == "vllm-interface-pr-summary.md"
    payload = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
    assert payload["summary"]["introduced_breaks"] == 0
    assert payload["summary"]["review_findings"] == 1
    assert payload["findings"] == []
    assert len(payload["review_findings"]) == 1
    assert payload["review_findings"][0]["details"]["new_delta_on_preexisting_break"] is True
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "historical incompatibilities are intentionally excluded" in markdown
    assert "new exact delta masked by a preexisting incompatibility" in markdown


def test_vllm_interface_cli_log_keeps_only_exact_masked_delta_review(
    tmp_path: Path,
    capsys,
) -> None:
    roots = _repositories(
        tmp_path,
        old_method="def run(self, value, already_required): return value",
        new_method="def run(self, value, already_required, new_required): return value",
    )
    output_dir = tmp_path / "ci-output"
    assert (
        cli_main(
            [
                "analyze-range",
                "--no-cache",
                "--vllm-root",
                str(roots[0]),
                "--ascend-root",
                str(roots[1]),
                "--old",
                roots[2],
                "--new",
                roots[3],
                "--expect-ascend-sha",
                roots[4],
                "--scenario",
                "vllm-interface",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    console = capsys.readouterr().out
    assert '"introduced_breaks": 0' in console
    assert '"review_findings": 1' in console
    assert '"preexisting":' not in console
    assert "new_delta_masked_by_preexisting_incompatibility" in console
    assert "analysis_unresolved" not in console


def _cache_events(report: dict[str, object]) -> list[dict[str, object]]:
    metadata = report["metadata"]
    assert isinstance(metadata, dict)
    cache = metadata["persistent_cache"]
    assert isinstance(cache, dict)
    events = cache["events"]
    assert isinstance(events, list)
    return events


def _component_statuses(report: dict[str, object], component: str) -> list[str]:
    return [str(event["status"]) for event in _cache_events(report) if event["component"] == component]


def _run_with_cache(
    roots: tuple[Path, Path, str, str, str],
    cache_dir: Path,
    *,
    new_sha: str | None = None,
    ascend_sha: str | None = None,
    scenario: str = "main2main",
    cache_enabled: bool = True,
) -> dict[str, object]:
    return analyze_range(
        vllm_root=roots[0],
        ascend_root=roots[1],
        old=roots[2],
        new=new_sha or roots[3],
        expect_ascend_sha=ascend_sha or roots[4],
        scenario=scenario,
        cache_dir=cache_dir,
        cache_enabled=cache_enabled,
    )


def test_persistent_cache_first_run_misses_and_second_run_hits(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"

    cold = _run_with_cache(roots, cache_dir)
    hot = _run_with_cache(roots, cache_dir)

    assert _component_statuses(cold, "downstream_relations") == ["miss"]
    assert _component_statuses(hot, "downstream_relations") == ["hit"]
    assert _component_statuses(hot, "downstream_direct_imports") == ["hit"]
    assert _component_statuses(hot, "downstream_direct_calls") == ["hit"]
    assert _component_statuses(hot, "downstream_direct_attributes") == ["hit"]
    assert _component_statuses(hot, "upstream_snapshot") == ["hit", "hit"]
    assert cold["metadata"]["repository_index_cache"]["downstream"]["status"] == "miss"
    assert hot["metadata"]["repository_index_cache"]["downstream"]["status"] == "hit"
    assert hot["metadata"]["repository_index_cache"]["upstream_file_fragments"]["status"] == "hit"
    assert hot["findings"] == cold["findings"]


def test_upstream_sha_change_invalidates_only_commit_dependent_entries(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    _run_with_cache(roots, cache_dir)
    _write(roots[0], "vllm/api.py", "def helper(value, required, optional=None): return value\n")
    third_sha = _commit(roots[0], "third")

    report = _run_with_cache(roots, cache_dir, new_sha=third_sha)

    assert _component_statuses(report, "downstream_relations") == ["miss"]
    assert _component_statuses(report, "downstream_direct_calls") == ["miss"]
    assert _component_statuses(report, "downstream_direct_attributes") == ["miss"]
    snapshot_status = {
        event["commit_sha"]: event["status"]
        for event in _cache_events(report)
        if event["component"] == "upstream_snapshot"
    }
    assert snapshot_status[roots[2]] == "hit"
    assert snapshot_status[third_sha] == "miss"
    downstream_index = report["metadata"]["repository_index_cache"]["downstream"]
    assert downstream_index["status"] == "hit"


def test_downstream_sha_change_invalidates_downstream_entries(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    _run_with_cache(roots, cache_dir)
    _write(
        roots[1],
        "vllm_ascend/consumer.py",
        "from vllm.api import helper\n\ndef use():\n    return helper(1)\n\ndef other():\n    return 1\n",
    )
    changed_ascend_sha = _commit(roots[1], "downstream change")

    report = _run_with_cache(roots, cache_dir, ascend_sha=changed_ascend_sha)

    assert _component_statuses(report, "downstream_relations") == ["miss"]
    assert _component_statuses(report, "downstream_direct_imports") == ["miss"]
    assert _component_statuses(report, "downstream_direct_calls") == ["miss"]
    assert _component_statuses(report, "downstream_direct_attributes") == ["miss"]
    assert _component_statuses(report, "upstream_snapshot") == ["hit", "hit"]
    downstream_index = report["metadata"]["repository_index_cache"]["downstream"]
    assert downstream_index["status"] == "miss"


def test_cache_schema_and_analysis_config_changes_invalidate_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    _run_with_cache(roots, cache_dir)

    configured = _run_with_cache(roots, cache_dir, scenario="vllm-interface")
    assert _component_statuses(configured, "downstream_relations") == ["miss"]

    monkeypatch.setattr(
        cache_module,
        "CACHE_SCHEMA_VERSION",
        cache_module.CACHE_SCHEMA_VERSION + 1,
    )
    schema_changed = _run_with_cache(roots, cache_dir)
    assert _component_statuses(schema_changed, "downstream_relations") == ["miss"]
    assert _component_statuses(schema_changed, "upstream_snapshot") == ["miss", "miss"]


def test_dirty_worktree_bypasses_persistent_cache(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    _run_with_cache(roots, cache_dir)
    _write(
        roots[1],
        "vllm_ascend/consumer.py",
        "from vllm.api import helper\n\ndef use():\n    return helper(2)\n",
    )

    report = _run_with_cache(roots, cache_dir)

    assert _component_statuses(report, "downstream_relations") == ["bypassed"]
    assert _component_statuses(report, "downstream_direct_imports") == ["bypassed"]
    assert _component_statuses(report, "downstream_direct_calls") == ["bypassed"]
    assert _component_statuses(report, "downstream_direct_attributes") == ["bypassed"]
    downstream_index = report["metadata"]["repository_index_cache"]["downstream"]
    assert downstream_index["status"] == "bypassed"


def test_corrupt_cache_is_deleted_and_rebuilt(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    cold = _run_with_cache(roots, cache_dir)
    relation_event = next(event for event in _cache_events(cold) if event["component"] == "downstream_relations")
    Path(str(relation_event["path"])).write_bytes(b"not a pickle")

    rebuilt = _run_with_cache(roots, cache_dir)
    assert _component_statuses(rebuilt, "downstream_relations") == ["corrupt_rebuilt"]
    hot = _run_with_cache(roots, cache_dir)
    assert _component_statuses(hot, "downstream_relations") == ["hit"]


def test_no_cache_performs_no_persistent_reads_or_writes(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "disabled-cache"

    report = _run_with_cache(roots, cache_dir, cache_enabled=False)

    assert not cache_dir.exists()
    assert {event["status"] for event in _cache_events(report)} == {"disabled"}
    repository_cache = report["metadata"]["repository_index_cache"]
    assert repository_cache["downstream"]["status"] == "disabled"
    assert repository_cache["upstream_file_fragments"]["status"] == "disabled"


def test_cache_clear_removes_only_analyzer_namespace(tmp_path: Path) -> None:
    cache_parent = tmp_path / "cache-parent"
    analyzer_entry = cache_parent / cache_module.CACHE_NAMESPACE / "entry"
    sibling = cache_parent / "keep.txt"
    analyzer_entry.parent.mkdir(parents=True)
    analyzer_entry.write_text("owned", encoding="utf-8")
    sibling.write_text("keep", encoding="utf-8")

    assert cli_main(["cache", "clear", "--cache-dir", str(cache_parent)]) == 0

    assert not analyzer_entry.parent.exists()
    assert sibling.read_text(encoding="utf-8") == "keep"


def test_concurrent_and_interrupted_cache_writes_remain_recoverable(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="def helper(value): return value\n",
        new_source="def helper(value, required): return value\n",
        consumer_source="from vllm.api import helper\n\ndef use():\n    return helper(1)\n",
    )
    cache_dir = tmp_path / "cache"
    orphan = (
        cache_dir
        / cache_module.CACHE_NAMESPACE
        / f"schema-{cache_module.CACHE_SCHEMA_VERSION}"
        / "downstream_relations"
        / ".interrupted.tmp"
    )
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"partial")

    with ThreadPoolExecutor(max_workers=2) as executor:
        reports = list(executor.map(lambda _index: _run_with_cache(roots, cache_dir), range(2)))

    assert reports[0]["findings"] == reports[1]["findings"]
    hot = _run_with_cache(roots, cache_dir)
    assert _component_statuses(hot, "downstream_relations") == ["hit"]
    assert orphan.is_file()


def test_inference_mode_override_break_is_commit_independent(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "import torch\n"
            "from vllm.api import Base\n\n"
            "class Child(Base):\n"
            "    @torch.inference_mode()\n"
            "    def run(self, value): return value\n"
        ),
    )
    report = _run(*roots)
    overrides = [
        item
        for item in report["findings"]
        if item["relation"] == "override" and item["contract_kind"] == "call_arguments"
    ]
    assert len(overrides) == 1
    assert overrides[0]["classification"] == "introduced_break"
    assert overrides[0]["downstream"]["signature_status"] == "exact"


def test_instrumented_direct_call_break_is_commit_independent(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source=("from vllm.tracing import instrument\n\n@instrument\ndef helper(value): return value\n"),
        new_source=("from vllm.tracing import instrument\n\n@instrument\ndef helper(value, required): return value\n"),
        consumer_source="from vllm.api import helper\n\ndef use():\n    helper(1)\n",
    )
    report = _run(*roots)
    calls = [
        item
        for item in report["findings"]
        if item["relation"] == "direct_call" and item["contract_kind"] == "call_arguments"
    ]
    assert len(calls) == 1
    assert calls[0]["classification"] == "introduced_break"
    assert calls[0]["upstream"]["old"]["signature_status"] == "exact"
    assert calls[0]["upstream"]["new"]["signature_status"] == "exact"


def test_triton_signature_contract_is_commit_independent(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="from vllm.triton_utils import triton\n\n@triton.jit\ndef kernel(value): pass\n",
        new_source=("from vllm.triton_utils import triton\n\n@triton.jit\ndef kernel(value, BLOCK_SIZE): pass\n"),
        consumer_source="VALUE = 1\n",
    )
    engine = InterfaceBoundaryGenerator(
        roots[0],
        roots[1],
        source_versions={"vllm": roots[3], "vllm_ascend": roots[4]},
    )
    callable_info = engine.upstream.find_callable("vllm.api.kernel")
    assert callable_info is not None
    contract = engine._signature_contract(callable_info)
    assert contract.status == "exact"
    assert contract.protocol == "triton_kernel_launch"
    assert contract.provenance[-1] == "vllm.triton_utils.triton.jit:kernel_launch"
