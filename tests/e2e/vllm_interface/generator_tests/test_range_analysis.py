from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.vllm_interface_contracts.range_analysis import (
    analyze_range,
    discover_imports,
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
    assert payload["metadata"]["vllm_old_sha"] == roots[2]
    assert "introduced_break" in introduced_csv
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
