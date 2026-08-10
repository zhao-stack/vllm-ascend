from __future__ import annotations

import json
import subprocess
from pathlib import Path

from tools.vllm_interface_contracts.cli import main as cli_main
from tools.vllm_interface_contracts.generator import InterfaceBoundaryGenerator
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
    assert payload["schema_version"] == 3
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
    dependencies, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert dependencies
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
    dependencies, findings = validate_current_contracts(
        engine,
        relations,
        GitSnapshot(vllm_root, new_sha),
    )
    assert len(dependencies) == 1
    direct = [item for item in findings if item["relation"] == "direct_call"]
    assert len(direct) == 1
    assert direct[0]["status"] == "risk"
    assert "does not exist" in direct[0]["reason"]


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
    assert payload["summary"]["contract_risks"] == 1
    assert payload["contract_findings"][0]["contract_kind"] == "call_arguments"


def test_vllm_interface_scenario_runs_only_upstream_pr_capabilities(tmp_path: Path) -> None:
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value): return value\n",
        new_source="class Base:\n    def run(self, value, required): return value\n",
        consumer_source=(
            "from vllm.api import Base\n\n"
            "def replacement(self, value):\n"
            "    return value\n\n"
            "Base.run = replacement\n"
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
    assert capabilities["direct_import"]["state"] == "skipped"
    assert report["metadata"]["timings_seconds"]["relation_generation.monkey_patch"] is None
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
    introduced = [
        item for item in report["findings"] if item["classification"] == "introduced_break"
    ]
    assert introduced
    assert {item["relation"] for item in introduced} == {"override"}


def test_vllm_interface_reports_only_actionable_introduced_breaks(tmp_path: Path) -> None:
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
    assert payload["findings"] == []
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "historical incompatibilities are intentionally excluded" in markdown
    assert "preexisting" not in markdown


def test_vllm_interface_cli_log_omits_historical_counts(tmp_path: Path, capsys) -> None:
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
    assert "preexisting" not in console
    assert "analysis_unresolved" not in console
