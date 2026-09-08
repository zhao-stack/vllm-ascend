"""Exact source contracts exposed by the final PR 14872 holdout replay."""

import ast
import inspect
from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _run, _write

from tools.vllm_interface_contracts.dataclass_contracts import ClassSource, dataclass_layout
from tools.vllm_interface_contracts.range_analysis import discover_imports


@pytest.mark.parametrize("import_source", ["import vllm", "import vllm.api"])
def test_patch_assignment_is_not_an_executable_import(tmp_path: Path, import_source: str):
    _write(
        tmp_path,
        "vllm_ascend/consumer.py",
        f"{import_source}\nvllm.api.removed = replacement\n"
        "read = vllm.api.actual_read\n"
        "vllm.api.target = vllm.api.replacement\n"
        "vllm.api.receiver.attribute = 1\n"
        "vllm.api.counter += 1\n",
    )
    references = {item.module for item in discover_imports(tmp_path)}
    assert "vllm.api.removed" not in references
    assert "vllm.api.target" not in references
    assert {"vllm.api.actual_read", "vllm.api.replacement"} <= references
    assert {"vllm.api.receiver", "vllm.api.counter"} <= references
    assert "vllm.api.receiver.attribute" not in references


def test_patch_target_removal_keeps_patch_finding_without_import_duplicate(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="def allocate(): pass\n",
        new_source="def allocate_new(): return 1\n",
        consumer_source="import vllm.api\ndef replacement(): pass\nvllm.api.allocate = replacement\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["action"] == "modify"]
    assert any(f["relation"] == "monkey_patch" for f in findings)
    assert not any(f["relation"] == "direct_import" for f in findings)


@pytest.mark.parametrize("consumer", ["Tensor(size=16, shared_by=['layer'])", "Tensor(16, ['layer'])"])
def test_dataclass_generated_constructor_break(tmp_path: Path, consumer: str):
    roots = _call_repositories(
        tmp_path,
        old_source=(
            "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n    size: int\n    shared_by: list[str]\n"
        ),
        new_source=(
            "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
            "    size: int\n    layers: list[str]\n    layer_stride: int\n"
        ),
        consumer_source=f"from vllm.api import Tensor\nvalue = {consumer}\n",
    )
    matches = [
        f for f in _run(*roots)["findings"] if f["relation"] == "direct_call" and f["contract_kind"] == "call_arguments"
    ]
    assert any(f["classification"] == "introduced_break" and f["action"] == "modify" for f in matches)


def test_dataclass_added_optional_field_keeps_existing_call_valid(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="from dataclasses import dataclass\n@dataclass\nclass Tensor:\n    size: int\n",
        new_source=(
            "from dataclasses import dataclass, field\n@dataclass\nclass Tensor:\n"
            "    size: int\n    layers: list[str] = field(default_factory=list, kw_only=True)\n"
        ),
        consumer_source="from vllm.api import Tensor\nvalue = Tensor(16)\n",
    )
    report = _run(*roots)
    assert not any(f["action"] == "modify" for f in report["findings"])
    assert any(
        f["classification"] == "compatibility_warning" and f["contract_kind"] == "call_arguments"
        for f in report["findings"]
    )


@pytest.mark.parametrize(
    "declaration",
    [
        "@dataclasses.dataclass\nclass Tensor:\n    size: int\n    value: int = 3\n",
        "@dataclasses.dataclass(kw_only=True)\nclass Tensor:\n    size: int\n",
        "@dataclasses.dataclass\nclass Tensor:\n    size: int\n    _: dataclasses.KW_ONLY\n    value: int\n",
        "@dataclasses.dataclass\nclass Tensor:\n    self: int\n    token: dataclasses.InitVar[int]\n"
        "    hidden: int = dataclasses.field(init=False, default=0)\n",
        "@dataclasses.dataclass\nclass Tensor:\n    value: int = dataclasses.MISSING\n"
        "    shared: typing.ClassVar[int] = dataclasses.field(default=1, init=True)\n",
        "@dataclasses.dataclass\nclass Base:\n    size: int\n    optional: int = 3\n"
        "@dataclasses.dataclass\nclass Tensor(Base):\n    required: int = dataclasses.field(kw_only=True)\n",
        "@dataclasses.dataclass(slots=True, frozen=True)\nclass Tensor:\n    size: int\n"
        "    values: list[int] = dataclasses.field(default_factory=list)\n",
    ],
)
def test_generated_protocol_matches_python_dataclass(declaration: str):
    source = "import dataclasses\nimport typing\n" + declaration
    tree = ast.parse(source)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    layout = dataclass_layout(
        "Tensor",
        lambda name: ClassSource(classes[name], lambda value: value, "fixture.py", name) if name in classes else None,
    )
    assert layout is not None
    initializer = layout.initializer()
    assert initializer is not None
    # Execute only these authored fixtures as an independent standard-library
    # oracle. Production analysis never imports or executes repository sources.
    namespace: dict[str, object] = {}
    exec(compile(tree, "fixture.py", "exec"), namespace)
    exec(compile(ast.Module(body=[initializer], type_ignores=[]), "generated.py", "exec"), namespace)

    def shape(parameters):
        return [(p.name, p.kind, p.default is inspect.Parameter.empty) for p in parameters]

    expected = shape(inspect.signature(namespace["Tensor"]).parameters.values())
    actual = shape(list(inspect.signature(namespace["__init__"]).parameters.values())[1:])
    assert actual == expected


@pytest.mark.parametrize(
    "custom",
    [
        "    def __new__(cls, token): return object.__new__(cls)\n",
        "    def __init_subclass__(cls): pass\n",
    ],
)
def test_dataclass_custom_allocation_stays_unresolved(tmp_path: Path, custom: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    size: int\n" + custom,
        new_source=prefix + "    size: int\n    added: int\n" + custom,
        consumer_source="from vllm.api import Tensor\nvalue = Tensor(16)\n",
    )
    findings = [
        f for f in _run(*roots)["findings"] if f["relation"] == "direct_call" and f["contract_kind"] == "call_arguments"
    ]
    assert findings
    assert all(f["classification"] == "analysis_unresolved" and f["action"] != "modify" for f in findings)
