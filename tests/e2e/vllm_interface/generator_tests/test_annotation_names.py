"""Annotation namespaces must not manufacture executable import bindings."""

import ast
from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _run

from tools.vllm_interface_contracts.annotation_names import AnnotationNamespace
from tools.vllm_interface_contracts.range_analysis import discover_imports


@pytest.mark.parametrize(
    "imports",
    [
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n",
        "from typing import TYPE_CHECKING as TC\nif TC:\n    from vllm.api import Config\n",
        "import typing as t\nif t.TYPE_CHECKING:\n    from vllm.api import Config\n",
    ],
)
def test_annotation_only_config_drives_field_read_not_import(tmp_path: Path, imports: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: int\n" + config,
        new_source=prefix + "    layers: int\n" + config,
        consumer_source=imports + 'def consume(config: "Config"):\n    return [t.shared_by for t in config.tensors]\n',
    )
    findings = _run(*roots)["findings"]
    assert not discover_imports(roots[1])
    fields = [f for f in findings if f["details"].get("member") == "shared_by"]
    assert len(fields) == 1
    assert fields[0]["classification"] == "introduced_break"
    assert not any(f["relation"] == "direct_import" for f in findings)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n", "vllm.api.Config"),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n"
            "else:\n    from other import Config\n",
            None,
        ),
        (
            "from typing import TYPE_CHECKING\nTYPE_CHECKING = unknown()\n"
            "if TYPE_CHECKING:\n    from vllm.api import Config\n",
            None,
        ),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n"
            "Config = unknown()\n",
            None,
        ),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n"
            "else:\n    Config = unknown()\n",
            None,
        ),
        ("if flag:\n    from vllm.api import Config\n", None),
        (
            "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from vllm.api import Config\n"
            "else:\n    from vllm.api import Config\n",
            "vllm.api.Config",
        ),
    ],
)
def test_annotation_namespace_keeps_ambiguous_bindings_unknown(source: str, expected: str | None):
    namespace = AnnotationNamespace(ast.parse(source), "vllm_ascend.consumer", False)
    assert namespace.resolve("Config") == expected


def test_annotation_only_builtin_alias_is_not_runtime_callable():
    namespace = AnnotationNamespace(
        ast.parse("from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from builtins import enumerate as walk\n"),
        "vllm_ascend.consumer",
        False,
    )
    assert namespace.resolve("walk") == "builtins.enumerate"
    assert namespace.runtime("walk") is None


def test_upstream_annotation_only_alias_resolves_in_each_snapshot(tmp_path: Path):
    prefix = "from dataclasses import dataclass\nfrom typing import TYPE_CHECKING\n@dataclass\nclass Tensor:\n"
    config = (
        "if TYPE_CHECKING:\n    from vllm.api import Tensor as Element\n"
        "@dataclass\nclass Config:\n    tensors: list[Element]\n"
    )
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: int\n" + config,
        new_source=prefix + "    layers: int\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    return [tensor.shared_by for tensor in config.tensors]\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"
    assert findings[0]["classification"] == "introduced_break"


@pytest.mark.parametrize("typing_only", [True, False])
def test_removed_annotation_import_is_not_runtime_import_break(tmp_path: Path, typing_only: bool):
    source = "from vllm.api import Removed\n"
    if typing_only:
        source = "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    " + source
    roots = _call_repositories(
        tmp_path,
        old_source="class Removed: pass\n",
        new_source="class Different: pass\n",
        consumer_source=source,
    )
    imports = [f for f in _run(*roots)["findings"] if f["relation"] == "direct_import"]
    assert len(imports) == (0 if typing_only else 1)
