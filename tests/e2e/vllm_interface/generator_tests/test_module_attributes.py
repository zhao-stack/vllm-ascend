# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime, not typing-only, module attribute contracts."""

import ast

import pytest

from tools.vllm_interface_contracts.module_attributes import module_getattr_contract, runtime_module_body


def _source(entries: str = '"SCALE": lambda: 200', suffix: str = "", getter: str | None = None) -> str:
    return (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    SCALE: float\n"
        f"registry = {{{entries}}}\n"
        + (
            getter
            if getter is not None
            else "def __getattr__(name):\n"
            "    if name in registry:\n        return registry[name]()\n"
            "    raise AttributeError(name)\n"
        )
        + suffix
    )


def _contract(source: str):
    body = runtime_module_body(ast.parse(source))
    getter = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == "__getattr__")
    return module_getattr_contract(body, getter)


def test_literal_registry_presence_and_absence() -> None:
    contract = _contract(_source())
    assert contract.resolve("SCALE")[0] == "attribute"
    assert contract.resolve("DELETED")[0] == "missing"
    assert _contract(_source(entries="")).resolve("SCALE")[0] == "missing"


def test_direct_value_registry() -> None:
    source = _source(entries='"SCALE": 200').replace("return registry[name]()", "return registry[name]")
    assert _contract(source).resolve("SCALE")[0] == "attribute"


def test_read_only_registry_iteration_does_not_hide_absence() -> None:
    source = _source(suffix="for key in registry:\n    pass\nfor key, value in registry.items():\n    pass\n")
    assert _contract(source).resolve("DELETED")[0] == "missing"


@pytest.mark.parametrize(
    "suffix",
    [
        'registry["DELETED"] = lambda: 1\n',
        'del registry["SCALE"]\n',
        'registry.update({"DELETED": lambda: 1})\n',
        "alias = registry\n",
        "consume(registry)\n",
        "registry = {}\n",
        "registry |= {}\n",
        'def mutate():\n    registry["DELETED"] = lambda: 1\n',
        'globals()["registry"] = {}\n',
        "def registry(): pass\n",
        "def shadow(registry): pass\n",
        "from external import *\n",
    ],
)
def test_mutated_or_escaping_registry_is_unresolved(suffix: str) -> None:
    contract = _contract(_source(suffix=suffix))
    assert contract.resolve("SCALE")[0] == "unknown"
    assert contract.resolve("DELETED")[0] == "unknown"


@pytest.mark.parametrize("entries", ["**external", "key: lambda: 1", "1: lambda: 1"])
def test_unknown_registry_key_set_is_unresolved(entries: str) -> None:
    assert _contract(_source(entries=entries)).resolve("DELETED")[0] == "unknown"


def test_unsupported_getter_is_unresolved() -> None:
    getter = "def __getattr__(name):\n    return fallback(name)\n"
    assert _contract(_source(getter=getter)).resolve("DELETED")[0] == "unknown"


@pytest.mark.parametrize("value", ["lambda required: 1", "lambda *, required: 1", "factory()", "None"])
def test_unknown_or_incompatible_registry_callback_is_unresolved(value: str) -> None:
    assert _contract(_source(entries=f'"SCALE": {value}')).resolve("SCALE")[0] == "unknown"


@pytest.mark.parametrize(
    ("imports", "guard"),
    [
        ("from typing import TYPE_CHECKING", "TYPE_CHECKING"),
        ("from typing import TYPE_CHECKING as TC", "TC"),
        ("import typing", "typing.TYPE_CHECKING"),
        ("import typing as t", "t.TYPE_CHECKING"),
    ],
)
def test_typing_only_assignments_are_not_runtime_bindings(imports: str, guard: str) -> None:
    tree = ast.parse(f"{imports}\nif {guard}:\n    SCALE = 1\nelse:\n    RUNTIME = 2\nANNOTATED: int\n")
    body = runtime_module_body(tree)
    assert not any(isinstance(node, (ast.If, ast.AnnAssign)) for node in body)
    assert [node.targets[0].id for node in body if isinstance(node, ast.Assign)] == ["RUNTIME"]
    assert any(isinstance(node, ast.If) for node in tree.body)  # Do not mutate the snapshot AST.


@pytest.mark.parametrize("rebind", ["TC = condition", "def TC(): pass", "import other as TC"])
def test_shadowed_typing_flag_is_not_constant_folded(rebind: str) -> None:
    body = runtime_module_body(ast.parse(f"from typing import TYPE_CHECKING as TC\n{rebind}\nif TC:\n    SCALE = 1\n"))
    assert any(isinstance(node, ast.If) for node in body)


def test_class_annotations_are_preserved() -> None:
    body = runtime_module_body(ast.parse("class Config:\n    SCALE: int\n"))
    assert isinstance(body[0], ast.ClassDef)
    assert isinstance(body[0].body[0], ast.AnnAssign)


def test_wildcard_import_prevents_typing_flag_constant_folding() -> None:
    body = runtime_module_body(
        ast.parse("from typing import TYPE_CHECKING\nfrom external import *\nif TYPE_CHECKING:\n    SCALE = 1\n")
    )
    assert any(isinstance(node, ast.If) for node in body)
