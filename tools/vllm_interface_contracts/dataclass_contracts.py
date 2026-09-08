# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-only dataclass field ordering and generated argument protocols.

No repository classes or decorators are executed. Unknown decorators, dynamic
field options, conditional declarations and multiple inheritance stay unknown.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassSource:
    node: ast.ClassDef
    resolve: Callable[[str], str | None]
    file: str
    qualified_name: str


@dataclass(frozen=True)
class DataclassField:
    name: str
    required: bool
    keyword_only: bool
    included: bool
    owner: str
    file: str
    line: int


@dataclass(frozen=True)
class DataclassLayout:
    fields: tuple[DataclassField, ...]
    generates_initializer: bool

    def initializer(self) -> ast.FunctionDef | None:
        """Represent only the generated argument protocol, without executing code."""
        if not self.generates_initializer or self.ordering_error() is not None:
            return None
        fields = [field for field in self.fields if field.included]
        positional = [field for field in fields if not field.keyword_only]
        keywords = [field for field in fields if field.keyword_only]
        node = ast.parse("def __init__(self): pass").body[0]
        assert isinstance(node, ast.FunctionDef)
        receiver = "__dataclass_self__" if any(field.name == "self" for field in fields) else "self"
        node.args = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=receiver), *(ast.arg(arg=field.name) for field in positional)],
            vararg=None,
            kwonlyargs=[ast.arg(arg=field.name) for field in keywords],
            kw_defaults=[None if field.required else ast.Constant(None) for field in keywords],
            kwarg=None,
            defaults=[ast.Constant(None) for field in positional if not field.required],
        )
        return ast.fix_missing_locations(node)

    def ordering_error(self) -> tuple[DataclassField, DataclassField] | None:
        if not self.generates_initializer:
            return None
        default: DataclassField | None = None
        for field in self.fields:
            if not field.included or field.keyword_only:
                continue
            if field.required and default is not None:
                return default, field
            if not field.required:
                default = field
        return None


def dataclass_layout(
    reference: str,
    lookup: Callable[[str], ClassSource | None],
    seen: frozenset[str] = frozenset(),
) -> DataclassLayout | None:
    """Resolve a unique source hierarchy and its ordered init fields."""
    if reference in seen or (source := lookup(reference)) is None:
        return None
    node = source.node
    if node.keywords or len(node.bases) > 1:
        return None
    init, default_kw_only = True, False
    decorated = False
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        target = ast.unparse(call.func if call else decorator)
        if decorated or source.resolve(target) != "dataclasses.dataclass":
            return None
        decorated = True
        if call is not None:
            if call.args:
                return None
            for keyword in call.keywords:
                if keyword.arg not in {
                    "init",
                    "repr",
                    "eq",
                    "order",
                    "unsafe_hash",
                    "frozen",
                    "match_args",
                    "kw_only",
                    "slots",
                    "weakref_slot",
                }:
                    return None
                if keyword.arg is None or not isinstance(keyword.value, ast.Constant):
                    return None
                if not isinstance(keyword.value.value, bool):
                    return None
                if keyword.arg == "init":
                    init = keyword.value.value
                elif keyword.arg == "kw_only":
                    default_kw_only = keyword.value.value

    fields: dict[str, DataclassField] = {}
    for base in node.bases:
        base_ref = source.resolve(ast.unparse(base))
        if base_ref in {"object", "builtins.object"}:
            continue
        if base_ref is None:
            return None
        parent = dataclass_layout(base_ref, lookup, seen | {reference})
        if parent is None:
            return None
        fields.update((field.name, field) for field in parent.fields)
    if not decorated:
        return DataclassLayout(tuple(fields.values()), False)

    keyword_only = default_kw_only
    for statement in node.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if statement.name == "__init__":
                init = False
            continue
        if isinstance(statement, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            # Choosing both paths would silently create an impossible layout.
            if any(isinstance(child, (ast.AnnAssign, ast.Assign)) for child in ast.walk(statement)):
                return None
            continue
        if isinstance(statement, ast.Assign):
            if any(
                isinstance(target, ast.Name) and (target.id in fields or target.id == "__init__")
                for target in statement.targets
            ):
                return None
            continue
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.target, ast.Name):
            continue
        name = statement.target.id
        annotation = statement.annotation
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            try:
                annotation = ast.parse(annotation.value, mode="eval").body
            except SyntaxError:
                return None
        root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
        annotation_ref = source.resolve(ast.unparse(root))
        if annotation_ref == "dataclasses.KW_ONLY":
            keyword_only = True
            continue
        included = annotation_ref != "typing.ClassVar"
        field_kw_only = keyword_only
        value = statement.value
        required = value is None or source.resolve(ast.unparse(value)) == "dataclasses.MISSING"
        if isinstance(value, ast.Call) and source.resolve(ast.unparse(value.func)) == "dataclasses.field":
            if value.args:
                return None
            required = True
            defaults = 0
            for keyword in value.keywords:
                if keyword.arg in {"default", "default_factory"}:
                    if source.resolve(ast.unparse(keyword.value)) != "dataclasses.MISSING":
                        required = False
                        defaults += 1
                        if keyword.arg == "default" and isinstance(keyword.value, (ast.List, ast.Dict, ast.Set)):
                            return None
                elif keyword.arg in {"init", "kw_only"}:
                    if not isinstance(keyword.value, ast.Constant) or not isinstance(keyword.value.value, bool):
                        return None
                    if keyword.arg == "init":
                        included = included and keyword.value.value
                    else:
                        field_kw_only = keyword.value.value
                elif keyword.arg not in {"repr", "hash", "compare", "metadata"}:
                    return None
            if defaults > 1:
                return None
        elif isinstance(value, (ast.Call, ast.List, ast.Dict, ast.Set)):
            # A descriptor factory can change requiredness through __get__;
            # mutable defaults can prevent the dataclass from being defined.
            return None
        elif value is None and name in fields:
            # An annotation without a value does not erase an inherited default.
            required = fields[name].required
        fields[name] = DataclassField(
            name,
            required,
            field_kw_only,
            included,
            reference,
            source.file,
            statement.lineno,
        )
    return DataclassLayout(tuple(fields.values()), init)
