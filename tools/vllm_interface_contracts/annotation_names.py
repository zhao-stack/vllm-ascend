# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Separate source annotation imports from executable module bindings."""

from __future__ import annotations

import ast

from .generator import _scope_reference_variants, _scope_state_before, _tag_guard_names
from .module_attributes import annotation_module_body, runtime_module_body


class AnnotationNamespace:
    """Resolve unique type names without importing or evaluating source code.

    TYPE_CHECKING-only imports can supply annotation evidence, not runtime
    callables. Conflicting or unknown runtime bindings invalidate a type hint.
    Both views reuse the ordinary scope interpreter, including rebinding,
    deletes and branch joins. The original AST and repository index stay intact.
    """

    def __init__(self, tree: ast.Module, module: str, is_package: bool):
        self.module = module
        self.is_package = is_package
        self.annotation_body = annotation_module_body(tree)
        self.runtime_body = runtime_module_body(tree)
        self.final_line = max((getattr(n, "end_lineno", 0) or 0 for n in tree.body), default=0) + 1
        self._cache: dict[tuple[str, int, bool], str | None] = {}

    def _reference(self, expression: str, line: int, *, annotation: bool) -> str | None:
        key = (expression, line, annotation)
        if key in self._cache:
            return self._cache[key]
        body = self.annotation_body if annotation else self.runtime_body
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            return None
        variants = _scope_reference_variants(
            node,
            statements=body,
            line=line,
            tag_guard_names=_tag_guard_names(body),
            module=self.module,
            is_package=self.is_package,
        )
        result = next(iter(variants)) if len(variants) == 1 else None
        self._cache[key] = result
        return result

    def runtime(self, expression: str, line: int | None = None) -> str | None:
        return self._reference(expression, self.final_line if line is None else line, annotation=False)

    def resolve(self, expression: str, line: int | None = None) -> str | None:
        point = self.final_line if line is None else line
        result = self._reference(expression, point, annotation=True)
        if result is None:
            return None
        root = expression.split(".", 1)[0]
        state = _scope_state_before(self.runtime_body, point, _tag_guard_names(self.runtime_body))
        if any(binding.kind != "unbound" for binding in state.get(root, ())):
            if self.runtime(expression, point) != result:
                return None
        return result
