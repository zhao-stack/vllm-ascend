# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-derived container element paths, shared by discovery and snapshots.

Paths retain their annotation root and field/iteration operations. Replaying a
path at each upstream SHA avoids borrowing the new element owner for the old
snapshot. No annotations, descriptors or repository code are executed.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .dataclass_contracts import ClassSource

Lookup = Callable[[str], ClassSource | None]
Resolve = Callable[[str], str | None]
TYPE_BUILTINS = frozenset(
    {
        "list",
        "set",
        "tuple",
        "dict",
        "int",
        "str",
        "bool",
        "float",
        "bytes",
        "object",
        "enumerate",
        "len",
        "print",
        "range",
    }
)
_IMMUTABLE_SCALARS = frozenset({"int", "str", "bool", "float", "bytes"})


@dataclass(frozen=True)
class TypeShape:
    reference: str
    arguments: tuple[TypeShape, ...] = ()

    def render(self) -> str:
        suffix = "[" + ",".join(arg.render() for arg in self.arguments) + "]" if self.arguments else ""
        return self.reference + suffix


def annotation_type(node: ast.AST | None, resolve: Resolve) -> TypeShape | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(node, (ast.Name, ast.Attribute)):
        reference = resolve(ast.unparse(node))
        return TypeShape(reference) if reference is not None else None
    if not isinstance(node, ast.Subscript):
        return None
    base = annotation_type(node.value, resolve)
    if base is None:
        return None
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    arguments = []
    for item in items:
        argument = (
            TypeShape("ellipsis")
            if isinstance(item, ast.Constant) and item.value is Ellipsis
            else (annotation_type(item, resolve))
        )
        if argument is None:
            return None
        arguments.append(argument)
    return TypeShape(base.reference, tuple(arguments))


def field_type(reference: str, member: str, lookup: Lookup, seen: frozenset[str] = frozenset()) -> TypeShape | None:
    if reference in seen or (source := lookup(reference)) is None:
        return None
    node = source.node
    if node.keywords or len(node.bases) > 1:
        return None
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if source.resolve(ast.unparse(target)) != "dataclasses.dataclass":
            return None
    candidates = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.target.id == member:
                candidates.append(statement.annotation)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if statement.name == member:
                return None  # A descriptor/method is not a stored annotated field.
        elif any(
            isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)) and child.id == member
            for child in ast.walk(statement)
        ):
            return None
    if candidates:
        return annotation_type(candidates[0], source.resolve) if len(candidates) == 1 else None
    for base in node.bases:
        base_reference = source.resolve(ast.unparse(base))
        if base_reference is not None:
            return field_type(base_reference, member, lookup, seen | {reference})
    return None


def type_step(value: TypeShape, step: str, lookup: Lookup) -> TypeShape | None:
    if step.startswith("field:"):
        return field_type(value.reference, step.removeprefix("field:"), lookup)
    name = value.reference
    args = value.arguments
    sequence = name in {
        "list",
        "set",
        "builtins.list",
        "builtins.set",
        "typing.List",
        "typing.Set",
        "typing.Sequence",
        "collections.abc.Sequence",
        "typing.Iterable",
        "collections.abc.Iterable",
    }
    mapping = name in {"dict", "builtins.dict", "typing.Dict", "typing.Mapping", "collections.abc.Mapping"}
    is_tuple = name in {"tuple", "builtins.tuple", "typing.Tuple"}
    if step == "collect":
        return TypeShape("list", (value,))
    if step == "enumerate":
        item = type_step(value, "item", lookup)
        return TypeShape("list", (TypeShape("tuple", (TypeShape("int"), item)),)) if item else None
    if mapping and len(args) == 2:
        if step in {"values", "keys", "items"}:
            item = args[1] if step == "values" else args[0] if step == "keys" else TypeShape("tuple", args)
            return TypeShape("list", (item,))
        if step == "item":
            return args[0]
        if step.startswith("index:"):
            return args[1]
    if sequence and len(args) == 1 and (step == "item" or step.startswith("index:")):
        return args[0]
    if is_tuple and args:
        if len(args) == 2 and args[1].reference == "ellipsis":
            return args[0] if step == "item" or step.startswith("index:") else None
        if step.startswith("index:"):
            try:
                return args[int(step.removeprefix("index:"))]
            except (ValueError, IndexError):
                return None
        if step == "item" and all(arg == args[0] for arg in args):
            return args[0]
    return None


def resolve_type_path(path: tuple[str, ...], lookup: Lookup) -> TypeShape | None:
    if not path:
        return None
    try:
        value = annotation_type(ast.parse(path[0], mode="eval").body, lambda name: name)
    except SyntaxError:
        return None
    for step in path[1:]:
        value = type_step(value, step, lookup) if value is not None else None
    return value


@dataclass(frozen=True)
class FlowValue:
    shape: TypeShape
    path: tuple[str, ...]

    def step(self, step: str, lookup: Lookup) -> FlowValue | None:
        shape = type_step(self.shape, step, lookup)
        return FlowValue(shape, (*self.path, step)) if shape is not None else None


class ContainerFlow:
    """Small ordered type-flow interpreter; unsupported writes kill bindings.

    It is deliberately separate from symbol/import flow: values are annotation
    paths, not nominal symbols. Branch joins retain only equal paths; loop and
    comprehension targets obey their distinct Python scopes.
    """

    def __init__(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        resolve: Resolve,
        lookup: Lookup,
        *,
        runtime_resolve: Resolve | None = None,
    ):
        self.lookup = lookup
        self.resolve = resolve
        self.runtime_resolve = runtime_resolve or resolve
        self.receivers: dict[int, FlowValue] = {}
        self.blocked_builtins = {
            n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        arguments = [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        self.blocked_builtins.update(arg.arg for arg in arguments)
        env = {}
        for argument in arguments:
            shape = annotation_type(argument.annotation, resolve)
            env[argument.arg] = FlowValue(shape, (shape.render(),)) if shape else None
        if any(value is not None and "vllm." in value.shape.render() for value in env.values()):
            self.statements(function.body, env)

    @staticmethod
    def join(left: dict[str, FlowValue | None], right: dict[str, FlowValue | None]) -> dict[str, FlowValue | None]:
        return {
            name: left.get(name) if left.get(name) == right.get(name) else None for name in left.keys() | right.keys()
        }

    def bind(self, target: ast.AST, value: FlowValue | None, env: dict[str, FlowValue | None]) -> None:
        if isinstance(target, ast.Name):
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self.bind(item, value.step(f"index:{index}", self.lookup) if value else None, env)
        else:
            self.invalidate(target, env)

    def invalidate(self, node: ast.AST, env: dict[str, FlowValue | None]) -> None:
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        value = env.get(root.id) if isinstance(root, ast.Name) else None
        if value is not None:
            # Aliases and derived element bindings share the same origin.
            for name, candidate in list(env.items()):
                if candidate is not None and candidate.path[0] == value.path[0]:
                    env[name] = None

    def expression(self, node: ast.AST | None, env: dict[str, FlowValue | None]) -> FlowValue | None:
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            value = self.expression(node.value, env)
            if value is not None:
                self.receivers[id(node)] = value
                return value.step(f"field:{node.attr}", self.lookup)
            return None
        if isinstance(node, ast.Subscript):
            value = self.expression(node.value, env)
            self.expression(node.slice, env)
            if isinstance(node.slice, ast.Slice):
                return value
            index = str(node.slice.value) if isinstance(node.slice, ast.Constant) else "dynamic"
            return value.step(f"index:{index}", self.lookup) if value else None
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            return self.comprehension(node, env)
        if isinstance(node, ast.Call):
            return self.call(node, env)
        if isinstance(node, ast.NamedExpr):
            value = self.expression(node.value, env)
            self.bind(node.target, value, env)
            return value
        if isinstance(node, ast.IfExp):
            self.expression(node.test, env)
            if isinstance(node.test, ast.Constant) and isinstance(node.test.value, bool):
                return self.expression(node.body if node.test.value else node.orelse, env)
            left_env, right_env = dict(env), dict(env)
            left = self.expression(node.body, left_env)
            right = self.expression(node.orelse, right_env)
            env.update(self.join(left_env, right_env))
            return left if left == right else None
        if isinstance(node, ast.BoolOp):
            for operand in node.values:
                self.expression(operand, env)
                if isinstance(operand, ast.Constant) and isinstance(operand.value, bool):
                    if (
                        isinstance(node.op, ast.And)
                        and not operand.value
                        or isinstance(node.op, ast.Or)
                        and operand.value
                    ):
                        break
            return None
        if node is not None and not isinstance(node, (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in ast.iter_child_nodes(node):
                self.expression(child, env)
        return None

    def call(self, node: ast.Call, env: dict[str, FlowValue | None]) -> FlowValue | None:
        self.expression(node.func, env)
        arguments = [self.expression(arg, env) for arg in node.args]
        keyword_arguments = [self.expression(keyword.value, env) for keyword in node.keywords]
        if isinstance(node.func, ast.Name) and node.func.id not in self.blocked_builtins:
            resolved = self.runtime_resolve(node.func.id)
            if resolved in {"enumerate", "builtins.enumerate"} and len(arguments) == 1 and not node.keywords:
                return arguments[0].step("enumerate", self.lookup) if arguments[0] else None
            if resolved in {"len", "builtins.len", "print", "builtins.print", "range", "builtins.range"}:
                return None
        if isinstance(node.func, ast.Attribute):
            receiver = self.receivers.get(id(node.func))
            if (
                receiver is not None
                and node.func.attr in {"values", "keys", "items"}
                and not (node.args or node.keywords)
            ):
                return receiver.step(node.func.attr, self.lookup)
            self.invalidate(node.func.value, env)
        for argument, value in zip([*node.args, *(kw.value for kw in node.keywords)], [*arguments, *keyword_arguments]):
            # Passing an immutable field value does not expose its owning
            # object/container for mutation. Unknown and mutable values still
            # invalidate aliases; do not infer read-only effects from a name.
            if value is None or value.shape.reference.removeprefix("builtins.") not in _IMMUTABLE_SCALARS:
                self.invalidate(argument, env)
        return None

    def comprehension(
        self, node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp, env: dict[str, FlowValue | None]
    ) -> FlowValue | None:
        local = dict(env)
        for generator in node.generators:
            iterable = self.expression(generator.iter, local)
            self.bind(
                generator.target,
                iterable.step("item", self.lookup) if iterable and not generator.is_async else None,
                local,
            )
            for condition in generator.ifs:
                self.expression(condition, local)
                if isinstance(condition, ast.Constant) and condition.value is False:
                    self.comprehension_effects(node, local, env)
                    return None
        if isinstance(node, ast.DictComp):
            self.expression(node.key, local)
            self.expression(node.value, local)
            result = None
        else:
            result = self.expression(node.elt, local)
        self.comprehension_effects(node, local, env)
        return result.step("collect", self.lookup) if result else None

    @staticmethod
    def comprehension_effects(
        node: ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp,
        local: dict[str, FlowValue | None],
        env: dict[str, FlowValue | None],
    ) -> None:
        targets = {child.id for gen in node.generators for child in ast.walk(gen.target) if isinstance(child, ast.Name)}
        for name in env.keys() - targets:
            if local.get(name) != env.get(name):
                env[name] = None
        for child in ast.walk(node):
            if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
                env[child.target.id] = None  # Walrus bindings escape; iteration may be empty.

    def statements(self, statements: Sequence[ast.stmt], env: dict[str, FlowValue | None]) -> bool:
        for statement in statements:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                if isinstance(statement, ast.AnnAssign) and statement.value is None:
                    continue  # An annotation alone does not rebind an existing value.
                value = self.expression(statement.value, env)
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                for target in targets:
                    self.bind(target, value, env)
            elif isinstance(statement, ast.AugAssign):
                self.expression(statement.target, env)
                self.expression(statement.value, env)
                self.invalidate(statement.target, env)
                self.bind(statement.target, None, env)
            elif isinstance(statement, ast.If):
                self.expression(statement.test, env)
                if isinstance(statement.test, ast.Constant) and isinstance(statement.test.value, bool):
                    if self.statements(statement.body if statement.test.value else statement.orelse, env):
                        return True
                    continue
                left, right = dict(env), dict(env)
                left_exits = self.statements(statement.body, left)
                right_exits = self.statements(statement.orelse, right)
                if left_exits and right_exits:
                    return True
                env.update(right if left_exits else left if right_exits else self.join(left, right))
            elif isinstance(statement, ast.For):
                iterable = self.expression(statement.iter, env)
                body = dict(env)
                self.bind(statement.target, iterable.step("item", self.lookup) if iterable else None, body)
                self.statements(statement.body, body)
                env.update(self.join(env, body))  # The loop may execute zero times.
                self.statements(statement.orelse, env)
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                env[statement.name] = None
            elif isinstance(statement, (ast.Expr, ast.Return, ast.Raise)):
                for child in ast.iter_child_nodes(statement):
                    self.expression(child, env)
                if isinstance(statement, (ast.Return, ast.Raise)):
                    return True
            elif isinstance(statement, (ast.Break, ast.Continue)):
                return True
            else:
                # Unsupported control flow must not manufacture evidence from
                # mutually exclusive paths or leak writes into a later read.
                for child in ast.walk(statement):
                    if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
                        env[child.id] = None
                    elif isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(
                        child.ctx, (ast.Store, ast.Del)
                    ):
                        self.invalidate(child, env)
                    elif isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            self.invalidate(child.func.value, env)
                        for argument in [*child.args, *(kw.value for kw in child.keywords)]:
                            self.invalidate(argument, env)
        return False
