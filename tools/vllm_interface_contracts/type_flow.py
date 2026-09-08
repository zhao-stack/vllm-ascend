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
        "isinstance",
    }
)
_IMMUTABLE_SCALARS = frozenset({"int", "str", "bool", "float", "bytes"})


def ordinary_instance_target(
    node: ast.AST, resolve: Resolve, lookup: Lookup, seen: frozenset[str] = frozenset()
) -> bool:
    """A custom metaclass may mutate an object during isinstance()."""
    if isinstance(node, ast.Tuple):
        return all(ordinary_instance_target(item, resolve, lookup, seen) for item in node.elts)
    if not isinstance(node, (ast.Name, ast.Attribute)):
        return False
    reference = resolve(ast.unparse(node))
    if reference is None or reference in seen:
        return False
    if reference.removeprefix("builtins.") in _IMMUTABLE_SCALARS | {"object", "list", "tuple", "dict", "set"}:
        return True
    source = lookup(reference)
    if source is None or source.node.keywords:
        return False
    for decorator in source.node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if source.resolve(ast.unparse(target)) != "dataclasses.dataclass":
            return False
    return all(ordinary_instance_target(base, source.resolve, lookup, seen | {reference}) for base in source.node.bases)


@dataclass(frozen=True)
class TypeShape:
    reference: str
    arguments: tuple[TypeShape, ...] = ()

    def render(self) -> str:
        suffix = "[" + ",".join(arg.render() for arg in self.arguments) + "]" if self.arguments else ""
        return self.reference + suffix


def dictionary_get_safe(receiver: TypeShape, key: TypeShape | None) -> bool:
    """Only builtin dictionaries with non-overloaded key protocols are queries."""
    scalar_keys = _IMMUTABLE_SCALARS | {"NoneType"}
    if key is None or key.reference.removeprefix("builtins.") not in scalar_keys:
        return False
    if receiver.reference == "dict_choice":
        return bool(receiver.arguments) and all(dictionary_get_safe(value, key) for value in receiver.arguments)
    if receiver.reference == "empty_dict":
        return True
    return (
        receiver.reference in {"dict", "builtins.dict", "typing.Dict"}
        and len(receiver.arguments) == 2
        and receiver.arguments[0].reference.removeprefix("builtins.") in scalar_keys
    )


def annotation_type(node: ast.AST | None, resolve: Resolve) -> TypeShape | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            node = ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        if isinstance(node.right, ast.Constant) and node.right.value is None:
            inner = annotation_type(node.left, resolve)
            return TypeShape("optional", (inner,)) if inner is not None else None
        if isinstance(node.left, ast.Constant) and node.left.value is None:
            inner = annotation_type(node.right, resolve)
            return TypeShape("optional", (inner,)) if inner is not None else None
    if isinstance(node, (ast.Name, ast.Attribute)):
        reference = resolve(ast.unparse(node))
        return TypeShape(reference) if reference is not None else None
    if not isinstance(node, ast.Subscript):
        return None
    base = annotation_type(node.value, resolve)
    if base is None:
        return None
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if base.reference == "typing.Optional" and len(items) == 1:
        inner = annotation_type(items[0], resolve)
        return TypeShape("optional", (inner,)) if inner is not None else None
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


def _plain_instance_storage(
    reference: str, member: str, lookup: Lookup, seen: frozenset[str] = frozenset(), *, reject_other_writes: bool = True
) -> bool:
    if reference in {"object", "builtins.object"}:
        return True
    if reference in seen or (source := lookup(reference)) is None:
        return False
    node = source.node
    if node.keywords or node.decorator_list or len(node.bases) > 1:
        return False
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if item.name in {member, "__getattribute__", "__getattr__", "__setattr__", "__delattr__", "__new__"}:
                return False
            if (
                reject_other_writes
                and item.name != "__init__"
                and any(
                    isinstance(child, ast.Attribute)
                    and child.attr == member
                    and isinstance(child.ctx, (ast.Store, ast.Del))
                    for child in ast.walk(item)
                )
            ):
                return False
        elif any(
            isinstance(child, ast.Name) and child.id == member and isinstance(child.ctx, (ast.Store, ast.Del))
            for child in ast.walk(item)
        ):
            return False
    return all(
        (base_reference := source.resolve(ast.unparse(base))) is not None
        and _plain_instance_storage(
            base_reference, member, lookup, seen | {reference}, reject_other_writes=reject_other_writes
        )
        for base in node.bases
    )


def instance_field_origin(
    reference: str, member: str, lookup: Lookup, seen: frozenset[str] = frozenset()
) -> tuple[str, TypeShape, str] | None:
    """Prove one unconditional constructor store of an annotated parameter.

    This is a stored-field contract, not arbitrary constructor execution. Other
    writes, descriptors, parameter rebinding and an overriding initializer
    block inherited evidence. Return the declaring owner for endpoint replay.
    """
    if reference in seen or (source := lookup(reference)) is None:
        return None
    if not _plain_instance_storage(reference, member, lookup):
        return None
    node = source.node
    if node.keywords or node.decorator_list or len(node.bases) > 1:
        return None
    initializers = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"]
    writes: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, ast.AST]] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in {
            "__getattribute__",
            "__getattr__",
            "__setattr__",
            "__delattr__",
        }:
            return None
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == member:
            return None
        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if any(
                isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == member
                for n in ast.walk(item)
            ):
                return None
            continue
        args = [*item.args.posonlyargs, *item.args.args]
        if not args:
            continue
        receiver = args[0].arg
        aliases = {receiver}
        for assignment in ast.walk(item):
            if isinstance(assignment, ast.Assign) and isinstance(assignment.value, ast.Name):
                if assignment.value.id in aliases:
                    aliases.update(t.id for t in assignment.targets if isinstance(t, ast.Name))
        for child in ast.walk(item):
            if (
                isinstance(child, ast.Attribute)
                and child.attr == member
                and isinstance(child.ctx, (ast.Store, ast.Del))
                and isinstance(child.value, ast.Name)
                and child.value.id in aliases
            ):
                writes.append((item, child))
    if writes:
        if len(writes) != 1 or len(initializers) != 1 or writes[0][0] is not initializers[0]:
            return None
        init = initializers[0]
        if init.decorator_list:
            return None
        assignments = [
            s
            for s in init.body
            if isinstance(s, (ast.Assign, ast.AnnAssign))
            and any(t is writes[0][1] for t in (s.targets if isinstance(s, ast.Assign) else [s.target]))
        ]
        if len(assignments) != 1 or not isinstance(assignments[0].value, ast.Name):
            return None
        if any(
            isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom))
            for statement in init.body[: init.body.index(assignments[0])]
            for child in ast.walk(statement)
        ):
            return None
        name = assignments[0].value.id
        parameters = [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
        argument = next((arg for arg in parameters if arg.arg == name), None)
        if argument is None or any(
            isinstance(n, ast.Name) and n.id in {name, parameters[0].arg} and isinstance(n.ctx, (ast.Store, ast.Del))
            for n in ast.walk(init)
        ):
            return None
        shape = annotation_type(argument.annotation, source.resolve)
        return (reference, shape, name) if shape is not None else None
    if initializers:
        return None
    for base in node.bases:
        base_reference = source.resolve(ast.unparse(base))
        if base_reference is not None:
            return instance_field_origin(base_reference, member, lookup, seen | {reference})
    return None


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
    if step == "nonnull":
        return value.arguments[0] if value.reference == "optional" and len(value.arguments) == 1 else value
    if step.startswith("instance_field:"):
        origin = instance_field_origin(value.reference, step.removeprefix("instance_field:"), lookup)
        return origin[1] if origin is not None else None
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
    origins: frozenset[str] = frozenset()

    @property
    def roots(self) -> frozenset[str]:
        return self.origins or frozenset(self.path[:1])

    def step(self, step: str, lookup: Lookup) -> FlowValue | None:
        shape = type_step(self.shape, step, lookup)
        if shape is not None:
            return FlowValue(shape, (*self.path, step), self.origins)
        return FlowValue(TypeShape("unknown"), ("unknown",), self.origins) if self.origins else None


@dataclass(frozen=True)
class HelperCallEffects:
    safe_arguments: frozenset[int]
    borrowed_arguments: frozenset[int]
    return_shape: TypeShape | None
    return_path: tuple[str, ...] | None


HelperResolver = Callable[[ast.Call, tuple[FlowValue | None, ...]], HelperCallEffects | None]


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
        helper_resolve: HelperResolver | None = None,
        instance_fields: dict[str, FlowValue] | None = None,
        instance_owner: str | None = None,
    ):
        self.lookup = lookup
        self.resolve = resolve
        self.runtime_resolve = runtime_resolve or resolve
        self.helper_resolve = helper_resolve
        self.instance_fields = instance_fields or {}
        self.instance_owner = instance_owner
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
        env.update(self.instance_fields)
        for receiver in {name.split(".")[0] for name in self.instance_fields}:
            env[receiver] = FlowValue(
                TypeShape("unknown"),
                ("unknown",),
                frozenset(
                    root
                    for name, value in self.instance_fields.items()
                    if name.startswith(receiver + ".")
                    for root in value.roots
                ),
            )
        if self.instance_fields or any(value is not None and "vllm." in value.shape.render() for value in env.values()):
            self.statements(function.body, env)

    @staticmethod
    def join(left: dict[str, FlowValue | None], right: dict[str, FlowValue | None]) -> dict[str, FlowValue | None]:
        return {
            name: left.get(name) if left.get(name) == right.get(name) else None for name in left.keys() | right.keys()
        }

    def class_reference(self, expression: str) -> str | None:
        if expression.split(".", 1)[0] in self.blocked_builtins:
            return None
        return self.runtime_resolve(expression)

    def bind(self, target: ast.AST, value: FlowValue | None, env: dict[str, FlowValue | None]) -> None:
        if isinstance(target, ast.Name):
            for field in self.instance_fields:
                if field.startswith(target.id + "."):
                    self.invalidate(ast.parse(field, mode="eval").body, env)
            env[target.id] = value
        elif isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                self.bind(item, value.step(f"index:{index}", self.lookup) if value else None, env)
        else:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and ast.unparse(target) not in self.instance_fields
                and any(name.startswith(target.value.id + ".") for name in self.instance_fields)
                and self.instance_owner is not None
                and _plain_instance_storage(self.instance_owner, target.attr, self.lookup, reject_other_writes=False)
            ):
                return  # A distinct plain stored field does not rebind the configuration.
            self.invalidate(target, env)

    def invalidate(self, node: ast.AST, env: dict[str, FlowValue | None]) -> None:
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            if ast.unparse(root) in self.instance_fields:
                value = env.get(ast.unparse(root))
                if value is not None:
                    for name, candidate in list(env.items()):
                        if candidate is not None and candidate.roots & value.roots:
                            env[name] = None
                return
            root = root.value
        if isinstance(root, ast.Name):
            for field in self.instance_fields:
                if field.startswith(root.id + ".") and isinstance(node, ast.Name):
                    self.invalidate(ast.parse(field, mode="eval").body, env)
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        value = env.get(root.id) if isinstance(root, ast.Name) else None
        if value is not None:
            # Aliases and derived element bindings share the same origin.
            for name, candidate in list(env.items()):
                if candidate is not None and candidate.roots & value.roots:
                    env[name] = None

    def expression(self, node: ast.AST | None, env: dict[str, FlowValue | None]) -> FlowValue | None:
        if isinstance(node, ast.Constant) and type(node.value).__name__ in _IMMUTABLE_SCALARS | {"NoneType"}:
            shape = TypeShape(type(node.value).__name__)
            return FlowValue(shape, (shape.render(),))
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Dict):
            keys, values = [], []
            for key_node, value_node in zip(node.keys, node.values):
                keys.append(self.expression(key_node, env))
                values.append(self.expression(value_node, env))
            if not node.keys:
                return FlowValue(TypeShape("empty_dict"), ("unknown",))
            key_shapes = {key.shape for key in keys if key is not None}
            value_shapes = {value.shape for value in values if value is not None}
            key_shape = next(iter(key_shapes)) if len(key_shapes) == 1 and all(keys) else TypeShape("unknown")
            value_shape = next(iter(value_shapes)) if len(value_shapes) == 1 and all(values) else TypeShape("unknown")
            for expression, value in zip(node.keys, keys):
                if expression is not None and (
                    value is None or value.shape.reference not in _IMMUTABLE_SCALARS | {"NoneType"}
                ):
                    self.invalidate(expression, env)
            origins = frozenset(root for value in (*keys, *values) if value is not None for root in value.roots)
            return FlowValue(TypeShape("dict", (key_shape, value_shape)), ("unknown",), origins)
        if isinstance(node, ast.Attribute):
            if ast.unparse(node) in self.instance_fields:
                return env.get(ast.unparse(node))
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
            self.narrow_condition(node.test, left_env, right_env)
            left = self.expression(node.body, left_env)
            right = self.expression(node.orelse, right_env)
            env.update(self.join(left_env, right_env))
            if (
                left is not None
                and right is not None
                and all(
                    value.shape.reference in {"dict", "builtins.dict", "typing.Dict", "empty_dict", "dict_choice"}
                    for value in (left, right)
                )
                and left != right
            ):
                return FlowValue(
                    TypeShape("dict_choice", (left.shape, right.shape)), ("unknown",), left.roots | right.roots
                )
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
            if resolved in {
                "len",
                "builtins.len",
                "print",
                "builtins.print",
                "range",
                "builtins.range",
            }:
                return None
            if (
                resolved in {"isinstance", "builtins.isinstance"}
                and len(node.args) == 2
                and not node.keywords
                and ordinary_instance_target(node.args[1], self.class_reference, self.lookup)
            ):
                return None
        if isinstance(node.func, ast.Attribute):
            receiver = self.receivers.get(id(node.func))
            if (
                receiver is not None
                and node.func.attr == "get"
                and not node.keywords
                and len(node.args) in {1, 2}
                and not any(isinstance(arg, ast.Starred) for arg in node.args)
                and dictionary_get_safe(receiver.shape, arguments[0].shape if arguments[0] else None)
            ):
                default = arguments[1] if len(arguments) == 2 else None
                if receiver.shape.reference == "empty_dict":
                    return default
                origins = receiver.roots | (default.roots if default is not None else frozenset())
                # The result may be the mapping element or the supplied default.
                # Retain both alias origins without inventing one historical path.
                return FlowValue(TypeShape("unknown"), ("unknown",), origins)
            if (
                receiver is not None
                and node.func.attr in {"values", "keys", "items"}
                and not (node.args or node.keywords)
                and type_step(receiver.shape, node.func.attr, self.lookup) is not None
            ):
                return receiver.step(node.func.attr, self.lookup)
            self.invalidate(node.func.value, env)
        values = (*arguments, *keyword_arguments)
        effects = self.helper_resolve(node, values) if self.helper_resolve is not None else None
        for index, (argument, value) in enumerate(zip([*node.args, *(kw.value for kw in node.keywords)], values)):
            # Passing an immutable field value does not expose its owning
            # object/container for mutation. Unknown and mutable values still
            # invalidate aliases; do not infer read-only effects from a name.
            if effects is not None and index in effects.safe_arguments:
                continue
            if value is None or value.shape.reference.removeprefix("builtins.") not in _IMMUTABLE_SCALARS:
                self.invalidate(argument, env)
        if effects is not None:
            origins = frozenset(
                root
                for index in effects.borrowed_arguments
                for value in (values[index],)
                if value is not None
                for root in value.roots
            )
            shape = effects.return_shape or TypeShape("unknown")
            # Borrowed results need a source path that both snapshots can
            # replay. A new-only inferred owner is not an old endpoint proof.
            path = effects.return_path or (("unknown",) if origins else (shape.render(),))
            return FlowValue(shape, path, origins)
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
                self.narrow_condition(statement.test, left, right)
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
            elif isinstance(statement, ast.Assert):
                self.expression(statement.test, env)
                self.narrow_condition(statement.test, env, dict(env))
                # A message may execute when the assertion fails; that path
                # does not continue to a following field read.
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

    def narrow_condition(
        self, test: ast.AST, positive: dict[str, FlowValue | None], negative: dict[str, FlowValue | None]
    ) -> None:
        if (
            isinstance(test, ast.Call)
            and isinstance(test.func, ast.Name)
            and self.class_reference(test.func.id) in {"isinstance", "builtins.isinstance"}
            and len(test.args) == 2
            and not test.keywords
            and isinstance(test.args[0], ast.Name)
            and ordinary_instance_target(test.args[1], self.class_reference, self.lookup)
        ):
            reference = self.class_reference(ast.unparse(test.args[1]))
            value = positive.get(test.args[0].id)
            if reference is not None and value is not None:
                positive[test.args[0].id] = FlowValue(TypeShape(reference), (reference,), value.roots)
            return
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is None
        ):
            return
        env = positive if isinstance(test.ops[0], ast.IsNot) else negative if isinstance(test.ops[0], ast.Is) else None
        name = ast.unparse(test.left)
        if env is not None and name in env and env[name] is not None:
            value = env[name]
            if value is not None:
                env[name] = value.step("nonnull", self.lookup)
