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
from dataclasses import dataclass, replace

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
        "zip",
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


def immutable_value(shape: TypeShape | None) -> bool:
    """Values that cannot carry a mutable alias back to their containing object."""
    if shape is None:
        return False
    name = shape.reference.removeprefix("builtins.")
    if name in _IMMUTABLE_SCALARS | {"NoneType"}:
        return True
    # PyTorch's exact dtype extension is a non-subclassable singleton carrying
    # a ScalarType and name, not a Tensor or a reference to its owning config.
    # See pytorch v2.8.0 torch/csrc/Dtype.h and Dtype.cpp.
    if name == "torch.dtype":
        return True
    if name in {"tuple", "typing.Tuple"} and shape.arguments:
        return all(arg.reference == "ellipsis" or immutable_value(arg) for arg in shape.arguments)
    return False


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


def empty_container_type(node: ast.AST | None, resolve: Resolve) -> TypeShape | None:
    if isinstance(node, ast.List) and not node.elts:
        return TypeShape("list")
    if isinstance(node, ast.Dict) and not node.keys:
        return TypeShape("empty_dict")
    if not isinstance(node, ast.Call) or node.args or node.keywords:
        return None
    target = node.func.value if isinstance(node.func, ast.Subscript) else node.func
    reference = resolve(ast.unparse(target))
    if reference not in {"list", "set", "dict", "builtins.list", "builtins.set", "builtins.dict"}:
        return None
    if isinstance(node.func, ast.Subscript):
        shape = annotation_type(node.func, resolve)
        return replace(shape, reference=shape.reference.removeprefix("builtins.")) if shape else None
    return TypeShape(
        "empty_dict" if reference.removeprefix("builtins.") == "dict" else reference.removeprefix("builtins.")
    )


def constructor_collection_type(node: ast.AST | None, resolve: Resolve) -> TypeShape | None:
    """Prove an allocated container's shape, not purity of its construction."""
    empty = empty_container_type(node, resolve)
    if empty is not None:
        return empty
    elements: list[ast.expr]
    if isinstance(node, ast.List):
        elements = node.elts
    elif isinstance(node, ast.ListComp) and not any(generator.is_async for generator in node.generators):
        # A comprehension always creates a list. Only literal scalar elements
        # establish an element contract independent of iteration bindings.
        elements = [node.elt]
    else:
        return None
    if not elements or any(
        not isinstance(element, ast.Constant) or type(element.value).__name__ not in _IMMUTABLE_SCALARS | {"NoneType"}
        for element in elements
    ):
        return None
    shapes = {TypeShape(type(element.value).__name__) for element in elements if isinstance(element, ast.Constant)}
    return TypeShape("list", (next(iter(shapes)),)) if len(shapes) == 1 else None


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
) -> tuple[str, TypeShape, str, bool] | None:
    """Prove one unconditional parameter store or builtin collection allocation.

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
        if len(assignments) != 1:
            return None
        if any(
            isinstance(child, (ast.Return, ast.Yield, ast.YieldFrom))
            for statement in init.body[: init.body.index(assignments[0])]
            for child in ast.walk(statement)
        ):
            return None
        parameters = [*init.args.posonlyargs, *init.args.args, *init.args.kwonlyargs]
        if any(
            isinstance(n, ast.Name) and n.id == parameters[0].arg and isinstance(n.ctx, (ast.Store, ast.Del))
            for n in ast.walk(init)
        ):
            return None
        if not isinstance(assignments[0].value, ast.Name):
            # A fresh allocation may already contain borrowed objects, or have
            # escaped, by the time __init__ returns. Do not seed an empty owned
            # field when the initializer observes it after the defining store.
            if any(
                isinstance(child, ast.Attribute)
                and isinstance(child.ctx, ast.Load)
                and isinstance(child.value, ast.Name)
                and child.value.id == parameters[0].arg
                and child.attr == member
                for child in ast.walk(init)
            ):
                return None
            local_names = {arg.arg for arg in parameters} | {
                n.id for n in ast.walk(init) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
            }
            shape = constructor_collection_type(
                assignments[0].value,
                lambda expression: None if expression.split(".")[0] in local_names else source.resolve(expression),
            )
            return (reference, shape, member, True) if shape is not None else None
        name = assignments[0].value.id
        argument = next((arg for arg in parameters if arg.arg == name), None)
        if argument is None or any(
            isinstance(n, ast.Name) and n.id in {name, parameters[0].arg} and isinstance(n.ctx, (ast.Store, ast.Del))
            for n in ast.walk(init)
        ):
            return None
        shape = annotation_type(argument.annotation, source.resolve)
        return (reference, shape, name, False) if shape is not None else None
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
    owned: str | None = None
    components: tuple[FlowValue, ...] = ()

    @property
    def roots(self) -> frozenset[str]:
        roots = self.origins or (frozenset() if self.owned else frozenset(self.path[:1]))
        return roots | frozenset({self.owned}) if self.owned is not None else roots

    def step(self, step: str, lookup: Lookup) -> FlowValue | None:
        if self.shape.reference == "zip" and step == "item":
            return FlowValue(
                TypeShape("tuple", self.shape.arguments), ("unknown",), self.roots, components=self.components
            )
        if self.components and self.shape.reference == "tuple" and step.startswith("index:"):
            try:
                return self.components[int(step.removeprefix("index:"))]
            except (ValueError, IndexError):
                return None
        shape = type_step(self.shape, step, lookup)
        if shape is not None:
            return FlowValue(shape, (*self.path, step), self.roots if self.owned else self.origins)
        return FlowValue(TypeShape("unknown"), ("unknown",), self.roots) if self.origins or self.owned else None


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
        borrowed_fields = frozenset(
            root
            for value in self.instance_fields.values()
            if value.owned is None and not immutable_value(value.shape)
            for root in value.roots
        )
        for name, value in list(self.instance_fields.items()):
            if value.owned is not None:
                # Constructor allocation proves container identity, not that its
                # contents are still empty after other methods have run. Stored
                # elements may borrow configuration objects through those methods.
                shape = value.shape
                if shape.reference == "list" and not shape.arguments:
                    shape = TypeShape("list", (TypeShape("unknown"),))
                if shape.reference == "empty_dict":
                    shape = TypeShape("dict", (TypeShape("unknown"), TypeShape("unknown")))
                self.instance_fields[name] = replace(value, shape=shape, origins=value.origins | borrowed_fields)
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
        result = {}
        for name in left.keys() | right.keys():
            first, second = left.get(name), right.get(name)
            merged = first if first == second else None
            if first is not None and second is not None and first.owned is not None and first.owned == second.owned:
                if ContainerFlow.is_empty(first):
                    merged = second
                elif ContainerFlow.is_empty(second):
                    merged = first
                elif first.shape == second.shape and first.path == second.path:
                    merged = replace(first, origins=first.origins | second.origins)
            result[name] = merged
        return result

    @staticmethod
    def is_empty(value: FlowValue) -> bool:
        return (
            value.owned is not None
            and value.shape.reference in {"empty_dict", "list", "set"}
            and not value.shape.arguments
        )

    @staticmethod
    def allocation(node: ast.AST) -> str:
        return f"owned:{getattr(node, 'lineno', 0)}:{getattr(node, 'col_offset', 0)}"

    @staticmethod
    def borrowed_origins(value: FlowValue | None) -> frozenset[str]:
        if value is None or immutable_value(value.shape):
            return frozenset()
        return value.roots

    @staticmethod
    def update_owned(old: FlowValue, new: FlowValue, env: dict[str, FlowValue | None]) -> None:
        for name, value in list(env.items()):
            if value is not None and old.owned is not None and value.owned == old.owned:
                env[name] = new

    @staticmethod
    def implicit_protocol_barrier(env: dict[str, FlowValue | None]) -> None:
        # An unknown hash/equality/iteration method may execute arbitrary
        # source behavior. Disjoint allocation alone is not a purity proof.
        for name in env:
            env[name] = None

    def class_reference(self, expression: str) -> str | None:
        if expression.split(".", 1)[0] in self.blocked_builtins:
            return None
        return self.runtime_resolve(expression)

    def bind(self, target: ast.AST, value: FlowValue | None, env: dict[str, FlowValue | None]) -> None:
        if isinstance(target, ast.Subscript):
            receiver = self.expression(target.value, env)
            key = self.expression(target.slice, env)
            if receiver is not None and receiver.owned is not None and receiver.shape.reference == "list":
                if key is None or key.shape.reference.removeprefix("builtins.") not in {"int", "bool"}:
                    # Unknown __index__, slice bounds or iterable replacement
                    # protocols can execute arbitrary effects.
                    self.implicit_protocol_barrier(env)
                    return
                if len(receiver.shape.arguments) == 1 and immutable_value(receiver.shape.arguments[0]):
                    element = value.shape if value is not None else TypeShape("unknown")
                    unchanged_type = element == receiver.shape.arguments[0]
                    updated = replace(
                        receiver,
                        shape=receiver.shape if unchanged_type else TypeShape("list", (TypeShape("unknown"),)),
                        path=receiver.path if unchanged_type else ("unknown",),
                        origins=receiver.origins | self.borrowed_origins(value),
                    )
                    self.update_owned(receiver, updated, env)
                    return
                # Releasing an unproven previous element may invoke __del__.
                self.invalidate(target, env)
                return
            if (
                receiver is not None
                and receiver.owned is not None
                and receiver.shape.reference in {"dict", "empty_dict"}
                and key is not None
                and key.shape.reference in _IMMUTABLE_SCALARS | {"NoneType"}
            ):
                shape = TypeShape("dict", (key.shape, value.shape if value else TypeShape("unknown")))
                if not self.is_empty(receiver) and receiver.shape != shape:
                    shape = TypeShape("dict", (TypeShape("unknown"), TypeShape("unknown")))
                updated = replace(receiver, shape=shape, origins=receiver.origins | self.borrowed_origins(value))
                self.update_owned(receiver, updated, env)
                return
            if (
                receiver is not None
                and receiver.owned is not None
                and receiver.shape.reference in {"dict", "empty_dict"}
            ):
                self.implicit_protocol_barrier(env)
                return
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
        if isinstance(node, (ast.List, ast.Set)):
            values = []
            unsafe = False
            for child in node.elts:
                if isinstance(child, ast.Starred):
                    iterable = self.expression(child.value, env)
                    item = iterable.step("item", self.lookup) if iterable is not None else None
                    if item is None or item.shape.reference == "unknown":
                        unsafe = True
                    values.append(item)
                else:
                    values.append(self.expression(child, env))
            if isinstance(node, ast.Set) and any(
                value is None or value.shape.reference not in _IMMUTABLE_SCALARS for value in values
            ):
                unsafe = True
            if unsafe:
                self.implicit_protocol_barrier(env)
            shapes = {value.shape for value in values if value is not None}
            element = next(iter(shapes)) if not unsafe and len(shapes) == 1 and all(values) else TypeShape("unknown")
            paths = {value.path for value in values if value is not None}
            path = (*next(iter(paths)), "collect") if not unsafe and len(paths) == 1 and all(values) else ("unknown",)
            shape = TypeShape("list" if isinstance(node, ast.List) else "set", (element,) if values else ())
            origins = frozenset(root for value in values for root in self.borrowed_origins(value))
            return FlowValue(shape, path, origins, self.allocation(node))
        if isinstance(node, ast.Dict):
            keys, values = [], []
            for key_node, value_node in zip(node.keys, node.values):
                keys.append(self.expression(key_node, env))
                values.append(self.expression(value_node, env))
            if not node.keys:
                return FlowValue(TypeShape("empty_dict"), ("unknown",), owned=self.allocation(node))
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
            return FlowValue(TypeShape("dict", (key_shape, value_shape)), ("unknown",), origins, self.allocation(node))
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
            if isinstance(node.slice, ast.Slice):
                bounds = [
                    self.expression(bound, env) for bound in (node.slice.lower, node.slice.upper, node.slice.step)
                ]
                if value is not None and value.owned is not None and value.shape.reference == "list":
                    if any(
                        bound is not None
                        and (result is None or result.shape.reference not in {"int", "bool", "NoneType"})
                        for bound, result in zip((node.slice.lower, node.slice.upper, node.slice.step), bounds)
                    ):
                        self.implicit_protocol_barrier(env)
                        return None
                    # Slicing copies the container, but its elements still alias
                    # the original source objects. Appending affects only the copy.
                    return replace(value, owned=self.allocation(node), origins=value.roots)
                return value
            key = self.expression(node.slice, env)
            if value is not None and value.owned is not None:
                safe = True
                if value.shape.reference in {"dict", "empty_dict"}:
                    safe = dictionary_get_safe(value.shape, key.shape if key else None)
                elif value.shape.reference == "list":
                    safe = key is not None and key.shape.reference in {"int", "bool"}
                if not safe:
                    self.implicit_protocol_barrier(env)
                    return None
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
            for expression_child in ast.iter_child_nodes(node):
                self.expression(expression_child, env)
        return None

    def call(self, node: ast.Call, env: dict[str, FlowValue | None]) -> FlowValue | None:
        self.expression(node.func, env)
        arguments = [self.expression(arg, env) for arg in node.args]
        keyword_arguments = [self.expression(keyword.value, env) for keyword in node.keywords]
        fresh = empty_container_type(node, self.class_reference)
        if fresh is not None:
            # Generic arguments describe allowed values, not existing elements.
            # A local zero-argument allocation is empty on its initial path.
            fresh = TypeShape("empty_dict" if fresh.reference == "dict" else fresh.reference)
            return FlowValue(fresh, ("unknown",), owned=self.allocation(node))
        if isinstance(node.func, ast.Name) and node.func.id not in self.blocked_builtins:
            resolved = self.runtime_resolve(node.func.id)
            if (
                resolved in {"zip", "builtins.zip"}
                and not any(isinstance(arg, ast.Starred) for arg in node.args)
                and all(
                    kw.arg == "strict" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool)
                    for kw in node.keywords
                )
                and all(
                    value is not None
                    and value.shape.reference
                    in {"list", "tuple", "builtins.list", "builtins.tuple", "typing.List", "typing.Tuple"}
                    for value in arguments
                )
            ):
                items = [value.step("item", self.lookup) for value in arguments if value is not None]
                if all(item is not None for item in items):
                    components = tuple(item for item in items if item is not None)
                    origins = frozenset(root for value in arguments for root in self.borrowed_origins(value))
                    return FlowValue(
                        TypeShape("zip", tuple(item.shape for item in components)),
                        ("unknown",),
                        origins,
                        components=components,
                    )
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
                and receiver.shape.reference.removeprefix("builtins.") == "str"
                and node.func.attr
                in {"lower", "upper", "strip", "lstrip", "rstrip", "casefold", "capitalize", "title", "swapcase"}
                and not node.args
                and not node.keywords
            ):
                return FlowValue(TypeShape("str"), ("str",))
            if (
                receiver is not None
                and receiver.owned is not None
                and not node.keywords
                and len(node.args) == 1
                and not isinstance(node.args[0], ast.Starred)
                and (
                    receiver.shape.reference == "list"
                    and node.func.attr == "append"
                    or receiver.shape.reference == "set"
                    and node.func.attr == "add"
                    and arguments[0] is not None
                    and arguments[0].shape.reference in _IMMUTABLE_SCALARS
                    and (not receiver.shape.arguments or receiver.shape.arguments[0].reference in _IMMUTABLE_SCALARS)
                )
            ):
                item = arguments[0]
                element = item.shape if item is not None else TypeShape("unknown")
                path = (*item.path, "collect") if item is not None else ("unknown",)
                shape = TypeShape(receiver.shape.reference, (element,))
                if not self.is_empty(receiver) and (
                    receiver.shape != shape or receiver.path not in {("unknown",), path}
                ):
                    shape, path = TypeShape(receiver.shape.reference, (TypeShape("unknown"),)), ("unknown",)
                updated = replace(
                    receiver, shape=shape, path=path, origins=receiver.origins | self.borrowed_origins(item)
                )
                self.update_owned(receiver, updated, env)
                return None
            if (
                receiver is not None
                and receiver.owned is not None
                and receiver.shape.reference == "set"
                and node.func.attr == "add"
            ):
                self.implicit_protocol_barrier(env)
                return None
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
            if value is None or not immutable_value(value.shape):
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
