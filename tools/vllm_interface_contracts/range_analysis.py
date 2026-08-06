"""Analyze an exact vLLM commit range against live vllm-ascend dependencies.

The dependency graph is generated consumer-first from the selected source pair.
The range analyzer then resolves every dependency at both upstream endpoints so
historical incompatibilities are separated from breaks introduced by the range.
"""

from __future__ import annotations

import ast
import copy
import csv
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.vllm_interface_contracts.generator import (
    GENERATOR_VERSION,
    InterfaceBoundaryGenerator,
    Relation,
    _accepts_signature_contract,
    _jsonable_signature,
)
from tools.vllm_interface_contracts.models import (
    CompatibilityState,
    RangeFinding,
    SourceEndpoint,
)

RANGE_SCHEMA_VERSION = 1
RANGE_ANALYZER_VERSION = "1.0.0"
CLASSIFICATIONS = (
    "introduced_break",
    "compatibility_warning",
    "preexisting",
    "fixed",
    "analysis_unresolved",
)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
    )
    return result.stdout.decode("utf-8", errors="replace").strip()


def git_head(repo: Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def resolve_commit(repo: Path, revision: str) -> str:
    try:
        return _git(repo, "rev-parse", f"{revision}^{{commit}}")
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Git commit does not exist: {revision}") from error


def verify_range(vllm_root: Path, old: str, new: str) -> tuple[str, str]:
    old_sha = resolve_commit(vllm_root, old)
    new_sha = resolve_commit(vllm_root, new)
    ancestor = subprocess.run(
        ["git", "-C", str(vllm_root), "merge-base", "--is-ancestor", old_sha, new_sha],
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValueError(f"old is not an ancestor of new: {old_sha} -> {new_sha}")
    return old_sha, new_sha


def verify_head(label: str, root: Path, expected: str) -> str:
    actual = git_head(root)
    resolved = resolve_commit(root, expected)
    if actual != resolved:
        raise ValueError(f"{label} checkout mismatch: expected {resolved}, got {actual}")
    return actual


def _module_file(module: str) -> tuple[str, str]:
    stem = module.replace(".", "/")
    return f"{stem}.py", f"{stem}/__init__.py"


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _decorator_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _descriptor(node: ast.AST) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    names = {_decorator_name(item).rsplit(".", 1)[-1] for item in node.decorator_list}
    for candidate in ("property", "classmethod", "staticmethod"):
        if candidate in names:
            return candidate
    return "ordinary"


def _class_nodes(tree: ast.Module) -> Iterator[tuple[tuple[str, ...], ast.ClassDef]]:
    def visit(body: list[ast.stmt], parents: tuple[str, ...]) -> Iterator[tuple[tuple[str, ...], ast.ClassDef]]:
        for item in body:
            if isinstance(item, ast.ClassDef):
                path = (*parents, item.name)
                yield path, item
                yield from visit(item.body, path)

    yield from visit(tree.body, ())


def _owner_node(tree: ast.Module, owner: str | None) -> ast.ClassDef | None:
    if not owner:
        return None
    expected = tuple(owner.split("."))
    matches = [node for path, node in _class_nodes(tree) if path == expected or path[-len(expected) :] == expected]
    return matches[0] if len(matches) == 1 else None


def _named_node(tree: ast.Module, owner: str | None, name: str) -> ast.AST | None:
    if owner:
        class_node = _owner_node(tree, owner)
        if class_node is None:
            return None
        return next(
            (
                item
                for item in class_node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == name
            ),
            None,
        )
    return next(
        (
            item
            for item in tree.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and item.name == name
        ),
        None,
    )


def _definition_fingerprint(node: ast.AST | None) -> str | None:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    normalized = copy.deepcopy(node)
    normalized.name = "__interface_callable__"
    normalized.decorator_list = []
    return hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()


class GitSnapshot:
    def __init__(self, root: Path, revision: str):
        self.root = root
        self.revision = revision
        self._files: set[str] | None = None
        self._source: dict[str, str | None] = {}
        self._trees: dict[str, ast.Module | None] = {}

    @property
    def files(self) -> set[str]:
        if self._files is None:
            output = _git(self.root, "ls-tree", "-r", "--name-only", self.revision)
            self._files = {line.strip() for line in output.splitlines() if line.strip()}
        return self._files

    def source(self, file_name: str) -> str | None:
        normalized = file_name.replace("\\", "/")
        if normalized not in self._source:
            if normalized not in self.files:
                self._source[normalized] = None
            else:
                raw = subprocess.run(
                    ["git", "-C", str(self.root), "show", f"{self.revision}:{normalized}"],
                    check=True,
                    capture_output=True,
                ).stdout
                self._source[normalized] = raw.decode("utf-8", errors="replace")
        return self._source[normalized]

    def tree(self, file_name: str) -> ast.Module | None:
        normalized = file_name.replace("\\", "/")
        if normalized not in self._trees:
            source = self.source(normalized)
            if source is None:
                self._trees[normalized] = None
            else:
                try:
                    self._trees[normalized] = ast.parse(source, filename=normalized)
                except SyntaxError:
                    self._trees[normalized] = None
        return self._trees[normalized]

    def resolve_module(self, module: str) -> str | None:
        return next((candidate for candidate in _module_file(module) if candidate in self.files), None)

    def endpoint(self, file_name: str, owner: str | None, name: str) -> SourceEndpoint:
        tree = self.tree(file_name)
        node = _named_node(tree, owner, name) if tree is not None else None
        return SourceEndpoint(
            file=file_name if file_name in self.files else None,
            owner=owner,
            name=name,
            line=getattr(node, "lineno", None),
            signature=_jsonable_signature(node),
            descriptor=_descriptor(node) if node is not None else None,
        )

    def expression_endpoint(self, expression: str) -> SourceEndpoint | None:
        parts = expression.strip().split(".")
        if not parts or parts[0] != "vllm":
            return None
        for split in range(len(parts), 0, -1):
            module = ".".join(parts[:split])
            file_name = self.resolve_module(module)
            if file_name is None:
                continue
            suffix = parts[split:]
            if not suffix:
                return SourceEndpoint(file=file_name, owner=None, name=None)
            owner = ".".join(suffix[:-1]) or None
            return self.endpoint(file_name, owner, suffix[-1])
        return None

    def unique_rename(
        self,
        file_name: str,
        owner: str | None,
        old_name: str,
        fingerprint: str | None,
    ) -> SourceEndpoint | None:
        if fingerprint is None:
            return None
        tree = self.tree(file_name)
        if tree is None:
            return None
        body = _owner_node(tree, owner).body if owner and _owner_node(tree, owner) is not None else tree.body
        matches = [
            item
            for item in body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name != old_name
            and _definition_fingerprint(item) == fingerprint
        ]
        if len(matches) != 1:
            return None
        node = matches[0]
        return SourceEndpoint(
            file=file_name,
            owner=owner,
            name=node.name,
            line=node.lineno,
            signature=_jsonable_signature(node),
            descriptor=_descriptor(node),
        )


def _rename_maps(root: Path, old: str, new: str) -> tuple[dict[str, str], dict[str, str]]:
    output = _git(root, "diff", "--name-status", "--find-renames", old, new)
    old_to_new: dict[str, str] = {}
    new_to_old: dict[str, str] = {}
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) == 3 and parts[0].startswith("R"):
            old_path, new_path = parts[1], parts[2]
            old_to_new[old_path] = new_path
            new_to_old[new_path] = old_path
    return old_to_new, new_to_old


def _state(upstream: SourceEndpoint, downstream_signature: list[object] | None, relation: str) -> CompatibilityState:
    if upstream.file is None or upstream.name is None:
        return CompatibilityState(False, False, "upstream target does not exist")
    if relation == "inheritance":
        return CompatibilityState(True, True, "upstream base class exists")
    if upstream.signature is None or downstream_signature is None:
        return CompatibilityState(True, None, "callable signature could not be compared")
    compatible = _accepts_signature_contract(upstream.signature, downstream_signature)
    return CompatibilityState(
        True,
        compatible,
        (
            "downstream accepts the upstream call contract"
            if compatible
            else "downstream does not accept the upstream call contract"
        ),
    )


def _classify(
    old_state: CompatibilityState,
    new_state: CompatibilityState,
    contract_changed: bool,
) -> str:
    if old_state.compatible is True and new_state.compatible is False:
        return "introduced_break"
    if old_state.compatible is False and new_state.compatible is True:
        return "fixed"
    if old_state.compatible is False and new_state.compatible is False:
        return "preexisting"
    if contract_changed and old_state.compatible is True and new_state.compatible is True:
        return "compatibility_warning"
    if old_state.exists is False and new_state.compatible is False:
        return "introduced_break"
    return "analysis_unresolved"


def _change_text(old: SourceEndpoint, new: SourceEndpoint) -> str:
    if old.file is None and new.file is not None:
        return "upstream target was added"
    if old.file is not None and new.file is None:
        return "upstream target was removed"
    if old.file != new.file:
        return f"upstream target moved: {old.file} -> {new.file}"
    if old.name != new.name:
        return f"upstream callable renamed: {old.name} -> {new.name}"
    if old.descriptor != new.descriptor:
        return f"descriptor changed: {old.descriptor} -> {new.descriptor}"
    if old.signature != new.signature:
        return "callable parameter contract changed"
    return "no exact callable contract delta"


def _suggestion(relation: str, classification: str, old: SourceEndpoint, new: SourceEndpoint) -> str:
    if classification == "preexisting":
        return "作为历史问题单独处理，不归因于本次上游升级。"
    if classification == "fixed":
        return "上游已经恢复兼容，确认下游兼容代码是否仍需保留。"
    if new.file is None:
        return "更新下游依赖目标；若上游已删除该能力，需要移除 patch/继承并补充替代实现。"
    if old.name != new.name:
        return f"把下游依赖从 {old.name} 更新到 {new.name}，并重新核对参数转发。"
    if relation == "monkey_patch":
        return "调整 replacement 签名，使其接受上游新调用方式，并确认 patch 安装路径仍生效。"
    if relation == "override":
        return "同步 override 参数并检查 super() 调用和关键字转发。"
    if relation == "inheritance":
        return "核对新基类路径和 MRO；不要在继承链不完整时猜测替代类。"
    if relation == "direct_import":
        return "更新 import 模块或符号路径，并补充导入边界测试。"
    return "根据上下游精确契约差异调整依赖，并补充接口级回归测试。"


def _finding_id(*parts: object) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _relation_finding(
    relation: Relation,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    new_to_old: dict[str, str],
) -> RangeFinding | None:
    old_file = new_to_old.get(relation.upstream_file, relation.upstream_file)
    old_endpoint = old_snapshot.endpoint(old_file, relation.upstream_owner, relation.upstream_name)
    new_endpoint = new_snapshot.endpoint(
        relation.upstream_file,
        relation.upstream_owner,
        relation.upstream_name,
    )
    if new_endpoint.file is None:
        old_tree = old_snapshot.tree(old_file)
        old_node = _named_node(old_tree, relation.upstream_owner, relation.upstream_name) if old_tree else None
        renamed = new_snapshot.unique_rename(
            relation.upstream_file,
            relation.upstream_owner,
            relation.upstream_name,
            _definition_fingerprint(old_node),
        )
        if renamed is not None:
            new_endpoint = renamed
    downstream = SourceEndpoint(
        file=relation.downstream_file,
        owner=relation.downstream_owner,
        name=relation.downstream_name,
        line=relation.evidence_line,
        signature=relation.downstream_signature,
        descriptor=relation.downstream_descriptor_kind,
    )
    old_state = _state(old_endpoint, relation.downstream_signature, relation.relation)
    new_state = _state(new_endpoint, relation.downstream_signature, relation.relation)
    contract_changed = (
        old_endpoint.file != new_endpoint.file
        or old_endpoint.name != new_endpoint.name
        or old_endpoint.signature != new_endpoint.signature
        or old_endpoint.descriptor != new_endpoint.descriptor
    )
    # The range report is a delta report. A proven relationship whose exact
    # upstream contract did not change is useful inventory, but it is not a
    # range risk and must not inflate unresolved or historical counts.
    if not contract_changed:
        return None
    classification = _classify(old_state, new_state, contract_changed)
    gates = {
        "relationship_verified": True,
        "contract_changed": contract_changed,
        "runtime_reachable": True,
        "version_lane_matches": True,
    }
    action = (
        "modify"
        if classification == "introduced_break" and all(gates.values())
        else ("dismiss" if classification in {"preexisting", "fixed"} else "review")
    )
    evidence = [item.as_dict() for item in relation.evidence] or [
        {"file": relation.evidence_file, "line": relation.evidence_line}
    ]
    return RangeFinding(
        finding_id=_finding_id(relation.exact_key(), old_snapshot.revision, new_snapshot.revision),
        classification=classification,
        relation=relation.relation,
        priority="P0"
        if relation.relation == "monkey_patch" and action == "modify"
        else ("P1" if action == "modify" else "P2"),
        action=action,
        confidence="high" if classification != "analysis_unresolved" else "medium",
        upstream_old=old_endpoint,
        upstream_new=new_endpoint,
        downstream=downstream,
        old_state=old_state,
        new_state=new_state,
        change=_change_text(old_endpoint, new_endpoint),
        evidence=evidence,
        gates=gates,
        suggestion=_suggestion(relation.relation, classification, old_endpoint, new_endpoint),
    )


@dataclass(frozen=True)
class ImportReference:
    module: str
    symbol: str | None
    file: str
    line: int


class _ImportVisitor(ast.NodeVisitor):
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.references: list[ImportReference] = []
        self._version_guard_depth = 0
        self._vllm_roots: set[str] = set()
        self._parents: dict[int, ast.AST] = {}

    def visit(self, node: ast.AST) -> Any:
        for child in ast.iter_child_nodes(node):
            self._parents[id(child)] = node
        return super().visit(node)

    @staticmethod
    def _is_version_guard(node: ast.AST) -> bool:
        return any(
            isinstance(item, ast.Call)
            and (
                isinstance(item.func, ast.Name)
                and item.func.id == "vllm_version_is"
                or isinstance(item.func, ast.Attribute)
                and item.func.attr == "vllm_version_is"
            )
            for item in ast.walk(node)
        )

    def visit_If(self, node: ast.If) -> None:
        guarded = self._is_version_guard(node.test)
        self._version_guard_depth += int(guarded)
        for item in (*node.body, *node.orelse):
            self.visit(item)
        self._version_guard_depth -= int(guarded)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._version_guard_depth == 0 and node.module and node.module.startswith("vllm"):
            for alias in node.names:
                if alias.name != "*":
                    self.references.append(ImportReference(node.module, alias.name, self.file_name, node.lineno))

    def visit_Import(self, node: ast.Import) -> None:
        if self._version_guard_depth:
            return
        for alias in node.names:
            if not alias.name.startswith("vllm"):
                continue
            self.references.append(ImportReference(alias.name, None, self.file_name, node.lineno))
            if alias.name == "vllm":
                self._vllm_roots.add(alias.asname or "vllm")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if self._version_guard_depth:
            return
        parent = self._parents.get(id(node))
        if isinstance(parent, ast.Attribute) and parent.value is node:
            return
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name) and current.id in self._vllm_roots:
            chain = ["vllm", *reversed(parts)]
            self.references.append(ImportReference(".".join(chain), None, self.file_name, node.lineno))
        self.generic_visit(node)


def discover_imports(ascend_root: Path) -> list[ImportReference]:
    references: list[ImportReference] = []
    for path in sorted((ascend_root / "vllm_ascend").rglob("*.py")):
        relative = path.relative_to(ascend_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError):
            continue
        visitor = _ImportVisitor(relative)
        visitor.visit(tree)
        references.extend(visitor.references)
    unique = {(item.module, item.symbol, item.file, item.line): item for item in references}
    ordered = sorted(
        unique,
        key=lambda item: (item[0], item[1] or "", item[2], item[3]),
    )
    return [unique[key] for key in ordered]


def _top_level_symbol(snapshot: GitSnapshot, file_name: str, name: str) -> SourceEndpoint:
    endpoint = snapshot.endpoint(file_name, None, name)
    if endpoint.line is not None:
        return endpoint
    tree = snapshot.tree(file_name)
    if tree is not None:
        for node in tree.body:
            names: list[str] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names = [item.id for target in targets for item in ast.walk(target) if isinstance(item, ast.Name)]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.asname or alias.name.rsplit(".", 1)[-1] for alias in node.names]
            if name in names:
                return SourceEndpoint(file=file_name, owner=None, name=name, line=node.lineno)
    return SourceEndpoint(file=None, owner=None, name=name)


def _import_findings(
    ascend_root: Path,
    old_snapshot: GitSnapshot,
    new_snapshot: GitSnapshot,
    old_to_new: dict[str, str],
) -> list[RangeFinding]:
    findings: list[RangeFinding] = []
    for reference in discover_imports(ascend_root):
        old_file = old_snapshot.resolve_module(reference.module)
        new_file = new_snapshot.resolve_module(reference.module)
        # For ``import vllm; vllm.a.b.symbol`` resolve the longest module prefix.
        symbol = reference.symbol
        if old_file is None and reference.symbol is None:
            parts = reference.module.split(".")
            for split in range(len(parts) - 1, 0, -1):
                old_file = old_snapshot.resolve_module(".".join(parts[:split]))
                if old_file is not None:
                    symbol = ".".join(parts[split:]) or None
                    new_file = new_snapshot.resolve_module(".".join(parts[:split]))
                    break
        if old_file is None:
            continue
        imported_submodule: str | None = None
        if symbol and "." not in symbol:
            imported_submodule = old_snapshot.resolve_module(f"{reference.module}.{symbol}")
            if imported_submodule is not None:
                old_file = imported_submodule
                new_file = new_snapshot.resolve_module(f"{reference.module}.{symbol}")
        moved_file = old_to_new.get(old_file)
        if imported_submodule is not None:
            old_endpoint = SourceEndpoint(file=old_file, owner=None, name=None)
            new_endpoint = SourceEndpoint(file=new_file, owner=None, name=None)
        elif symbol and "." not in symbol:
            old_endpoint = _top_level_symbol(old_snapshot, old_file, symbol)
            new_endpoint = (
                _top_level_symbol(new_snapshot, new_file, symbol)
                if new_file is not None
                else SourceEndpoint(file=None, owner=None, name=symbol)
            )
        else:
            old_endpoint = SourceEndpoint(file=old_file, owner=None, name=symbol)
            new_endpoint = SourceEndpoint(file=new_file, owner=None, name=symbol)
        # An import can only be attributed to this upgrade when its exact old
        # module or exported symbol was proven to resolve at the old endpoint.
        if old_endpoint.file is None:
            continue
        if new_endpoint.file is not None:
            continue
        relocated = moved_file is not None and moved_file in new_snapshot.files
        if relocated:
            new_endpoint = SourceEndpoint(file=moved_file, owner=None, name=symbol)
        old_state = CompatibilityState(True, True, "import target exists at old")
        new_state = CompatibilityState(False, False, "old import path no longer resolves at new")
        gates = {
            "relationship_verified": True,
            "contract_changed": True,
            "runtime_reachable": True,
            "version_lane_matches": True,
        }
        findings.append(
            RangeFinding(
                finding_id=_finding_id("import", reference, old_snapshot.revision, new_snapshot.revision),
                classification="introduced_break",
                relation="direct_import",
                priority="P1",
                action="modify",
                confidence="high",
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=SourceEndpoint(reference.file, None, reference.symbol or reference.module, reference.line),
                old_state=old_state,
                new_state=new_state,
                change=(
                    f"import module moved: {old_file} -> {moved_file}"
                    if relocated
                    else "import module or symbol was removed"
                ),
                evidence=[
                    {
                        "file": reference.file,
                        "line": reference.line,
                        "import": reference.module,
                        "symbol": reference.symbol,
                    }
                ],
                gates=gates,
                suggestion=_suggestion("direct_import", "introduced_break", old_endpoint, new_endpoint),
                source="direct_import_detector",
            )
        )
    return findings


def analyze_range(
    *,
    vllm_root: Path,
    ascend_root: Path,
    old: str,
    new: str,
    expect_ascend_sha: str,
    external_roots: dict[str, Path] | None = None,
    external_shas: dict[str, str] | None = None,
    profile: str = "exact-contracts",
) -> dict[str, Any]:
    if profile not in {"exact-contracts", "expanded"}:
        raise ValueError(f"unsupported profile: {profile}")
    old_sha, new_sha = verify_range(vllm_root, old, new)
    verify_head("vLLM new", vllm_root, new_sha)
    ascend_sha = verify_head("vllm-ascend", ascend_root, expect_ascend_sha)
    external_roots = external_roots or {}
    external_shas = external_shas or {}
    if set(external_roots) != set(external_shas):
        raise ValueError("external roots and SHAs must name the same packages")
    for package, root in external_roots.items():
        verify_head(f"external {package}", root, external_shas[package])

    generator = InterfaceBoundaryGenerator(
        vllm_root,
        ascend_root,
        external_roots,
        source_versions={"vllm": new_sha, "vllm_ascend": ascend_sha, **external_shas},
    )
    relations, generator_findings = generator.generate()
    old_snapshot = GitSnapshot(vllm_root, old_sha)
    new_snapshot = GitSnapshot(vllm_root, new_sha)
    old_to_new, new_to_old = _rename_maps(vllm_root, old_sha, new_sha)

    findings = [
        finding
        for relation in relations
        if relation.upstream_package == "vllm"
        if (finding := _relation_finding(relation, old_snapshot, new_snapshot, new_to_old)) is not None
    ]
    findings.extend(_import_findings(ascend_root, old_snapshot, new_snapshot, old_to_new))

    for candidate in generator_findings:
        if candidate.status not in {"review", "risk"}:
            continue
        old_endpoint = old_snapshot.expression_endpoint(candidate.target_expression)
        new_endpoint = new_snapshot.expression_endpoint(candidate.target_expression)
        old_endpoint = old_endpoint or SourceEndpoint(None, None, candidate.target_expression)
        new_endpoint = new_endpoint or SourceEndpoint(None, None, candidate.target_expression)
        old_exists = old_endpoint.file is not None and (old_endpoint.name is None or old_endpoint.line is not None)
        new_exists = new_endpoint.file is not None and (new_endpoint.name is None or new_endpoint.line is not None)
        verified_removal = (
            old_exists and not new_exists and candidate.status == "risk" and not candidate.generator_issue
        )
        classification = "introduced_break" if verified_removal else "analysis_unresolved"
        old_state = CompatibilityState(
            old_exists,
            True if verified_removal else None,
            "upstream target exists at old" if verified_removal else candidate.reason,
        )
        new_state = CompatibilityState(
            new_exists,
            False if verified_removal else None,
            "upstream target was removed at new" if verified_removal else candidate.reason,
        )
        gates = {
            "relationship_verified": not candidate.generator_issue,
            "contract_changed": verified_removal,
            "runtime_reachable": verified_removal,
            "version_lane_matches": True,
        }
        findings.append(
            RangeFinding(
                finding_id=_finding_id(candidate.as_dict(), old_sha, new_sha),
                classification=classification,
                relation=candidate.relation,
                priority=(
                    "P0"
                    if verified_removal and candidate.relation == "monkey_patch"
                    else "P1"
                    if verified_removal
                    else "P2"
                ),
                action="modify" if verified_removal else "review",
                confidence=("high" if verified_removal else "low" if candidate.generator_issue else "medium"),
                upstream_old=old_endpoint,
                upstream_new=new_endpoint,
                downstream=SourceEndpoint(
                    candidate.downstream_file,
                    candidate.downstream_owner,
                    candidate.downstream_name,
                    candidate.evidence_line,
                ),
                old_state=old_state,
                new_state=new_state,
                change=(
                    "upstream target existed at old and was removed at new" if verified_removal else candidate.reason
                ),
                evidence=[candidate.as_dict()["evidence"]],
                gates=gates,
                suggestion=(
                    _suggestion(
                        candidate.relation,
                        classification,
                        old_endpoint,
                        new_endpoint,
                    )
                    if verified_removal
                    else "静态证据不足，先人工确认目标绑定或补充分析规则，不要直接修改下游代码。"
                ),
                source="generator_finding",
            )
        )

    deduplicated = {item.finding_id: item for item in findings}
    ordered = sorted(
        deduplicated.values(),
        key=lambda item: (
            CLASSIFICATIONS.index(item.classification),
            item.relation,
            item.downstream.file or "",
            item.downstream.line or 0,
            item.finding_id,
        ),
    )
    counts = Counter(item.classification for item in ordered)
    return {
        "schema_version": RANGE_SCHEMA_VERSION,
        "metadata": {
            "range_analyzer_version": RANGE_ANALYZER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "profile": profile,
            "vllm_old_sha": old_sha,
            "vllm_new_sha": new_sha,
            "vllm_ascend_sha": ascend_sha,
            "external_sources": dict(sorted(external_shas.items())),
        },
        "summary": {
            "relations": len(relations),
            "generator_findings": len(generator_findings),
            "total": len(ordered),
            **{name: counts[name] for name in CLASSIFICATIONS},
        },
        "findings": [item.as_dict() for item in ordered],
    }


def _csv_rows(findings: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for item in findings:
        old = item["upstream"]["old"]
        new = item["upstream"]["new"]
        downstream = item["downstream"]
        yield {
            "classification": item["classification"],
            "priority": item["priority"],
            "action": item["action"],
            "relation": item["relation"],
            "upstream_old": ":".join(str(value or "") for value in (old["file"], old["owner"], old["name"])),
            "upstream_new": ":".join(str(value or "") for value in (new["file"], new["owner"], new["name"])),
            "downstream": ":".join(
                str(value or "") for value in (downstream["file"], downstream["owner"], downstream["name"])
            ),
            "downstream_line": downstream["line"],
            "change": item["change"],
            "old_compatible": item["compatibility"]["old"]["compatible"],
            "new_compatible": item["compatibility"]["new"]["compatible"],
            "confidence": item["confidence"],
            "suggestion": item["suggestion"],
        }


def _write_csv(path: Path, findings: list[dict[str, Any]]) -> None:
    rows = list(_csv_rows(findings))
    fields = (
        list(rows[0])
        if rows
        else [
            "classification",
            "priority",
            "action",
            "relation",
            "upstream_old",
            "upstream_new",
            "downstream",
            "downstream_line",
            "change",
            "old_compatible",
            "new_compatible",
            "confidence",
            "suggestion",
        ]
    )
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _markdown(report: dict[str, Any]) -> str:
    meta = report["metadata"]
    summary = report["summary"]
    lines = [
        "# vLLM main2main 接口兼容报告",
        "",
        f"- vLLM 区间：`{meta['vllm_old_sha']}` → `{meta['vllm_new_sha']}`",
        f"- vllm-ascend 基线：`{meta['vllm_ascend_sha']}`",
        f"- 本次升级引入：{summary['introduced_break']}",
        f"- 兼容性提醒：{summary['compatibility_warning']}",
        f"- 历史问题：{summary['preexisting']}",
        f"- 无法静态确认：{summary['analysis_unresolved']}",
        "",
        "## 本次升级需要处理",
        "",
    ]
    introduced = [item for item in report["findings"] if item["classification"] == "introduced_break"]
    if not introduced:
        lines.append("没有发现能够确认由本次区间引入的接口 break。")
    for item in introduced:
        downstream = item["downstream"]
        lines.extend(
            [
                f"### {item['relation']}：{downstream['file']}:{downstream['line'] or ''}",
                "",
                f"- 变化：{item['change']}",
                f"- 下游接口：`{downstream['owner'] or ''}.{downstream['name'] or ''}`",
                f"- 建议：{item['suggestion']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 说明",
            "",
            "`preexisting` 表示 old 和 new 都不兼容，不归因于这次升级；"
            "`analysis_unresolved` 表示证据不足，脚本没有猜测。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(report: dict[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "main2main-range-report.json"
    all_csv = output_dir / "main2main-all-findings.csv"
    introduced_csv = output_dir / "main2main-introduced-breaks.csv"
    markdown_path = output_dir / "main2main-range-report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(all_csv, report["findings"])
    _write_csv(
        introduced_csv,
        [item for item in report["findings"] if item["classification"] == "introduced_break"],
    )
    markdown_path.write_text(_markdown(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "all_csv": str(all_csv),
        "introduced_csv": str(introduced_csv),
        "markdown": str(markdown_path),
    }
