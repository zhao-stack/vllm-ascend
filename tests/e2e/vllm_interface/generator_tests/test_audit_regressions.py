"""Independent contract regressions discovered by held-out upgrade audits."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _commit, _git, _run, _write

from tools.vllm_interface_contracts.condition_contracts import conditional_state_evidence


def actionable(report):
    return [item for item in report["findings"] if item["action"] == "modify"]


def test_new_keyword_has_independent_witness_despite_reordered_adapter(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="def build(a, b): pass\n",
        new_source="def build(a, b, lookahead=0): pass\n",
        consumer_source="import vllm.api as api\ndef patched(b, a): pass\napi.build = patched\n",
    )
    report = _run(*roots)
    historical = [f for f in report["findings"] if f["classification"] == "preexisting"]
    # Reordering alone is not represented as a positional semantic mismatch by
    # every signature oracle. Require the independently provable delta either way.
    matches = [f for f in actionable(report) if f["contract_kind"] == "call_arguments"]
    assert matches
    if historical:
        assert matches[0]["details"]["historical_finding_id"] == historical[0]["id"]


def test_historical_missing_required_argument_is_not_wholesale_promoted(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="def build(a, old_required): pass\n",
        new_source="def build(a, old_required, lookahead=0): pass\n",
        consumer_source="import vllm.api as api\ndef patched(a): pass\napi.build = patched\n",
    )
    assert not actionable(_run(*roots))


def test_local_factory_proves_new_constructor_alignment_despite_extra_context(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self, config): pass\n",
        new_source="class Base:\n    def __init__(self, config, varlen_decode=False): pass\n",
        consumer_source=(
            "from vllm.api import Base\nclass Child(Base):\n"
            "    def __init__(self, config, model_runner): pass\n"
            "def factory(config, runner): return Child(config, runner)\n"
        ),
    )
    report = _run(*roots)
    matches = [f for f in actionable(report) if f["relation"] == "override"]
    assert len(matches) == 1
    assert matches[0]["details"]["delta_witness"]["kind"] == "local_constructor_interface_alignment"
    assert any(f["classification"] == "preexisting" for f in report["findings"])


def test_factory_and_constructor_override_share_parameter_root(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def __init__(self, config): pass\n",
        new_source="class Base:\n    def __init__(self, config, varlen_decode=False): pass\n",
        consumer_source=(
            "import vllm.api as api\nclass Child(api.Base):\n"
            "    def __init__(self, config, runner): pass\n"
            "def factory(config): return Child(config, object())\napi.Base = factory\n"
        ),
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "call_arguments"]
    assert {f["relation"] for f in matches} == {"override", "monkey_patch"}
    assert len({f["root_cause_id"] for f in matches}) == 1


@pytest.mark.parametrize("assignment", ["count = 0", "aligned = count - count", "aligned = count % count"])
def test_constant_false_numeric_condition_is_not_activation_evidence(assignment):
    variable = "count" if assignment.startswith("count =") else "aligned"
    source = (
        f"def cache(self, count: int):\n    {assignment}\n    for manager in self.managers:\n"
        f"        if manager.enabled and {variable} > 0:\n            return self.lookahead\n"
    )
    initializer = ast.parse(
        "def __init__(self, managers):\n    self.managers = managers\n    self.managers[0].enabled = True\n"
    ).body[0]
    function = ast.parse(source).body[0]
    assert conditional_state_evidence(function, initializer, "lookahead", 5) == []


@pytest.mark.parametrize("block, expected", [("2", True), ("0", False)])
def test_aligned_numeric_condition_retains_supported_configuration(block, expected):
    function = ast.parse(
        "def cache(self, count: int):\n    aligned = count // self.block * self.block\n"
        "    for manager in self.managers:\n        if manager.enabled and aligned > 0:\n"
        "            return self.lookahead\n"
    ).body[0]
    initializer = ast.parse(
        f"def __init__(self, managers):\n    self.managers = managers\n    self.block = {block}\n"
        "    self.managers[0].enabled = True\n"
    ).body[0]
    assert bool(conditional_state_evidence(function, initializer, "lookahead", 5)) is expected


@pytest.mark.parametrize("call_helper,early_return", [(True, False), (False, False), (True, True)])
def test_conditional_state_follows_only_a_proven_initializer_helper(tmp_path, call_helper, early_return):
    base = "class Base:\n    def __init__(self):\n        self.lookahead = 0\n"
    roots = _call_repositories(
        tmp_path,
        old_source=base + "    def cache(self, count: int): pass\n",
        new_source=base + "    def cache(self, count: int):\n        for manager in self.managers:\n"
        "            if manager.enabled and count > 0:\n                count -= self.lookahead\n",
        consumer_source="from vllm.api import Base\nclass Child(Base):\n"
        "    def __init__(self, managers):\n        self.managers = managers\n"
        + ("        self.activate()\n" if call_helper else "")
        + "    def activate(self):\n"
        + ("        return\n" if early_return else "")
        + "        self.managers[0].enabled = True\n",
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "required_instance_attribute"]
    assert bool(matches) is (call_helper and not early_return)
    if matches:
        assert matches[0]["details"]["condition_evidence"][0]["initialization_helper_calls"]


@pytest.mark.parametrize("class_prefix", ["", "(metaclass=Meta)"])
def test_class_replaced_by_scoped_factory_uses_constructor_protocol(tmp_path: Path, class_prefix: str):
    source = "class Meta(type): pass\nclass Manager" + class_prefix + ":\n    def __init__(self, config{extra}): pass\n"
    roots = _call_repositories(
        tmp_path,
        old_source=source.format(extra=""),
        new_source=source.format(extra=", varlen_decode=False"),
        consumer_source=(
            "import vllm.api as api\nfrom contextlib import contextmanager\n"
            "@contextmanager\ndef wrapper():\n"
            "    original = api.Manager\n"
            "    def factory(config): return object()\n"
            "    try:\n        api.Manager = factory\n        yield\n"
            "    finally:\n        api.Manager = original\n"
        ),
    )
    matches = [f for f in actionable(_run(*roots)) if f["relation"] == "monkey_patch"]
    assert bool(matches) is (not class_prefix)
    if matches:
        assert matches[0]["upstream"]["new"]["symbol_kind"] == "constructor"


def test_call_and_override_parameter_change_share_root(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="class Base:\n    def run(self, value, removed): pass\n",
        new_source="class Base:\n    def run(self, value): pass\n",
        consumer_source=(
            "from vllm.api import Base\nclass Child(Base):\n"
            "    def run(self, value, removed):\n"
            "        return super().run(value, removed=removed)\n"
        ),
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "call_arguments"]
    assert {f["relation"] for f in matches} == {"direct_call", "override"}
    assert len({f["root_cause_id"] for f in matches}) == 1


def test_moved_import_and_call_share_root_without_parent_module_misattribution(tmp_path: Path):
    roots = _call_repositories(
        tmp_path,
        old_source="def gather(value): return value\n",
        new_source="def gather(value): return value\n# changed\n",
        consumer_source="from vllm.api import gather\ndef run(value): return gather(value)\n",
    )
    upstream, downstream, old, _, baseline = roots
    _write(upstream, "vllm/attention/__init__.py", "")
    _git(upstream, "mv", "vllm/api.py", "vllm/attention/pcp.py")
    new = _commit(upstream, "relocate callable module")
    matches = actionable(_run(upstream, downstream, old, new, baseline))
    assert {item["relation"] for item in matches} == {"direct_call", "direct_import"}
    assert len({item["root_cause_id"] for item in matches}) == 1
    call = next(item for item in matches if item["relation"] == "direct_call")
    assert call["upstream"]["new"]["file"] == "vllm/api.py"
    assert call["upstream"]["new"]["symbol_kind"] == "missing"
    assert call["details"]["relocation_destination"]["file"] == "vllm/attention/pcp.py"


@pytest.mark.parametrize("dynamic", [False, True])
def test_free_function_method_patch_reads_removed_literal_buffer(tmp_path: Path, dynamic: bool):
    common = "from torch import nn\nclass Norm(nn.Module):\n    def __init__(self, flag):\n        super().__init__()\n"
    old = common + (
        "        if flag:\n            self.register_buffer('mean', None)\n"
        "        else:\n            self.register_buffer('mean', None)\n"
        "    def forward(self, x): return x\n"
    )
    new = (
        common
        + (
            "        self.register_buffer(name, None)\n"
            if dynamic
            else "        self.register_buffer('weight', None)\n"
        )
        + "    def forward(self, x): return x\n"
    )
    roots = _call_repositories(
        tmp_path,
        old_source=old,
        new_source=new,
        consumer_source=(
            "from vllm.api import Norm\ndef patched(self, x): return x - self.mean\nNorm.forward = patched\n"
        ),
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "attribute_presence"]
    assert bool(matches) is (not dynamic)
    if matches:
        assert matches[0]["details"]["resolution_basis"] == "verified_method_patch_receiver"
        assert matches[0]["details"]["registered_buffer_evidence"] is True


@pytest.mark.parametrize(
    "child_field, decorator, expected",
    [
        ("values: int", "dataclass", True),
        ("values: int = 0", "dataclass", False),
        ("values: int", "dataclass(kw_only=True)", False),
        ("values: int = field(init=False)", "dataclass", False),
        ("values: int", "dataclass(init=False)", False),
        ("values: ClassVar[int]", "dataclass", False),
        ("values: InitVar[int]", "dataclass", True),
    ],
)
def test_upstream_dataclass_default_invalidates_child_definition(tmp_path: Path, child_field, decorator, expected):
    roots = _call_repositories(
        tmp_path,
        old_source="from dataclasses import dataclass\n@dataclass\nclass Batch:\n    tokens: int\n",
        new_source=(
            "from dataclasses import dataclass\n@dataclass\nclass Batch:\n    tokens: int\n    maximum: int = 0\n"
        ),
        consumer_source=(
            "from dataclasses import dataclass, field, InitVar\nfrom typing import ClassVar\n"
            f"from vllm.api import Batch\n@{decorator}\nclass Child(Batch):\n    {child_field}\n"
        ),
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "dataclass_field_order"]
    assert bool(matches) is expected


@pytest.mark.parametrize(
    "enabled,condition,expected",
    [
        ("True", "manager.enabled and count > 0", True),
        ("False", "manager.enabled and count > 0", False),
        ("flag", "manager.enabled and count > 0", False),
        ("True", "False and manager.enabled and count > 0", False),
        ("True", "manager.enabled and not manager.enabled", False),
    ],
)
def test_conditional_inherited_field_requires_activation_evidence(tmp_path: Path, enabled, condition, expected):
    base = "class Base:\n    def __init__(self):\n        self.managers = []\n        self.lookahead = 0\n"
    roots = _call_repositories(
        tmp_path,
        old_source=base + "    def cache(self, count: int): return count\n",
        new_source=base
        + f"    def cache(self, count: int):\n        for manager in self.managers:\n            if {condition}:\n"
        "                count -= self.lookahead\n",
        consumer_source="from vllm.api import Base\nclass Child(Base):\n    def __init__(self, managers, flag=True):\n"
        "        self.managers = managers\n        self.managers[0].enabled = " + enabled + "\n",
    )
    matches = [f for f in actionable(_run(*roots)) if f["contract_kind"] == "required_instance_attribute"]
    assert bool(matches) is expected
    if expected:
        assert matches[0]["details"]["read_condition"] == "supported_conditional"
        assert matches[0]["details"]["condition_evidence"]
