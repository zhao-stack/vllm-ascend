"""Read-only helper evidence and mutations through borrowed return values."""

from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _run


def _report(tmp_path: Path, helper: str, call: str = "view = helper(config)", after: str = ""):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\n" + helper + "\ndef consume(config: Config):\n"
        f"    {call}\n" + after + "    return [t.shared_by for t in config.tensors]\n",
    )
    return [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]


@pytest.mark.parametrize(
    "helper",
    [
        "def helper(value):\n    return len(value.tensors)\n",
        "def helper(value):\n    alias = value.tensors\n    return alias\n",
        "def helper(value):\n    result = {}\n    for t in value.tensors:\n        result[0] = t\n    return result\n",
        "def helper(value):\n    return [t for t in value.tensors]\n",
    ],
)
def test_read_only_helper_preserves_parameter_type(tmp_path: Path, helper: str):
    findings = _report(tmp_path, helper)
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"


@pytest.mark.parametrize(
    "helper,after",
    [
        ("def helper(value):\n    value.tensors.clear()\n", ""),
        ("def helper(value):\n    x = value.tensors\n    x.append(object())\n", ""),
        ("def helper(value):\n    unknown(value)\n", ""),
        ("def helper(value):\n    global stored\n    stored = value\n", ""),
        ("def helper(value):\n    return value.tensors\n", "    view.clear()\n"),
        (
            "def helper(value):\n    result = {}\n    for t in value.tensors:\n"
            "        result[0] = t\n    return result\n",
            "    mutate(view[0])\n",
        ),
        ("@unknown_decorator\ndef helper(value):\n    return len(value.tensors)\n", ""),
        ("def helper(value):\n    return value.tensors\nhelper = unknown()\n", ""),
    ],
)
def test_unknown_or_mutating_helpers_do_not_preserve_type(tmp_path: Path, helper: str, after: str):
    assert not any(f["action"] == "modify" for f in _report(tmp_path, helper, after=after))


def test_helper_effects_bind_keywords_and_distinguish_parameters(tmp_path: Path):
    helper = "def helper(output, value):\n    output.clear()\n    return len(value.tensors)\n"
    findings = _report(tmp_path, helper, "view = helper(value=config, output=[])")
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"


def test_parameter_rebinding_of_helper_name_cannot_use_module_proof(tmp_path: Path):
    helper = "def helper(value):\n    return len(value.tensors)\n"
    assert not _report(tmp_path, helper, "helper = unknown(); view = helper(config)")


@pytest.mark.parametrize("mutate", [False, True])
def test_group_spec_helper_tracks_read_only_mapping_and_borrowed_values(tmp_path: Path, mutate: bool):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = (
        "@dataclass\nclass Spec: pass\n@dataclass\nclass Uniform(Spec):\n    specs: dict[str, Spec]\n"
        "@dataclass\nclass Group:\n    spec: Spec\n    names: list[str]\n"
        "@dataclass\nclass Config:\n    tensors: list[Tensor]\n    groups: list[Group]\n"
    )
    helper = (
        "from vllm.api import Config, Uniform, Spec\ndef helper(config):\n"
        "    result = {}\n    for group in config.groups:\n        spec = group.spec\n"
        "        for name in group.names:\n            if isinstance(spec, Uniform):\n"
        "                result[name] = spec.specs[name]\n            else:\n                result[name] = spec\n"
        "    return result\ndef consume(config: Config):\n    specs = helper(config)\n"
        "    has_spec = any(isinstance(spec, Spec) for spec in specs.values())\n"
    )
    if mutate:
        helper += "    mutate_borrowed(specs['layer'])\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source=helper + "    return [t.shared_by for t in config.tensors]\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert bool(findings) is not mutate


@pytest.mark.parametrize(
    "helper",
    [
        "def helper(value):\n    result = {}\n    alias = result\n    alias[0] = value\n    unknown(result)\n",
        "def helper(value):\n    result = [value]\n    unknown(result[0])\n",
        "def helper(value):\n    for t in value.tensors:\n        t.layers = []\n",
        "def helper(value):\n    return unknown(value)\n",
        "def helper(value):\n    return len(value)\n",
    ],
)
def test_alias_escape_or_implicit_protocol_is_not_read_only(tmp_path: Path, helper: str):
    assert not any(f["action"] == "modify" for f in _report(tmp_path, helper))


def test_helper_return_owner_is_not_borrowed_from_new_for_old(tmp_path: Path):
    common = "from dataclasses import dataclass\n@dataclass\nclass OldTensor:\n    other: int\n"
    roots = _call_repositories(
        tmp_path,
        old_source=common + "@dataclass\nclass NewTensor:\n    shared_by: int\n"
        "@dataclass\nclass Config:\n    tensors: list[OldTensor]\n",
        new_source=common + "@dataclass\nclass NewTensor:\n    layers: int\n"
        "@dataclass\nclass Config:\n    tensors: list[NewTensor]\n",
        consumer_source="from vllm.api import Config\ndef helper(config):\n    return config.tensors\n"
        "def consume(config: Config):\n    tensors = helper(config)\n    return [t.shared_by for t in tensors]\n",
    )
    # Old actually returns OldTensor, which already lacks shared_by. Using the
    # new inferred NewTensor owner in old would invent an introduced break.
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert not any(f["classification"] == "introduced_break" for f in findings)


@pytest.mark.parametrize("returned", ["config.tensors", "[t for t in config.tensors]"])
def test_helper_return_alias_retains_source_path(tmp_path: Path, returned: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\ndef helper(config):\n"
        f"    return {returned}\n"
        "def consume(config: Config):\n    tensors = helper(config)\n"
        "    return [t.shared_by for t in tensors]\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"
    assert findings[0]["evidence"][0]["receiver_path"][0] == "vllm.api.Config"


def test_custom_instance_check_is_not_a_read_only_builtin(tmp_path: Path):
    helper = (
        "class Meta(type):\n    def __instancecheck__(cls, value):\n"
        "        value.tensors.clear()\n        return True\n"
        "class Check(metaclass=Meta): pass\n"
        "def helper(value):\n    return len(value.tensors)\n"
    )
    assert not _report(tmp_path, helper, after="    isinstance(config, Check)\n")


def test_local_class_alias_cannot_borrow_module_instance_check_proof(tmp_path: Path):
    helper = "class Check: pass\ndef helper(value):\n    return len(value.tensors)\n"
    assert not _report(tmp_path, helper, after="    Check = unknown()\n    isinstance(config, Check)\n")


def test_mutated_parameter_return_cannot_restore_invalidated_type(tmp_path: Path):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\ndef helper(config):\n"
        "    config.tensors.clear()\n    return config.tensors\n"
        "def consume(config: Config):\n    tensors = helper(config)\n"
        "    return [t.shared_by for t in tensors]\n",
    )
    assert not any(
        f["action"] == "modify" and f["details"].get("member") == "shared_by" for f in _run(*roots)["findings"]
    )
