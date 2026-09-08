"""Container element field/call contracts and evidence-path counterexamples."""

from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _component_statuses, _run, _run_with_cache


@pytest.mark.parametrize(
    "annotation,body",
    [
        ("list[Tensor]", "    for tensor in config.tensors:\n        print(tensor.shared_by)\n"),
        ("list[Tensor]", "    values = config.tensors\n    for tensor in values:\n        print(tensor.shared_by)\n"),
        ("list[Tensor]", "    return [tensor.shared_by for tensor in config.tensors]\n"),
        ("list[Tensor]", "    for index, tensor in enumerate(config.tensors):\n        print(tensor.shared_by)\n"),
        ("list[Tensor]", "    tensor = config.tensors[0]\n    return tensor.shared_by\n"),
        ("list[Tensor]", "    values = [t for t in config.tensors]\n    return [t.shared_by for t in values]\n"),
        ("dict[str, Tensor]", "    for tensor in config.tensors.values():\n        print(tensor.shared_by)\n"),
        ("dict[str, Tensor]", "    for name, tensor in config.tensors.items():\n        print(tensor.shared_by)\n"),
        ("tuple[Tensor, ...]", "    return [tensor.shared_by for tensor in config.tensors]\n"),
        (
            "list[Tensor]",
            "    values = config.tensors\n    values: list[int]\n    return [t.shared_by for t in values]\n",
        ),
        ("list[Tensor]", "    for tensor in config.tensors:\n        tensor.shared_by += ['layer']\n"),
    ],
)
def test_removed_container_element_field(tmp_path: Path, annotation: str, body: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = f"\n@dataclass\nclass Config:\n    tensors: {annotation}\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n" + body,
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["action"] == "modify"
    assert findings[0]["details"]["resolution_basis"] == "typed_container_flow"
    assert findings[0]["evidence"][0]["receiver_path"][0] == "vllm.api.Config"


@pytest.mark.parametrize(
    "body",
    [
        "    for tensor in config.tensors:\n        tensor = object()\n        print(tensor.shared_by)\n",
        "    values = config.tensors\n    if flag:\n        values = unknown()\n"
        "    for t in values:\n        print(t.shared_by)\n",
        "    values = config.tensors\n    values.append(unknown())\n    for t in values:\n        print(t.shared_by)\n",
        "    values = config.tensors\n    mutate(values)\n    for t in values:\n        print(t.shared_by)\n",
        "    config.tensors = unknown()\n    for t in config.tensors:\n        print(t.shared_by)\n",
        "    for t in config.tensors:\n        pass\n    print(t.shared_by)\n",
        "    values = [t for t in config.tensors]\n    print(t.shared_by)\n",
        "    if False:\n        for t in config.tensors:\n            print(t.shared_by)\n",
        "    values = config.tensors\n    try:\n        mutate(values)\n    except Exception:\n        pass\n"
        "    for t in values:\n        print(t.shared_by)\n",
        "    return [t.shared_by for t in config.tensors if False]\n",
        "    values = config.tensors\n    _ = mutate(values) if flag else None\n"
        "    for t in values:\n        print(t.shared_by)\n",
        "    values = config.tensors\n    _ = [values.clear() for _ in range(1)]\n"
        "    for t in values:\n        print(t.shared_by)\n",
        "    for t in config.tensors:\n        break\n        print(t.shared_by)\n",
        "    for t in config.tensors:\n        continue\n        print(t.shared_by)\n",
        "    if True:\n        return\n    for t in config.tensors:\n        print(t.shared_by)\n",
        "    for t in config.tensors:\n        False and t.shared_by\n",
        "    for t in config.tensors:\n        t.shared_by if False else None\n",
        "    values = config.tensors\n    _ = [t for t in values if mutate(values) if False]\n"
        "    for t in values:\n        print(t.shared_by)\n",
    ],
)
def test_container_flow_does_not_guess_after_unknown_writes(tmp_path: Path, body: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "\n@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config, flag):\n" + body,
    )
    assert not any(
        f["action"] == "modify" and f["details"].get("member") == "shared_by" for f in _run(*roots)["findings"]
    )


def test_container_owner_is_resolved_independently_at_each_sha(tmp_path: Path):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Old:\n    shared_by: int\n"
    old = prefix + "@dataclass\nclass New:\n    other: int\n@dataclass\nclass Config:\n    tensors: list[Old]\n"
    new = prefix + "@dataclass\nclass New:\n    shared_by: int\n@dataclass\nclass Config:\n    tensors: list[New]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=old,
        new_source=new,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    return [t.shared_by for t in config.tensors]\n",
    )
    # Both actual element owners retain the field; resolving the new owner in
    # the old snapshot would incorrectly invent an old missing/new present delta.
    assert not any(f["details"].get("member") == "shared_by" for f in _run(*roots)["findings"])


def test_container_element_method_removal(tmp_path: Path):
    config = "class Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source="class Tensor:\n    def run(self): pass\n" + config,
        new_source="class Tensor:\n    def renamed(self): pass\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    for tensor in config.tensors:\n        tensor.run()\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["relation"] == "direct_call"]
    assert any(f["classification"] == "introduced_break" and f["action"] == "modify" for f in findings)


def test_comprehension_element_shadows_same_named_parameter(tmp_path: Path):
    common = (
        "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n    shared_by: int\n"
        "@dataclass\nclass Config:\n    tensors: list[Tensor]\n@dataclass\nclass Other:\n"
    )
    roots = _call_repositories(
        tmp_path,
        old_source=common + "    shared_by: int\n",
        new_source=common + "    changed: int\n",
        consumer_source="from vllm.api import Config, Other\ndef consume(config: Config, tensor: Other):\n"
        "    return [tensor.shared_by for tensor in config.tensors]\n",
    )
    assert not any(f["action"] == "modify" for f in _run(*roots)["findings"])


def test_unknown_old_container_path_stays_unresolved(tmp_path: Path):
    tensor = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n    layers: int\n"
    roots = _call_repositories(
        tmp_path,
        old_source=tensor + "@dataclass\nclass Config:\n    other: int\n",
        new_source=tensor + "@dataclass\nclass Config:\n    tensors: list[Tensor]\n",
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    return [t.shared_by for t in config.tensors]\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert len(findings) == 1
    assert findings[0]["classification"] == "analysis_unresolved"
    assert findings[0]["action"] != "modify"


def test_container_path_survives_persistent_cache_roundtrip(tmp_path: Path):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: int\n" + config,
        new_source=prefix + "    changed: int\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    return [t.shared_by for t in config.tensors]\n",
    )
    cold = _run_with_cache(roots, tmp_path / "cache")
    hot = _run_with_cache(roots, tmp_path / "cache")
    assert _component_statuses(cold, "downstream_direct_attributes") == ["miss"]
    assert _component_statuses(hot, "downstream_direct_attributes") == ["hit"]
    assert hot["findings"] == cold["findings"]
    assert any(f["action"] == "modify" and f["details"].get("receiver_path") for f in hot["findings"])


@pytest.mark.parametrize("argument", ["tensor.size", "size=tensor.size", "tensor.shared_by"])
def test_helper_primitive_argument_does_not_invalidate_parent(tmp_path: Path, argument: str):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n    size: int\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\ndef consume(config: Config):\n"
        "    for tensor in config.tensors:\n"
        f"        helper({argument})\n"
        "        print(tensor.shared_by)\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    # Mutable/unknown values remain barriers; their own argument read is still
    # a valid dependency, but the later read must not borrow the old binding.
    assert len(findings) == 1
    expected_line = 4 if argument == "tensor.shared_by" else 5
    assert findings[0]["downstream"]["line"] == expected_line
    assert findings[0]["action"] == "modify"
