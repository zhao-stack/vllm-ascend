"""Dictionary queries retain parameter evidence without hiding borrowed aliases."""

from pathlib import Path

import pytest
from test_helper_effects import _report as _helper_report
from test_range_analysis import _call_repositories, _run


def _report(tmp_path: Path, body: str, extra: str = "", annotation: str = "dict[str, Spec]"):
    prefix = "from dataclasses import dataclass\nfrom typing import Mapping\n@dataclass\nclass Tensor:\n"
    config = (
        "@dataclass\nclass Spec: pass\n@dataclass\nclass Uniform(Spec):\n    specs: dict[str, Spec]\n"
        + extra
        + "@dataclass\nclass Config:\n    tensors: list[Tensor]\n    spec: Spec\n"
        f"    mapping: {annotation}\n"
    )
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config, Uniform\ndef consume(config: Config, key):\n"
        + body
        + "    return [t.shared_by for t in config.tensors]\n",
    )
    return [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]


@pytest.mark.parametrize(
    "body",
    [
        "    result = config.mapping.get('key', config.spec)\n",
        "    result = config.mapping.get('key')\n",
        "    mapping = {}\n    result = mapping.get('key', config.spec)\n",
        "    spec = config.spec\n    mapping = spec.specs if isinstance(spec, Uniform) else {}\n"
        "    result = mapping.get('key', spec)\n",
    ],
)
def test_builtin_dictionary_get_preserves_configuration(tmp_path: Path, body: str):
    findings = _report(tmp_path, body)
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"


@pytest.mark.parametrize(
    "body,extra,annotation",
    [
        ("    result = config.mapping.get('key', config.spec)\n", "", "Mapping[str, Spec]"),
        (
            "    result = config.mapping.get('key', config.spec)\n",
            "class Custom:\n    def get(self, key, default):\n        mutate(default)\n",
            "Custom",
        ),
        (
            "    result = config.mapping.get('key', config.spec)\n",
            "class Custom(dict):\n    def get(self, key, default):\n        mutate(default)\n",
            "Custom",
        ),
        ("    result = config.mapping.get(key, config.spec)\n", "", "dict[str, Spec]"),
        (
            "    result = config.mapping.get('key', config.spec)\n",
            "class Key:\n    def __hash__(self):\n        mutate(self)\n        return 1\n",
            "dict[Key, Spec]",
        ),
        ("    result = config.mapping.get(key='key', default=config.spec)\n", "", "dict[str, Spec]"),
        ("    result = config.mapping.get('key', config.spec)\n    mutate(result)\n", "", "dict[str, Spec]"),
        ("    mapping = {}\n    result = mapping.get('key', config.spec)\n    mutate(result)\n", "", "dict[str, Spec]"),
    ],
)
def test_unknown_query_or_borrowed_result_is_not_read_only(tmp_path: Path, body: str, extra: str, annotation: str):
    assert not _report(tmp_path, body, extra, annotation)


def test_literal_dictionary_does_not_hide_borrowed_default(tmp_path: Path):
    assert not _report(
        tmp_path, "    mapping = {'key': config.spec}\n    result = mapping.get('key')\n    mutate(result)\n"
    )


def test_narrowing_does_not_hide_custom_instance_check(tmp_path: Path):
    extra = "class Meta(type):\n    def __instancecheck__(cls, value):\n        mutate(value)\n        return True\n"
    # The body uses a local unknown check target, not the imported ordinary class.
    body = "    Uniform = key\n    spec = config.spec\n    mapping = spec.specs if isinstance(spec, Uniform) else {}\n"
    body += "    result = mapping.get('key', spec)\n"
    assert not _report(tmp_path, body, extra)


@pytest.mark.parametrize("after", ["", "    mutate(view)\n"])
def test_helper_dictionary_queries_retain_borrowed_aliases(tmp_path: Path, after: str):
    helper = "def helper(value):\n    mapping = {'key': value.tensors}\n    return mapping.get('key', value.tensors)\n"
    findings = _helper_report(tmp_path, helper, after=after)
    assert bool(findings) is (not after)


def test_conditional_mapping_query_retains_borrowed_default(tmp_path: Path):
    body = "    spec = config.spec\n    mapping = spec.specs if isinstance(spec, Uniform) else {}\n"
    body += "    result = mapping.get('key', spec)\n    mutate(result)\n"
    assert not _report(tmp_path, body)
