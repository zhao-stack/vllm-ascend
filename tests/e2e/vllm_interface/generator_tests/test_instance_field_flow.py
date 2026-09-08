"""Stored configuration provenance must survive only proven receiver paths."""

from pathlib import Path

import pytest
from test_range_analysis import _call_repositories, _run


def _report(tmp_path: Path, consumer: str, upstream: str = ""):
    prefix = "from dataclasses import dataclass\n@dataclass\nclass Tensor:\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n" + upstream
    roots = _call_repositories(
        tmp_path,
        old_source=prefix + "    shared_by: list[str]\n" + config,
        new_source=prefix + "    layers: list[str]\n" + config,
        consumer_source="from vllm.api import Config\n" + consumer,
    )
    return [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]


@pytest.mark.parametrize(
    "annotation,guard,body",
    [
        ("Config", "", "return [t.shared_by for t in self.config.tensors]"),
        ("Config", "config = self.config\n        ", "return [t.shared_by for t in config.tensors]"),
        (
            "'Config | None'",
            "assert self.config is not None\n        ",
            "return [t.shared_by for t in self.config.tensors]",
        ),
        (
            "'Config | None'",
            "if self.config is None:\n            return\n        ",
            "return [t.shared_by for t in self.config.tensors]",
        ),
    ],
)
def test_local_stored_parameter_field(tmp_path: Path, annotation: str, guard: str, body: str):
    consumer = (
        f"class Worker:\n    def __init__(self, config: {annotation}):\n        self.config = config\n"
        "    def consume(self):\n        self.unrelated = []\n        " + guard + body + "\n"
    )
    findings = _report(tmp_path, consumer)
    assert len(findings) == 1
    assert findings[0]["classification"] == "introduced_break"
    assert findings[0]["action"] == "modify"


@pytest.mark.parametrize(
    "change",
    [
        "self.config = unknown()",
        "self = unknown()",
        "mutate(self)",
        "mutate(self.config)",
        "self.config.tensors.clear()",
        "holder = self; mutate(holder)",
    ],
)
def test_stored_parameter_mutation_invalidates_evidence(tmp_path: Path, change: str):
    consumer = (
        "class Worker:\n    def __init__(self, config: Config):\n        self.config = config\n"
        f"    def consume(self):\n        alias = self.config\n        {change}\n"
        "        return [t.shared_by for t in alias.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


@pytest.mark.parametrize(
    "initializer,extra",
    [
        ("if flag:\n            self.config = config", ""),
        ("config = unknown()\n        self.config = config", ""),
        ("self.config = config", "    def reset(self):\n        self.config = unknown()\n"),
        ("self.config = config", "    @property\n    def config(self):\n        return unknown()\n"),
        ("self.config = config", "    async def reset(self):\n        self.config = unknown()\n"),
        ("self.config = config", "    def __getattribute__(self, name):\n        return unknown()\n"),
        ("self.config = config", "    def __new__(cls, *args):\n        return unknown()\n"),
    ],
)
def test_ambiguous_initialization_is_not_an_exact_field(tmp_path: Path, initializer: str, extra: str):
    consumer = (
        "class Worker:\n    def __init__(self, config: Config):\n        "
        + initializer
        + "\n"
        + extra
        + "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


def test_optional_stored_parameter_requires_narrowing(tmp_path: Path):
    consumer = (
        "class Worker:\n    def __init__(self, config: 'Config | None'):\n        self.config = config\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


def test_inherited_stored_parameter_retains_upstream_owner_path(tmp_path: Path):
    upstream = "class Base:\n    def __init__(self, config: Config):\n        self.config = config\n"
    consumer = (
        "from vllm.api import Base\nclass Worker(Base):\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    findings = _report(tmp_path, consumer, upstream)
    assert len(findings) == 1
    assert findings[0]["action"] == "modify"
    assert tuple(findings[0]["evidence"][0]["receiver_path"][:2]) == ("vllm.api.Base", "instance_field:config")


def test_overridden_init_cannot_assume_base_field_exists(tmp_path: Path):
    upstream = "class Base:\n    def __init__(self, config: Config):\n        self.config = config\n"
    consumer = (
        "from vllm.api import Base\nclass Worker(Base):\n    def __init__(self):\n        pass\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer, upstream)


def test_inherited_stored_parameter_does_not_borrow_new_type_for_old(tmp_path: Path):
    common = "from dataclasses import dataclass\n@dataclass\nclass Other:\n    tensors: list[int]\n"
    old = common + "@dataclass\nclass Tensor:\n    shared_by: int\n"
    new = common + "@dataclass\nclass Tensor:\n    layers: int\n"
    config = "@dataclass\nclass Config:\n    tensors: list[Tensor]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=old + config + "class Base:\n    def __init__(self, config: Other):\n        self.config = config\n",
        new_source=new + config + "class Base:\n    def __init__(self, config: Config):\n"
        "        self.config = config\n",
        consumer_source="from vllm.api import Base\nclass Worker(Base):\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["details"].get("member") == "shared_by"]
    assert not any(f["classification"] == "introduced_break" for f in findings)


@pytest.mark.parametrize(
    "base",
    [
        "class Base:\n    @property\n    def config(self):\n        return unknown()\n",
        "class Base:\n    def __getattribute__(self, name):\n        return unknown()\n",
        "class Base:\n    def reset(self):\n        self.config = unknown()\n",
    ],
)
def test_local_constructor_does_not_override_inherited_storage_barriers(tmp_path: Path, base: str):
    consumer = (
        "from vllm.api import Base\nclass Worker(Base):\n"
        "    def __init__(self, config: Config):\n        self.config = config\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer, base)


def test_other_property_store_may_mutate_configuration(tmp_path: Path):
    consumer = (
        "class Worker:\n    def __init__(self, config: Config):\n        self.config = config\n"
        "    @property\n    def unrelated(self):\n        return 0\n"
        "    @unrelated.setter\n    def unrelated(self, value):\n        mutate(self)\n"
        "    def consume(self):\n        self.unrelated = 1\n"
        "        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


def test_early_initializer_return_does_not_prove_field_initialization(tmp_path: Path):
    consumer = (
        "class Worker:\n    def __init__(self, config: Config, flag):\n"
        "        if flag:\n            return\n        self.config = config\n"
        "    def consume(self):\n        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


def test_fields_can_alias_the_same_constructor_parameter(tmp_path: Path):
    consumer = (
        "class Worker:\n    def __init__(self, config: Config):\n"
        "        self.config = config\n        self.other = config\n"
        "    def consume(self):\n        mutate(self.other)\n"
        "        return [t.shared_by for t in self.config.tensors]\n"
    )
    assert not _report(tmp_path, consumer)


def test_stored_configuration_can_prove_a_call_argument_break(tmp_path: Path):
    old = "class Client:\n    def run(self, legacy=False): pass\n"
    new = "class Client:\n    def run(self): pass\n"
    config = "from dataclasses import dataclass\n@dataclass\nclass Config:\n    clients: list[Client]\n"
    roots = _call_repositories(
        tmp_path,
        old_source=old + config,
        new_source=new + config,
        consumer_source="from vllm.api import Config\nclass Worker:\n"
        "    def __init__(self, config: Config):\n        self.config = config\n"
        "    def consume(self):\n        for client in self.config.clients:\n            client.run(legacy=True)\n",
    )
    findings = [f for f in _run(*roots)["findings"] if f["relation"] == "direct_call"]
    assert any(f["classification"] == "introduced_break" and f["action"] == "modify" for f in findings)
