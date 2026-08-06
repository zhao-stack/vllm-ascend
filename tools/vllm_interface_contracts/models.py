"""Public data models for range-level interface compatibility reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceEndpoint:
    file: str | None
    owner: str | None
    name: str | None
    line: int | None = None
    signature: list[object] | None = field(default=None, compare=False)
    descriptor: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityState:
    exists: bool | None
    compatible: bool | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RangeFinding:
    finding_id: str
    classification: str
    relation: str
    priority: str
    action: str
    confidence: str
    upstream_old: SourceEndpoint
    upstream_new: SourceEndpoint
    downstream: SourceEndpoint
    old_state: CompatibilityState
    new_state: CompatibilityState
    change: str
    evidence: list[dict[str, Any]]
    gates: dict[str, bool]
    suggestion: str
    source: str = "dynamic_relation_graph"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.finding_id,
            "classification": self.classification,
            "relation": self.relation,
            "priority": self.priority,
            "action": self.action,
            "confidence": self.confidence,
            "upstream": {
                "old": self.upstream_old.as_dict(),
                "new": self.upstream_new.as_dict(),
            },
            "downstream": self.downstream.as_dict(),
            "compatibility": {
                "old": self.old_state.as_dict(),
                "new": self.new_state.as_dict(),
            },
            "change": self.change,
            "evidence": self.evidence,
            "gates": self.gates,
            "suggestion": self.suggestion,
            "source": self.source,
        }
