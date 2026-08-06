"""Static vLLM/vllm-ascend interface contract analysis engine."""

from tools.vllm_interface_contracts.generator import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    InterfaceBoundaryGenerator,
    RepositoryIndex,
)

__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "InterfaceBoundaryGenerator",
    "RepositoryIndex",
]
