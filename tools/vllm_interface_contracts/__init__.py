"""Static vLLM/vllm-ascend interface contract analysis engine."""

from tools.vllm_interface_contracts.analysis_plans import (
    ANALYSIS_PLAN_VERSION,
    MAIN2MAIN_SCENARIO,
    VLLM_INTERFACE_SCENARIO,
    AnalysisPlan,
    resolve_analysis_plan,
)
from tools.vllm_interface_contracts.call_contracts import (
    DirectCallDetector,
    ReturnContract,
    ReturnUse,
)
from tools.vllm_interface_contracts.generator import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    InterfaceBoundaryGenerator,
    RepositoryIndex,
)

__all__ = [
    "GENERATOR_VERSION",
    "SCHEMA_VERSION",
    "ANALYSIS_PLAN_VERSION",
    "MAIN2MAIN_SCENARIO",
    "VLLM_INTERFACE_SCENARIO",
    "AnalysisPlan",
    "DirectCallDetector",
    "InterfaceBoundaryGenerator",
    "RepositoryIndex",
    "ReturnContract",
    "ReturnUse",
    "resolve_analysis_plan",
]
