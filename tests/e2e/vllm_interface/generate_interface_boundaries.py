#!/usr/bin/env python3
"""Compatibility facade for the repository interface-contract engine.

Keep this path stable because existing local commands and vllm-interface tests
load it directly.  All analysis code lives in ``tools.vllm_interface_contracts``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from tools.vllm_interface_contracts import generator as _implementation  # noqa: E402

# Preserve the complete legacy module surface, including private helpers used
# by the focused regression suite.  New callers should import the tools package.
globals().update({name: getattr(_implementation, name) for name in dir(_implementation) if not name.startswith("__")})


class _CompatibilityModule(types.ModuleType):
    """Forward test monkeypatches to the canonical implementation module."""

    def __setattr__(self, name: str, value: object) -> None:
        if hasattr(_implementation, name):
            setattr(_implementation, name, value)
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CompatibilityModule


if __name__ == "__main__":
    _implementation.main()
