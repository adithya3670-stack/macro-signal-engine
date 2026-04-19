"""
Legacy training helper placeholder.

This module previously contained an indented method fragment that made the
repository fail import/compile checks. The training logic now lives inside
`analysis.deep_learning_model.DLMacroModel`.
"""

from typing import Any, Dict


def legacy_train_model_instance(*_: Any, **__: Any) -> Dict[str, Any]:
    """
    Backward-compatible shim for old experimental imports.

    Raises:
        NotImplementedError: Always, because this helper is no longer active.
    """
    raise NotImplementedError(
        "legacy_train_model_instance has been retired; use DLMacroModel APIs instead."
    )
