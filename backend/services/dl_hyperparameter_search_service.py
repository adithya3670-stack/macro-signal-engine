from __future__ import annotations

from typing import Any, Dict, MutableMapping


class DLHyperparameterSearchService:
    """Typed boundary for DL random-search orchestration."""

    def optimize_architecture(
        self,
        *,
        builder: Any,
        model_type: str,
        iterations: int,
        base_config: MutableMapping[str, Any],
    ) -> Dict[str, Any]:
        result = builder.optimize_models(
            model_type=model_type,
            iterations=max(1, int(iterations)),
            save_config=True,
            base_config=dict(base_config),
        )
        if isinstance(result, dict):
            base_config.update(result)
        return dict(base_config)
