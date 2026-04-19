from __future__ import annotations

from typing import Any, Dict, Optional

from backend.domain.dto import ModelSourceSelection
from backend.infrastructure.model_artifacts import ModelArtifactResolver
from backend.services.backtest_orchestration import BacktestOrchestrationService


class ModelSourceSelectionService:
    """Resolves live/holdout/rolling model source selections."""

    def __init__(
        self,
        artifact_resolver: ModelArtifactResolver,
        orchestration: BacktestOrchestrationService,
    ) -> None:
        self.artifacts = artifact_resolver
        self.orchestration = orchestration

    def list_holdout_folders(self) -> Dict[str, list[str]]:
        return {"folders": ["default"] + self.artifacts.list_holdout_dl_folders()}

    def resolve_live_selection(self, model_type: str = "ensemble") -> ModelSourceSelection:
        return self.artifacts.resolve_model_source(
            model_category="dl",
            model_year=None,
            dl_folder="default",
            model_type=model_type,
        )

    def resolve_portfolio_selection(
        self,
        model_category: Optional[str],
        model_year: Optional[str],
        strat_config: Dict[str, Any],
    ) -> ModelSourceSelection:
        return self.orchestration.resolve_portfolio_selection(
            model_category=model_category,
            model_year=model_year,
            strat_config=strat_config,
        )
