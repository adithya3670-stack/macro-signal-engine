from backend.domain.dto import ModelSourceSelection
from backend.services.model_source_selection_service import ModelSourceSelectionService


class _Artifacts:
    def __init__(self):
        self.resolve_calls = []

    def list_holdout_dl_folders(self):
        return ["2024", "2023"]

    def resolve_model_source(self, **kwargs):
        self.resolve_calls.append(kwargs)
        return ModelSourceSelection(
            source="live",
            category="dl",
            model_type=kwargs["model_type"],
            model_dir="models_dl",
            label="Live Auto-Pilot",
        )


class _Orchestration:
    def __init__(self):
        self.calls = []

    def resolve_portfolio_selection(self, model_category, model_year, strat_config):
        self.calls.append((model_category, model_year, strat_config))
        return ModelSourceSelection(
            source="holdout",
            category="dl",
            model_type="ensemble",
            year="2024",
            model_dir="models/holdout_dl/2024",
            label="DL 2024",
        )


def test_model_source_selection_service_delegates_and_shapes():
    artifacts = _Artifacts()
    orchestration = _Orchestration()
    svc = ModelSourceSelectionService(artifact_resolver=artifacts, orchestration=orchestration)

    assert svc.list_holdout_folders() == {"folders": ["default", "2024", "2023"]}

    live = svc.resolve_live_selection(model_type="winner_ensemble")
    assert live.model_type == "winner_ensemble"
    assert artifacts.resolve_calls[-1]["dl_folder"] == "default"

    selected = svc.resolve_portfolio_selection("dl", "2024", {"model_type": "ensemble"})
    assert selected.source == "holdout"
    assert orchestration.calls[-1][0] == "dl"
