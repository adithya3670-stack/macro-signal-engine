import pandas as pd

from backend.services.dl_inference_service import DLInferenceService
from backend.services.dl_snapshot_lifecycle_service import DLSnapshotLifecycleService
from backend.services.dl_training_orchestration_service import DLTrainingOrchestrationService


class _FakeDLBuilder:
    def __init__(self):
        self.train_calls = []
        self.optimize_calls = []

    def train_all_models(self, **kwargs):
        self.train_calls.append(kwargs)

    def optimize_models(self, **kwargs):
        self.optimize_calls.append(kwargs)
        model_type = kwargs["model_type"]
        return {f"cfg_{model_type}": {"window_size": 30}}


def _sse_stub(progress, phase, details=""):
    return f"{progress}|{phase}|{details}"


def test_dl_training_orchestration_lite_mode_runs_search_and_training():
    builder = _FakeDLBuilder()
    service = DLTrainingOrchestrationService()
    config = {}

    chunks = list(
        service.train_holdout_mode(
            builder=builder,
            cutoff_date="2025-12-31",
            dl_mode="lite",
            dl_config=config,
            send_update=_sse_stub,
        )
    )

    assert len(chunks) == 6
    assert len(builder.optimize_calls) == 3
    assert len(builder.train_calls) == 3
    assert all(call["iterations"] == 5 for call in builder.optimize_calls)
    assert all(call["epochs"] == 30 for call in builder.train_calls)
    assert config["cfg_lstm"]["window_size"] == 30
    assert config["cfg_transformer"]["window_size"] == 30
    assert config["cfg_nbeats"]["window_size"] == 30


def test_dl_training_orchestration_unknown_mode_falls_back_to_balanced():
    builder = _FakeDLBuilder()
    service = DLTrainingOrchestrationService()
    config = {}

    chunks = list(
        service.train_holdout_mode(
            builder=builder,
            cutoff_date="2025-12-31",
            dl_mode="unknown-mode",
            dl_config=config,
            send_update=_sse_stub,
        )
    )

    assert len(chunks) == 1
    assert len(builder.train_calls) == 1
    call = builder.train_calls[0]
    assert call["model_type"] == "lstm"
    assert call["epochs"] == 30


def test_dl_inference_service_build_weighted_signals_matches_expected_weighting():
    service = DLInferenceService()
    idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
    results = pd.DataFrame(
        {
            "Pred_SP500_1m_lstm": [0.8, 0.6],
            "Pred_SP500_1m_transformer": [0.2, 0.4],
        },
        index=idx,
    )
    metrics = {
        "lstm": {"SP500_1m": {"accuracy": 0.9, "precision": 0.9}},
        "transformer": {"SP500_1m": {"accuracy": 0.5, "precision": 0.5}},
    }

    signals, assets = service.build_weighted_signals(results=results, metrics_db=metrics)

    assert assets == {"SP500"}
    assert "SP500" in signals.columns
    assert round(float(signals.iloc[0]["SP500"]), 4) == 0.5857


def test_dl_snapshot_lifecycle_service_config_roundtrip(tmp_path):
    version_dir = tmp_path / "holdout_dl" / "2025"
    service = DLSnapshotLifecycleService()

    assert service.load_config(str(version_dir)) == {}
    saved_path = service.save_config(str(version_dir), {"dl_mode": "balanced", "epochs": 30})
    loaded = service.load_config(str(version_dir))

    assert saved_path.endswith("dl_config.json")
    assert loaded["dl_mode"] == "balanced"
    assert loaded["epochs"] == 30
