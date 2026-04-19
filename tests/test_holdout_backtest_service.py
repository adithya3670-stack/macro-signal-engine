import pandas as pd

from backend.domain.dto import HoldoutRunRequest, HoldoutSplitRequest
from backend.services import holdout_backtest_service as holdout_module
from backend.services.holdout_backtest_service import HoldoutBacktestService


def test_prepare_split_creates_manifest(tmp_path, monkeypatch):
    engineered_path = tmp_path / "engineered_features.csv"
    source_df = pd.DataFrame(
        {"SP500": [100.0, 101.5, 102.2]},
        index=pd.to_datetime(["2022-01-03", "2022-12-30", "2023-01-04"]),
    )
    source_df.to_csv(engineered_path)

    holdout_root = tmp_path / "models" / "holdout"

    monkeypatch.setattr(holdout_module, "ENGINEERED_FEATURES_FILE", str(engineered_path))
    monkeypatch.setattr(holdout_module, "MASTER_DATA_FILE", str(tmp_path / "master_missing.csv"))
    monkeypatch.setattr(holdout_module, "HOLDOUT_DIR", str(holdout_root))

    svc = HoldoutBacktestService()
    payload = svc.prepare_split(HoldoutSplitRequest(cutoff_year=2022))

    assert payload["cutoff_year"] == 2022
    assert payload["train_rows"] == 2
    assert payload["test_rows"] == 1

    manifest_path = holdout_root / "2022" / "manifest.json"
    assert manifest_path.exists()


def test_list_models_reads_holdout_dl_directories(tmp_path, monkeypatch):
    models_root = tmp_path / "models"
    holdout_dl = models_root / "holdout_dl"
    year_2022 = holdout_dl / "2022"
    year_2024 = holdout_dl / "2024"
    year_2022.mkdir(parents=True)
    year_2024.mkdir(parents=True)

    (year_2022 / "dl_config.json").write_text('{"dl_mode":"quick"}', encoding="utf-8")
    (year_2024 / "dl_config.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(holdout_module, "MODELS_DIR", str(models_root))

    svc = HoldoutBacktestService()
    rows = svc.list_models()

    assert [row["cutoff_year"] for row in rows] == ["2024", "2022"]
    assert rows[0]["category"] == "dl"
    assert rows[1]["dl_mode"] == "quick"


def test_stream_holdout_ml_mode_returns_sse_error_without_crash():
    svc = HoldoutBacktestService()
    chunks = list(
        svc.stream_holdout(
            HoldoutRunRequest(
                cutoff_year=2024,
                model_type="ml",
                dl_mode="balanced",
                use_existing=False,
            )
        )
    )

    assert any('"message": "Error"' in chunk for chunk in chunks)
    assert chunks[-1] == "data: DONE\n\n"


def test_build_weighted_signals_respects_metric_weights():
    svc = HoldoutBacktestService()
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

    signals, assets = svc._build_weighted_signals(results, metrics)

    assert assets == {"SP500"}
    assert "SP500" in signals.columns
    assert round(float(signals.iloc[0]["SP500"]), 4) == 0.5857
