import numpy as np
import pandas as pd

from analysis.deep_learning_model import DLMacroModel
from analysis.dl.snapshot_store import create_model_snapshot, list_model_snapshots, restore_model_snapshot


def test_snapshot_store_roundtrip(tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact = model_dir / "example_model.pt"
    artifact.write_text("baseline", encoding="utf-8")

    created = create_model_snapshot(str(model_dir), tag="unit")
    assert created["id"].startswith("snap_")
    assert created["files_count"] >= 1

    snapshots = list_model_snapshots(str(model_dir))
    assert snapshots
    assert snapshots[0]["id"] == created["id"]

    artifact.write_text("mutated", encoding="utf-8")
    restored = restore_model_snapshot(str(model_dir), created["id"])
    assert restored is True
    assert artifact.read_text(encoding="utf-8") == "baseline"


def test_dlmacro_snapshot_delegation(tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "weights.pt").write_text("w", encoding="utf-8")
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")

    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))
    created = dl.create_model_snapshot(tag="delegate")
    listed = dl.list_model_snapshots()

    assert created["id"]
    assert any(item["id"] == created["id"] for item in listed)


def test_dlmacro_training_and_inference_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    def fake_train_model_instance(**kwargs):
        assert kwargs["model_type"] == "lstm"
        return {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc": 1.0, "threshold": 0.5}

    monkeypatch.setattr("analysis.dl.training_core.train_model_instance", fake_train_model_instance)
    metrics = dl._train_model_instance(
        X_train=np.zeros((4, 3, 2)),
        y_train=np.zeros((4, 1)),
        X_val=np.zeros((2, 3, 2)),
        y_val=np.zeros((2, 1)),
        X_test=np.zeros((2, 3, 2)),
        y_test=np.zeros((2, 1)),
        model_type="lstm",
        params={"hidden_size": 16, "num_layers": 1, "dropout": 0.1},
        top_features=["f1", "f2"],
        save_path=str(model_dir / "fake.pth"),
        force_full_training=False,
        window_size=3,
    )
    assert metrics["f1"] == 1.0

    monkeypatch.setattr(
        "analysis.dl.inference_engine.predict_latest_by_model",
        lambda **kwargs: [{"asset": "SP500", "horizon": "1m", "direction": "BULLISH", "confidence": "60.0%"}],
    )
    preds = dl._predict_latest_by_model(model_type="lstm")
    assert preds[0]["asset"] == "SP500"

    monkeypatch.setattr(
        "analysis.dl.inference_engine.predict_ensemble",
        lambda **kwargs: [{"asset": "SP500", "horizon": "1m", "direction": "BULLISH", "confidence": "55.0%"}],
    )
    ens = dl.predict_ensemble()
    assert ens[0]["horizon"] == "1m"


def test_dlmacro_train_all_models_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    def fake_train_all_models_core(**kwargs):
        assert kwargs["dl_model"] is dl
        assert kwargs["model_type"] == "lstm"
        assert kwargs["train_cutoff_date"] == "2025-12-31"
        assert kwargs["epochs"] == 11
        assert kwargs["use_bagging_ensemble"] is True
        assert kwargs["n_folds"] == 1
        assert kwargs["custom_flag"] is True
        return [{"asset": "SP500", "horizon": "1m", "metrics": {"f1": 0.5}}]

    monkeypatch.setattr(
        "analysis.dl.training_pipeline.train_all_models_core",
        fake_train_all_models_core,
    )
    result = dl.train_all_models(
        model_type="lstm",
        train_cutoff_date="2025-12-31",
        epochs=11,
        use_bagging_ensemble=True,
        n_folds=1,
        custom_flag=True,
    )
    assert result[0]["asset"] == "SP500"


def test_dlmacro_optimize_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    monkeypatch.setattr(
        "analysis.dl.hyperparameter_search.optimize_models_core",
        lambda **kwargs: {"SP500_1m_lstm": {"window_size": 30}},
    )
    result = dl.optimize_models(model_type="lstm", iterations=1)
    assert "SP500_1m_lstm" in result


def test_dlmacro_backtest_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    def fake_backtest_simple_standard_core(**kwargs):
        assert kwargs["dl_model"] is dl
        assert kwargs["snapshot_id"] == "latest"
        assert kwargs["model_type"] == "ensemble"
        assert kwargs["asset"] == "SP500_1m"
        assert kwargs["initial_capital"] == 10000
        assert kwargs["threshold"] == 0.6
        return {"status": "success", "metrics": {"final_equity": 10100.0}}

    monkeypatch.setattr(
        "analysis.dl.backtest_engine.backtest_simple_standard_core",
        fake_backtest_simple_standard_core,
    )

    result = dl.backtest_simple_standard(
        snapshot_id="latest",
        model_type="ensemble",
        asset="SP500_1m",
        initial_capital=10000,
        start_date="2026-01-01",
        end_date="2026-01-31",
        threshold=0.6,
    )
    assert result["status"] == "success"
    assert result["metrics"]["final_equity"] == 10100.0


def test_dlmacro_range_prediction_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    expected_df = pd.DataFrame(
        {"Pred_SP500_1m_lstm": [0.62]},
        index=pd.to_datetime(["2026-01-02"]),
    )

    def fake_predict_range_core(**kwargs):
        assert kwargs["dl_model"] is dl
        assert kwargs["start_date"] == "2026-01-01"
        assert kwargs["end_date"] == "2026-01-31"
        return expected_df

    monkeypatch.setattr(
        "analysis.dl.range_predictor.predict_range_core",
        fake_predict_range_core,
    )
    got_df = dl.predict_range("2026-01-01", "2026-01-31")
    assert got_df.equals(expected_df)

    monkeypatch.setattr(
        "analysis.dl.range_predictor.predict_latest_from_range_core",
        lambda **kwargs: [
            {
                "asset": "SP500",
                "horizon": "1m",
                "model_type": "lstm",
                "direction": "BULLISH",
                "confidence": "62.0%",
                "raw_score": 0.62,
            },
        ],
    )
    latest = dl._predict_latest_from_range()
    assert latest[0]["direction"] == "BULLISH"


def test_dlmacro_data_pipeline_delegation(monkeypatch, tmp_path):
    model_dir = tmp_path / "models_dl"
    model_dir.mkdir(parents=True, exist_ok=True)
    dummy_data = tmp_path / "engineered_features.csv"
    dummy_data.write_text("Date,SP500\n2026-01-01,1\n", encoding="utf-8")
    dl = DLMacroModel(data_path=str(dummy_data), model_dir=str(model_dir))

    expected_df = pd.DataFrame({"SP500": [1]}, index=pd.to_datetime(["2026-01-01"]))

    def fake_load_and_preprocess_core(**kwargs):
        assert kwargs["dl_model"] is dl
        assert "engineered_features_file" in kwargs
        assert "master_data_file" in kwargs
        return expected_df

    monkeypatch.setattr(
        "analysis.dl.data_pipeline.load_and_preprocess_core",
        fake_load_and_preprocess_core,
    )
    got_df = dl.load_and_preprocess()
    assert got_df.equals(expected_df)

    expected_x = np.zeros((2, 3, 4))
    expected_y = np.array([1.0, 0.0])

    def fake_create_sequences_core(**kwargs):
        assert kwargs["window_size"] == 3
        assert kwargs["target_alignment"] == "next"
        return expected_x, expected_y

    monkeypatch.setattr(
        "analysis.dl.data_pipeline.create_sequences_core",
        fake_create_sequences_core,
    )
    X_seq, y_seq = dl.create_sequences(X_data=np.zeros((5, 4)), window_size=3, y_data=np.ones(5))
    assert np.array_equal(X_seq, expected_x)
    assert np.array_equal(y_seq, expected_y)
