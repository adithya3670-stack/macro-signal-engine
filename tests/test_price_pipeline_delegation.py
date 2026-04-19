import numpy as np
import pandas as pd
import pytest
import torch

from analysis.price_1m_regression import Price1MRegressionManager
from analysis.price_1w_regression import Price1WRegressionManager
from analysis.price_3d_regression import Price3DRegressionManager


@pytest.mark.parametrize(
    "manager_cls,missing_date_msg,artifact_prefix,horizon_label,metrics_filename",
    [
        (
            Price3DRegressionManager,
            "Price3D features file is missing a Date column.",
            "price3d",
            "price-3d",
            "price3d_metrics.json",
        ),
        (
            Price1WRegressionManager,
            "Price1W features file is missing a Date column.",
            "price1w",
            "price-1w",
            "price1w_metrics.json",
        ),
        (
            Price1MRegressionManager,
            "Price1M features file is missing a Date column.",
            "price1m",
            "price-1m",
            "price1m_metrics.json",
        ),
    ],
)
def test_price_pipeline_common_delegation(
    monkeypatch,
    tmp_path,
    manager_cls,
    missing_date_msg,
    artifact_prefix,
    horizon_label,
    metrics_filename,
):
    data_path = tmp_path / f"{manager_cls.__name__}_features.csv"
    model_dir = tmp_path / f"{manager_cls.__name__}_models"
    holdout_root = tmp_path / f"{manager_cls.__name__}_holdout"
    mgr = manager_cls(
        data_path=str(data_path),
        model_dir=str(model_dir),
        holdout_root=str(holdout_root),
        assets=["SP500"],
    )
    model_type = next(iter(mgr.MODEL_CONFIGS.keys()))

    expected_df = pd.DataFrame({"SP500": [1]}, index=pd.to_datetime(["2026-01-01"]))

    def fake_load_and_preprocess_price_features(**kwargs):
        assert kwargs["data_path"] == str(data_path)
        assert kwargs["feature_builder"] is mgr.feature_builder
        assert kwargs["force_refresh_features"] is False
        assert kwargs["missing_date_message"] == missing_date_msg
        return expected_df

    monkeypatch.setattr(
        "analysis.price_pipeline_common.load_and_preprocess_price_features",
        fake_load_and_preprocess_price_features,
    )
    got_df = mgr.load_and_preprocess(force_refresh_features=False)
    assert got_df.equals(expected_df)

    monkeypatch.setattr(
        "analysis.price_pipeline_common.refresh_price_feature_cache",
        lambda **kwargs: {
            "path": kwargs["data_path"],
            "rows": 1,
            "columns": ["SP500"],
        },
    )
    refreshed = mgr.refresh_feature_cache()
    assert refreshed["rows"] == 1

    monkeypatch.setattr(
        "analysis.price_pipeline_common.promote_price_champions",
        lambda **kwargs: {
            "status": "ok",
            "prefix": kwargs["artifact_prefix"],
            "year": kwargs["holdout_year"],
        },
    )
    promoted = mgr.promote_champions(holdout_year=2025)
    assert promoted["status"] == "ok"
    assert promoted["prefix"] == artifact_prefix
    assert promoted["year"] == 2025

    monkeypatch.setattr(
        "analysis.price_pipeline_common.predict_latest_price",
        lambda **kwargs: [{"asset": "SP500", "prefix": kwargs["artifact_prefix"]}],
    )
    latest = mgr.predict_latest(model_dir=str(model_dir), year=2025)
    assert latest[0]["asset"] == "SP500"
    assert latest[0]["prefix"] == artifact_prefix

    monkeypatch.setattr(
        "analysis.price_pipeline_common.load_price_metrics",
        lambda **kwargs: {"metrics": [], "prefix": kwargs["artifact_prefix"]},
    )
    metrics = mgr.load_metrics(holdout_year=2025)
    assert metrics["prefix"] == artifact_prefix

    monkeypatch.setattr(
        "analysis.price_pipeline_common.load_price_feature_registry",
        lambda **kwargs: {"SP500": ["feat_1", "feat_2"], "prefix": kwargs["artifact_prefix"]},
    )
    registry = mgr.load_feature_registry(holdout_year=2025)
    assert registry["SP500"] == ["feat_1", "feat_2"]

    monkeypatch.setattr(
        "analysis.price_pipeline_common.load_price_model_metadata",
        lambda **kwargs: {"asset": kwargs["asset"], "model_type": kwargs["model_type"]},
    )
    metadata = mgr.load_model_metadata(asset="SP500", model_type=model_type, holdout_year=2025)
    assert metadata["asset"] == "SP500"
    assert metadata["model_type"] == model_type

    monkeypatch.setattr(
        "analysis.price_pipeline_common.load_price_model_scaler",
        lambda **kwargs: "artifact::scaler",
    )
    loaded_scaler = mgr.load_model_scaler(asset="SP500", model_type=model_type, holdout_year=2025)
    assert loaded_scaler == "artifact::scaler"

    monkeypatch.setattr(
        "analysis.price_pipeline_common.resolve_holdout_dir",
        lambda **kwargs: f"holdout::{kwargs['horizon_label']}::{kwargs['holdout_year']}",
    )
    resolved_holdout = mgr._resolve_holdout_dir(holdout_year=2025)
    assert resolved_holdout == f"holdout::{horizon_label}::2025"

    monkeypatch.setattr(
        "analysis.price_pipeline_common.resolve_default_metrics_dir",
        lambda **kwargs: f"default::{kwargs['metrics_filename']}",
    )
    resolved_default = mgr._resolve_default_metrics_dir()
    assert resolved_default == f"default::{metrics_filename}"

    monkeypatch.setattr(
        "analysis.price_pipeline_common.estimate_target_date",
        lambda **kwargs: pd.Timestamp("2026-01-08"),
    )
    est_target = mgr._estimate_target_date(pd.Timestamp("2026-01-01"))
    assert est_target == pd.Timestamp("2026-01-08")

    monkeypatch.setattr(
        "analysis.price_pipeline_common.own_asset_features",
        lambda asset: [asset, f"{asset}_lag_1"],
    )
    own_feats = mgr._own_asset_features("SP500")
    assert own_feats == ["SP500", "SP500_lag_1"]

    monkeypatch.setattr(
        "analysis.price_pipeline_common.asset_group",
        lambda asset, groups: "equities" if asset in groups["equities"] else "other",
    )
    assert mgr._asset_group("SP500") == "equities"

    monkeypatch.setattr("analysis.price_pipeline_common.safe_stat", lambda values, fn: 42.0)
    assert mgr._safe_stat([1.0, 2.0], min) == 42.0

    monkeypatch.setattr("analysis.price_pipeline_common.sort_nan", lambda value: 123.0)
    assert mgr._sort_nan(float("nan")) == 123.0

    monkeypatch.setattr(
        "analysis.price_pipeline_common.train_holdout_pipeline_core",
        lambda **kwargs: {
            "status": "ok",
            "artifact_prefix": kwargs["artifact_prefix"],
            "progress_label": kwargs["progress_label"],
        },
    )
    holdout_result = mgr.train_holdout_pipeline(holdout_year=2025, epochs=1)
    assert holdout_result["status"] == "ok"
    assert holdout_result["artifact_prefix"] == artifact_prefix

    monkeypatch.setattr(
        "analysis.price_pipeline_common.train_asset_model_core",
        lambda **kwargs: {"asset": kwargs["asset"], "model_type": kwargs["model_type"]},
    )
    asset_result = mgr._train_asset_model(
        df=pd.DataFrame({"SP500": [1.0]}, index=pd.to_datetime(["2026-01-01"])),
        asset="SP500",
        model_type=model_type,
        feature_cols=["SP500"],
        save_dir=str(tmp_path),
        epochs=1,
    )
    assert asset_result["asset"] == "SP500"
    assert asset_result["model_type"] == model_type

    monkeypatch.setattr(
        "analysis.price_pipeline_common.run_discovery_fold_core",
        lambda **kwargs: {"fold": kwargs["fold"].name, "metrics": {"rows": 1}},
    )
    fold = mgr.DISCOVERY_FOLDS[0]
    discovery = mgr._run_discovery_fold(
        sequence_payload={
            "X": np.zeros((2, 3, 1)),
            "y": np.zeros(2),
            "current_prices": np.ones(2),
            "future_prices": np.ones(2),
            "mape_valid": np.array([True, True]),
            "dates": np.array(pd.to_datetime(["2026-01-01", "2026-01-02"])),
        },
        model_type=model_type,
        model_config={"window_size": 3, "batch_size": 2, "lr": 0.001},
        fold=fold,
        epochs=1,
    )
    assert discovery["fold"] == fold.name

    monkeypatch.setattr(
        "analysis.price_pipeline_common.train_final_model_core",
        lambda **kwargs: {"best_epochs": 3, "target_suffix": kwargs["target_suffix"]},
    )
    final_out = mgr._train_final_model(
        df=pd.DataFrame({"SP500": [1.0]}, index=pd.to_datetime(["2026-01-01"])),
        asset="SP500",
        model_type=model_type,
        model_config={"window_size": 3, "batch_size": 2, "lr": 0.001},
        feature_cols=["SP500"],
        epochs=1,
        save_dir=str(tmp_path),
    )
    assert final_out["best_epochs"] == 3
    assert final_out["target_suffix"] == artifact_prefix[-2:]

    base_df = pd.DataFrame(
        {"feat": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
    )
    train_mask = pd.Series([True, True, False], index=base_df.index)

    monkeypatch.setattr(
        "analysis.price_pipeline_common.fit_price_scaler",
        lambda **kwargs: "scaler::ok",
    )
    scaler_obj = mgr.fit_scaler(base_df, ["feat"], train_mask)
    assert scaler_obj == "scaler::ok"

    monkeypatch.setattr(
        "analysis.price_pipeline_common.scale_price_frame",
        lambda **kwargs: np.array([[0.1], [0.2], [0.3]]),
    )
    scaled = mgr._scale_frame(base_df, ["feat"], scaler="ignored")
    assert scaled.shape == (3, 1)

    monkeypatch.setattr(
        "analysis.price_pipeline_common.create_price_sequences",
        lambda **kwargs: {
            "X": np.array([[[1.0]]]),
            "y": np.array([0.0]),
            "current_prices": np.array([100.0]),
            "future_prices": np.array([101.0]),
            "mape_valid": np.array([True]),
            "dates": np.array([np.datetime64("2026-01-05")]),
        },
    )
    seq = mgr._create_sequences(
        scaled_features=np.array([[0.1], [0.2], [0.3]]),
        centered_targets=np.array([0.01, 0.02, 0.03]),
        current_prices=np.array([100.0, 101.0, 102.0]),
        future_prices=np.array([101.0, 102.0, 103.0]),
        mape_valid=np.array([True, True, True]),
        dates=np.array(pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])),
        window_size=2,
    )
    assert seq["X"].shape == (1, 1, 1)

    monkeypatch.setattr(
        "analysis.price_pipeline_common.evaluate_price_predictions",
        lambda **kwargs: {"mape": 0.12, "rows": 3},
    )
    eval_metrics = mgr._evaluate_predictions(
        current_prices=[100.0, 101.0],
        actual_future=[101.0, 103.0],
        pred_centered=[0.01, 0.02],
        mape_valid=[True, True],
    )
    assert eval_metrics["mape"] == 0.12

    suffix = artifact_prefix[-2:]
    payload_df = pd.DataFrame(
        {
            "feat": [1.0, 2.0, 3.0],
            "SP500": [100.0, 101.0, 102.0],
            f"CenteredNormPrice_SP500_{suffix}": [0.01, 0.02, 0.03],
            f"FuturePrice_SP500_{suffix}": [101.0, 102.0, 103.0],
            f"MAPEValid_SP500_{suffix}": [1, 1, 1],
        },
        index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]),
    )

    def fake_build_price_sequence_payload(**kwargs):
        assert kwargs["train_end"] == mgr.FINAL_TUNE_TRAIN_END
        assert kwargs["holdout_start"] == mgr.OUTER_HOLDOUT_START
        assert kwargs["holdout_end"] == mgr.OUTER_HOLDOUT_END
        assert kwargs["window_size"] == 3
        assert len(kwargs["centered_targets"]) == 3
        assert len(kwargs["current_prices"]) == 3
        assert len(kwargs["future_prices"]) == 3
        assert len(kwargs["mape_valid"]) == 3
        return {"X": np.array([[[0.1]]]), "holdout_mask": np.array([True])}

    monkeypatch.setattr(
        "analysis.price_pipeline_common.build_price_sequence_payload",
        fake_build_price_sequence_payload,
    )
    payload = mgr._build_sequence_payload(payload_df, asset="SP500", feature_cols=["feat"], window_size=3)
    assert bool(payload["holdout_mask"][0]) is True

    monkeypatch.setattr(
        "analysis.price_pipeline_common.price_loss_function",
        lambda **kwargs: torch.tensor(1.25),
    )
    loss_val = mgr._loss_function(torch.tensor([[0.1]]), torch.tensor([[0.0]]))
    assert float(loss_val) == 1.25

    monkeypatch.setattr(
        "analysis.price_pipeline_common.predict_price_array",
        lambda **kwargs: np.array([0.11]),
    )
    pred_arr = mgr._predict_array(model="fake-model", X=np.zeros((1, 3, 1)))
    assert np.array_equal(pred_arr, np.array([0.11]))

    monkeypatch.setattr(
        "analysis.price_pipeline_common.fit_fixed_epochs_core",
        lambda **kwargs: "fixed-model",
    )
    fixed_model = mgr._fit_fixed_epochs(
        model="seed-model",
        model_config={"batch_size": 2, "lr": 0.001},
        X_train=np.zeros((2, 3, 1)),
        y_train=np.zeros(2),
        epochs=2,
    )
    assert fixed_model == "fixed-model"

    monkeypatch.setattr(
        "analysis.price_pipeline_common.fit_with_early_stopping_core",
        lambda **kwargs: ("early-model", 4),
    )
    early_model, best_epoch = mgr._fit_with_early_stopping(
        model_type="nlinear",
        model_config={"batch_size": 2, "lr": 0.001},
        input_size=1,
        X_train=np.zeros((3, 3, 1)),
        y_train=np.zeros(3),
        X_val=np.zeros((2, 3, 1)),
        y_val=np.zeros(2),
        current_val=np.ones(2),
        future_val=np.ones(2),
        mape_valid_val=np.array([True, True]),
        epochs=2,
    )
    assert early_model == "early-model"
    assert best_epoch == 4
