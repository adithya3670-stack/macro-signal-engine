from __future__ import annotations

import copy
import datetime as dt
import json
import os
import shutil
import time
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader


def load_and_preprocess_price_features(
    *,
    data_path: str,
    feature_builder,
    force_refresh_features: bool,
    missing_date_message: str,
) -> pd.DataFrame:
    if force_refresh_features or not os.path.exists(data_path):
        feature_builder.ensure_feature_file(force=force_refresh_features)

    df = pd.read_csv(data_path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
    elif "Unnamed: 0" in df.columns:
        df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.set_index("Date", inplace=True)
    else:
        raise ValueError(missing_date_message)

    df = df[df.index.notna()].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan)

    target_prefixes = ("FuturePrice_", "NormPrice_", "CenteredNormPrice_", "MAPEValid_")
    feature_cols = [c for c in df.columns if not c.startswith(target_prefixes)]
    df.loc[:, feature_cols] = df.loc[:, feature_cols].ffill().fillna(0)

    mask_cols = [c for c in df.columns if c.startswith("MAPEValid_")]
    for col in mask_cols:
        df[col] = df[col].fillna(0).astype(int)

    return df


def refresh_price_feature_cache(*, data_path: str, feature_builder) -> Dict[str, object]:
    df = feature_builder.build_and_save()
    return {
        "path": data_path,
        "rows": int(len(df)),
        "columns": list(df.columns),
    }


def fit_price_scaler(*, df: pd.DataFrame, feature_cols: List[str], train_mask: pd.Series) -> RobustScaler:
    scaler = RobustScaler()
    train_frame = df.loc[train_mask, feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    scaler.fit(train_frame)
    return scaler


def scale_price_frame(*, df: pd.DataFrame, feature_cols: List[str], scaler: RobustScaler) -> np.ndarray:
    frame = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return scaler.transform(frame)


def create_price_sequences(
    *,
    scaled_features: np.ndarray,
    centered_targets: np.ndarray,
    current_prices: np.ndarray,
    future_prices: np.ndarray,
    mape_valid: np.ndarray,
    dates: np.ndarray,
    window_size: int,
) -> Dict[str, np.ndarray]:
    X_seq, y_seq, cur_seq, fut_seq, mape_seq, date_seq = [], [], [], [], [], []

    for end_idx in range(window_size - 1, len(scaled_features)):
        y_val = centered_targets[end_idx]
        cur_val = current_prices[end_idx]
        fut_val = future_prices[end_idx]
        if not np.isfinite(y_val) or not np.isfinite(cur_val) or not np.isfinite(fut_val):
            continue

        seq = scaled_features[end_idx - window_size + 1 : end_idx + 1]
        if np.isnan(seq).any():
            continue

        X_seq.append(seq)
        y_seq.append(float(y_val))
        cur_seq.append(float(cur_val))
        fut_seq.append(float(fut_val))
        mape_seq.append(bool(mape_valid[end_idx]))
        date_seq.append(dates[end_idx])

    return {
        "X": np.array(X_seq, dtype=float),
        "y": np.array(y_seq, dtype=float),
        "current_prices": np.array(cur_seq, dtype=float),
        "future_prices": np.array(fut_seq, dtype=float),
        "mape_valid": np.array(mape_seq, dtype=bool),
        "dates": np.array(date_seq),
    }


def build_price_sequence_payload(
    *,
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end: str,
    centered_targets: np.ndarray,
    current_prices: np.ndarray,
    future_prices: np.ndarray,
    mape_valid: np.ndarray,
    dates: np.ndarray,
    window_size: int,
    holdout_start: str,
    holdout_end: str,
) -> Dict[str, np.ndarray]:
    train_mask = df.index <= pd.Timestamp(train_end)
    scaler = fit_price_scaler(df=df, feature_cols=feature_cols, train_mask=train_mask)
    scaled = scale_price_frame(df=df, feature_cols=feature_cols, scaler=scaler)
    payload = create_price_sequences(
        scaled_features=scaled,
        centered_targets=centered_targets,
        current_prices=current_prices,
        future_prices=future_prices,
        mape_valid=mape_valid,
        dates=dates,
        window_size=window_size,
    )
    payload["holdout_mask"] = (payload["dates"] >= np.datetime64(holdout_start)) & (
        payload["dates"] <= np.datetime64(holdout_end)
    )
    return payload


def evaluate_price_predictions(
    *,
    current_prices,
    actual_future,
    pred_centered,
    mape_valid,
    latency_ms: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    current_prices = np.asarray(current_prices, dtype=float)
    actual_future = np.asarray(actual_future, dtype=float)
    pred_centered = np.asarray(pred_centered, dtype=float)
    mape_valid = np.asarray(mape_valid, dtype=bool)

    if len(actual_future) == 0:
        return {"mape": None, "smape": None, "mae": None, "rows": 0, "mape_rows": 0, "latency_ms": latency_ms}

    pred_norm = np.clip(pred_centered + 1.0, -5.0, 5.0)
    pred_future = current_prices * pred_norm
    abs_err = np.abs(pred_future - actual_future)

    smape_denom = np.clip(np.abs(actual_future) + np.abs(pred_future), 1e-6, None)
    smape = float(np.mean((2.0 * abs_err / smape_denom) * 100.0))
    mae = float(np.mean(abs_err))

    valid_mask = mape_valid & np.isfinite(actual_future) & np.isfinite(pred_future) & (np.abs(actual_future) > 1e-6)
    mape = None
    median_mape = None
    if valid_mask.any():
        ape = (abs_err[valid_mask] / np.abs(actual_future[valid_mask])) * 100.0
        mape = float(np.mean(ape))
        median_mape = float(np.median(ape))

    return {
        "mape": mape,
        "median_mape": median_mape,
        "smape": smape,
        "mae": mae,
        "rows": int(len(actual_future)),
        "mape_rows": int(valid_mask.sum()),
        "latency_ms": None if latency_ms is None else float(latency_ms),
    }


def price_loss_function(*, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    huber = torch.nn.functional.smooth_l1_loss(preds, targets)
    pred_norm = preds.squeeze(1) + 1.0
    true_norm = targets.squeeze(1) + 1.0
    valid = torch.abs(true_norm) > 1e-6
    if valid.any():
        rel = torch.abs(pred_norm[valid] - true_norm[valid]) / torch.clamp(torch.abs(true_norm[valid]), min=1e-6)
        return huber + (0.25 * rel.mean())
    return huber


def predict_price_array(*, model: torch.nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    if len(X) == 0:
        return np.array([], dtype=float)

    model.eval()
    preds = []
    with torch.no_grad():
        tensor = torch.tensor(X, dtype=torch.float32).to(device)
        batch_size = 512
        for idx in range(0, len(tensor), batch_size):
            batch = tensor[idx : idx + batch_size]
            out = model(batch).detach().cpu().numpy().flatten()
            preds.extend(out.tolist())
    return np.array(preds, dtype=float)


def fit_fixed_epochs_core(
    *,
    model: torch.nn.Module,
    model_config: Dict[str, object],
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    device: torch.device,
    dataset_cls,
    loss_function_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> torch.nn.Module:
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_config.get("lr", 1e-3)))
    train_loader = DataLoader(
        dataset_cls(X_train, y_train),
        batch_size=int(model_config.get("batch_size", 128)),
        shuffle=True,
        num_workers=0,
    )
    model.train()
    for _ in range(max(1, epochs)):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = loss_function_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def fit_with_early_stopping_core(
    *,
    model_type: str,
    model_config: Dict[str, object],
    input_size: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    current_val: np.ndarray,
    future_val: np.ndarray,
    mape_valid_val: np.ndarray,
    epochs: int,
    device: torch.device,
    build_model_fn: Callable[[str, int, Dict[str, object]], torch.nn.Module],
    dataset_cls,
    loss_function_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    predict_array_fn: Callable[[torch.nn.Module, np.ndarray], np.ndarray],
    evaluate_predictions_fn: Callable[..., Dict[str, Optional[float]]],
) -> Tuple[torch.nn.Module, int]:
    if len(X_val) == 0:
        model = build_model_fn(model_type, input_size, model_config).to(device)
        model = fit_fixed_epochs_core(
            model=model,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            epochs=max(1, epochs),
            device=device,
            dataset_cls=dataset_cls,
            loss_function_fn=loss_function_fn,
        )
        return model, max(1, epochs)

    model = build_model_fn(model_type, input_size, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(model_config.get("lr", 1e-3)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    train_loader = DataLoader(
        dataset_cls(X_train, y_train),
        batch_size=int(model_config.get("batch_size", 128)),
        shuffle=True,
        num_workers=0,
    )

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 1
    best_score = float("inf")
    patience = 6
    wait = 0

    for epoch in range(max(1, epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = loss_function_fn(preds, y_batch)
            loss.backward()
            optimizer.step()

        preds_val = predict_array_fn(model, X_val)
        metrics = evaluate_predictions_fn(current_val, future_val, preds_val, mape_valid_val)
        monitor = metrics["mape"] if metrics["mape"] is not None else metrics["smape"]
        monitor = float(monitor if monitor is not None else 1e9)
        scheduler.step(monitor)

        if monitor < best_score:
            best_score = monitor
            best_epoch = epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch


def train_holdout_pipeline_core(
    *,
    manager,
    holdout_year: Optional[int] = None,
    progress_callback=None,
    epochs: int = 20,
    asset_subset: Optional[Iterable[str]] = None,
    candidate_override: Optional[Dict[str, List[str]]] = None,
    force_refresh_features: bool = False,
    artifact_prefix: str,
    progress_label: str,
) -> Dict[str, object]:
    holdout_year = int(holdout_year or manager.DEFAULT_HOLDOUT_YEAR)
    holdout_dir = os.path.join(manager.holdout_root, str(holdout_year))
    os.makedirs(holdout_dir, exist_ok=True)

    df = manager.load_and_preprocess(force_refresh_features=force_refresh_features)
    feature_registry = manager.build_feature_registry(df)
    active_assets = [a for a in (asset_subset or manager.assets) if a in feature_registry]
    total_jobs = sum(len((candidate_override or {}).get(asset, manager.get_model_candidates(asset))) for asset in active_assets)
    job_index = 0
    metrics: List[Dict[str, object]] = []

    with open(os.path.join(holdout_dir, f"{artifact_prefix}_feature_registry.json"), "w", encoding="utf-8") as handle:
        json.dump(feature_registry, handle, indent=2)

    for asset in active_assets:
        candidates = list((candidate_override or {}).get(asset, manager.get_model_candidates(asset)))
        for model_type in candidates:
            job_index += 1
            if progress_callback:
                pct = int(((job_index - 1) / max(total_jobs, 1)) * 100)
                progress_callback(pct, f"Training {progress_label} price model {asset} {model_type}...")

            entry = manager._train_asset_model(
                df=df,
                asset=asset,
                model_type=model_type,
                feature_cols=feature_registry[asset],
                save_dir=holdout_dir,
                epochs=epochs,
            )
            metrics.append(entry)

            if progress_callback:
                pct = int((job_index / max(total_jobs, 1)) * 100)
                progress_callback(pct, f"Finished {progress_label} price model {asset} {model_type}")

    metrics_path = os.path.join(holdout_dir, f"{artifact_prefix}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    champions = manager.select_champions_from_metrics(metrics, holdout_year=holdout_year)
    champions_path = os.path.join(holdout_dir, f"{artifact_prefix}_champions.json")
    with open(champions_path, "w", encoding="utf-8") as handle:
        json.dump(champions, handle, indent=2)

    if progress_callback:
        progress_callback(100, f"{progress_label} price training complete for holdout {holdout_year}")

    return {
        "holdout_dir": holdout_dir,
        "metrics_path": metrics_path,
        "champions_path": champions_path,
        "metrics": metrics,
        "champions": champions,
    }


def train_asset_model_core(
    *,
    manager,
    df: pd.DataFrame,
    asset: str,
    model_type: str,
    feature_cols: List[str],
    save_dir: str,
    epochs: int,
) -> Dict[str, object]:
    model_config = copy.deepcopy(manager.MODEL_CONFIGS[model_type])
    window_size = int(model_config["window_size"])
    sequence_payload = manager._build_sequence_payload(df, asset, feature_cols, window_size)

    baseline_holdout = manager._evaluate_predictions(
        current_prices=sequence_payload["current_prices"][sequence_payload["holdout_mask"]],
        actual_future=sequence_payload["future_prices"][sequence_payload["holdout_mask"]],
        pred_centered=np.zeros(np.sum(sequence_payload["holdout_mask"])),
        mape_valid=sequence_payload["mape_valid"][sequence_payload["holdout_mask"]],
    )

    discovery_results = []
    for fold in manager.DISCOVERY_FOLDS:
        discovery_results.append(
            manager._run_discovery_fold(
                sequence_payload=sequence_payload,
                model_type=model_type,
                model_config=model_config,
                fold=fold,
                epochs=epochs,
            )
        )

    discovery_mapes = [x["metrics"]["mape"] for x in discovery_results if x["metrics"]["mape"] is not None]
    discovery_summary = {
        "median_mape": manager._safe_stat(discovery_mapes, np.median),
        "std_mape": manager._safe_stat(discovery_mapes, np.std),
        "folds": discovery_results,
    }

    final_artifacts = manager._train_final_model(
        df=df,
        asset=asset,
        model_type=model_type,
        model_config=model_config,
        feature_cols=feature_cols,
        epochs=epochs,
        save_dir=save_dir,
    )

    holdout_metrics = manager._evaluate_predictions(
        current_prices=final_artifacts["holdout_current_prices"],
        actual_future=final_artifacts["holdout_future_prices"],
        pred_centered=final_artifacts["holdout_predictions"],
        mape_valid=final_artifacts["holdout_mape_valid"],
        latency_ms=final_artifacts["latency_ms"],
    )

    shadow_metrics = manager._evaluate_predictions(
        current_prices=final_artifacts["shadow_current_prices"],
        actual_future=final_artifacts["shadow_future_prices"],
        pred_centered=final_artifacts["shadow_predictions"],
        mape_valid=final_artifacts["shadow_mape_valid"],
    )

    metadata = {
        "asset": asset,
        "model_type": model_type,
        "window_size": window_size,
        "feature_cols": feature_cols,
        "model_config": model_config,
        "best_epochs": int(final_artifacts["best_epochs"]),
        "trained_until": manager.OUTER_HOLDOUT_START,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "weight_file": os.path.basename(final_artifacts["weight_file"]),
        "scaler_file": os.path.basename(final_artifacts["scaler_file"]),
    }
    with open(final_artifacts["meta_file"], "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    return {
        "asset": asset,
        "model_type": model_type,
        "group": manager._asset_group(asset),
        "feature_count": len(feature_cols),
        "discovery_summary": discovery_summary,
        "holdout_metrics": holdout_metrics,
        "shadow_metrics": shadow_metrics,
        "baseline_2025": baseline_holdout,
        "artifacts": {
            "weight_file": final_artifacts["weight_file"],
            "scaler_file": final_artifacts["scaler_file"],
            "meta_file": final_artifacts["meta_file"],
        },
    }


def run_discovery_fold_core(
    *,
    manager,
    sequence_payload: Dict[str, np.ndarray],
    model_type: str,
    model_config: Dict[str, object],
    fold,
    epochs: int,
) -> Dict[str, object]:
    seq_dates = sequence_payload["dates"]
    train_mask = seq_dates <= np.datetime64(fold.train_end)
    val_mask = (seq_dates >= np.datetime64(fold.val_start)) & (seq_dates <= np.datetime64(fold.val_end))
    if train_mask.sum() < 32 or val_mask.sum() < 8:
        return {"fold": fold.name, "metrics": {"mape": None, "smape": None, "mae": None, "rows": int(val_mask.sum())}}

    X_train = sequence_payload["X"][train_mask]
    y_train = sequence_payload["y"][train_mask]
    X_val = sequence_payload["X"][val_mask]
    y_val = sequence_payload["y"][val_mask]
    current_val = sequence_payload["current_prices"][val_mask]
    future_val = sequence_payload["future_prices"][val_mask]
    mape_val = sequence_payload["mape_valid"][val_mask]

    model, best_epochs = manager._fit_with_early_stopping(
        model_type=model_type,
        model_config=model_config,
        input_size=X_train.shape[2],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        current_val=current_val,
        future_val=future_val,
        mape_valid_val=mape_val,
        epochs=epochs,
    )
    preds = manager._predict_array(model, X_val)
    metrics = manager._evaluate_predictions(current_val, future_val, preds, mape_val)
    metrics["best_epochs"] = int(best_epochs)
    return {"fold": fold.name, "metrics": metrics}


def train_final_model_core(
    *,
    manager,
    df: pd.DataFrame,
    asset: str,
    model_type: str,
    model_config: Dict[str, object],
    feature_cols: List[str],
    epochs: int,
    save_dir: str,
    target_suffix: str,
    build_model_fn: Callable[[str, int, Dict[str, object]], torch.nn.Module],
) -> Dict[str, object]:
    window_size = int(model_config["window_size"])
    centered_col = f"CenteredNormPrice_{asset}_{target_suffix}"
    future_col = f"FuturePrice_{asset}_{target_suffix}"
    mape_col = f"MAPEValid_{asset}_{target_suffix}"

    tune_train_mask = df.index <= pd.Timestamp(manager.FINAL_TUNE_TRAIN_END)
    full_train_mask = df.index <= pd.Timestamp(f"{manager.DEFAULT_HOLDOUT_YEAR - 1}-12-31")
    scaler_tune = manager.fit_scaler(df, feature_cols, tune_train_mask)
    scaled_tune = manager._scale_frame(df, feature_cols, scaler_tune)
    tune_payload = manager._create_sequences(
        scaled_features=scaled_tune,
        centered_targets=df[centered_col].values,
        current_prices=df[asset].values,
        future_prices=df[future_col].values,
        mape_valid=df[mape_col].values.astype(bool),
        dates=df.index.values,
        window_size=window_size,
    )

    tune_seq_dates = tune_payload["dates"]
    tune_train_seq_mask = tune_seq_dates <= np.datetime64(manager.FINAL_TUNE_TRAIN_END)
    tune_val_seq_mask = (tune_seq_dates >= np.datetime64(manager.FINAL_TUNE_VAL_START)) & (
        tune_seq_dates <= np.datetime64(manager.FINAL_TUNE_VAL_END)
    )

    _, best_epochs = manager._fit_with_early_stopping(
        model_type=model_type,
        model_config=model_config,
        input_size=tune_payload["X"][tune_train_seq_mask].shape[2],
        X_train=tune_payload["X"][tune_train_seq_mask],
        y_train=tune_payload["y"][tune_train_seq_mask],
        X_val=tune_payload["X"][tune_val_seq_mask],
        y_val=tune_payload["y"][tune_val_seq_mask],
        current_val=tune_payload["current_prices"][tune_val_seq_mask],
        future_val=tune_payload["future_prices"][tune_val_seq_mask],
        mape_valid_val=tune_payload["mape_valid"][tune_val_seq_mask],
        epochs=epochs,
    )
    best_epochs = max(1, int(best_epochs))

    scaler_final = manager.fit_scaler(df, feature_cols, full_train_mask)
    scaled_final = manager._scale_frame(df, feature_cols, scaler_final)
    final_payload = manager._create_sequences(
        scaled_features=scaled_final,
        centered_targets=df[centered_col].values,
        current_prices=df[asset].values,
        future_prices=df[future_col].values,
        mape_valid=df[mape_col].values.astype(bool),
        dates=df.index.values,
        window_size=window_size,
    )

    train_seq_mask = final_payload["dates"] <= np.datetime64(f"{manager.DEFAULT_HOLDOUT_YEAR - 1}-12-31")
    final_model = build_model_fn(model_type, final_payload["X"][train_seq_mask].shape[2], model_config).to(manager.device)
    final_model = manager._fit_fixed_epochs(
        model=final_model,
        model_config=model_config,
        X_train=final_payload["X"][train_seq_mask],
        y_train=final_payload["y"][train_seq_mask],
        epochs=best_epochs,
    )

    holdout_mask = (final_payload["dates"] >= np.datetime64(manager.OUTER_HOLDOUT_START)) & (
        final_payload["dates"] <= np.datetime64(manager.OUTER_HOLDOUT_END)
    )
    shadow_mask = final_payload["dates"] >= np.datetime64(manager.SHADOW_START)

    start = time.perf_counter()
    holdout_predictions = manager._predict_array(final_model, final_payload["X"][holdout_mask])
    latency_ms = ((time.perf_counter() - start) * 1000.0) / max(len(holdout_predictions), 1)
    shadow_predictions = manager._predict_array(final_model, final_payload["X"][shadow_mask])

    prefix = os.path.join(save_dir, f"{asset}_{model_type}")
    weight_file = f"{prefix}.pth"
    scaler_file = f"{prefix}_scaler.pkl"
    meta_file = f"{prefix}_meta.json"
    torch.save(final_model.state_dict(), weight_file)
    joblib.dump(scaler_final, scaler_file)

    return {
        "weight_file": weight_file,
        "scaler_file": scaler_file,
        "meta_file": meta_file,
        "best_epochs": best_epochs,
        "holdout_predictions": holdout_predictions.tolist(),
        "holdout_current_prices": final_payload["current_prices"][holdout_mask].tolist(),
        "holdout_future_prices": final_payload["future_prices"][holdout_mask].tolist(),
        "holdout_mape_valid": final_payload["mape_valid"][holdout_mask].tolist(),
        "shadow_predictions": shadow_predictions.tolist(),
        "shadow_current_prices": final_payload["current_prices"][shadow_mask].tolist(),
        "shadow_future_prices": final_payload["future_prices"][shadow_mask].tolist(),
        "shadow_mape_valid": final_payload["mape_valid"][shadow_mask].tolist(),
        "latency_ms": latency_ms,
    }


def promote_price_champions(
    *,
    manager,
    holdout_year: Optional[int],
    artifact_prefix: str,
) -> Dict[str, object]:
    holdout_dir = manager._resolve_holdout_dir(holdout_year)
    champions_path = os.path.join(holdout_dir, f"{artifact_prefix}_champions.json")
    metrics_path = os.path.join(holdout_dir, f"{artifact_prefix}_metrics.json")
    registry_path = os.path.join(holdout_dir, f"{artifact_prefix}_feature_registry.json")

    if not os.path.exists(champions_path):
        raise FileNotFoundError(f"Champion file not found at {champions_path}")

    with open(champions_path, "r", encoding="utf-8") as handle:
        champions = json.load(handle)

    os.makedirs(manager.model_dir, exist_ok=True)
    copied = []
    for _, info in champions.get("assets", {}).items():
        if info.get("deployment_strategy") != "model":
            continue
        prefix = info.get("artifact_prefix")
        if not prefix:
            continue
        for suffix in [".pth", "_scaler.pkl", "_meta.json"]:
            src = os.path.join(holdout_dir, f"{prefix}{suffix}")
            if os.path.exists(src):
                dst = os.path.join(manager.model_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                copied.append(dst)

    for path in [metrics_path, champions_path, registry_path]:
        if os.path.exists(path):
            shutil.copy2(path, os.path.join(manager.model_dir, os.path.basename(path)))

    champions["promoted_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    champions["promoted_from"] = holdout_dir
    with open(os.path.join(manager.model_dir, f"{artifact_prefix}_champions.json"), "w", encoding="utf-8") as handle:
        json.dump(champions, handle, indent=2)

    return {
        "model_dir": manager.model_dir,
        "copied_files": copied,
        "champions": champions,
    }


def predict_latest_price(
    *,
    manager,
    build_model_fn: Callable[[str, int, Dict[str, object]], torch.nn.Module],
    artifact_prefix: str,
    model_dir: Optional[str] = None,
    year: Optional[int] = None,
):
    artifact_dir = model_dir or (manager._resolve_holdout_dir(year) if year is not None else manager.model_dir)
    champions_path = os.path.join(artifact_dir, f"{artifact_prefix}_champions.json")
    if not os.path.exists(champions_path):
        raise FileNotFoundError(f"Champion file not found at {champions_path}")

    with open(champions_path, "r", encoding="utf-8") as handle:
        champions = json.load(handle).get("assets", {})

    df = manager.load_and_preprocess(force_refresh_features=False)
    results = []
    latest_date = df.index.max()
    target_date = estimate_target_date(latest_date=latest_date, horizon_days=manager.HORIZON_DAYS)

    for asset in manager.assets:
        if asset not in df.columns or asset not in champions:
            continue

        current_price = float(df[asset].iloc[-1])
        champion = champions[asset]
        if champion.get("deployment_strategy") != "model":
            predicted_future = current_price
            predicted_norm = 1.0
            selected_model = "naive_last"
        else:
            prefix = champion.get("artifact_prefix")
            weight_path = os.path.join(artifact_dir, f"{prefix}.pth")
            meta = load_price_model_metadata(
                manager=manager,
                asset=asset,
                model_type=str(champion.get("selected_model", "")),
                model_dir=artifact_dir,
            )
            feature_cols = list(meta["feature_cols"])
            scaler = load_price_model_scaler(
                manager=manager,
                asset=asset,
                model_type=str(champion.get("selected_model", "")),
                model_dir=artifact_dir,
            )
            frame = df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
            scaled = scaler.transform(frame)
            window_size = int(meta["model_config"]["window_size"])
            sequence = scaled[-window_size:]
            if len(sequence) < window_size:
                continue

            model = build_model_fn(meta["model_type"], len(feature_cols), meta["model_config"]).to(manager.device)
            model.load_state_dict(torch.load(weight_path, map_location=manager.device, weights_only=False))
            pred_centered = manager._predict_array(model, np.array([sequence], dtype=float))[0]
            predicted_norm = float(np.clip(pred_centered + 1.0, -5.0, 5.0))
            predicted_future = float(current_price * predicted_norm)
            selected_model = meta["model_type"]

        results.append(
            {
                "asset": asset,
                "model": selected_model,
                "deployment_strategy": champion.get("deployment_strategy"),
                "as_of_date": latest_date.strftime("%Y-%m-%d"),
                "target_date": target_date.strftime("%Y-%m-%d"),
                "current_price": current_price,
                "predicted_norm_price": predicted_norm,
                "predicted_future_price": predicted_future,
                "implied_return_pct": round((predicted_norm - 1.0) * 100.0, 4),
                "baseline_2025_mape": champion.get("baseline_2025_mape"),
                "selected_holdout_mape": champion.get("selected_holdout_mape"),
            }
        )

    return results


def load_price_metrics(
    *,
    manager,
    artifact_prefix: str,
    holdout_year: Optional[int] = None,
    model_dir: Optional[str] = None,
) -> Dict[str, object]:
    if model_dir:
        base_dir = model_dir
    elif holdout_year is not None:
        base_dir = manager._resolve_holdout_dir(holdout_year)
    else:
        base_dir = manager._resolve_default_metrics_dir()

    metrics_path = os.path.join(base_dir, f"{artifact_prefix}_metrics.json")
    champions_path = os.path.join(base_dir, f"{artifact_prefix}_champions.json")
    registry_path = os.path.join(base_dir, f"{artifact_prefix}_feature_registry.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found at {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    champions = {}
    if os.path.exists(champions_path):
        with open(champions_path, "r", encoding="utf-8") as handle:
            champions = json.load(handle)

    feature_registry = {}
    if os.path.exists(registry_path):
        feature_registry = load_price_feature_registry(
            manager=manager,
            artifact_prefix=artifact_prefix,
            model_dir=base_dir,
        )

    return {
        "directory": base_dir,
        "metrics": metrics,
        "champions": champions,
        "feature_registry": feature_registry,
    }


def resolve_price_artifact_dir(
    *,
    manager,
    holdout_year: Optional[int] = None,
    model_dir: Optional[str] = None,
) -> str:
    if model_dir:
        return model_dir
    if holdout_year is not None:
        return manager._resolve_holdout_dir(holdout_year)
    return manager._resolve_default_metrics_dir()


def load_price_feature_registry(
    *,
    manager,
    artifact_prefix: str,
    holdout_year: Optional[int] = None,
    model_dir: Optional[str] = None,
) -> Dict[str, object]:
    base_dir = resolve_price_artifact_dir(
        manager=manager,
        holdout_year=holdout_year,
        model_dir=model_dir,
    )
    registry_path = os.path.join(base_dir, f"{artifact_prefix}_feature_registry.json")
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"Feature registry not found at {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def load_price_model_metadata(
    *,
    manager,
    asset: str,
    model_type: str,
    holdout_year: Optional[int] = None,
    model_dir: Optional[str] = None,
) -> Dict[str, object]:
    base_dir = resolve_price_artifact_dir(
        manager=manager,
        holdout_year=holdout_year,
        model_dir=model_dir,
    )
    prefix = f"{asset}_{model_type}"
    meta_path = os.path.join(base_dir, f"{prefix}_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Model metadata not found at {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Invalid metadata payload at {meta_path}")
    return loaded


def load_price_model_scaler(
    *,
    manager,
    asset: str,
    model_type: str,
    holdout_year: Optional[int] = None,
    model_dir: Optional[str] = None,
):
    base_dir = resolve_price_artifact_dir(
        manager=manager,
        holdout_year=holdout_year,
        model_dir=model_dir,
    )
    prefix = f"{asset}_{model_type}"
    scaler_path = os.path.join(base_dir, f"{prefix}_scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler artifact not found at {scaler_path}")
    return joblib.load(scaler_path)


def resolve_holdout_dir(*, holdout_root: str, holdout_year: Optional[int], horizon_label: str) -> str:
    if holdout_year is not None:
        directory = os.path.join(holdout_root, str(holdout_year))
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Holdout directory not found at {directory}")
        return directory

    candidates = [x for x in os.listdir(holdout_root) if os.path.isdir(os.path.join(holdout_root, x))]
    if not candidates:
        raise FileNotFoundError(f"No holdout {horizon_label} directories are available.")
    latest = sorted(candidates)[-1]
    return os.path.join(holdout_root, latest)


def resolve_default_metrics_dir(*, model_dir: str, metrics_filename: str, resolve_holdout_dir_fn: Callable[[], str]) -> str:
    if os.path.exists(os.path.join(model_dir, metrics_filename)):
        return model_dir
    return resolve_holdout_dir_fn()


def estimate_target_date(*, latest_date: pd.Timestamp, horizon_days: int) -> pd.Timestamp:
    return pd.bdate_range(latest_date, periods=horizon_days + 1)[-1]


def own_asset_features(asset: str) -> List[str]:
    cols = [asset]
    for lag in [1, 2, 3, 5, 10]:
        cols.extend([f"{asset}_lag_{lag}", f"{asset}_ret_{lag}"])
    for window in [5, 10, 20]:
        cols.append(f"{asset}_vol_{window}")
    for window in [10, 20, 60]:
        cols.extend([f"{asset}_drawdown_{window}", f"{asset}_trend_{window}", f"{asset}_zscore_{window}"])
    return cols


def asset_group(asset: str, asset_groups: Dict[str, List[str]]) -> str:
    for name, group_assets in asset_groups.items():
        if asset in group_assets:
            return name
    return "other"


def safe_stat(values: List[float], fn):
    clean = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not clean:
        return None
    return float(fn(clean))


def sort_nan(value):
    if value is None or not np.isfinite(value):
        return 1e12
    return float(value)
