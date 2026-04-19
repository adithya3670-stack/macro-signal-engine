from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict

import joblib
import numpy as np
import pandas as pd
import torch


def backtest_simple_standard_core(
    *,
    dl_model,
    snapshot_id: str,
    model_type: str,
    asset: str,
    initial_capital: float,
    start_date,
    end_date,
    threshold: float = 0.6,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
) -> Dict[str, Any]:
    """
    Robust backtesting engine extracted from DLMacroModel monolith.
    """
    if snapshot_id == "latest":
        base_path = dl_model.model_dir
    else:
        base_path = os.path.join(dl_model.model_dir, "snapshots", snapshot_id)
        if not os.path.exists(base_path):
            return {"error": f"Snapshot {snapshot_id} not found."}

    df = dl_model.load_and_preprocess()

    scaler_path = os.path.join(base_path, "scaler.pkl")
    if not os.path.exists(scaler_path):
        return {"error": "Scaler not found for backtest."}
    scaler = joblib.load(scaler_path)

    exclude_cols = [c for c in df.columns if "Target_" in c or "Regime_" in c or "Date" in c]
    numeric_cols = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    df_scaled = df.copy()
    try:
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
    except Exception:
        valid_cols = [c for c in numeric_cols if c in getattr(scaler, "feature_names_in_", numeric_cols)]
        if valid_cols:
            df_scaled[valid_cols] = scaler.transform(df[valid_cols])

    config_path = os.path.join(base_path, "dl_config.json")
    dl_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            dl_config = json.load(handle)

    models_to_run = ["lstm", "transformer", "nbeats"] if model_type == "ensemble" else [model_type]
    final_preds_df = pd.DataFrame(index=df.index, columns=models_to_run)

    for current_model_type in models_to_run:
        model_key = f"{asset}_{current_model_type}"
        params = dl_config.get(model_key, {})
        if not params:
            params = {"window_size": 60, "hidden_size": 128, "num_layers": 2, "dropout": 0.2, "num_features": 20}

        window_size = int(params.get("window_size", 60))

        feat_file = os.path.join(base_path, f"dl_features_{current_model_type}.json")
        spec_features = []
        if os.path.exists(feat_file):
            with open(feat_file, "r", encoding="utf-8") as handle:
                feat_map = json.load(handle)
                spec_features = feat_map.get(model_key, [])

        if not spec_features:
            spec_features = numeric_cols[: int(params.get("num_features", 20))]

        try:
            X_values = df_scaled[spec_features].values
        except Exception:
            continue

        X_seq = []
        valid_indices = []
        for i in range(window_size, len(X_values)):
            seq = X_values[i - window_size : i]
            X_seq.append(seq)
            valid_indices.append(i)

        if not X_seq:
            continue

        X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32).to(dl_model.device)

        try:
            inp = len(spec_features)
            if current_model_type == "lstm":
                model = lstm_cls(
                    input_size=inp,
                    hidden_size=params.get("hidden_size", 128),
                    num_layers=params.get("num_layers", 2),
                    num_heads=4,
                    dropout=params.get("dropout", 0.2),
                ).to(dl_model.device)
            elif current_model_type == "transformer":
                dm = params.get("trans_d_model", 128)
                nh = params.get("trans_nhead", 8)
                if dm % nh != 0:
                    dm = (dm // nh) * nh
                model = transformer_cls(
                    input_size=inp,
                    d_model=dm,
                    nhead=nh,
                    num_layers=params.get("trans_layers", 3),
                    dropout=params.get("dropout", 0.2),
                ).to(dl_model.device)
            elif current_model_type == "nbeats":
                model = nbeats_cls(
                    num_features=inp,
                    window_size=window_size,
                    num_stacks=params.get("nb_stacks", 3),
                    num_blocks=params.get("nb_blocks", 3),
                    layer_width=params.get("nb_width", 256),
                ).to(dl_model.device)
            else:
                continue

            weight_file = os.path.join(base_path, f"{asset}_{current_model_type}.pt")
            if os.path.exists(weight_file):
                model.load_state_dict(torch.load(weight_file, map_location=dl_model.device))
            else:
                continue

            model.eval()
            with torch.no_grad():
                raw = []
                for k in range(0, len(X_tensor), 512):
                    out = model(X_tensor[k : k + 512])
                    raw.extend(out.cpu().numpy().flatten())

                probs = 1 / (1 + np.exp(-np.array(raw)))
                target_dates = df.index[valid_indices]
                final_preds_df.loc[target_dates, current_model_type] = probs
        except Exception as exc:
            print(f"Error {current_model_type}: {exc}")

    if model_type == "ensemble":
        final_preds_df["ensemble"] = final_preds_df[models_to_run].mean(axis=1)
    else:
        final_preds_df["ensemble"] = final_preds_df[model_type]

    start_dt = pd.to_datetime(start_date)
    if end_date:
        end_dt = pd.to_datetime(end_date)
        sim = final_preds_df[(final_preds_df.index >= start_dt) & (final_preds_df.index <= end_dt)]
    else:
        sim = final_preds_df[final_preds_df.index >= start_dt]
    sim = sim.dropna(subset=["ensemble"])

    if len(sim) == 0:
        return {"error": "No predictions."}

    prices = df.loc[sim.index]["Close"] if "Close" in df.columns else df.loc[sim.index]["SP500"]
    probs = sim["ensemble"].values.astype(float)
    dates = sim.index

    cash = float(initial_capital)
    units = 0.0
    pos = 0
    eq_curve = []
    trades = []

    review_days = 1
    if "_1w" in asset:
        review_days = 5
    elif "_1m" in asset:
        review_days = 21
    elif "_3m" in asset:
        review_days = 63

    next_review_idx = 0
    for i in range(len(dates)):
        date = dates[i].strftime("%Y-%m-%d")
        price = float(prices.iloc[i])

        if i >= next_review_idx:
            prob = probs[i]
            if prob > threshold and pos == 0:
                units = cash / price
                cash = 0
                pos = 1
                trades.append({"date": date, "type": "BUY", "price": price, "prob": prob})
            elif prob < (1.0 - threshold) and pos == 1:
                cash = units * price
                units = 0
                pos = 0
                trades.append({"date": date, "type": "SELL", "price": price, "prob": prob})
            next_review_idx = i + review_days

        equity = cash + (units * price)
        eq_curve.append({"date": date, "equity": equity, "price": price, "signal": float(probs[i])})

    return {
        "status": "success",
        "metrics": {
            "total_return": (eq_curve[-1]["equity"] - initial_capital) / initial_capital,
            "final_equity": eq_curve[-1]["equity"],
            "trades": len(trades),
            "signal_min": float(np.min(probs)),
            "signal_max": float(np.max(probs)),
            "signal_std": float(np.std(probs)),
        },
        "equity_curve": eq_curve,
        "trades": trades,
    }
