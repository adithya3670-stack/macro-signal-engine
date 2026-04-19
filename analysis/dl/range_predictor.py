from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch


def predict_range_core(
    *,
    dl_model,
    start_date,
    end_date,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
) -> pd.DataFrame:
    """
    Range prediction engine extracted from DLMacroModel monolith.
    """
    df = dl_model.load_and_preprocess()

    max_window = 90
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    if pd.isna(end_dt):
        end_dt = df.index.max()

    buffer_start = start_dt - timedelta(days=max_window * 3)
    mask = (df.index >= buffer_start) & (df.index <= end_dt)
    df_range = df.loc[mask].copy()
    df_range = df_range.ffill().fillna(0)

    if len(df_range) < max_window:
        return pd.DataFrame()

    try:
        scaler = joblib.load(f"{dl_model.model_dir}/scaler.pkl")
        superset_cols = joblib.load(f"{dl_model.model_dir}/features.pkl")
    except Exception:
        print("DL Predict: Scaler or Features not found.")
        return pd.DataFrame()

    results_df = pd.DataFrame(index=df.index)

    types_found = []
    for model_type in ["lstm", "transformer", "nbeats"]:
        if os.path.exists(f"{dl_model.model_dir}/model_features_{model_type}.json"):
            types_found.append(model_type)
    if not types_found:
        return pd.DataFrame()

    for model_type in types_found:
        try:
            with open(f"{dl_model.model_dir}/model_features_{model_type}.json", "r", encoding="utf-8") as handle:
                feat_map = json.load(handle)
        except Exception:
            continue

        try:
            X_all_scaled = scaler.transform(df_range[superset_cols])
            df_scaled = pd.DataFrame(X_all_scaled, columns=superset_cols, index=df_range.index)
        except Exception as exc:
            print(f"DL Predict: Scaling error {exc}")
            continue

        for asset in dl_model.assets:
            for horizon in dl_model.horizons:
                model_key = f"{asset}_{horizon}_{model_type}"
                if model_key not in feat_map:
                    continue

                spec_features = feat_map[model_key]
                window_size, params = dl_model.get_horizon_config(horizon, model_type, asset=asset)

                interest_mask = (df_scaled.index >= start_dt) & (df_scaled.index <= end_dt)
                interest_dates = df_scaled.index[interest_mask]
                if len(interest_dates) == 0:
                    continue

                X_vals = df_scaled[spec_features].values
                dates_vals = df_scaled.index

                sequences = []
                target_dates = []
                try:
                    start_idx = df_scaled.index.get_loc(interest_dates[0])
                    if isinstance(start_idx, slice):
                        start_idx = start_idx.start
                except Exception:
                    continue

                end_idx = df_scaled.index.get_loc(interest_dates[-1])
                if isinstance(end_idx, slice):
                    end_idx = end_idx.stop - 1

                for i in range(start_idx, end_idx + 1):
                    if i < window_size - 1:
                        continue
                    seq = X_vals[i - window_size + 1 : i + 1]
                    if len(seq) == window_size:
                        sequences.append(seq)
                        target_dates.append(dates_vals[i])

                if not sequences:
                    continue

                try:
                    inp_dim = len(spec_features)
                    if model_type == "transformer":
                        model = transformer_cls(
                            inp_dim,
                            params["trans_d_model"],
                            params["trans_nhead"],
                            params["trans_layers"],
                            params["trans_dropout"],
                        ).to(dl_model.device)
                    elif model_type == "nbeats":
                        model = nbeats_cls(
                            inp_dim,
                            window_size,
                            params["nb_stacks"],
                            params["nb_blocks"],
                            params["nb_width"],
                        ).to(dl_model.device)
                    else:
                        model = lstm_cls(
                            inp_dim,
                            params["hidden_size"],
                            params["num_layers"],
                            16,
                            params["dropout"],
                        ).to(dl_model.device)

                    model_path = f"{dl_model.model_dir}/{asset}_{horizon}_{model_type}.pt"
                    if not os.path.exists(model_path):
                        continue

                    model.load_state_dict(torch.load(model_path, map_location=dl_model.device, weights_only=True))
                    model.eval()

                    X_tensor = torch.FloatTensor(np.array(sequences)).to(dl_model.device)
                    preds = []
                    bs = 512
                    with torch.no_grad():
                        for k in range(0, len(X_tensor), bs):
                            batch = X_tensor[k : k + bs]
                            out = model(batch)
                            probs = torch.sigmoid(out).cpu().numpy().flatten()
                            preds.extend(probs)

                    col_name = f"Pred_{asset}_{horizon}_{model_type}"
                    results_df.loc[target_dates, col_name] = preds
                except Exception as exc:
                    print(f"DL Predict Error {asset} {horizon}: {exc}")
                    continue

    final_mask = (results_df.index >= start_dt) & (results_df.index <= end_dt)
    return results_df.loc[final_mask]


def predict_latest_from_range_core(
    *,
    dl_model,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
) -> List[Dict[str, object]]:
    """
    Latest-point prediction wrapper around range predictions.
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    df = predict_range_core(
        dl_model=dl_model,
        start_date=start_date,
        end_date=end_date,
        lstm_cls=lstm_cls,
        transformer_cls=transformer_cls,
        nbeats_cls=nbeats_cls,
    )
    if df.empty:
        return []

    latest_row = df.iloc[-1]
    predictions = []
    for col in df.columns:
        parts = col.split("_")
        if len(parts) < 4 or parts[0] != "Pred":
            continue

        asset = parts[1]
        horizon = parts[2]
        model_type = parts[3]
        try:
            score = float(latest_row[col])
        except Exception:
            score = 0.5

        direction = "BULLISH" if score > 0.5 else "BEARISH"
        conf = score if direction == "BULLISH" else (1 - score)
        predictions.append(
            {
                "asset": asset,
                "horizon": horizon,
                "model_type": model_type,
                "direction": direction,
                "confidence": f"{conf:.1%}",
                "raw_score": round(score, 2),
            },
        )
    return predictions
