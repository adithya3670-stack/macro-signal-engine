from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import pandas as pd
import torch


def predict_latest_by_model(
    *,
    dl_model,
    model_type: str,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
) -> List[Dict[str, Any]]:
    df = dl_model.load_and_preprocess()

    if not os.path.exists(f"{dl_model.model_dir}/scaler.pkl"):
        return []

    scaler = joblib.load(f"{dl_model.model_dir}/scaler.pkl")
    features_map_file = f"{dl_model.model_dir}/model_features_{model_type}.json"

    if not os.path.exists(features_map_file):
        if os.path.exists(f"{dl_model.model_dir}/features.pkl"):
            all_feature_cols = joblib.load(f"{dl_model.model_dir}/features.pkl")
            model_features_map = {
                f"{asset}_{horizon}_{model_type}": all_feature_cols
                for asset in dl_model.assets
                for horizon in dl_model.horizons
            }
        else:
            return []
    else:
        with open(features_map_file, "r", encoding="utf-8") as handle:
            model_features_map = json.load(handle)

    if os.path.exists(f"{dl_model.model_dir}/features.pkl"):
        all_feature_cols = joblib.load(f"{dl_model.model_dir}/features.pkl")
    else:
        return []

    last_chunk_all = df[all_feature_cols].tail(150)
    if len(last_chunk_all) < 30:
        return []

    try:
        X_raw_all = scaler.transform(last_chunk_all)
        X_df_scaled = pd.DataFrame(X_raw_all, columns=all_feature_cols)
    except Exception as exc:
        print(f"Prediction Scaling Error: {exc}")
        return []

    if model_type == "ensemble":
        models = ["lstm", "transformer", "nbeats"]
        all_preds_map: Dict[str, List[Dict[str, Any]]] = {}

        for sub_model in models:
            try:
                preds = predict_latest_by_model(
                    dl_model=dl_model,
                    model_type=sub_model,
                    lstm_cls=lstm_cls,
                    transformer_cls=transformer_cls,
                    nbeats_cls=nbeats_cls,
                )
                for pred in preds:
                    key = f"{pred['asset']}_{pred['horizon']}"
                    all_preds_map.setdefault(key, []).append(
                        {
                            "raw": pred.get("raw_score", 0.5),
                            "metrics": pred.get("backtest_metrics", {}),
                        },
                    )
            except Exception:
                continue

        final_predictions = []
        for key, pred_list in all_preds_map.items():
            if not pred_list:
                continue
            asset, horizon = key.split("_")
            probs = []
            weights = []
            for item in pred_list:
                prob = item["raw"]
                metrics = item["metrics"]
                m_f1 = metrics.get("f1", 0.5)
                m_acc = metrics.get("accuracy", 0.5)
                score = (0.7 * m_f1 + 0.3 * m_acc) ** 2
                probs.append(prob)
                weights.append(score)

            total_weight = sum(weights)
            avg_prob = np.mean(probs) if total_weight == 0 else sum(p * w for p, w in zip(probs, weights)) / total_weight
            direction = "BULLISH" if avg_prob > 0.5 else "BEARISH"
            confidence = abs(avg_prob - 0.5) * 2
            avg_acc = np.mean([item["metrics"].get("accuracy", 0.5) for item in pred_list])
            avg_f1 = np.mean([item["metrics"].get("f1", 0.5) for item in pred_list])
            avg_auc = np.mean([item["metrics"].get("auc", 0.5) for item in pred_list])

            final_predictions.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "direction": direction,
                    "confidence": f"{confidence*100:.1f}%",
                    "raw_score": avg_prob,
                    "drivers": ["Ensemble Consensus"],
                    "model_scores": {"RF": f"{avg_prob:.2f}", "XGB": "N/A"},
                    "backtest_metrics": {"accuracy": avg_acc, "f1": avg_f1, "roc_auc": avg_auc},
                },
            )
        return final_predictions

    predictions = []
    metrics_file = f"dl_metrics_{model_type}.json"
    metrics_map = {}
    metrics_path = f"{dl_model.model_dir}/{metrics_file}"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as handle:
            for metric_row in json.load(handle):
                metrics_map[f"{metric_row['asset']}_{metric_row['horizon']}"] = metric_row["metrics"]

    for asset in dl_model.assets:
        for horizon in dl_model.horizons:
            threshold = 0.5
            lookup_key = f"{asset}_{horizon}"
            if lookup_key in metrics_map:
                threshold = metrics_map[lookup_key].get("threshold", 0.5)

            window_size, params = dl_model.get_horizon_config(horizon, model_type, asset=asset)
            model_key_full = f"{asset}_{horizon}_{model_type}"
            if model_key_full not in model_features_map:
                continue

            selected_feats = model_features_map[model_key_full]
            last_chunk = X_df_scaled[selected_feats].tail(window_size)
            if len(last_chunk) < window_size:
                continue

            X_seq = torch.FloatTensor(np.array([last_chunk.values])).to(dl_model.device)
            base_name = f"{asset}_{horizon}_{model_type}"
            base_path = f"{dl_model.model_dir}/{base_name}"

            ensemble_folds = []
            for i in range(3):
                fold_path = f"{base_path}_fold{i}.pth"
                if os.path.exists(fold_path):
                    ensemble_folds.append(fold_path)

            single_model_path = None
            if not ensemble_folds:
                if os.path.exists(f"{base_path}.pth"):
                    single_model_path = f"{base_path}.pth"
                elif os.path.exists(f"{base_path}.pt"):
                    single_model_path = f"{base_path}.pt"
                elif model_type == "lstm" and os.path.exists(f"{dl_model.model_dir}/{asset}_{horizon}.pt"):
                    single_model_path = f"{dl_model.model_dir}/{asset}_{horizon}.pt"
                else:
                    continue

            try:
                if model_type == "transformer":
                    model = transformer_cls(
                        input_size=len(selected_feats),
                        d_model=params.get("trans_d_model", 128),
                        nhead=params.get("trans_nhead", 8),
                        num_layers=params.get("trans_layers", 3),
                        dropout=params.get("trans_dropout", 0.1),
                    ).to(dl_model.device)
                elif model_type == "nbeats":
                    model = nbeats_cls(
                        num_features=len(selected_feats),
                        window_size=window_size,
                        num_stacks=params.get("nb_stacks", 2),
                        num_blocks=params.get("nb_blocks", 3),
                        layer_width=params.get("nb_width", 128),
                    ).to(dl_model.device)
                else:
                    model = lstm_cls(
                        input_size=len(selected_feats),
                        hidden_size=params.get("hidden_size", 128),
                        num_layers=params.get("num_layers", 2),
                        dropout=params.get("dropout", 0.2),
                    ).to(dl_model.device)

                probs_list = []
                paths_to_run = ensemble_folds if ensemble_folds else [single_model_path]
                for model_path in paths_to_run:
                    model.load_state_dict(torch.load(model_path, map_location=dl_model.device, weights_only=False))
                    model.eval()
                    with torch.no_grad():
                        if torch.cuda.is_available():
                            with torch.amp.autocast("cuda"):
                                logit = model(X_seq).cpu().item()
                                probs_list.append(1 / (1 + np.exp(-logit)))
                        else:
                            logit = model(X_seq).cpu().item()
                            probs_list.append(1 / (1 + np.exp(-logit)))

                prob = np.mean(probs_list)
                direction = "BULLISH" if prob > threshold else "BEARISH"
                conf = prob if direction == "BULLISH" else (1 - prob)
                hist_metrics = metrics_map.get(f"{asset}_{horizon}", {})

                predictions.append(
                    {
                        "asset": asset,
                        "horizon": horizon,
                        "direction": direction,
                        "confidence": f"{conf:.1%}",
                        "raw_score": round(float(conf), 2),
                        "threshold_used": round(float(threshold), 2),
                        "metrics": hist_metrics,
                    },
                )
            except Exception as exc:
                print(f"DL Pred Error {asset}: {exc}")

    return predictions


def predict_ensemble(
    *,
    dl_model,
    lstm_cls: Callable[..., torch.nn.Module],
    transformer_cls: Callable[..., torch.nn.Module],
    nbeats_cls: Callable[..., torch.nn.Module],
) -> List[Dict[str, Any]]:
    models = ["lstm", "transformer", "nbeats"]
    all_preds: Dict[str, List[Dict[str, Any]]] = {}

    for model_name in models:
        try:
            preds = predict_latest_by_model(
                dl_model=dl_model,
                model_type=model_name,
                lstm_cls=lstm_cls,
                transformer_cls=transformer_cls,
                nbeats_cls=nbeats_cls,
            )
            for pred in preds:
                key = f"{pred['asset']}_{pred['horizon']}"
                all_preds.setdefault(key, [])
                conf_val = float(pred["confidence"].strip("%")) / 100.0
                prob = conf_val if pred["direction"] == "BULLISH" else (1.0 - conf_val)
                metrics = pred.get("metrics", {})
                acc = metrics.get("accuracy", 0.5)
                f1 = metrics.get("f1", 0.5)
                weight = (acc + f1) / 2.0
                all_preds[key].append({"prob": prob, "weight": weight, "model": model_name})
        except Exception as exc:
            print(f"Ensemble: Failed to get {model_name} preds: {exc}")

    final_results = []
    for key, items in all_preds.items():
        if not items:
            continue
        total_weight = sum(item["weight"] for item in items)
        if total_weight == 0:
            total_weight = 1e-6
        weighted_prob = sum(item["prob"] * item["weight"] for item in items) / total_weight
        asset, horizon = key.split("_")
        direction = "BULLISH" if weighted_prob > 0.5 else "BEARISH"
        confidence = weighted_prob if direction == "BULLISH" else (1 - weighted_prob)
        final_results.append(
            {
                "asset": asset,
                "horizon": horizon,
                "direction": direction,
                "confidence": f"{confidence:.1%}",
                "raw_score": round(float(confidence), 2),
                "models_used": [item["model"] for item in items],
            },
        )
    return final_results
