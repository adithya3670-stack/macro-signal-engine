import argparse
import json
import os
import sys
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.price_1w_regression import Price1WRegressionManager
from analysis.price_3d_models import build_price_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Slice 1W price holdout errors by regime, trend, volatility, and calendar buckets.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing the model .pth, scaler, and meta artifacts.")
    parser.add_argument("--asset", required=True, help="Asset to analyze.")
    parser.add_argument("--model-type", required=True, help="Model type for the saved artifact prefix.")
    parser.add_argument("--output-json", default=None, help="Optional explicit output path for the summary JSON.")
    return parser.parse_args()


def safe_qcut(series: pd.Series, labels: List[str]) -> pd.Series:
    valid = series.dropna()
    if valid.nunique() < len(labels):
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(["all"] * len(series), index=series.index, dtype="object")


def bucket_summary(frame: pd.DataFrame, column: str) -> List[Dict[str, object]]:
    rows = []
    for bucket, subset in frame.groupby(column, dropna=False, observed=False):
        if len(subset) == 0:
            continue
        rows.append(
            {
                "bucket": None if pd.isna(bucket) else str(bucket),
                "rows": int(len(subset)),
                "mape": float(subset["ape"].mean()),
                "median_mape": float(subset["ape"].median()),
                "smape": float(subset["smape_component"].mean()),
                "mae": float(subset["abs_err"].mean()),
                "signed_error_mean": float(subset["signed_err"].mean()),
                "avg_implied_return_pct": float(subset["pred_return_pct"].mean()),
                "avg_actual_return_pct": float(subset["actual_return_pct"].mean()),
            }
        )
    rows.sort(key=lambda item: item["mape"], reverse=True)
    return rows


def main() -> None:
    args = parse_args()
    manager = Price1WRegressionManager()
    df = manager.load_and_preprocess(force_refresh_features=False)

    prefix = f"{args.asset}_{args.model_type}"
    meta_path = os.path.join(args.artifact_dir, f"{prefix}_meta.json")
    scaler_path = os.path.join(args.artifact_dir, f"{prefix}_scaler.pkl")
    weight_path = os.path.join(args.artifact_dir, f"{prefix}.pth")
    if not all(os.path.exists(path) for path in [meta_path, scaler_path, weight_path]):
        raise FileNotFoundError(f"Missing one or more artifact files for {prefix} in {args.artifact_dir}")

    with open(meta_path, "r") as handle:
        meta = json.load(handle)

    feature_cols = meta["feature_cols"]
    model_config = meta["model_config"]
    scaler = joblib.load(scaler_path)
    scaled = manager._scale_frame(df, feature_cols, scaler)
    payload = manager._create_sequences(
        scaled_features=scaled,
        centered_targets=df[f"CenteredNormPrice_{args.asset}_1w"].values,
        current_prices=df[args.asset].values,
        future_prices=df[f"FuturePrice_{args.asset}_1w"].values,
        mape_valid=df[f"MAPEValid_{args.asset}_1w"].values.astype(bool),
        dates=df.index.values,
        window_size=int(model_config["window_size"]),
    )
    holdout_mask = (payload["dates"] >= np.datetime64(manager.OUTER_HOLDOUT_START)) & (
        payload["dates"] <= np.datetime64(manager.OUTER_HOLDOUT_END)
    )

    X_holdout = payload["X"][holdout_mask]
    current_prices = payload["current_prices"][holdout_mask]
    future_prices = payload["future_prices"][holdout_mask]
    mape_valid = payload["mape_valid"][holdout_mask]
    dates = pd.to_datetime(payload["dates"][holdout_mask])

    model = build_price_model(args.model_type, len(feature_cols), model_config).to(manager.device)
    model.load_state_dict(torch.load(weight_path, map_location=manager.device, weights_only=False))
    pred_centered = manager._predict_array(model, X_holdout)
    pred_norm = np.clip(pred_centered + 1.0, -5.0, 5.0)
    pred_future = current_prices * pred_norm

    actual_return_pct = ((future_prices / current_prices) - 1.0) * 100.0
    pred_return_pct = (pred_norm - 1.0) * 100.0
    abs_err = np.abs(pred_future - future_prices)
    ape = (abs_err / np.abs(future_prices)) * 100.0
    smape_component = (2.0 * abs_err / np.clip(np.abs(future_prices) + np.abs(pred_future), 1e-6, None)) * 100.0

    analysis = pd.DataFrame(
        {
            "Date": dates,
            "current_price": current_prices,
            "future_price": future_prices,
            "pred_future_price": pred_future,
            "actual_return_pct": actual_return_pct,
            "pred_return_pct": pred_return_pct,
            "abs_err": abs_err,
            "ape": ape,
            "smape_component": smape_component,
            "signed_err": pred_future - future_prices,
            "mape_valid": mape_valid.astype(bool),
        }
    )
    analysis = analysis[analysis["mape_valid"]].copy()
    analysis["month"] = analysis["Date"].dt.to_period("M").astype(str)
    analysis["trend_bucket"] = np.where(analysis["actual_return_pct"] >= 0.0, "up_move", "down_move")
    analysis["pred_direction"] = np.where(analysis["pred_return_pct"] >= 0.0, "predict_up", "predict_down")
    analysis["direction_match"] = np.where(
        np.sign(analysis["actual_return_pct"]) == np.sign(analysis["pred_return_pct"]),
        "matched",
        "mismatched",
    )

    trend_source = f"{args.asset}_trend_10"
    vol_source = f"{args.asset}_vol_10"
    risk_source = "VIX" if "VIX" in df.columns else vol_source
    analysis["trend_10"] = df.loc[analysis["Date"], trend_source].values if trend_source in df.columns else np.nan
    analysis["vol_10"] = df.loc[analysis["Date"], vol_source].values if vol_source in df.columns else np.nan
    analysis["risk_level"] = df.loc[analysis["Date"], risk_source].values if risk_source in df.columns else np.nan
    analysis["trend_regime"] = pd.cut(
        analysis["trend_10"],
        bins=[-np.inf, -1e-9, 1e-9, np.inf],
        labels=["downtrend", "flat", "uptrend"],
    )
    analysis["vol_bucket"] = safe_qcut(analysis["vol_10"], ["low_vol", "mid_vol", "high_vol"])
    analysis["risk_bucket"] = safe_qcut(analysis["risk_level"], ["low_risk", "mid_risk", "high_risk"])

    overall = {
        "rows": int(len(analysis)),
        "mape": float(analysis["ape"].mean()),
        "median_mape": float(analysis["ape"].median()),
        "smape": float(analysis["smape_component"].mean()),
        "mae": float(analysis["abs_err"].mean()),
        "directional_accuracy": float((analysis["direction_match"] == "matched").mean()),
        "overprediction_rate": float((analysis["signed_err"] > 0).mean()),
        "avg_implied_return_pct": float(analysis["pred_return_pct"].mean()),
        "avg_actual_return_pct": float(analysis["actual_return_pct"].mean()),
    }

    top_errors = (
        analysis.sort_values("ape", ascending=False)
        .head(10)[["Date", "ape", "abs_err", "actual_return_pct", "pred_return_pct", "trend_10", "vol_10", "risk_level"]]
        .assign(Date=lambda frame: frame["Date"].dt.strftime("%Y-%m-%d"))
        .to_dict(orient="records")
    )

    summary = {
        "asset": args.asset,
        "model_type": args.model_type,
        "artifact_dir": os.path.abspath(args.artifact_dir),
        "overall": overall,
        "by_month": bucket_summary(analysis, "month"),
        "by_trend_regime": bucket_summary(analysis, "trend_regime"),
        "by_vol_bucket": bucket_summary(analysis, "vol_bucket"),
        "by_risk_bucket": bucket_summary(analysis, "risk_bucket"),
        "by_direction_match": bucket_summary(analysis, "direction_match"),
        "top_errors": top_errors,
    }

    output_path = args.output_json or os.path.join(args.artifact_dir, f"{prefix}_holdout_error_slices.json")
    with open(output_path, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"{args.asset} {args.model_type} overall MAPE={overall['mape']:.4f} direction_acc={overall['directional_accuracy']:.4f}")
    print(f"Saved error slices to {output_path}")


if __name__ == "__main__":
    main()
