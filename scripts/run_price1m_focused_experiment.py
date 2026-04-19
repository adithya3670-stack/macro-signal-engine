import argparse
import json
import os
import sys
from typing import Dict, Iterable, List

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.price_1m_regression import Price1MRegressionManager


FOCUSED_MODELS = {
    "SP500": "patchtst",
    "Nasdaq": "patchtst",
    "DJIA": "patchtst",
    "Russell2000": "patchtst",
    "Gold": "tide",
    "Silver": "tide",
    "Copper": "lstm_reg_revin",
    "Oil": "lstm_reg_revin",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a focused 1M price experiment on the best current model family per asset.")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs per fold/model fit.")
    parser.add_argument("--holdout-year", type=int, default=2025, help="Outer holdout year.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("models", "holdout_price_1m_experiments", "focused_run"),
        help="Directory root where the experiment folder will be written.",
    )
    parser.add_argument("--refresh-features", action="store_true", help="Rebuild the cached feature file before training.")
    parser.add_argument(
        "--assets",
        nargs="*",
        default=None,
        help="Optional asset subset. Defaults to the full focused universe.",
    )
    return parser.parse_args()


def compute_moving_average_baselines(
    manager: Price1MRegressionManager,
    df,
    asset: str,
    windows: Iterable[int] = (20, 40, 60),
) -> Dict[str, Dict[str, float]]:
    holdout_mask = (df.index >= manager.OUTER_HOLDOUT_START) & (df.index <= manager.OUTER_HOLDOUT_END)
    current_prices = df[asset]
    future_prices = df[f"FuturePrice_{asset}_1m"]
    mape_valid = df[f"MAPEValid_{asset}_1m"].astype(bool)
    baselines: Dict[str, Dict[str, float]] = {}

    for window in windows:
        predicted_future = current_prices.rolling(window).mean()
        valid = holdout_mask & predicted_future.notna() & current_prices.notna() & future_prices.notna()
        pred_centered = predicted_future[valid] / current_prices[valid] - 1.0
        metrics = manager._evaluate_predictions(
            current_prices=current_prices[valid].values,
            actual_future=future_prices[valid].values,
            pred_centered=pred_centered.values,
            mape_valid=mape_valid[valid].values,
        )
        baselines[f"ma_{window}"] = metrics

    ranked = sorted(
        baselines.items(),
        key=lambda item: (
            float("inf") if item[1].get("mape") is None else item[1]["mape"],
            float("inf") if item[1].get("smape") is None else item[1]["smape"],
        ),
    )
    best_name, best_metrics = ranked[0]
    baselines["best"] = {"name": best_name, **best_metrics}
    return baselines


def build_leaderboard(metrics: List[Dict[str, object]], ma_baselines: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    leaderboard = []
    for entry in metrics:
        asset = entry["asset"]
        holdout = entry.get("holdout_metrics", {})
        naive = entry.get("baseline_2025", {})
        best_ma = ma_baselines.get(asset, {}).get("best", {})
        holdout_mape = holdout.get("mape")
        naive_mape = naive.get("mape")
        ma_mape = best_ma.get("mape")

        leaderboard.append(
            {
                "asset": asset,
                "model_type": entry["model_type"],
                "holdout_mape": holdout_mape,
                "naive_last_mape": naive_mape,
                "best_ma_name": best_ma.get("name"),
                "best_ma_mape": ma_mape,
                "beat_naive_last": None if holdout_mape is None or naive_mape is None else holdout_mape < naive_mape,
                "beat_best_ma": None if holdout_mape is None or ma_mape is None else holdout_mape < ma_mape,
                "delta_vs_naive": None if holdout_mape is None or naive_mape is None else holdout_mape - naive_mape,
                "delta_vs_best_ma": None if holdout_mape is None or ma_mape is None else holdout_mape - ma_mape,
            }
        )

    leaderboard.sort(key=lambda item: (float("inf") if item["holdout_mape"] is None else item["holdout_mape"]))
    return leaderboard


def main() -> None:
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    manager = Price1MRegressionManager(holdout_root=output_root)

    os.makedirs(output_root, exist_ok=True)
    holdout_dir = os.path.join(output_root, str(args.holdout_year))
    os.makedirs(holdout_dir, exist_ok=True)

    selected_assets = [asset for asset in (args.assets or list(FOCUSED_MODELS)) if asset in FOCUSED_MODELS]
    df = manager.load_and_preprocess(force_refresh_features=args.refresh_features)
    feature_registry = manager.build_feature_registry(df)

    metrics: List[Dict[str, object]] = []
    ma_baselines: Dict[str, Dict[str, object]] = {}
    model_map: Dict[str, List[str]] = {}

    print(f"Running focused 1M price experiment for {len(selected_assets)} assets with max {args.epochs} epochs.")
    for asset in selected_assets:
        model_type = FOCUSED_MODELS[asset]
        model_map[asset] = [model_type]
        print(f"[train] {asset} -> {model_type}")
        entry = manager._train_asset_model(
            df=df,
            asset=asset,
            model_type=model_type,
            feature_cols=feature_registry[asset],
            save_dir=holdout_dir,
            epochs=args.epochs,
        )
        metrics.append(entry)
        ma_baselines[asset] = compute_moving_average_baselines(manager, df, asset)
        holdout_mape = entry["holdout_metrics"].get("mape")
        naive_mape = entry["baseline_2025"].get("mape")
        best_ma = ma_baselines[asset]["best"]
        print(
            "[done] "
            f"{asset}: holdout MAPE={holdout_mape:.4f} "
            f"naive={naive_mape:.4f} "
            f"{best_ma['name']}={best_ma['mape']:.4f}"
        )

    champions = manager.select_champions_from_metrics(metrics, holdout_year=args.holdout_year)
    leaderboard = build_leaderboard(metrics, ma_baselines)

    metrics_path = os.path.join(holdout_dir, "price1m_metrics.json")
    champions_path = os.path.join(holdout_dir, "price1m_champions.json")
    registry_path = os.path.join(holdout_dir, "price1m_feature_registry.json")
    ma_path = os.path.join(holdout_dir, "price1m_moving_average_baselines.json")
    leaderboard_path = os.path.join(holdout_dir, "price1m_leaderboard.json")
    experiment_path = os.path.join(holdout_dir, "price1m_focused_experiment.json")

    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    with open(champions_path, "w") as handle:
        json.dump(champions, handle, indent=2)
    with open(registry_path, "w") as handle:
        json.dump({asset: feature_registry[asset] for asset in selected_assets}, handle, indent=2)
    with open(ma_path, "w") as handle:
        json.dump(ma_baselines, handle, indent=2)
    with open(leaderboard_path, "w") as handle:
        json.dump(leaderboard, handle, indent=2)
    with open(experiment_path, "w") as handle:
        json.dump(
            {
                "holdout_year": args.holdout_year,
                "epochs": args.epochs,
                "output_root": output_root,
                "assets": selected_assets,
                "model_map": model_map,
                "artifacts": {
                    "metrics_path": metrics_path,
                    "champions_path": champions_path,
                    "feature_registry_path": registry_path,
                    "moving_average_baselines_path": ma_path,
                    "leaderboard_path": leaderboard_path,
                },
            },
            handle,
            indent=2,
        )

    print("\nLeaderboard:")
    for row in leaderboard:
        print(
            f"{row['asset']:12s} {row['model_type']:16s} "
            f"holdout={row['holdout_mape']:.4f} "
            f"naive={row['naive_last_mape']:.4f} "
            f"{row['best_ma_name']}={row['best_ma_mape']:.4f}"
        )

    print(f"\nSaved experiment artifacts to {holdout_dir}")


if __name__ == "__main__":
    main()
