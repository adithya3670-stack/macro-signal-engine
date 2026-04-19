import argparse
import copy
import json
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from analysis.price_3d_regression import Price3DRegressionManager
from scripts.run_price3d_focused_experiment import compute_moving_average_baselines
from scripts.run_price3d_round3_search import compact_registry, score_entry


EXTREMES_EXPERIMENTS = [
    {
        "name": "silver_patchtst_default",
        "asset": "Silver",
        "model_type": "patchtst",
        "feature_profile": "default",
        "config_override": {
            "window_size": 40,
            "lr": 5e-4,
            "patch_len": 5,
            "stride": 2,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.05,
        },
    },
    {
        "name": "silver_patchtst_compact_short",
        "asset": "Silver",
        "model_type": "patchtst",
        "feature_profile": "precious_compact",
        "config_override": {
            "window_size": 30,
            "lr": 6e-4,
            "patch_len": 5,
            "stride": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.05,
        },
    },
    {
        "name": "silver_lstm_default",
        "asset": "Silver",
        "model_type": "lstm_reg_revin",
        "feature_profile": "default",
        "config_override": {
            "window_size": 35,
            "lr": 6e-4,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.1,
        },
    },
    {
        "name": "oil_patchtst_default_short",
        "asset": "Oil",
        "model_type": "patchtst",
        "feature_profile": "default",
        "config_override": {
            "window_size": 30,
            "lr": 6e-4,
            "patch_len": 5,
            "stride": 1,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.05,
        },
    },
    {
        "name": "oil_nhits_default_short",
        "asset": "Oil",
        "model_type": "nhits",
        "feature_profile": "default",
        "config_override": {
            "window_size": 35,
            "lr": 7e-4,
            "hidden_size": 192,
            "pool_sizes": [1, 2, 3],
            "dropout": 0.05,
        },
    },
    {
        "name": "oil_lstm_default_short",
        "asset": "Oil",
        "model_type": "lstm_reg_revin",
        "feature_profile": "default",
        "config_override": {
            "window_size": 30,
            "lr": 7e-4,
            "hidden_size": 160,
            "num_layers": 2,
            "dropout": 0.1,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted Silver/Oil model search for 3D price MAPE.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs per experiment.")
    parser.add_argument("--holdout-year", type=int, default=2025, help="Outer holdout year.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("models", "holdout_price_3d_experiments", "extremes_search"),
        help="Directory root for Silver/Oil search artifacts.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    manager = Price3DRegressionManager(holdout_root=output_root)
    os.makedirs(output_root, exist_ok=True)
    holdout_dir = os.path.join(output_root, str(args.holdout_year))
    os.makedirs(holdout_dir, exist_ok=True)

    experiments = EXTREMES_EXPERIMENTS
    if args.experiments:
        allowed = set(args.experiments)
        experiments = [exp for exp in experiments if exp["name"] in allowed]

    df = manager.load_and_preprocess(force_refresh_features=False)
    default_registry = manager.build_feature_registry(df)
    profiled_registry = compact_registry(manager, df, default_registry)
    ma_cache: Dict[str, Dict[str, object]] = {}
    results = []

    print(f"Running {len(experiments)} Silver/Oil experiments with max {args.epochs} epochs.")
    for idx, exp in enumerate(experiments, start=1):
        asset = exp["asset"]
        model_type = exp["model_type"]
        feature_profile = exp["feature_profile"]
        feature_cols = profiled_registry[asset].get(feature_profile, default_registry[asset])
        model_config = copy.deepcopy(manager.MODEL_CONFIGS.get(model_type, {}))
        model_config.update(exp.get("config_override", {}))
        manager.MODEL_CONFIGS[model_type] = model_config

        print(f"[{idx}/{len(experiments)}] {exp['name']} -> {asset} / {model_type} / {feature_profile} ({len(feature_cols)} features)")
        entry = manager._train_asset_model(
            df=df,
            asset=asset,
            model_type=model_type,
            feature_cols=feature_cols,
            save_dir=holdout_dir,
            epochs=args.epochs,
        )
        if asset not in ma_cache:
            ma_cache[asset] = compute_moving_average_baselines(manager, df, asset)
        scored = score_entry(entry, ma_cache[asset]["best"])
        results.append(
            {
                "experiment_name": exp["name"],
                "asset": asset,
                "model_type": model_type,
                "feature_profile": feature_profile,
                "feature_count": len(feature_cols),
                "config": model_config,
                "score": scored,
                "entry": entry,
            }
        )
        print(
            f"     holdout={scored['holdout_mape']:.4f} "
            f"naive={scored['naive_last_mape']:.4f} "
            f"{scored['best_ma_name']}={scored['best_ma_mape']:.4f} "
            f"beat_naive={scored['beat_naive_last']}"
        )

    leaderboard = sorted(
        [
            {
                "experiment_name": row["experiment_name"],
                "asset": row["asset"],
                "model_type": row["model_type"],
                "feature_profile": row["feature_profile"],
                **row["score"],
            }
            for row in results
        ],
        key=lambda row: (float("inf") if row["holdout_mape"] is None else row["holdout_mape"]),
    )

    best_by_asset: Dict[str, Dict[str, object]] = {}
    for row in leaderboard:
        if row["asset"] not in best_by_asset:
            best_by_asset[row["asset"]] = row

    with open(os.path.join(holdout_dir, "extremes_results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    with open(os.path.join(holdout_dir, "extremes_leaderboard.json"), "w") as handle:
        json.dump(leaderboard, handle, indent=2)
    with open(os.path.join(holdout_dir, "extremes_best_by_asset.json"), "w") as handle:
        json.dump(best_by_asset, handle, indent=2)
    with open(os.path.join(holdout_dir, "extremes_moving_average_baselines.json"), "w") as handle:
        json.dump(ma_cache, handle, indent=2)

    print("\nBest by asset:")
    for asset, row in best_by_asset.items():
        print(
            f"{asset:12s} {row['model_type']:16s} {row['feature_profile']:18s} "
            f"holdout={row['holdout_mape']:.4f} naive={row['naive_last_mape']:.4f} "
            f"beat_naive={row['beat_naive_last']} promotion={row['passes_promotion']}"
        )

    print(f"\nSaved extremes search artifacts to {holdout_dir}")


if __name__ == "__main__":
    main()
