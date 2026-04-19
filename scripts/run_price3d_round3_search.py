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


ROUND3_EXPERIMENTS = [
    {
        "name": "sp500_patchtst_compact",
        "asset": "SP500",
        "model_type": "patchtst",
        "feature_profile": "equity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 3e-4,
            "patch_len": 10,
            "stride": 5,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 3,
            "dropout": 0.05,
        },
    },
    {
        "name": "djia_patchtst_compact",
        "asset": "DJIA",
        "model_type": "patchtst",
        "feature_profile": "equity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 3e-4,
            "patch_len": 10,
            "stride": 5,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 3,
            "dropout": 0.05,
        },
    },
    {
        "name": "russell_patchtst_compact",
        "asset": "Russell2000",
        "model_type": "patchtst",
        "feature_profile": "equity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 3e-4,
            "patch_len": 10,
            "stride": 5,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 3,
            "dropout": 0.05,
        },
    },
    {
        "name": "nasdaq_patchtst_compact",
        "asset": "Nasdaq",
        "model_type": "patchtst",
        "feature_profile": "equity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 3e-4,
            "patch_len": 10,
            "stride": 5,
            "d_model": 96,
            "nhead": 4,
            "num_layers": 3,
            "dropout": 0.05,
        },
    },
    {
        "name": "gold_patchtst_compact",
        "asset": "Gold",
        "model_type": "patchtst",
        "feature_profile": "precious_compact",
        "config_override": {
            "window_size": 50,
            "lr": 4e-4,
            "patch_len": 10,
            "stride": 5,
            "d_model": 80,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.05,
        },
    },
    {
        "name": "silver_nbeats_compact",
        "asset": "Silver",
        "model_type": "nbeats_reg",
        "feature_profile": "precious_compact",
        "config_override": {
            "window_size": 40,
            "lr": 5e-4,
            "nb_stacks": 2,
            "nb_blocks": 4,
            "nb_width": 160,
            "dropout": 0.05,
        },
    },
    {
        "name": "copper_lstm_compact",
        "asset": "Copper",
        "model_type": "lstm_reg_revin",
        "feature_profile": "commodity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 5e-4,
            "hidden_size": 160,
            "num_layers": 2,
            "dropout": 0.15,
        },
    },
    {
        "name": "oil_nhits_compact",
        "asset": "Oil",
        "model_type": "nhits",
        "feature_profile": "commodity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 5e-4,
            "hidden_size": 160,
            "pool_sizes": [1, 3, 6],
            "dropout": 0.05,
        },
    },
    {
        "name": "oil_lstm_compact",
        "asset": "Oil",
        "model_type": "lstm_reg_revin",
        "feature_profile": "commodity_compact",
        "config_override": {
            "window_size": 60,
            "lr": 5e-4,
            "hidden_size": 160,
            "num_layers": 2,
            "dropout": 0.15,
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a targeted round-3 search for 3-day price MAPE improvements.")
    parser.add_argument("--epochs", type=int, default=25, help="Maximum epochs per experiment.")
    parser.add_argument("--holdout-year", type=int, default=2025, help="Outer holdout year.")
    parser.add_argument(
        "--output-root",
        default=os.path.join("models", "holdout_price_3d_experiments", "round3_search"),
        help="Directory root for round-3 artifacts.",
    )
    parser.add_argument("--refresh-features", action="store_true", help="Rebuild the cached feature file before the search.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional list of experiment names to run. Defaults to the full round-3 set.",
    )
    return parser.parse_args()


def compact_registry(manager: Price3DRegressionManager, df, default_registry: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
    weekday_cols = ["is_month_end", "dow_0", "dow_1", "dow_2", "dow_3", "dow_4"]
    core_macro = ["VIX", "FEDFUNDS", "DGS10", "Real_Yield", "Liquidity_Impulse", "Curve_Steepening", "Breadth_Vol"]

    def keep(cols):
        return [col for col in cols if col in df.columns]

    registry: Dict[str, Dict[str, List[str]]] = {}
    for asset in manager.assets:
        own = manager._own_asset_features(asset)
        if asset in manager.ASSET_GROUPS["equities"]:
            peers = ["SP500", "Nasdaq", "DJIA", "Russell2000", "Tech_vs_Broad", "Gold", "Oil"]
            zscores = ["SP500_ZScore", "Nasdaq_ZScore", "DJIA_ZScore", "Russell2000_ZScore"]
            registry[asset] = {
                "default": default_registry[asset],
                "equity_compact": keep(own + peers + zscores + core_macro + weekday_cols),
            }
        elif asset in manager.ASSET_GROUPS["precious"]:
            peers = ["Gold", "Silver", "Silver_Gold_Ratio", "Safe_Haven_Demand", "Real_Yield", "Liquidity_Impulse", "DGS10", "VIX"]
            zscores = ["Gold_ZScore", "Silver_ZScore"]
            registry[asset] = {
                "default": default_registry[asset],
                "precious_compact": keep(own + peers + zscores + weekday_cols),
            }
        else:
            peers = ["Oil", "Copper", "Gold", "Silver", "Oil_Gold_Ratio", "Silver_Gold_Ratio"]
            macro = ["CPI_YoY", "PPI_YoY", "Curve_Steepening", "Liquidity_Impulse", "UNRATE", "PAYEMS", "Breadth_Vol", "VIX"]
            zscores = ["Oil_ZScore", "Copper_ZScore"]
            registry[asset] = {
                "default": default_registry[asset],
                "commodity_compact": keep(own + peers + macro + zscores + weekday_cols),
            }
    return registry


def score_entry(entry: Dict[str, object], ma_best: Dict[str, object]) -> Dict[str, object]:
    holdout = entry.get("holdout_metrics", {})
    naive = entry.get("baseline_2025", {})
    holdout_mape = holdout.get("mape")
    naive_mape = naive.get("mape")
    ma_mape = ma_best.get("mape")
    improvement_absolute = None if holdout_mape is None or naive_mape is None else naive_mape - holdout_mape
    improvement_relative = None if holdout_mape is None or naive_mape in (None, 0) else (naive_mape - holdout_mape) / naive_mape
    passes_promotion = bool(
        improvement_absolute is not None
        and (
            improvement_absolute >= Price3DRegressionManager.PROMOTION_ABSOLUTE_THRESHOLD
            or (improvement_relative is not None and improvement_relative >= Price3DRegressionManager.PROMOTION_RELATIVE_THRESHOLD)
        )
    )
    return {
        "holdout_mape": holdout_mape,
        "naive_last_mape": naive_mape,
        "best_ma_name": ma_best.get("name"),
        "best_ma_mape": ma_mape,
        "beat_naive_last": None if holdout_mape is None or naive_mape is None else holdout_mape < naive_mape,
        "beat_best_ma": None if holdout_mape is None or ma_mape is None else holdout_mape < ma_mape,
        "improvement_absolute": improvement_absolute,
        "improvement_relative": improvement_relative,
        "passes_promotion": passes_promotion,
    }


def main() -> None:
    args = parse_args()
    output_root = os.path.abspath(args.output_root)
    manager = Price3DRegressionManager(holdout_root=output_root)
    os.makedirs(output_root, exist_ok=True)
    holdout_dir = os.path.join(output_root, str(args.holdout_year))
    os.makedirs(holdout_dir, exist_ok=True)

    experiments = ROUND3_EXPERIMENTS
    if args.experiments:
        allowed = set(args.experiments)
        experiments = [exp for exp in experiments if exp["name"] in allowed]

    df = manager.load_and_preprocess(force_refresh_features=args.refresh_features)
    default_registry = manager.build_feature_registry(df)
    profiled_registry = compact_registry(manager, df, default_registry)

    results = []
    ma_cache: Dict[str, Dict[str, object]] = {}
    total = len(experiments)

    print(f"Running {total} round-3 experiments with max {args.epochs} epochs.")
    for idx, exp in enumerate(experiments, start=1):
        asset = exp["asset"]
        model_type = exp["model_type"]
        feature_profile = exp["feature_profile"]
        feature_cols = profiled_registry[asset].get(feature_profile, default_registry[asset])
        model_config = copy.deepcopy(manager.MODEL_CONFIGS.get(model_type, {}))
        model_config.update(exp.get("config_override", {}))
        manager.MODEL_CONFIGS[model_type] = model_config

        print(f"[{idx}/{total}] {exp['name']} -> {asset} / {model_type} / {feature_profile} ({len(feature_cols)} features)")
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
        result = {
            "experiment_name": exp["name"],
            "asset": asset,
            "model_type": model_type,
            "feature_profile": feature_profile,
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
            "config": model_config,
            "entry": entry,
            "score": scored,
        }
        results.append(result)
        print(
            f"     holdout={scored['holdout_mape']:.4f} "
            f"naive={scored['naive_last_mape']:.4f} "
            f"{scored['best_ma_name']}={scored['best_ma_mape']:.4f} "
            f"beat_naive={scored['beat_naive_last']}"
        )

    leaderboard = sorted(
        [
            {
                "experiment_name": item["experiment_name"],
                "asset": item["asset"],
                "model_type": item["model_type"],
                "feature_profile": item["feature_profile"],
                **item["score"],
            }
            for item in results
        ],
        key=lambda row: (
            float("inf") if row["holdout_mape"] is None else row["holdout_mape"],
            float("inf") if row["naive_last_mape"] is None else row["naive_last_mape"],
        ),
    )

    best_by_asset: Dict[str, Dict[str, object]] = {}
    for row in leaderboard:
        if row["asset"] not in best_by_asset:
            best_by_asset[row["asset"]] = row

    results_path = os.path.join(holdout_dir, "round3_results.json")
    leaderboard_path = os.path.join(holdout_dir, "round3_leaderboard.json")
    best_path = os.path.join(holdout_dir, "round3_best_by_asset.json")
    ma_path = os.path.join(holdout_dir, "round3_moving_average_baselines.json")

    with open(results_path, "w") as handle:
        json.dump(results, handle, indent=2)
    with open(leaderboard_path, "w") as handle:
        json.dump(leaderboard, handle, indent=2)
    with open(best_path, "w") as handle:
        json.dump(best_by_asset, handle, indent=2)
    with open(ma_path, "w") as handle:
        json.dump(ma_cache, handle, indent=2)

    print("\nBest by asset:")
    for asset, row in best_by_asset.items():
        print(
            f"{asset:12s} {row['model_type']:16s} {row['feature_profile']:18s} "
            f"holdout={row['holdout_mape']:.4f} naive={row['naive_last_mape']:.4f} "
            f"beat_naive={row['beat_naive_last']} promotion={row['passes_promotion']}"
        )

    print(f"\nSaved round-3 artifacts to {holdout_dir}")


if __name__ == "__main__":
    main()
