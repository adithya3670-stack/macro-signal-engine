from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


class DLInferenceService:
    """Inference and weighted-signal utilities for DL holdout workflows."""

    def predict_range(
        self,
        *,
        builder: Any,
        start_date: str,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return builder.predict_range(start_date=start_date, end_date=end_date)

    def save_predictions(self, *, predictions: pd.DataFrame, output_path: str) -> None:
        predictions.to_csv(output_path)

    def load_metrics_db(self, working_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        metrics_db: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for model_type in ["lstm", "transformer", "nbeats"]:
            metrics_path = os.path.join(working_dir, f"dl_metrics_{model_type}.json")
            if not os.path.exists(metrics_path):
                continue

            try:
                with open(metrics_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, list):
                    continue
            except Exception:
                continue

            metrics_db[model_type] = {}
            for item in data:
                if not isinstance(item, dict):
                    continue
                asset = item.get("asset")
                horizon = item.get("horizon")
                metrics = item.get("metrics")
                if not asset or not horizon or not isinstance(metrics, dict):
                    continue
                key = f"{asset}_{horizon}"
                metrics_db[model_type][key] = metrics

        return metrics_db

    def build_weighted_signals(
        self,
        *,
        results: pd.DataFrame,
        metrics_db: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Tuple[pd.DataFrame, Set[str]]:
        signals = pd.DataFrame(index=results.index)
        unique_assets: Set[str] = set()

        for col in results.columns:
            parts = str(col).split("_")
            if len(parts) >= 2:
                unique_assets.add(parts[1])

        for asset in unique_assets:
            relevant_columns = [col for col in results.columns if f"_{asset}_" in str(col)]
            if not relevant_columns:
                continue

            valid_preds: List[pd.Series] = []
            valid_weights: List[float] = []

            for col in relevant_columns:
                column = str(col)
                model_type = ""
                for candidate in ["lstm", "transformer", "nbeats"]:
                    if column.endswith(f"_{candidate}"):
                        model_type = candidate
                        break

                if not model_type:
                    continue

                weight = 0.5
                if model_type in metrics_db:
                    prefix = f"Pred_{asset}_"
                    suffix = f"_{model_type}"
                    if column.startswith(prefix) and column.endswith(suffix):
                        horizon = column[len(prefix) : -len(suffix)]
                        key = f"{asset}_{horizon}"
                        if key in metrics_db[model_type]:
                            metrics = metrics_db[model_type][key]
                            weight = (
                                float(metrics.get("accuracy", 0.5)) + float(metrics.get("precision", 0.5))
                            ) / 2.0

                valid_preds.append(results[column])
                valid_weights.append(weight)

            if valid_preds:
                total_weight = sum(valid_weights)
                if total_weight > 0:
                    normalized = [w / total_weight for w in valid_weights]
                else:
                    normalized = [1.0 / len(valid_weights)] * len(valid_weights)
                signals[asset] = sum(pred * weight for pred, weight in zip(valid_preds, normalized))
            else:
                signals[asset] = 0.5

        return signals, unique_assets
