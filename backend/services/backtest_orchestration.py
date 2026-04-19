from __future__ import annotations

import os
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.domain.dto import ModelSourceSelection
from backend.infrastructure.model_artifacts import ModelArtifactResolver


HYBRID_WEIGHT_PROFILES: Dict[str, Dict[str, Dict[str, float] | str]] = {
    "1": {"name": "short_max", "weights": {"1w": 0.70, "1m": 0.20, "3m": 0.10}},
    "2": {"name": "short_tilt", "weights": {"1w": 0.60, "1m": 0.30, "3m": 0.10}},
    "3": {"name": "short_heavy", "weights": {"1w": 0.60, "1m": 0.20, "3m": 0.20}},
    "4": {"name": "mid_tilt", "weights": {"1w": 0.50, "1m": 0.30, "3m": 0.20}},
    "5": {"name": "mid_blend", "weights": {"1w": 0.40, "1m": 0.40, "3m": 0.20}},
    "6": {"name": "mid_heavy", "weights": {"1w": 0.40, "1m": 0.30, "3m": 0.30}},
    "7": {"name": "balanced", "weights": {"1w": 1 / 3, "1m": 1 / 3, "3m": 1 / 3}},
    "8": {"name": "barbell", "weights": {"1w": 0.30, "1m": 0.40, "3m": 0.30}},
    "9": {"name": "defensive", "weights": {"1w": 0.30, "1m": 0.30, "3m": 0.40}},
    "10": {"name": "long_tilt", "weights": {"1w": 0.20, "1m": 0.40, "3m": 0.40}},
    "11": {"name": "long_heavy", "weights": {"1w": 0.20, "1m": 0.30, "3m": 0.50}},
    "12": {"name": "long_max", "weights": {"1w": 0.10, "1m": 0.30, "3m": 0.60}},
    "13": {"name": "mid_core", "weights": {"1w": 0.20, "1m": 0.60, "3m": 0.20}},
    "14": {"name": "mid_anchor", "weights": {"1w": 0.15, "1m": 0.60, "3m": 0.25}},
}


class BacktestOrchestrationService:
    """Shared orchestration for signal-universe, model source, and blending."""

    def __init__(self, artifact_resolver: Optional[ModelArtifactResolver] = None) -> None:
        self.artifacts = artifact_resolver or ModelArtifactResolver()

    def normalize_model_type(self, model_type: Optional[str]) -> str:
        normalized = (model_type or "ensemble").strip()
        if normalized.startswith("winner_") and not normalized.startswith("winner_ensemble"):
            return "winner_ensemble"
        return normalized

    def resolve_portfolio_selection(
        self,
        model_category: Optional[str],
        model_year: Optional[str],
        strat_config: Dict[str, object],
    ) -> ModelSourceSelection:
        model_type = self.normalize_model_type(str(strat_config.get("model_type", "ensemble")))
        dl_folder = str(strat_config.get("dl_folder", "default"))
        return self.artifacts.resolve_model_source(
            model_category=model_category,
            model_year=str(model_year) if model_year else None,
            dl_folder=dl_folder,
            model_type=model_type,
        )

    def build_signal_universe(
        self,
        universe: Sequence[str],
        horizon: str,
        use_hybrid: bool,
    ) -> List[str]:
        signal_universe: List[str] = []
        if use_hybrid:
            horizons = ["1w", "1m", "3m"]
            for asset in universe:
                for h in horizons:
                    signal_universe.append(asset if "_" in asset else f"{asset}_{h}")
            return signal_universe

        for asset in universe:
            signal_universe.append(asset if "_" in asset else f"{asset}_{horizon}")
        return signal_universe

    def get_hybrid_weights(self, profile_id: Optional[str]) -> Dict[str, float]:
        if profile_id is None:
            return HYBRID_WEIGHT_PROFILES["7"]["weights"]  # balanced default

        key = str(profile_id).strip().lower()
        if key in HYBRID_WEIGHT_PROFILES:
            return HYBRID_WEIGHT_PROFILES[key]["weights"]  # type: ignore[return-value]

        for profile in HYBRID_WEIGHT_PROFILES.values():
            if str(profile["name"]).lower() == key:
                return profile["weights"]  # type: ignore[return-value]

        return HYBRID_WEIGHT_PROFILES["7"]["weights"]  # type: ignore[return-value]

    def build_hybrid_signals(
        self,
        raw_signals: pd.DataFrame,
        universe: Sequence[str],
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        hybrid = pd.DataFrame(index=raw_signals.index)

        for asset in universe:
            cols: List[str] = []
            wts: List[float] = []
            for horizon in ["1w", "1m", "3m"]:
                col = f"{asset}_{horizon}"
                if col in raw_signals.columns:
                    cols.append(col)
                    wts.append(float(weights.get(horizon, 0.0)))

            if not cols:
                continue

            weight_sum = float(np.sum(wts))
            if weight_sum > 0:
                normalized = [w / weight_sum for w in wts]
                hybrid[asset] = (raw_signals[cols] * normalized).sum(axis=1)
            else:
                hybrid[asset] = raw_signals[cols].mean(axis=1)

        return hybrid

    def rename_single_horizon_columns(
        self,
        raw_signals: pd.DataFrame,
        universe: Sequence[str],
    ) -> pd.DataFrame:
        rename_map: Dict[str, str] = {}
        for col in raw_signals.columns:
            for original_asset in universe:
                if col == original_asset:
                    break
                if col.startswith(f"{original_asset}_"):
                    rename_map[col] = original_asset
                    break
        if rename_map:
            raw_signals = raw_signals.rename(columns=rename_map)
        return raw_signals

    def generate_signals(
        self,
        signal_universe: Sequence[str],
        start_date: Optional[str],
        end_date: Optional[str],
        selection: ModelSourceSelection,
    ) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        from backtesting.signal_generator import SignalGenerator

        if selection.source == "holdout":
            generator = SignalGenerator(model_dir=str(selection.model_dir))
            signals = generator.generate_signals(
                list(signal_universe),
                start_date,
                end_date,
                force_refresh=True,
                model_type=selection.model_type,
            )
            return signals, []

        if selection.source == "rolling_master":
            return self._generate_rolling_signals(
                universe=list(signal_universe),
                start_date=start_date,
                end_date=end_date,
                model_type=selection.model_type,
                master_root=selection.model_dir,
            )

        generator = SignalGenerator(model_dir=str(selection.model_dir))
        signals = generator.generate_signals(
            list(signal_universe),
            start_date,
            end_date,
            force_refresh=True,
            model_type=selection.model_type,
        )
        return signals, []

    def _generate_rolling_signals(
        self,
        universe: Sequence[str],
        start_date: Optional[str],
        end_date: Optional[str],
        model_type: str,
        master_root: Optional[str],
    ) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        from backtesting.signal_generator import SignalGenerator

        root = master_root or str(self.artifacts.base_dir / "MasterDl")
        if not os.path.exists(root):
            return pd.DataFrame(), []

        folders: List[Dict[str, object]] = []
        for dirname in os.listdir(root):
            path = os.path.join(root, dirname)
            if not os.path.isdir(path):
                continue
            try:
                parts = dirname.split("_")
                if len(parts) < 2:
                    continue
                dt_str = f"{parts[0]} {parts[1].replace('-', ':')}"
                folder_date = pd.to_datetime(dt_str)
                folders.append({"path": path, "date": folder_date})
            except Exception:
                continue

        if not folders:
            return pd.DataFrame(), []

        folders.sort(key=lambda x: x["date"])  # type: ignore[index]

        sim_start = pd.to_datetime(start_date) if start_date else pd.Timestamp.now() - pd.Timedelta(days=365)
        sim_end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

        current_model = None
        for folder in folders:
            if folder["date"] <= sim_start:
                current_model = folder
            else:
                break
        if current_model is None:
            current_model = folders[0]

        segments: List[Dict[str, str]] = []
        cursor = sim_start
        while cursor < sim_end:
            next_model = None
            for folder in folders:
                if folder["date"] > current_model["date"] and folder["date"] > cursor:
                    next_model = folder
                    break

            seg_end = next_model["date"] if next_model else sim_end
            if seg_end > sim_end:
                seg_end = sim_end

            segments.append(
                {
                    "model": str(current_model["path"]),
                    "start": cursor.strftime("%Y-%m-%d"),
                    "end": pd.Timestamp(seg_end).strftime("%Y-%m-%d"),
                }
            )
            cursor = pd.Timestamp(seg_end)
            if next_model:
                current_model = next_model
            else:
                break

        chunks: List[pd.DataFrame] = []
        for seg in segments:
            generator = SignalGenerator(model_dir=seg["model"])
            try:
                chunk = generator.generate_signals(
                    list(universe),
                    seg["start"],
                    seg["end"],
                    force_refresh=True,
                    model_type=model_type,
                )
            except Exception:
                chunk = pd.DataFrame()

            if chunk.empty:
                continue

            is_last = seg is segments[-1]
            if is_last:
                mask = (chunk.index >= seg["start"]) & (chunk.index <= seg["end"])
            else:
                mask = (chunk.index >= seg["start"]) & (chunk.index < seg["end"])
            trimmed = chunk.loc[mask]
            if not trimmed.empty:
                chunks.append(trimmed)

        if not chunks:
            return pd.DataFrame(), segments

        full_df = pd.concat(chunks).sort_index()
        full_df = full_df[~full_df.index.duplicated(keep="last")]
        return full_df, segments

    def selection_to_dict(self, selection: ModelSourceSelection) -> Dict[str, object]:
        return asdict(selection)
