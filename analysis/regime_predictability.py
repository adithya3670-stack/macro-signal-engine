import datetime as dt
import glob
import json
import os
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import torch

from analysis.price_1m_features import Price1MFeatureBuilder
from analysis.price_1m_regression import Price1MRegressionManager
from analysis.price_1w_features import Price1WFeatureBuilder
from analysis.price_1w_regression import Price1WRegressionManager
from analysis.price_3d_features import Price3DFeatureBuilder
from analysis.price_3d_models import build_price_model
from analysis.price_3d_regression import Price3DRegressionManager
from analysis.regime_engine import RegimeStateEngine
from analysis.regime_forecast import RegimeForecastEngine
from config.settings import (
    ASSETS,
    BASE_DIR,
    HOLDOUT_PRICE_1M_DIR,
    HOLDOUT_PRICE_1W_DIR,
    HOLDOUT_PRICE_3D_DIR,
    MASTER_DATA_FILE,
    MODELS_PRICE_1M_DIR,
    MODELS_PRICE_1W_DIR,
    MODELS_PRICE_3D_DIR,
    MODELS_REGIME_DIR,
    PRICE_1M_FEATURES_FILE,
    PRICE_1W_FEATURES_FILE,
    PRICE_3D_FEATURES_FILE,
)


class RegimePredictabilityManager:
    """Hybrid regime diagnostics + gating over 3d/1w/1m price pipelines."""

    HORIZON_CONFIG = {
        "3d": {
            "manager_cls": Price3DRegressionManager,
            "feature_builder_cls": Price3DFeatureBuilder,
            "feature_path": PRICE_3D_FEATURES_FILE,
            "production_dir": MODELS_PRICE_3D_DIR,
            "holdout_dir": HOLDOUT_PRICE_3D_DIR,
            "metrics_file": "price3d_metrics.json",
            "champions_file": "price3d_champions.json",
        },
        "1w": {
            "manager_cls": Price1WRegressionManager,
            "feature_builder_cls": Price1WFeatureBuilder,
            "feature_path": PRICE_1W_FEATURES_FILE,
            "production_dir": MODELS_PRICE_1W_DIR,
            "holdout_dir": HOLDOUT_PRICE_1W_DIR,
            "metrics_file": "price1w_metrics.json",
            "champions_file": "price1w_champions.json",
        },
        "1m": {
            "manager_cls": Price1MRegressionManager,
            "feature_builder_cls": Price1MFeatureBuilder,
            "feature_path": PRICE_1M_FEATURES_FILE,
            "production_dir": MODELS_PRICE_1M_DIR,
            "holdout_dir": HOLDOUT_PRICE_1M_DIR,
            "metrics_file": "price1m_metrics.json",
            "champions_file": "price1m_champions.json",
        },
    }

    MIN_SUPPORT = {"3d": 60, "1w": 40, "1m": 25}
    FORECAST_MIN_SUPPORT = {"3d": 30, "1w": 20, "1m": 12}
    CONF_THRESHOLD = 0.55
    IMPROVE_ABS_THRESHOLD = 0.03
    IMPROVE_REL_THRESHOLD = 0.02
    HOLDOUT_DEFAULT_YEAR = 2025

    def __init__(
        self,
        regime_dir: str = MODELS_REGIME_DIR,
        master_data_path: str = MASTER_DATA_FILE,
        feature_paths: Optional[Dict[str, str]] = None,
        production_dirs: Optional[Dict[str, str]] = None,
        holdout_dirs: Optional[Dict[str, str]] = None,
    ):
        self.regime_dir = regime_dir
        self.master_data_path = master_data_path
        self.feature_paths = feature_paths or {k: v["feature_path"] for k, v in self.HORIZON_CONFIG.items()}
        self.production_dirs = production_dirs or {k: v["production_dir"] for k, v in self.HORIZON_CONFIG.items()}
        self.holdout_dirs = holdout_dirs or {k: v["holdout_dir"] for k, v in self.HORIZON_CONFIG.items()}
        self.assets = list(ASSETS)
        self.state_engine = RegimeStateEngine()
        self.forecast_engine = RegimeForecastEngine()
        os.makedirs(self.regime_dir, exist_ok=True)

    def rebuild(self, year: Optional[int] = None, source: str = "production", refresh_features: bool = False) -> Dict[str, object]:
        holdout_year = int(year or self.HOLDOUT_DEFAULT_YEAR)
        source = (source or "production").lower()
        if source not in {"production", "holdout"}:
            raise ValueError("source must be 'production' or 'holdout'")

        if refresh_features:
            self._refresh_feature_tables()

        master_df = self._load_dataframe(self.master_data_path)
        feature_tables = {h: self._load_dataframe(path) for h, path in self.feature_paths.items()}
        state_df = self.state_engine.build_state_history(
            master_df=master_df,
            feature_tables=feature_tables,
            holdout_start=f"{holdout_year}-01-01",
        )

        horizon_outputs: Dict[str, Dict[str, object]] = {}
        for horizon in self.HORIZON_CONFIG:
            bundle = self._load_horizon_bundle(horizon=horizon, source=source, year=holdout_year)
            manager = self._build_manager(horizon=horizon, artifact_dir=bundle["artifact_dir"])
            df = manager.load_and_preprocess(force_refresh_features=False)
            predictability_rows, latest_predictions = self._evaluate_horizon(
                horizon=horizon,
                manager=manager,
                df=df,
                state_df=state_df,
                bundle=bundle,
            )
            horizon_outputs[horizon] = {
                "bundle": bundle,
                "predictability_rows": predictability_rows,
                "latest_predictions": latest_predictions,
            }

        forecast_outputs = self._build_forecast_outputs(
            state_df=state_df,
            feature_tables=feature_tables,
            holdout_year=holdout_year,
        )
        policy = self._build_policy(
            state_df=state_df,
            horizon_outputs=horizon_outputs,
            forecast_outputs=forecast_outputs,
        )
        latest = self._build_latest_predictions(
            policy=policy,
            horizon_outputs=horizon_outputs,
            forecast_outputs=forecast_outputs,
        )
        artifact_info = self._save_artifacts(
            state_df=state_df,
            horizon_outputs=horizon_outputs,
            forecast_outputs=forecast_outputs,
            policy=policy,
            latest=latest,
            holdout_year=holdout_year,
            source=source,
        )

        return {
            "status": "success",
            "regime_dir": self.regime_dir,
            "as_of_date": latest.get("as_of_date"),
            "holdout_year": holdout_year,
            "source": source,
            "state_rows": int(len(state_df)),
            "policy_items": int(sum(len(v) for v in policy.get("horizons", {}).values())),
            **artifact_info,
        }

    def load_state_latest(self) -> Dict[str, object]:
        history = self._read_state_history()
        if history.empty:
            return {"status": "empty", "latest": {}}
        latest = history.iloc[-1].to_dict()
        latest["date"] = history.index[-1].strftime("%Y-%m-%d")
        return {"status": "success", "latest": latest}

    def load_predictability(self, horizon: Optional[str] = None, asset: Optional[str] = None, group: Optional[str] = None) -> Dict[str, object]:
        horizons = [horizon] if horizon else list(self.HORIZON_CONFIG.keys())
        payload = {}
        forecast_diag = {}
        for hz in horizons:
            hz = hz.lower()
            if hz not in self.HORIZON_CONFIG:
                continue
            path = os.path.join(self.regime_dir, f"regime_predictability_{hz}.json")
            if not os.path.exists(path):
                payload[hz] = []
                continue
            rows = self._read_json(path).get("rows", [])
            if asset:
                rows = [r for r in rows if r.get("asset") == asset]
            if group:
                rows = [r for r in rows if r.get("group") == group]
            payload[hz] = rows
            forecast_path = os.path.join(self.regime_dir, f"regime_forecast_{hz}.json")
            if os.path.exists(forecast_path):
                forecast_diag[hz] = self._read_json(forecast_path).get("groups", {})
            else:
                forecast_diag[hz] = {}
        return {"status": "success", "predictability": payload, "forecast_diagnostics": forecast_diag}

    def load_policy_latest(self) -> Dict[str, object]:
        path = os.path.join(self.regime_dir, "regime_policy.json")
        if not os.path.exists(path):
            return {"status": "empty", "horizons": {}}
        data = self._read_json(path)
        return {"status": "success", **data}

    def load_forecast_latest(self) -> Dict[str, object]:
        path = os.path.join(self.regime_dir, "regime_forecast_latest.json")
        if not os.path.exists(path):
            return {"status": "empty", "horizons": {}}
        data = self._read_json(path)
        return {"status": "success", **data}

    def predict_latest(self, force_rebuild: bool = False, source: str = "production", year: Optional[int] = None) -> Dict[str, object]:
        latest_path = os.path.join(self.regime_dir, "regime_latest.json")
        if force_rebuild or not os.path.exists(latest_path):
            self.rebuild(year=year, source=source, refresh_features=False)
        if not os.path.exists(latest_path):
            return {"status": "empty", "predictions": []}
        data = self._read_json(latest_path)
        return {"status": "success", **data}

    def _refresh_feature_tables(self):
        for horizon, cfg in self.HORIZON_CONFIG.items():
            builder = cfg["feature_builder_cls"](master_data_path=self.master_data_path, output_path=self.feature_paths[horizon], assets=self.assets)
            builder.ensure_feature_file(force=True)

    def _load_dataframe(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")
        df = pd.read_csv(path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        elif "Unnamed: 0" in df.columns:
            df = df.rename(columns={"Unnamed: 0": "Date"})
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.set_index("Date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(f"Expected Date column in {path}")
        df = df[df.index.notna()].sort_index()
        return df

    def _build_manager(self, horizon: str, artifact_dir: str):
        cfg = self.HORIZON_CONFIG[horizon]
        manager_cls = cfg["manager_cls"]
        return manager_cls(
            data_path=self.feature_paths[horizon],
            model_dir=artifact_dir,
            holdout_root=self.holdout_dirs[horizon],
            assets=self.assets,
            master_data_path=self.master_data_path,
        )

    def _load_horizon_bundle(self, horizon: str, source: str, year: int) -> Dict[str, object]:
        cfg = self.HORIZON_CONFIG[horizon]
        artifact_dir = self._resolve_artifact_dir(horizon=horizon, source=source, year=year)
        metrics_file = os.path.join(artifact_dir, cfg["metrics_file"])
        champions_file = os.path.join(artifact_dir, cfg["champions_file"])
        round3_results = os.path.join(artifact_dir, "round3_results.json")
        round3_best = os.path.join(artifact_dir, "round3_best_by_asset.json")

        if os.path.exists(metrics_file) and os.path.exists(champions_file):
            metrics = self._read_json(metrics_file)
            champions = self._read_json(champions_file)
            return self._normalize_standard_bundle(
                horizon=horizon,
                artifact_dir=artifact_dir,
                metrics_data=metrics,
                champions_data=champions,
            )

        if os.path.exists(round3_results) and os.path.exists(round3_best):
            metrics = self._read_json(round3_results)
            champions = self._read_json(round3_best)
            return self._normalize_round3_bundle(
                horizon=horizon,
                artifact_dir=artifact_dir,
                metrics_data=metrics,
                champions_data=champions,
                year=year,
            )

        raise FileNotFoundError(f"No recognizable artifacts found for horizon={horizon} in {artifact_dir}")

    def _resolve_artifact_dir(self, horizon: str, source: str, year: int) -> str:
        cfg = self.HORIZON_CONFIG[horizon]
        candidates = []
        production_dir = self.production_dirs[horizon]
        holdout_dir = self.holdout_dirs[horizon]

        if source == "production":
            candidates.append(production_dir)
        if year:
            candidates.append(os.path.join(holdout_dir, str(year)))

        latest_holdout = self._latest_numeric_subdir(holdout_dir)
        if latest_holdout:
            candidates.append(latest_holdout)

        candidates.extend(self._experiment_candidates(horizon=horizon, year=year))
        candidates.extend(self._experiment_candidates(horizon=horizon, year=None))

        seen = set()
        ordered = []
        for path in candidates:
            if not path:
                continue
            key = os.path.abspath(path)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(path)

        for candidate in ordered:
            if self._has_valid_artifacts(candidate, cfg):
                return candidate

        raise FileNotFoundError(f"Could not resolve artifacts for horizon={horizon}, source={source}")

    def _has_valid_artifacts(self, folder: str, cfg: dict) -> bool:
        if not folder or not os.path.isdir(folder):
            return False
        metrics_file = os.path.join(folder, cfg["metrics_file"])
        champions_file = os.path.join(folder, cfg["champions_file"])
        if os.path.exists(metrics_file) and os.path.exists(champions_file):
            return True
        if os.path.exists(os.path.join(folder, "round3_results.json")) and os.path.exists(
            os.path.join(folder, "round3_best_by_asset.json")
        ):
            return True
        return False

    def _experiment_candidates(self, horizon: str, year: Optional[int]) -> List[str]:
        root = os.path.join(BASE_DIR, "models", f"holdout_price_{horizon}_experiments")
        if not os.path.isdir(root):
            return []
        pattern = os.path.join(root, "*", str(year)) if year else os.path.join(root, "*")
        candidates = [p for p in glob.glob(pattern) if os.path.isdir(p)]
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates

    def _latest_numeric_subdir(self, folder: str) -> Optional[str]:
        if not os.path.isdir(folder):
            return None
        options = []
        for name in os.listdir(folder):
            full = os.path.join(folder, name)
            if os.path.isdir(full) and name.isdigit():
                options.append((name, full))
        if not options:
            return None
        options.sort(key=lambda t: int(t[0]), reverse=True)
        return options[0][1]

    def _normalize_standard_bundle(self, horizon: str, artifact_dir: str, metrics_data, champions_data) -> Dict[str, object]:
        metrics_list = metrics_data if isinstance(metrics_data, list) else []
        assets = champions_data.get("assets", {}) if isinstance(champions_data, dict) else {}

        metrics_by_key = {}
        for item in metrics_list:
            asset = item.get("asset")
            model_type = item.get("model_type")
            if asset and model_type:
                metrics_by_key[(asset, model_type)] = item

        normalized_assets = {}
        for asset, info in assets.items():
            selected_model = info.get("selected_model") or "naive_last"
            normalized_assets[asset] = {
                "selected_model": selected_model,
                "artifact_prefix": info.get("artifact_prefix", f"{asset}_{selected_model}"),
                "deployment_strategy": info.get("deployment_strategy", "naive_last"),
                "baseline_2025_mape": info.get("baseline_2025_mape"),
                "selected_holdout_mape": info.get("selected_holdout_mape"),
                "improvement_absolute": info.get("improvement_absolute"),
                "improvement_relative": info.get("improvement_relative"),
            }

        return {
            "horizon": horizon,
            "format": "standard",
            "artifact_dir": artifact_dir,
            "assets": normalized_assets,
            "metrics_by_key": metrics_by_key,
        }

    def _normalize_round3_bundle(self, horizon: str, artifact_dir: str, metrics_data, champions_data, year: int) -> Dict[str, object]:
        metrics_list = metrics_data if isinstance(metrics_data, list) else []
        metrics_by_key = {}
        for item in metrics_list:
            entry = item.get("entry", {})
            asset = entry.get("asset", item.get("asset"))
            model_type = entry.get("model_type", item.get("model_type"))
            if not asset or not model_type:
                continue
            if "asset" not in entry:
                entry["asset"] = asset
            if "model_type" not in entry:
                entry["model_type"] = model_type
            if "holdout_metrics" not in entry:
                score = item.get("score", {})
                entry["holdout_metrics"] = {"mape": score.get("holdout_mape")}
            if "baseline_2025" not in entry:
                score = item.get("score", {})
                entry["baseline_2025"] = {"mape": score.get("naive_last_mape")}
            metrics_by_key[(asset, model_type)] = entry

        normalized_assets = {}
        if isinstance(champions_data, dict):
            for asset, info in champions_data.items():
                selected_model = info.get("model_type", "naive_last")
                normalized_assets[asset] = {
                    "selected_model": selected_model,
                    "artifact_prefix": f"{asset}_{selected_model}",
                    "deployment_strategy": "model" if info.get("passes_promotion") else "naive_last",
                    "baseline_2025_mape": info.get("naive_last_mape"),
                    "selected_holdout_mape": info.get("holdout_mape"),
                    "improvement_absolute": info.get("improvement_absolute"),
                    "improvement_relative": info.get("improvement_relative"),
                }

        for asset, info in normalized_assets.items():
            key = (asset, info["selected_model"])
            if key not in metrics_by_key:
                metrics_by_key[key] = {
                    "asset": asset,
                    "model_type": info["selected_model"],
                    "holdout_metrics": {"mape": info.get("selected_holdout_mape")},
                    "shadow_metrics": {"mape": None},
                    "baseline_2025": {"mape": info.get("baseline_2025_mape")},
                    "discovery_summary": {"std_mape": None},
                }

        return {
            "horizon": horizon,
            "format": "round3",
            "artifact_dir": artifact_dir,
            "assets": normalized_assets,
            "metrics_by_key": metrics_by_key,
            "holdout_year": year,
        }

    def _evaluate_horizon(self, horizon: str, manager, df: pd.DataFrame, state_df: pd.DataFrame, bundle: Dict[str, object]):
        predictability_rows: List[Dict[str, object]] = []
        latest_predictions: Dict[str, Dict[str, object]] = {}
        feature_registry = manager.build_feature_registry(df)

        for asset in self.assets:
            champion = bundle["assets"].get(asset)
            if not champion:
                continue
            selected_model = champion.get("selected_model", "naive_last")
            default_feature_cols = feature_registry.get(asset, [])
            history = self._predict_asset_history(
                horizon=horizon,
                manager=manager,
                df=df,
                asset=asset,
                selected_model=selected_model,
                default_feature_cols=default_feature_cols,
                artifact_dir=bundle["artifact_dir"],
                artifact_prefix=champion.get("artifact_prefix"),
            )
            if history.empty:
                continue

            history = self._attach_regime_keys(horizon=horizon, asset=asset, history_df=history, state_df=state_df)
            entry = bundle["metrics_by_key"].get((asset, selected_model), {})
            predictability_rows.extend(
                self._aggregate_predictability(
                    horizon=horizon,
                    asset=asset,
                    group=self._asset_group(asset),
                    history_df=history,
                    metrics_entry=entry,
                )
            )
            latest_predictions[asset] = self._predict_latest_asset(
                horizon=horizon,
                manager=manager,
                df=df,
                asset=asset,
                selected_model=selected_model,
                default_feature_cols=default_feature_cols,
                artifact_dir=bundle["artifact_dir"],
                artifact_prefix=champion.get("artifact_prefix"),
            )

        predictability_rows.sort(
            key=lambda row: (
                row.get("asset", ""),
                {"composite": 0, "latent": 1, "rule": 2, "global": 3}.get(row.get("key_type"), 99),
                row.get("key", ""),
            )
        )
        return predictability_rows, latest_predictions

    def _build_forecast_outputs(
        self,
        state_df: pd.DataFrame,
        feature_tables: Dict[str, pd.DataFrame],
        holdout_year: int,
    ) -> Dict[str, object]:
        holdout_start = f"{int(holdout_year)}-01-01"
        holdout_end = f"{int(holdout_year)}-12-31"
        shadow_start = f"{int(holdout_year) + 1}-01-01"
        return self.forecast_engine.build_forecasts(
            state_df=state_df,
            feature_tables=feature_tables,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            shadow_start=shadow_start,
        )

    def _predict_asset_history(
        self,
        horizon: str,
        manager,
        df: pd.DataFrame,
        asset: str,
        selected_model: str,
        default_feature_cols: List[str],
        artifact_dir: str,
        artifact_prefix: Optional[str],
    ) -> pd.DataFrame:
        suffix = horizon
        centered_col = f"CenteredNormPrice_{asset}_{suffix}"
        future_col = f"FuturePrice_{asset}_{suffix}"
        mape_col = f"MAPEValid_{asset}_{suffix}"
        if centered_col not in df.columns or future_col not in df.columns or mape_col not in df.columns:
            return pd.DataFrame()

        model_info = self._resolve_model_info(
            manager=manager,
            asset=asset,
            selected_model=selected_model,
            default_feature_cols=default_feature_cols,
            artifact_dir=artifact_dir,
            artifact_prefix=artifact_prefix,
        )
        if not model_info["feature_cols"]:
            return pd.DataFrame()

        scaled = self._scale_with_model_info(manager=manager, df=df, model_info=model_info)
        if scaled.size == 0:
            return pd.DataFrame()

        payload = manager._create_sequences(
            scaled_features=scaled,
            centered_targets=df[centered_col].values,
            current_prices=df[asset].values,
            future_prices=df[future_col].values,
            mape_valid=df[mape_col].values.astype(bool),
            dates=df.index.values,
            window_size=model_info["window_size"],
        )
        if len(payload["dates"]) == 0:
            return pd.DataFrame()

        eval_mask = payload["dates"] >= np.datetime64(manager.OUTER_HOLDOUT_START)
        if not np.any(eval_mask):
            return pd.DataFrame()

        X_eval = payload["X"][eval_mask]
        dates = pd.to_datetime(payload["dates"][eval_mask])
        current_prices = payload["current_prices"][eval_mask].astype(float)
        actual_future = payload["future_prices"][eval_mask].astype(float)
        mape_valid = payload["mape_valid"][eval_mask].astype(bool)

        pred_centered = self._predict_centered(
            manager=manager,
            model_info=model_info,
            X_eval=X_eval,
            fallback_len=len(X_eval),
        )
        pred_norm = np.clip(pred_centered + 1.0, -5.0, 5.0)
        model_future = current_prices * pred_norm

        return pd.DataFrame(
            {
                "date": dates,
                "asset": asset,
                "selected_model": model_info["model_type"],
                "current_price": current_prices,
                "actual_future": actual_future,
                "model_future": model_future,
                "naive_future": current_prices,
                "mape_valid": mape_valid,
            }
        )

    def _predict_latest_asset(
        self,
        horizon: str,
        manager,
        df: pd.DataFrame,
        asset: str,
        selected_model: str,
        default_feature_cols: List[str],
        artifact_dir: str,
        artifact_prefix: Optional[str],
    ) -> Dict[str, object]:
        model_info = self._resolve_model_info(
            manager=manager,
            asset=asset,
            selected_model=selected_model,
            default_feature_cols=default_feature_cols,
            artifact_dir=artifact_dir,
            artifact_prefix=artifact_prefix,
        )
        if not model_info["feature_cols"]:
            return {}

        scaled = self._scale_with_model_info(manager=manager, df=df, model_info=model_info)
        if scaled.size == 0:
            return {}
        window_size = model_info["window_size"]
        if len(scaled) < window_size:
            return {}

        latest_sequence = np.array([scaled[-window_size:]], dtype=float)
        centered = self._predict_centered(
            manager=manager,
            model_info=model_info,
            X_eval=latest_sequence,
            fallback_len=1,
        )[0]
        pred_norm = float(np.clip(centered + 1.0, -5.0, 5.0))
        current_price = float(pd.to_numeric(df[asset].iloc[-1], errors="coerce"))
        predicted_future = float(current_price * pred_norm)
        naive_future = float(current_price)
        as_of = pd.Timestamp(df.index[-1])
        target_date = manager._estimate_target_date(as_of)

        return {
            "asset": asset,
            "horizon": horizon,
            "model_type": model_info["model_type"],
            "as_of_date": as_of.strftime("%Y-%m-%d"),
            "target_date": target_date.strftime("%Y-%m-%d"),
            "current_price": current_price,
            "champion_norm_price": pred_norm,
            "champion_future_price": predicted_future,
            "naive_future_price": naive_future,
        }

    def _resolve_model_info(
        self,
        manager,
        asset: str,
        selected_model: str,
        default_feature_cols: List[str],
        artifact_dir: str,
        artifact_prefix: Optional[str],
    ) -> Dict[str, object]:
        prefix = artifact_prefix or f"{asset}_{selected_model}"
        meta_path = os.path.join(artifact_dir, f"{prefix}_meta.json")
        scaler_path = os.path.join(artifact_dir, f"{prefix}_scaler.pkl")
        weight_path = os.path.join(artifact_dir, f"{prefix}.pth")

        meta = {}
        if os.path.exists(meta_path):
            meta = self._read_json(meta_path)

        model_type = meta.get("model_type", selected_model if selected_model != "naive_last" else "nlinear")
        model_config = meta.get("model_config", manager.MODEL_CONFIGS.get(model_type, manager.MODEL_CONFIGS["nlinear"]))
        feature_cols = [c for c in meta.get("feature_cols", default_feature_cols)]
        window_size = int(model_config.get("window_size", manager.MODEL_CONFIGS["nlinear"]["window_size"]))
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        return {
            "asset": asset,
            "prefix": prefix,
            "meta_path": meta_path,
            "scaler_path": scaler_path,
            "weight_path": weight_path,
            "model_type": model_type,
            "model_config": model_config,
            "feature_cols": feature_cols,
            "window_size": window_size,
            "scaler": scaler,
        }

    def _scale_with_model_info(self, manager, df: pd.DataFrame, model_info: Dict[str, object]) -> np.ndarray:
        feature_cols = [c for c in model_info["feature_cols"] if c in df.columns]
        if not feature_cols:
            model_info["feature_cols"] = []
            return np.zeros((0, 0), dtype=float)
        model_info["feature_cols"] = feature_cols

        scaler = model_info["scaler"]
        if scaler is None:
            train_mask = df.index <= pd.Timestamp(manager.FINAL_TUNE_TRAIN_END)
            scaler = manager.fit_scaler(df=df, feature_cols=feature_cols, train_mask=train_mask)
            model_info["scaler"] = scaler

        return manager._scale_frame(df, feature_cols, scaler)

    def _predict_centered(self, manager, model_info: Dict[str, object], X_eval: np.ndarray, fallback_len: int) -> np.ndarray:
        if model_info["model_type"] == "naive_last":
            return np.zeros(fallback_len, dtype=float)
        if not os.path.exists(model_info["weight_path"]):
            return np.zeros(fallback_len, dtype=float)
        if len(X_eval) == 0:
            return np.array([], dtype=float)

        try:
            model = build_price_model(
                model_info["model_type"],
                input_size=int(X_eval.shape[-1]),
                config=model_info["model_config"],
            ).to(manager.device)
            state = torch.load(model_info["weight_path"], map_location=manager.device, weights_only=False)
            model.load_state_dict(state)
            return manager._predict_array(model, X_eval)
        except Exception:
            return np.zeros(fallback_len, dtype=float)

    def _attach_regime_keys(self, horizon: str, asset: str, history_df: pd.DataFrame, state_df: pd.DataFrame) -> pd.DataFrame:
        group = self._asset_group(asset)
        comp_col = f"{horizon}_{group}_composite_id"
        latent_col = f"{horizon}_{group}_latent_id"
        conf_col = f"{horizon}_{group}_latent_conf"
        rule_col = f"{horizon}_rule_id"

        frame = history_df.copy()
        state_cols = [rule_col, comp_col, latent_col, conf_col]
        state_src = state_df.copy()
        if rule_col not in state_src.columns:
            state_src[rule_col] = f"H={horizon}|R=UNK"
        if comp_col not in state_src.columns:
            state_src[comp_col] = f"H={horizon}|G={group}|R=UNK|L=-1"
        if latent_col not in state_src.columns:
            state_src[latent_col] = f"H={horizon}|G={group}|L=-1"
        if conf_col not in state_src.columns:
            state_src[conf_col] = 0.0

        joined = state_src.reindex(frame["date"])[state_cols].ffill().bfill().reset_index(drop=True)

        frame["rule_id"] = joined[rule_col].astype(str)
        frame["composite_id"] = joined[comp_col].astype(str)
        frame["latent_id"] = joined[latent_col].astype(str)
        frame["regime_conf"] = pd.to_numeric(joined[conf_col], errors="coerce").fillna(0.0).astype(float)
        frame["group"] = group
        frame["global_id"] = f"H={horizon}|GLOBAL"
        return frame

    def _aggregate_predictability(self, horizon: str, asset: str, group: str, history_df: pd.DataFrame, metrics_entry: Dict[str, object]):
        rows = []
        stability_score = self._stability_score(metrics_entry)
        for key_type, key_col in [("composite", "composite_id"), ("latent", "latent_id"), ("rule", "rule_id")]:
            grouped = history_df.groupby(key_col, dropna=False)
            for key, subset in grouped:
                rows.append(
                    self._build_predictability_row(
                        horizon=horizon,
                        asset=asset,
                        group=group,
                        key_type=key_type,
                        key=str(key),
                        subset=subset,
                        stability_score=stability_score,
                    )
                )

        rows.append(
            self._build_predictability_row(
                horizon=horizon,
                asset=asset,
                group=group,
                key_type="global",
                key=f"H={horizon}|GLOBAL",
                subset=history_df,
                stability_score=stability_score,
            )
        )
        return rows

    def _build_predictability_row(self, horizon: str, asset: str, group: str, key_type: str, key: str, subset: pd.DataFrame, stability_score: float):
        champion = self._metrics_from_prices(
            actual_future=subset["actual_future"].values,
            pred_future=subset["model_future"].values,
            mape_valid=subset["mape_valid"].values,
        )
        naive = self._metrics_from_prices(
            actual_future=subset["actual_future"].values,
            pred_future=subset["naive_future"].values,
            mape_valid=subset["mape_valid"].values,
        )

        improvement_abs = None
        improvement_rel = None
        if champion["mape"] is not None and naive["mape"] is not None:
            improvement_abs = float(naive["mape"] - champion["mape"])
            if naive["mape"] != 0:
                improvement_rel = float(improvement_abs / naive["mape"])

        support = int(len(subset))
        support_score = min(1.0, support / max(self.MIN_SUPPORT[horizon], 1))
        edge_score = 0.0
        if improvement_rel is not None:
            edge_score = float(np.clip(improvement_rel / 0.20, 0.0, 1.0))
        regime_conf_mean = float(pd.to_numeric(subset["regime_conf"], errors="coerce").fillna(0.0).mean())
        predictability_conf = (
            0.40 * support_score
            + 0.30 * stability_score
            + 0.20 * edge_score
            + 0.10 * float(np.clip(regime_conf_mean, 0.0, 1.0))
        )

        return {
            "horizon": horizon,
            "asset": asset,
            "group": group,
            "key_type": key_type,
            "key": key,
            "rows": support,
            "mape_champion": champion["mape"],
            "smape_champion": champion["smape"],
            "mae_champion": champion["mae"],
            "mape_naive": naive["mape"],
            "smape_naive": naive["smape"],
            "mae_naive": naive["mae"],
            "improvement_abs": improvement_abs,
            "improvement_rel": improvement_rel,
            "regime_conf_mean": regime_conf_mean,
            "stability_score": stability_score,
            "predictability_confidence": float(np.clip(predictability_conf, 0.0, 1.0)),
        }

    def _build_policy(
        self,
        state_df: pd.DataFrame,
        horizon_outputs: Dict[str, Dict[str, object]],
        forecast_outputs: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        latest_state = state_df.iloc[-1]
        forecast_latest = (forecast_outputs or {}).get("latest", {})
        policy = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "as_of_date": state_df.index[-1].strftime("%Y-%m-%d"), "horizons": {}}

        for horizon, output in horizon_outputs.items():
            rows = output["predictability_rows"]
            champions = output["bundle"]["assets"]
            by_key = {}
            for row in rows:
                by_key[(row["asset"], row["key_type"], row["key"])] = row

            horizon_policy = {}
            for asset, champion in champions.items():
                group = self._asset_group(asset)
                observed_composite_key = str(latest_state.get(f"{horizon}_{group}_composite_id", f"H={horizon}|G={group}|R=UNK|L=-1"))
                latent_key = str(latest_state.get(f"{horizon}_{group}_latent_id", f"H={horizon}|G={group}|L=-1"))
                rule_key = str(latest_state.get(f"{horizon}_rule_id", f"H={horizon}|R=UNK"))
                regime_conf = float(pd.to_numeric(latest_state.get(f"{horizon}_{group}_latent_conf", 0.0), errors="coerce"))
                global_key = f"H={horizon}|GLOBAL"

                forecast_group = (((forecast_latest or {}).get(horizon, {}) or {}).get(group, {}) or {})
                forecast_key = forecast_group.get("selected_key")
                forecast_confidence = float(pd.to_numeric(forecast_group.get("confidence", 0.0), errors="coerce"))
                forecast_support = int(forecast_group.get("transition_support", 0) or 0)
                forecast_top3 = forecast_group.get("top3", []) or []
                forecast_available = bool(forecast_group)

                forecast_eligible = bool(
                    forecast_available
                    and forecast_key
                    and (forecast_confidence >= self.CONF_THRESHOLD)
                    and (forecast_support >= self.FORECAST_MIN_SUPPORT.get(horizon, 1))
                )

                forecast_rejection_reason = None
                if forecast_available:
                    if not forecast_key:
                        forecast_rejection_reason = "MISSING_FORECAST_KEY"
                    elif forecast_confidence < self.CONF_THRESHOLD:
                        forecast_rejection_reason = "LOW_FORECAST_CONFIDENCE"
                    elif forecast_support < self.FORECAST_MIN_SUPPORT.get(horizon, 1):
                        forecast_rejection_reason = "LOW_SUPPORT"

                key_chain = []
                if forecast_eligible:
                    key_chain.append(("forecast", "composite", str(forecast_key)))
                key_chain.extend(
                    [
                        ("observed", "composite", observed_composite_key),
                        ("latent", "latent", latent_key),
                        ("rule", "rule", rule_key),
                        ("global", "global", global_key),
                    ]
                )

                chosen = None
                policy_key_source = "global"
                fallback_level = "global"
                policy_key = global_key
                fallback_path = []
                for source_name, key_type, key in key_chain:
                    fallback_path.append(source_name)
                    candidate = by_key.get((asset, key_type, key))
                    if candidate:
                        chosen = candidate
                        policy_key_source = source_name
                        fallback_level = key_type
                        policy_key = key
                        break

                if chosen is None:
                    chosen = {
                        "rows": 0,
                        "improvement_abs": None,
                        "improvement_rel": None,
                        "mape_champion": None,
                        "mape_naive": None,
                        "predictability_confidence": 0.0,
                    }

                support_ok = int(chosen.get("rows", 0)) >= self.MIN_SUPPORT[horizon]
                confidence_value = forecast_confidence if policy_key_source == "forecast" else regime_conf
                confidence_ok = confidence_value >= self.CONF_THRESHOLD
                improvement_abs = chosen.get("improvement_abs")
                improvement_rel = chosen.get("improvement_rel")
                edge_ok = bool(
                    (improvement_abs is not None and improvement_abs >= self.IMPROVE_ABS_THRESHOLD)
                    or (improvement_rel is not None and improvement_rel >= self.IMPROVE_REL_THRESHOLD)
                )

                passes_all = support_ok and confidence_ok and edge_ok
                selected_strategy = "champion_model" if passes_all else "naive_last"
                if not support_ok:
                    reason = "LOW_SUPPORT"
                elif not confidence_ok and forecast_rejection_reason == "LOW_FORECAST_CONFIDENCE":
                    reason = "LOW_FORECAST_CONFIDENCE"
                elif not confidence_ok:
                    reason = "LOW_CONFIDENCE"
                elif not edge_ok:
                    reason = "NO_EDGE"
                else:
                    if policy_key_source == "forecast":
                        reason = "USE_CHAMPION_FORECAST"
                    elif policy_key_source == "observed" and forecast_available and forecast_rejection_reason is not None:
                        reason = "FALLBACK_OBSERVED"
                    else:
                        reason = "USE_CHAMPION"

                horizon_policy[asset] = {
                    "asset": asset,
                    "horizon": horizon,
                    "group": group,
                    "selected_model": champion.get("selected_model", "naive_last"),
                    "selected_strategy": selected_strategy,
                    "reason_code": reason,
                    "fallback_level": fallback_level,
                    "policy_key_source": policy_key_source,
                    "policy_key": policy_key,
                    "fallback_path": fallback_path,
                    "forecast_key": forecast_key,
                    "forecast_confidence": forecast_confidence,
                    "forecast_support": forecast_support,
                    "forecast_top3": forecast_top3,
                    "forecast_eligible": forecast_eligible,
                    "forecast_rejection_reason": forecast_rejection_reason,
                    "composite_key": observed_composite_key,
                    "latent_key": latent_key,
                    "rule_key": rule_key,
                    "rows": int(chosen.get("rows", 0)),
                    "regime_confidence": regime_conf,
                    "predictability_confidence": chosen.get("predictability_confidence"),
                    "improvement_abs": improvement_abs,
                    "improvement_rel": improvement_rel,
                    "mape_champion": chosen.get("mape_champion"),
                    "mape_naive": chosen.get("mape_naive"),
                    "baseline_2025_mape": champion.get("baseline_2025_mape"),
                    "selected_holdout_mape": champion.get("selected_holdout_mape"),
                }

            policy["horizons"][horizon] = horizon_policy

        return policy

    def _build_latest_predictions(
        self,
        policy: Dict[str, object],
        horizon_outputs: Dict[str, Dict[str, object]],
        forecast_outputs: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        rows = []
        as_of_date = policy.get("as_of_date")
        for horizon, output in horizon_outputs.items():
            latest_map = output["latest_predictions"]
            horizon_policy = policy.get("horizons", {}).get(horizon, {})
            for asset, policy_row in horizon_policy.items():
                latest_pred = latest_map.get(asset)
                if not latest_pred:
                    continue
                if policy_row["selected_strategy"] == "champion_model":
                    selected_future = latest_pred["champion_future_price"]
                else:
                    selected_future = latest_pred["naive_future_price"]
                current = latest_pred["current_price"]
                implied = ((selected_future / current) - 1.0) * 100.0 if current else 0.0
                rows.append(
                    {
                        "asset": asset,
                        "horizon": horizon,
                        "group": policy_row["group"],
                        "selected_model": policy_row["selected_model"],
                        "selected_strategy": policy_row["selected_strategy"],
                        "reason_code": policy_row["reason_code"],
                        "policy_key_source": policy_row.get("policy_key_source"),
                        "policy_key": policy_row.get("policy_key"),
                        "as_of_date": latest_pred["as_of_date"],
                        "target_date": latest_pred["target_date"],
                        "current_price": current,
                        "champion_future_price": latest_pred["champion_future_price"],
                        "naive_future_price": latest_pred["naive_future_price"],
                        "selected_future_price": selected_future,
                        "implied_return_pct": float(round(implied, 4)),
                        "regime_confidence": policy_row["regime_confidence"],
                        "predictability_confidence": policy_row["predictability_confidence"],
                        "forecast_key": policy_row.get("forecast_key"),
                        "forecast_confidence": policy_row.get("forecast_confidence"),
                        "forecast_support": policy_row.get("forecast_support"),
                        "forecast_top3": policy_row.get("forecast_top3", []),
                        "baseline_2025_mape": policy_row["baseline_2025_mape"],
                        "selected_holdout_mape": policy_row["selected_holdout_mape"],
                    }
                )
        rows.sort(key=lambda item: (item["horizon"], item["asset"]))
        return {"as_of_date": as_of_date, "predictions": rows}

    def _save_artifacts(
        self,
        state_df: pd.DataFrame,
        horizon_outputs: Dict[str, Dict[str, object]],
        forecast_outputs: Dict[str, object],
        policy: Dict[str, object],
        latest: Dict[str, object],
        holdout_year: int,
        source: str,
    ) -> Dict[str, object]:
        os.makedirs(self.regime_dir, exist_ok=True)
        state_path = os.path.join(self.regime_dir, "regime_state_history.parquet")
        state_format = "parquet"
        try:
            state_df.to_parquet(state_path)
        except Exception:
            state_df.to_pickle(state_path)
            state_format = "pickle_fallback"

        predictability_paths = {}
        for horizon, output in horizon_outputs.items():
            path = os.path.join(self.regime_dir, f"regime_predictability_{horizon}.json")
            self._write_json(path, {"horizon": horizon, "rows": output["predictability_rows"]})
            predictability_paths[horizon] = path

        forecast_paths = {}
        forecast_horizons = (forecast_outputs or {}).get("horizons", {})
        for horizon, payload in forecast_horizons.items():
            path = os.path.join(self.regime_dir, f"regime_forecast_{horizon}.json")
            self._write_json(path, {"horizon": horizon, "groups": payload.get("groups", {}), "generated_at": payload.get("generated_at")})
            forecast_paths[horizon] = path

        forecast_latest_path = os.path.join(self.regime_dir, "regime_forecast_latest.json")
        self._write_json(
            forecast_latest_path,
            {
                "generated_at": (forecast_outputs or {}).get("generated_at"),
                "horizons": (forecast_outputs or {}).get("latest", {}),
            },
        )

        policy_path = os.path.join(self.regime_dir, "regime_policy.json")
        latest_path = os.path.join(self.regime_dir, "regime_latest.json")
        meta_path = os.path.join(self.regime_dir, "regime_build_meta.json")

        self._write_json(policy_path, policy)
        self._write_json(latest_path, latest)
        self._write_json(
            meta_path,
            {
                "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "holdout_year": holdout_year,
                "source": source,
                "state_format": state_format,
                "state_rows": int(len(state_df)),
                "horizon_sources": {
                    h: {
                        "artifact_dir": out["bundle"]["artifact_dir"],
                        "format": out["bundle"]["format"],
                        "rows": len(out["predictability_rows"]),
                    }
                    for h, out in horizon_outputs.items()
                },
                "forecast": {
                    "generated_at": (forecast_outputs or {}).get("generated_at"),
                    "horizons": {
                        h: {
                            "groups": {
                                g: {
                                    "train_rows": payload.get("train_rows"),
                                    "class_count": payload.get("class_count"),
                                    "holdout_top1": ((payload.get("diagnostics") or {}).get("holdout") or {}).get("top1_accuracy"),
                                    "holdout_top3": ((payload.get("diagnostics") or {}).get("holdout") or {}).get("top3_recall"),
                                    "holdout_brier": ((payload.get("diagnostics") or {}).get("holdout") or {}).get("brier_score"),
                                }
                                for g, payload in (((forecast_horizons.get(h) or {}).get("groups")) or {}).items()
                            }
                        }
                        for h in forecast_horizons.keys()
                    },
                },
            },
        )

        return {
            "state_path": state_path,
            "predictability_paths": predictability_paths,
            "forecast_paths": forecast_paths,
            "forecast_latest_path": forecast_latest_path,
            "policy_path": policy_path,
            "latest_path": latest_path,
            "meta_path": meta_path,
        }

    def _read_state_history(self) -> pd.DataFrame:
        state_path = os.path.join(self.regime_dir, "regime_state_history.parquet")
        if not os.path.exists(state_path):
            return pd.DataFrame()
        try:
            df = pd.read_parquet(state_path)
        except Exception:
            df = pd.read_pickle(state_path)
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.set_index("Date")
        df = df[df.index.notna()].sort_index()
        return df

    def _metrics_from_prices(self, actual_future, pred_future, mape_valid):
        actual = np.asarray(actual_future, dtype=float)
        pred = np.asarray(pred_future, dtype=float)
        mask = np.asarray(mape_valid, dtype=bool)
        valid = np.isfinite(actual) & np.isfinite(pred)
        if not valid.any():
            return {"mape": None, "smape": None, "mae": None}

        actual = actual[valid]
        pred = pred[valid]
        mask = mask[valid]

        abs_err = np.abs(actual - pred)
        smape_denom = np.clip(np.abs(actual) + np.abs(pred), 1e-6, None)
        smape = float(np.mean((2.0 * abs_err / smape_denom) * 100.0))
        mae = float(np.mean(abs_err))

        mape_mask = mask & (np.abs(actual) > 1e-6)
        if mape_mask.any():
            mape = float(np.mean(np.abs((actual[mape_mask] - pred[mape_mask]) / actual[mape_mask])) * 100.0)
        else:
            mape = None

        return {"mape": mape, "smape": smape, "mae": mae}

    def _stability_score(self, metrics_entry: Dict[str, object]) -> float:
        discovery_std = ((metrics_entry.get("discovery_summary") or {}).get("std_mape"))
        holdout_mape = ((metrics_entry.get("holdout_metrics") or {}).get("mape"))
        shadow_mape = ((metrics_entry.get("shadow_metrics") or {}).get("mape"))

        std_term = 0.5
        if discovery_std is not None and np.isfinite(discovery_std):
            std_term = float(1.0 / (1.0 + abs(discovery_std)))

        spread_term = 0.5
        if holdout_mape is not None and shadow_mape is not None and np.isfinite(holdout_mape) and np.isfinite(shadow_mape):
            spread_term = float(1.0 / (1.0 + abs(float(shadow_mape) - float(holdout_mape))))

        return float(np.clip(0.5 * std_term + 0.5 * spread_term, 0.0, 1.0))

    def _asset_group(self, asset: str) -> str:
        for name, group_assets in RegimeStateEngine.ASSET_GROUPS.items():
            if asset in group_assets:
                return name
        return "other"

    def _read_json(self, path: str):
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_json(self, path: str, payload):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
