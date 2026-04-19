import copy
import datetime as dt
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

from analysis.base_model import BaseModel
from analysis.price_1w_features import Price1WFeatureBuilder
from analysis.price_3d_models import build_price_model
from backend.shared.device import resolve_torch_device
from config.settings import (
    ASSETS,
    HOLDOUT_PRICE_1W_DIR,
    MASTER_DATA_FILE,
    MODELS_PRICE_1W_DIR,
    PRICE_1W_FEATURES_FILE,
)


DEVICE = resolve_torch_device()


@dataclass
class DiscoveryFold:
    name: str
    train_end: str
    val_start: str
    val_end: str


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class Price1WRegressionManager(BaseModel):
    DISCOVERY_FOLDS = [
        DiscoveryFold("2022_H1", "2021-12-31", "2022-01-01", "2022-06-30"),
        DiscoveryFold("2023_H1", "2022-12-31", "2023-01-01", "2023-06-30"),
        DiscoveryFold("2024_H1", "2023-12-31", "2024-01-01", "2024-06-30"),
        DiscoveryFold("2024_H2", "2024-06-30", "2024-07-01", "2024-12-31"),
    ]
    FINAL_TUNE_TRAIN_END = "2024-06-30"
    FINAL_TUNE_VAL_START = "2024-07-01"
    FINAL_TUNE_VAL_END = "2024-12-31"
    DEFAULT_HOLDOUT_YEAR = 2025
    OUTER_HOLDOUT_START = "2025-01-01"
    OUTER_HOLDOUT_END = "2025-12-31"
    SHADOW_START = "2026-01-01"
    PROMOTION_RELATIVE_THRESHOLD = 0.03
    PROMOTION_ABSOLUTE_THRESHOLD = 0.05
    HORIZON_DAYS = 5

    ASSET_GROUPS = {
        "equities": ["SP500", "Nasdaq", "DJIA", "Russell2000"],
        "precious": ["Gold", "Silver"],
        "commodities": ["Copper", "Oil"],
    }

    MODEL_CANDIDATES = {
        "SP500": ["nlinear", "patchtst", "tide"],
        "Nasdaq": ["nlinear", "patchtst", "tide"],
        "DJIA": ["nlinear", "patchtst", "tide"],
        "Russell2000": ["nlinear", "patchtst", "tide"],
        "Gold": ["nlinear", "tide", "nbeats_reg"],
        "Silver": ["nlinear", "tide", "nbeats_reg"],
        "Copper": ["nlinear", "nhits", "lstm_reg_revin"],
        "Oil": ["nlinear", "nhits", "lstm_reg_revin"],
    }

    MODEL_CONFIGS = {
        "nlinear": {"window_size": 20, "batch_size": 256, "lr": 1e-3, "dropout": 0.0},
        "patchtst": {
            "window_size": 40,
            "batch_size": 128,
            "lr": 5e-4,
            "patch_len": 5,
            "stride": 2,
            "d_model": 64,
            "nhead": 4,
            "num_layers": 2,
            "dropout": 0.1,
        },
        "tide": {
            "window_size": 60,
            "batch_size": 128,
            "lr": 7e-4,
            "hidden_size": 128,
            "depth": 3,
            "dropout": 0.1,
        },
        "nbeats_reg": {
            "window_size": 50,
            "batch_size": 128,
            "lr": 8e-4,
            "nb_stacks": 2,
            "nb_blocks": 3,
            "nb_width": 128,
            "dropout": 0.1,
        },
        "nhits": {
            "window_size": 50,
            "batch_size": 128,
            "lr": 8e-4,
            "hidden_size": 128,
            "pool_sizes": [1, 2, 5],
            "dropout": 0.1,
        },
        "lstm_reg_revin": {
            "window_size": 45,
            "batch_size": 128,
            "lr": 7e-4,
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.2,
        },
    }

    def __init__(
        self,
        data_path: str = PRICE_1W_FEATURES_FILE,
        model_dir: str = MODELS_PRICE_1W_DIR,
        holdout_root: str = HOLDOUT_PRICE_1W_DIR,
        assets: Optional[Iterable[str]] = None,
        master_data_path: str = MASTER_DATA_FILE,
    ):
        super().__init__(data_path=data_path, model_dir=model_dir)
        self.holdout_root = holdout_root
        self.assets = list(assets or ASSETS)
        self.device = DEVICE
        self.feature_builder = Price1WFeatureBuilder(
            master_data_path=master_data_path,
            output_path=data_path,
            assets=self.assets,
        )
        os.makedirs(self.holdout_root, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

    def train_all_models(self, progress_callback=None, **kwargs):
        return self.train_holdout_pipeline(progress_callback=progress_callback, **kwargs)

    def load_and_preprocess(self, force_refresh_features: bool = False) -> pd.DataFrame:
        from analysis.price_pipeline_common import load_and_preprocess_price_features

        return load_and_preprocess_price_features(
            data_path=self.data_path,
            feature_builder=self.feature_builder,
            force_refresh_features=force_refresh_features,
            missing_date_message="Price1W features file is missing a Date column.",
        )

    def refresh_feature_cache(self) -> Dict[str, object]:
        from analysis.price_pipeline_common import refresh_price_feature_cache

        return refresh_price_feature_cache(data_path=self.data_path, feature_builder=self.feature_builder)

    def fit_scaler(self, df: pd.DataFrame, feature_cols: List[str], train_mask: pd.Series) -> RobustScaler:
        from analysis.price_pipeline_common import fit_price_scaler

        return fit_price_scaler(df=df, feature_cols=feature_cols, train_mask=train_mask)

    def build_feature_registry(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        registry = {}
        common = [
            "VIX",
            "Momentum",
            "Strength_RSI",
            "Breadth_Vol",
            "Options_VIX",
            "Junk_Bond_Demand",
            "Volatility_Spread",
            "Safe_Haven_Demand",
            "FEDFUNDS",
            "CPIAUCSL",
            "PPIACO",
            "UNRATE",
            "PAYEMS",
            "M2SL",
            "T10Y3M",
            "UMCSENT",
            "WALCL",
            "DGS10",
            "A191RL1Q225SBEA",
            "CPI_YoY",
            "PPI_YoY",
            "Real_Yield",
            "Liquidity_Impulse",
            "Curve_Steepening",
            "Bond_Stock_Corr",
            "VIX_Regime",
            "Safety_vs_Risk",
            "Regime_Inflation",
            "Regime_Liquidity",
            "Regime_Risk",
            "Regime_Rates",
            "is_month_end",
            "dow_0",
            "dow_1",
            "dow_2",
            "dow_3",
            "dow_4",
        ]

        for asset in self.assets:
            own = self._own_asset_features(asset)
            if asset in self.ASSET_GROUPS["equities"]:
                extras = [
                    "SP500",
                    "Nasdaq",
                    "DJIA",
                    "Russell2000",
                    "Tech_vs_Broad",
                    "Gold",
                    "Oil",
                    "SP500_ZScore",
                    "Nasdaq_ZScore",
                    "DJIA_ZScore",
                    "Russell2000_ZScore",
                ]
            elif asset in self.ASSET_GROUPS["precious"]:
                extras = [
                    "Gold",
                    "Silver",
                    "Silver_Gold_Ratio",
                    "Safe_Haven_Demand",
                    "Real_Yield",
                    "Liquidity_Impulse",
                    "DGS10",
                    "VIX",
                    "Gold_ZScore",
                    "Silver_ZScore",
                ]
            else:
                extras = [
                    "Oil",
                    "Copper",
                    "Gold",
                    "Silver",
                    "Oil_Gold_Ratio",
                    "Silver_Gold_Ratio",
                    "CPI_YoY",
                    "PPI_YoY",
                    "Curve_Steepening",
                    "Liquidity_Impulse",
                    "UNRATE",
                    "PAYEMS",
                    "Breadth_Vol",
                    "Oil_ZScore",
                    "Copper_ZScore",
                ]

            ordered = []
            for col in own + extras + common:
                if col in df.columns and col not in ordered:
                    ordered.append(col)
            registry[asset] = ordered

        return registry

    def get_model_candidates(self, asset: str) -> List[str]:
        return list(self.MODEL_CANDIDATES.get(asset, ["nlinear"]))

    def train_holdout_pipeline(
        self,
        holdout_year: Optional[int] = None,
        progress_callback=None,
        epochs: int = 20,
        asset_subset: Optional[Iterable[str]] = None,
        candidate_override: Optional[Dict[str, List[str]]] = None,
        force_refresh_features: bool = False,
    ) -> Dict[str, object]:
        from analysis.price_pipeline_common import train_holdout_pipeline_core

        return train_holdout_pipeline_core(
            manager=self,
            holdout_year=holdout_year,
            progress_callback=progress_callback,
            epochs=epochs,
            asset_subset=asset_subset,
            candidate_override=candidate_override,
            force_refresh_features=force_refresh_features,
            artifact_prefix="price1w",
            progress_label="1W",
        )

    def _train_asset_model(
        self,
        df: pd.DataFrame,
        asset: str,
        model_type: str,
        feature_cols: List[str],
        save_dir: str,
        epochs: int,
    ) -> Dict[str, object]:
        from analysis.price_pipeline_common import train_asset_model_core

        return train_asset_model_core(
            manager=self,
            df=df,
            asset=asset,
            model_type=model_type,
            feature_cols=feature_cols,
            save_dir=save_dir,
            epochs=epochs,
        )

    def _run_discovery_fold(
        self,
        sequence_payload: Dict[str, np.ndarray],
        model_type: str,
        model_config: Dict[str, object],
        fold: DiscoveryFold,
        epochs: int,
    ) -> Dict[str, object]:
        from analysis.price_pipeline_common import run_discovery_fold_core

        return run_discovery_fold_core(
            manager=self,
            sequence_payload=sequence_payload,
            model_type=model_type,
            model_config=model_config,
            fold=fold,
            epochs=epochs,
        )

    def _train_final_model(
        self,
        df: pd.DataFrame,
        asset: str,
        model_type: str,
        model_config: Dict[str, object],
        feature_cols: List[str],
        epochs: int,
        save_dir: str,
    ) -> Dict[str, object]:
        from analysis.price_pipeline_common import train_final_model_core

        return train_final_model_core(
            manager=self,
            df=df,
            asset=asset,
            model_type=model_type,
            model_config=model_config,
            feature_cols=feature_cols,
            epochs=epochs,
            save_dir=save_dir,
            target_suffix="1w",
            build_model_fn=build_price_model,
        )

    def _fit_with_early_stopping(
        self,
        model_type: str,
        model_config: Dict[str, object],
        input_size: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        current_val: np.ndarray,
        future_val: np.ndarray,
        mape_valid_val: np.ndarray,
        epochs: int,
    ) -> Tuple[nn.Module, int]:
        from analysis.price_pipeline_common import fit_with_early_stopping_core

        return fit_with_early_stopping_core(
            model_type=model_type,
            model_config=model_config,
            input_size=input_size,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            current_val=current_val,
            future_val=future_val,
            mape_valid_val=mape_valid_val,
            epochs=epochs,
            device=self.device,
            build_model_fn=build_price_model,
            dataset_cls=SequenceDataset,
            loss_function_fn=self._loss_function,
            predict_array_fn=self._predict_array,
            evaluate_predictions_fn=self._evaluate_predictions,
        )

    def _fit_fixed_epochs(self, model: nn.Module, model_config: Dict[str, object], X_train: np.ndarray, y_train: np.ndarray, epochs: int) -> nn.Module:
        from analysis.price_pipeline_common import fit_fixed_epochs_core

        return fit_fixed_epochs_core(
            model=model,
            model_config=model_config,
            X_train=X_train,
            y_train=y_train,
            epochs=epochs,
            device=self.device,
            dataset_cls=SequenceDataset,
            loss_function_fn=self._loss_function,
        )

    def _loss_function(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        from analysis.price_pipeline_common import price_loss_function

        return price_loss_function(preds=preds, targets=targets)

    def _predict_array(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        from analysis.price_pipeline_common import predict_price_array

        return predict_price_array(model=model, X=X, device=self.device)

    def _build_sequence_payload(self, df: pd.DataFrame, asset: str, feature_cols: List[str], window_size: int) -> Dict[str, np.ndarray]:
        from analysis.price_pipeline_common import build_price_sequence_payload

        return build_price_sequence_payload(
            df=df,
            feature_cols=feature_cols,
            train_end=self.FINAL_TUNE_TRAIN_END,
            centered_targets=df[f"CenteredNormPrice_{asset}_1w"].values,
            current_prices=df[asset].values,
            future_prices=df[f"FuturePrice_{asset}_1w"].values,
            mape_valid=df[f"MAPEValid_{asset}_1w"].values.astype(bool),
            dates=df.index.values,
            window_size=window_size,
            holdout_start=self.OUTER_HOLDOUT_START,
            holdout_end=self.OUTER_HOLDOUT_END,
        )

    def _scale_frame(self, df: pd.DataFrame, feature_cols: List[str], scaler: RobustScaler) -> np.ndarray:
        from analysis.price_pipeline_common import scale_price_frame

        return scale_price_frame(df=df, feature_cols=feature_cols, scaler=scaler)

    def _create_sequences(
        self,
        scaled_features: np.ndarray,
        centered_targets: np.ndarray,
        current_prices: np.ndarray,
        future_prices: np.ndarray,
        mape_valid: np.ndarray,
        dates: np.ndarray,
        window_size: int,
    ) -> Dict[str, np.ndarray]:
        from analysis.price_pipeline_common import create_price_sequences

        return create_price_sequences(
            scaled_features=scaled_features,
            centered_targets=centered_targets,
            current_prices=current_prices,
            future_prices=future_prices,
            mape_valid=mape_valid,
            dates=dates,
            window_size=window_size,
        )

    def _evaluate_predictions(
        self,
        current_prices,
        actual_future,
        pred_centered,
        mape_valid,
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        from analysis.price_pipeline_common import evaluate_price_predictions

        return evaluate_price_predictions(
            current_prices=current_prices,
            actual_future=actual_future,
            pred_centered=pred_centered,
            mape_valid=mape_valid,
            latency_ms=latency_ms,
        )

    def select_champions_from_metrics(self, metrics: List[Dict[str, object]], holdout_year: int) -> Dict[str, object]:
        champions = {
            "holdout_year": int(holdout_year),
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "assets": {},
        }

        for asset in self.assets:
            asset_metrics = [m for m in metrics if m["asset"] == asset]
            baseline = asset_metrics[0].get("baseline_2025", {}) if asset_metrics else {}
            ranked = sorted(
                asset_metrics,
                key=lambda item: (
                    self._sort_nan(item["discovery_summary"]["median_mape"]),
                    self._sort_nan(item["discovery_summary"]["std_mape"]),
                    self._sort_nan(item["holdout_metrics"].get("latency_ms")),
                ),
            )
            best = ranked[0] if ranked else None

            selected_model = best["model_type"] if best else "naive_last"
            holdout_mape = best["holdout_metrics"].get("mape") if best else None
            baseline_mape = baseline.get("mape")
            improvement_relative = None
            improvement_absolute = None
            deployment_strategy = "naive_last"

            if holdout_mape is not None and baseline_mape is not None:
                improvement_absolute = float(baseline_mape - holdout_mape)
                improvement_relative = float(improvement_absolute / baseline_mape) if baseline_mape else None
                if improvement_absolute >= self.PROMOTION_ABSOLUTE_THRESHOLD or (
                    improvement_relative is not None and improvement_relative >= self.PROMOTION_RELATIVE_THRESHOLD
                ):
                    deployment_strategy = "model"

            champions["assets"][asset] = {
                "selected_model": selected_model,
                "deployment_strategy": deployment_strategy,
                "baseline_2025_mape": baseline_mape,
                "selected_holdout_mape": holdout_mape,
                "improvement_relative": improvement_relative,
                "improvement_absolute": improvement_absolute,
                "discovery_median_mape": None if not best else best["discovery_summary"]["median_mape"],
                "discovery_std_mape": None if not best else best["discovery_summary"]["std_mape"],
                "latency_ms": None if not best else best["holdout_metrics"].get("latency_ms"),
                "artifact_prefix": None if not best else f"{asset}_{selected_model}",
            }

        return champions

    def promote_champions(self, holdout_year: Optional[int] = None) -> Dict[str, object]:
        from analysis.price_pipeline_common import promote_price_champions

        return promote_price_champions(
            manager=self,
            holdout_year=holdout_year,
            artifact_prefix="price1w",
        )

    def predict_latest(self, model_dir: Optional[str] = None, year: Optional[int] = None):
        from analysis.price_pipeline_common import predict_latest_price

        return predict_latest_price(
            manager=self,
            build_model_fn=build_price_model,
            artifact_prefix="price1w",
            model_dir=model_dir,
            year=year,
        )

    def load_metrics(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None) -> Dict[str, object]:
        from analysis.price_pipeline_common import load_price_metrics

        return load_price_metrics(
            manager=self,
            artifact_prefix="price1w",
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_feature_registry(self, holdout_year: Optional[int] = None, model_dir: Optional[str] = None) -> Dict[str, object]:
        from analysis.price_pipeline_common import load_price_feature_registry

        return load_price_feature_registry(
            manager=self,
            artifact_prefix="price1w",
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_model_metadata(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        from analysis.price_pipeline_common import load_price_model_metadata

        return load_price_model_metadata(
            manager=self,
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def load_model_scaler(
        self,
        asset: str,
        model_type: str,
        holdout_year: Optional[int] = None,
        model_dir: Optional[str] = None,
    ):
        from analysis.price_pipeline_common import load_price_model_scaler

        return load_price_model_scaler(
            manager=self,
            asset=asset,
            model_type=model_type,
            holdout_year=holdout_year,
            model_dir=model_dir,
        )

    def _resolve_holdout_dir(self, holdout_year: Optional[int] = None) -> str:
        from analysis.price_pipeline_common import resolve_holdout_dir

        return resolve_holdout_dir(
            holdout_root=self.holdout_root,
            holdout_year=holdout_year,
            horizon_label="price-1w",
        )

    def _resolve_default_metrics_dir(self) -> str:
        from analysis.price_pipeline_common import resolve_default_metrics_dir

        return resolve_default_metrics_dir(
            model_dir=self.model_dir,
            metrics_filename="price1w_metrics.json",
            resolve_holdout_dir_fn=self._resolve_holdout_dir,
        )

    def _estimate_target_date(self, latest_date: pd.Timestamp) -> pd.Timestamp:
        from analysis.price_pipeline_common import estimate_target_date

        return estimate_target_date(latest_date=latest_date, horizon_days=self.HORIZON_DAYS)

    def _own_asset_features(self, asset: str) -> List[str]:
        from analysis.price_pipeline_common import own_asset_features

        return own_asset_features(asset)

    def _asset_group(self, asset: str) -> str:
        from analysis.price_pipeline_common import asset_group

        return asset_group(asset, self.ASSET_GROUPS)

    @staticmethod
    def _safe_stat(values: List[float], fn):
        from analysis.price_pipeline_common import safe_stat

        return safe_stat(values, fn)

    @staticmethod
    def _sort_nan(value):
        from analysis.price_pipeline_common import sort_nan

        return sort_nan(value)
