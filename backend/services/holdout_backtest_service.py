from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Set

import numpy as np
import pandas as pd

from backend.domain.dto import HoldoutRunRequest, HoldoutSplitRequest
from backend.services.dl_inference_service import DLInferenceService
from backend.services.dl_snapshot_lifecycle_service import DLSnapshotLifecycleService
from backend.services.dl_training_orchestration_service import DLTrainingOrchestrationService
from backend.shared.http import ServiceError
from config.settings import ENGINEERED_FEATURES_FILE, HOLDOUT_DIR, MASTER_DATA_FILE, MODELS_DIR


class HoldoutBacktestService:
    """Domain service for holdout split prep, holdout runs, and holdout model listing."""

    def __init__(
        self,
        *,
        training_orchestration_service: Optional[DLTrainingOrchestrationService] = None,
        inference_service: Optional[DLInferenceService] = None,
        snapshot_lifecycle_service: Optional[DLSnapshotLifecycleService] = None,
    ) -> None:
        self.training_orchestration = training_orchestration_service or DLTrainingOrchestrationService()
        self.inference_service = inference_service or DLInferenceService()
        self.snapshot_lifecycle = snapshot_lifecycle_service or DLSnapshotLifecycleService()

    @staticmethod
    def _sse_data(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    @staticmethod
    def _format_time(seconds: float) -> str:
        if seconds < 60:
            return f"{int(seconds)}s"
        minutes, rem_seconds = divmod(seconds, 60)
        return f"{int(minutes)}m {int(rem_seconds)}s"

    def prepare_split(self, request: HoldoutSplitRequest) -> Dict[str, Any]:
        cutoff_year = int(request.cutoff_year)
        target_file = ENGINEERED_FEATURES_FILE if os.path.exists(ENGINEERED_FEATURES_FILE) else MASTER_DATA_FILE

        if not os.path.exists(target_file):
            raise ServiceError(
                "No data file found (Engineered or Master). Run Data Collection first.",
                status_code=404,
            )

        df = pd.read_csv(target_file, index_col=0, parse_dates=True)
        if df.empty:
            raise ServiceError("Data file is empty.", status_code=400)

        cutoff_date = f"{cutoff_year}-12-31"
        train_df = df.loc[df.index <= cutoff_date]
        test_df = df.loc[df.index > cutoff_date]

        response: Dict[str, Any] = {
            "train_rows": len(train_df),
            "train_start": train_df.index.min().strftime("%Y-%m-%d") if not train_df.empty else "-",
            "train_end": train_df.index.max().strftime("%Y-%m-%d") if not train_df.empty else "-",
            "test_rows": len(test_df),
            "test_start": test_df.index.min().strftime("%Y-%m-%d") if not test_df.empty else "-",
            "test_end": test_df.index.max().strftime("%Y-%m-%d") if not test_df.empty else "-",
            "cutoff_year": cutoff_year,
        }

        version_dir = os.path.join(HOLDOUT_DIR, str(cutoff_year))
        os.makedirs(version_dir, exist_ok=True)

        manifest_path = os.path.join(version_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(response, handle)

        return response

    def list_models(self) -> List[Dict[str, Any]]:
        models: List[Dict[str, Any]] = []
        dl_base = os.path.join(MODELS_DIR, "holdout_dl")

        if os.path.exists(dl_base):
            for year in os.listdir(dl_base):
                path = os.path.join(dl_base, year)
                if not os.path.isdir(path):
                    continue

                ctime = os.path.getctime(path)
                dl_mode = "balanced"

                config_path = os.path.join(path, "dl_config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, "r", encoding="utf-8") as handle:
                            conf = json.load(handle)
                        dl_mode = str(conf.get("dl_mode", "balanced"))
                    except Exception:
                        pass

                models.append(
                    {
                        "id": f"dl_{year}",
                        "cutoff_year": year,
                        "category": "dl",
                        "model_type": "dl_auto",
                        "dl_mode": dl_mode,
                        "created_at": datetime.fromtimestamp(ctime).isoformat(),
                        "cutoff_date": f"{year}-12-31",
                    }
                )

        models.sort(key=lambda item: str(item.get("cutoff_year", "")), reverse=True)
        return models

    def _load_dl_config(self, version_dir: str) -> Dict[str, Any]:
        return self.snapshot_lifecycle.load_config(version_dir)

    def _train_dl_models(
        self,
        builder: Any,
        cutoff_date: str,
        dl_mode: str,
        dl_config: Dict[str, Any],
        send_update: Any,
    ) -> Iterator[str]:
        yield from self.training_orchestration.train_holdout_mode(
            builder=builder,
            cutoff_date=cutoff_date,
            dl_mode=dl_mode,
            dl_config=dl_config,
            send_update=send_update,
        )

    def _load_metrics_db(self, working_dir: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
        return self.inference_service.load_metrics_db(working_dir)

    def _build_weighted_signals(
        self,
        results: pd.DataFrame,
        metrics_db: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> tuple[pd.DataFrame, Set[str]]:
        return self.inference_service.build_weighted_signals(
            results=results,
            metrics_db=metrics_db,
        )

    def stream_holdout(self, request: HoldoutRunRequest) -> Iterator[str]:
        cutoff_year = int(request.cutoff_year)
        model_type = str(request.model_type or "ml").strip().lower()
        dl_mode = str(request.dl_mode or "balanced").strip().lower()
        use_existing = bool(request.use_existing)

        relative_base = "holdout_dl" if model_type == "dl" else "holdout"
        base_dir = os.path.join(MODELS_DIR, relative_base)
        version_dir = os.path.join(base_dir, str(cutoff_year))

        start_time = time.time()

        def send_update(progress: int, phase: str, details: str = "", force_eta: str | None = None) -> str:
            elapsed = time.time() - start_time
            eta = ""
            if progress > 0:
                total_est = elapsed / (progress / 100.0)
                eta = self._format_time(max(0.0, total_est - elapsed))
            if force_eta is not None:
                eta = force_eta
            return self._sse_data(
                {
                    "progress": progress,
                    "message": phase,
                    "details": details,
                    "eta": eta,
                }
            )

        try:
            yield send_update(5, "Initializing Environment...")

            cutoff_date = f"{cutoff_year}-12-31"
            start_test_date = f"{cutoff_year + 1}-01-01"

            should_train = True
            manifest_path = os.path.join(version_dir, "manifest.json")
            if use_existing and os.path.exists(manifest_path):
                should_train = False
                yield send_update(20, "Loading Saved Models...", "Checking manifest...")

            os.makedirs(version_dir, exist_ok=True)

            if model_type != "dl":
                yield send_update(100, "Error", "ML Mode is deprecated. Use Deep Learning.")
                yield "data: DONE\n\n"
                return

            from analysis.deep_learning_model import DLMacroModel

            builder = DLMacroModel(model_dir=version_dir)
            dl_config = self._load_dl_config(version_dir)

            if should_train:
                yield from self._train_dl_models(
                    builder=builder,
                    cutoff_date=cutoff_date,
                    dl_mode=dl_mode,
                    dl_config=dl_config,
                    send_update=send_update,
                )
                yield send_update(90, "Finalizing...", "Saving models...")

            working_dir = builder.model_dir

            yield send_update(95, "Running Inference...", "Predicting test set...")
            results = self.inference_service.predict_range(
                builder=builder,
                start_date=start_test_date,
                end_date=None,
            )

            pred_path = os.path.join(working_dir, "holdout_predictions.csv")
            self.inference_service.save_predictions(predictions=results, output_path=pred_path)

            if results.empty:
                yield self._sse_data({"error": "No predictions generated"})
                return

            yield send_update(92, "Loading Market Data...", "For Simulation...")
            from data.etl import load_and_merge_data

            df = load_and_merge_data(start_date=f"{cutoff_year}-01-01", end_date=None, save_to_disk=False)
            if df.empty:
                df = pd.read_csv(ENGINEERED_FEATURES_FILE, index_col=0, parse_dates=True)

            max_date = df.index.max().strftime("%Y-%m-%d")
            prices = df.loc[start_test_date:max_date]

            yield send_update(93, "Applying Weighted Ensemble...", "Calculating weights...")
            metrics_db = self._load_metrics_db(working_dir)
            signals, unique_assets = self._build_weighted_signals(results, metrics_db)

            yield send_update(95, "Running Strategy Simulation...", "Executing trades...")
            from backtesting.engine import VectorizedBacktester
            from backtesting.strategies import RotationalStrategy

            full_vols = df[list(unique_assets)].pct_change().rolling(60).std() * np.sqrt(252)
            asset_vols = full_vols.loc[start_test_date:max_date]

            risk_df = pd.DataFrame(index=prices.index)
            risk_df["VIX"] = prices["VIX"] if "VIX" in prices.columns else 20.0

            strategy = RotationalStrategy(
                top_n=3,
                vol_target=0.15,
                use_regime_filter=True,
                rebalance_freq="monthly",
            )
            weights = strategy.generate_weights(signals, risk_df, asset_vols)

            engine = VectorizedBacktester(initial_capital=10000)
            sim_results = engine.run_portfolio_simulation(prices, weights, trade_threshold=0.01)

            benchmark_equity = prices["SP500"] / prices["SP500"].iloc[0] * 10000

            if dl_config:
                self.snapshot_lifecycle.save_config(version_dir=version_dir, config=dl_config)

            payload: Dict[str, Any] = {
                "metrics": sim_results["metrics"],
                "equity": json.loads(sim_results["equity_curve"].to_json()),
                "drawdown": json.loads(sim_results["drawdown"].to_json()),
                "benchmark": json.loads(benchmark_equity.to_json()),
                "trades": sim_results["trades"],
                "saved_path": version_dir,
            }

            yield self._sse_data({"progress": 100, "message": "Complete!", "result": payload})
            yield "data: DONE\n\n"
        except Exception as exc:
            traceback.print_exc()
            yield self._sse_data({"error": str(exc)})
