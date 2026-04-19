from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from backtesting.data_loader import DataLoader
from backtesting.signal_vectorizer import SignalVectorizer
from backend.infrastructure.model_artifacts import ModelArtifactResolver
from backend.services.backtest_orchestration import BacktestOrchestrationService
from backend.services.model_source_selection_service import ModelSourceSelectionService
from backend.services.portfolio_runtime_support_service import PortfolioRuntimeSupportService
from backend.services.signal_resolution_service import SignalResolutionService
from backend.services.simulation_execution_service import SimulationExecutionService
from backend.shared.http import ServiceError


class PortfolioSimulationService:
    def __init__(
        self,
        data_loader_factory: Optional[Callable[[], Any]] = None,
        artifact_resolver: Optional[ModelArtifactResolver] = None,
        orchestration: Optional[BacktestOrchestrationService] = None,
        vectorizer: Optional[SignalVectorizer] = None,
        model_selection_service: Optional[ModelSourceSelectionService] = None,
        signal_resolution_service: Optional[SignalResolutionService] = None,
        execution_service: Optional[SimulationExecutionService] = None,
        runtime_support_service: Optional[PortfolioRuntimeSupportService] = None,
    ) -> None:
        self.artifacts = artifact_resolver or ModelArtifactResolver()
        self.orchestration = orchestration or BacktestOrchestrationService(artifact_resolver=self.artifacts)
        self.vectorizer = vectorizer or SignalVectorizer()

        self.model_selection = model_selection_service or ModelSourceSelectionService(
            artifact_resolver=self.artifacts,
            orchestration=self.orchestration,
        )
        self.signal_resolution = signal_resolution_service or SignalResolutionService(
            orchestration=self.orchestration,
            vectorizer=self.vectorizer,
        )
        self.execution = execution_service or SimulationExecutionService()
        self.runtime_support = runtime_support_service or PortfolioRuntimeSupportService()
        self.data_loader_factory = data_loader_factory or DataLoader

    def list_holdout_folders(self) -> Dict[str, List[str]]:
        return self.model_selection.list_holdout_folders()

    def _format_backtest_response(
        self,
        results: Dict[str, Any],
        prices: pd.DataFrame,
        initial_capital: float,
    ) -> Dict[str, Any]:
        equity_curve = results["equity_curve"].reset_index()
        equity_curve.columns = ["Date", "Equity"]
        equity_curve["PctReturn"] = ((equity_curve["Equity"] / initial_capital) - 1) * 100
        equity_curve["Date"] = equity_curve["Date"].dt.strftime("%Y-%m-%d")

        drawdown = results["drawdown"].reset_index()
        drawdown.columns = ["Date", "Drawdown"]
        drawdown["Date"] = drawdown["Date"].dt.strftime("%Y-%m-%d")

        weight_data = results["weights"].reset_index()
        weight_data["Date"] = weight_data["Date"].dt.strftime("%Y-%m-%d")

        benchmarks: Dict[str, List[float]] = {}
        for benchmark_asset in ["SP500", "Gold"]:
            if benchmark_asset not in prices.columns:
                benchmarks[benchmark_asset] = []
                continue
            benchmark_prices = prices[benchmark_asset]
            if benchmark_prices.empty or benchmark_prices.iloc[0] <= 0:
                benchmarks[benchmark_asset] = []
                continue
            benchmark_curve = ((benchmark_prices / benchmark_prices.iloc[0]) - 1) * 100
            benchmarks[benchmark_asset] = benchmark_curve.tolist()

        return {
            "metrics": results["metrics"],
            "equity_curve": equity_curve.to_dict(orient="records"),
            "drawdown": drawdown.to_dict(orient="records"),
            "weights": weight_data.to_dict(orient="records"),
            "benchmarks": benchmarks,
            "trades": results.get("trades", []),
        }

    def run_backtest_v2(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_date = payload.get("start_date", "2015-01-01")
        end_date = payload.get("end_date")
        initial_capital = float(payload.get("initial_capital", 10000.0))

        strat_config = payload.get("strategy_config", {}) or {}
        top_n = int(strat_config.get("top_n", 2))
        vol_target = float(strat_config.get("vol_target", 0.15))
        use_regime = bool(strat_config.get("use_regime", True))
        horizon = str(strat_config.get("forecast_horizon", "1m"))
        use_hybrid = bool(strat_config.get("use_hybrid", False))
        hybrid_profile = strat_config.get("hybrid_profile")

        universe = payload.get("universe", ["SP500", "Nasdaq", "Gold", "Silver", "Oil"])
        if not isinstance(universe, list) or not universe:
            raise ServiceError("Universe must be a non-empty list.", status_code=400)

        loader = self.data_loader_factory()
        prices = loader.get_asset_prices(universe)
        risk_data = loader.get_risk_factors()
        prices, risk_data = self.runtime_support.filter_dates(prices, risk_data, start_date, end_date)

        signal_universe = self.signal_resolution.build_signal_universe(universe, horizon=horizon, use_hybrid=use_hybrid)
        live_selection = self.model_selection.resolve_live_selection(model_type="ensemble")
        raw_signals, _timeline = self.signal_resolution.generate_signals(signal_universe, start_date, end_date, live_selection)

        if raw_signals.empty:
            raise ServiceError("No signals generated. Models might be missing.", status_code=400)

        normalized_signals = self.signal_resolution.normalize_signals(
            raw_signals=raw_signals,
            universe=universe,
            use_hybrid=use_hybrid,
            hybrid_profile=hybrid_profile,
        )
        aligned_signals = self.signal_resolution.align_for_backtest(normalized_signals, prices)

        trade_threshold = float(strat_config.get("trade_threshold", 0.5)) / 100.0
        results = self.execution.run_backtest(
            prices=prices,
            risk_data=risk_data,
            aligned_signals=aligned_signals,
            initial_capital=initial_capital,
            top_n=top_n,
            vol_target=vol_target,
            use_regime=use_regime,
            trade_threshold=trade_threshold,
        )

        return self._format_backtest_response(results, prices, initial_capital)

    def run_portfolio(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_date = payload.get("start_date", "2015-01-01")
        end_date = payload.get("end_date")
        initial_capital = float(payload.get("initial_capital", 10000.0))
        monthly_contribution = float(payload.get("monthly_contribution", 0.0))
        custom_cashflows = list(payload.get("custom_cashflows", []) or [])
        benchmark_ticker = payload.get("benchmark_ticker", "SP500")

        model_category = payload.get("model_category")
        model_year = payload.get("model_year")
        if model_year:
            model_year = str(model_year)
            if model_year.startswith("ml_") or model_year.startswith("dl_"):
                model_year = model_year.split("_", 1)[1]

        strat_config = payload.get("strategy_config", {}) or {}
        top_n = int(strat_config.get("top_n", 2))
        vol_target = float(strat_config.get("vol_target", 0.15))
        use_regime = bool(strat_config.get("use_regime", True))
        trade_threshold = float(strat_config.get("trade_threshold", 0.005))
        horizon = str(strat_config.get("forecast_horizon", "1m"))
        use_hybrid = bool(strat_config.get("use_hybrid", False))
        hybrid_profile = strat_config.get("hybrid_profile")

        universe = payload.get("universe", ["SP500", "Nasdaq", "Gold", "Silver", "Oil"])
        if not isinstance(universe, list) or not universe:
            raise ServiceError("Universe must be a non-empty list.", status_code=400)

        selection = self.model_selection.resolve_portfolio_selection(model_category, model_year, strat_config)
        if selection.category == "ml":
            raise ServiceError("ML models are deprecated. Use Deep Learning.", status_code=400)

        loader = self.data_loader_factory()
        fetch_universe = list(set(universe + [benchmark_ticker]))
        prices = loader.get_asset_prices(fetch_universe)
        risk_data = loader.get_risk_factors()
        full_universe_prices = loader.get_asset_prices(universe)
        full_asset_vols = full_universe_prices.pct_change().rolling(60).std() * np.sqrt(252)

        prices, risk_data = self.runtime_support.filter_dates(prices, risk_data, start_date, end_date)
        (
            prices,
            risk_data,
            original_start_date,
            initial_capital,
            custom_cashflows,
            auto_corrected,
            effective_start_date,
        ) = self.runtime_support.apply_recent_date_auto_correction(
            prices=prices,
            risk_data=risk_data,
            loader=loader,
            fetch_universe=fetch_universe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            custom_cashflows=custom_cashflows,
        )

        if len(prices) < 100:
            raise ServiceError(
                f"Insufficient data: Only {len(prices)} rows available after filtering by start_date={effective_start_date}. "
                "Please select an earlier start date (recommended: at least 6 months before today).",
                status_code=400,
            )

        signal_universe = self.signal_resolution.build_signal_universe(universe, horizon=horizon, use_hybrid=use_hybrid)
        raw_signals, timeline_segments = self.signal_resolution.generate_signals(
            signal_universe=signal_universe,
            start_date=effective_start_date,
            end_date=end_date,
            selection=selection,
        )

        if raw_signals.empty:
            raise ServiceError("No signals generated. Model data might be missing.", status_code=400)

        normalized_signals = self.signal_resolution.normalize_signals(
            raw_signals=raw_signals,
            universe=universe,
            use_hybrid=use_hybrid,
            hybrid_profile=hybrid_profile,
        )

        strat_prices = prices[universe]
        strat_prices, aligned_signals = self.signal_resolution.align_for_portfolio(normalized_signals, strat_prices)

        min_confidence = float(strat_config.get("min_confidence", 0.5))
        rebalance_freq = str(strat_config.get("rebalance_freq", "daily"))

        results, bench_results, _weights = self.execution.run_portfolio(
            strat_prices=strat_prices,
            prices=prices,
            risk_data=risk_data,
            aligned_signals=aligned_signals,
            full_asset_vols=full_asset_vols,
            benchmark_ticker=benchmark_ticker,
            initial_capital=initial_capital,
            monthly_contribution=monthly_contribution,
            custom_cashflows=custom_cashflows,
            trade_threshold=trade_threshold,
            top_n=top_n,
            vol_target=vol_target,
            use_regime=use_regime,
            min_confidence=min_confidence,
            rebalance_freq=rebalance_freq,
        )

        equity_curve = results["equity_curve"].reset_index()
        equity_curve.columns = ["Date", "Equity"]
        equity_curve["Date"] = equity_curve["Date"].dt.strftime("%Y-%m-%d")

        benchmark_curve = bench_results["equity_curve"].reset_index()
        benchmark_curve.columns = ["Date", "Equity"]
        benchmark_curve["Date"] = benchmark_curve["Date"].dt.strftime("%Y-%m-%d")

        trades = list(results.get("trades", []))
        if auto_corrected and original_start_date:
            equity_curve = equity_curve[equity_curve["Date"] >= original_start_date].reset_index(drop=True)
            benchmark_curve = benchmark_curve[benchmark_curve["Date"] >= original_start_date].reset_index(drop=True)
            trades = [t for t in trades if t.get("Date", "") >= original_start_date]

        if "latest_status" in results and results["latest_status"] and not aligned_signals.empty:
            results["latest_status"]["latest_signals"] = aligned_signals.iloc[-1].to_dict()
            results["latest_status"]["model_source"] = selection.label

        if auto_corrected and not equity_curve.empty and not benchmark_curve.empty:
            filtered_equity = pd.Series(equity_curve["Equity"].values, index=pd.to_datetime(equity_curve["Date"]))
            filtered_bench = pd.Series(benchmark_curve["Equity"].values, index=pd.to_datetime(benchmark_curve["Date"]))
            results["metrics"] = self.runtime_support.recalculate_metrics(filtered_equity)
            bench_results["metrics"] = self.runtime_support.recalculate_metrics(filtered_bench)

        final_balance = float(equity_curve["Equity"].iloc[-1]) if not equity_curve.empty else float(initial_capital)

        return {
            "benchmark_name": benchmark_ticker,
            "metrics": results.get("metrics", {}),
            "benchmark_metrics": bench_results.get("metrics", {}),
            "equity_curve": equity_curve.to_dict(orient="records"),
            "benchmark_curve": benchmark_curve.to_dict(orient="records"),
            "trades": trades,
            "final_balance": round(final_balance, 2),
            "latest_status": results.get("latest_status"),
            "model_timeline": timeline_segments,
        }
