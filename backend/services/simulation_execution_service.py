from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from backtesting.engine import VectorizedBacktester
from backtesting.strategies import RotationalStrategy


class SimulationExecutionService:
    """Runs strategy allocation and simulation engines."""

    def run_backtest(
        self,
        prices: pd.DataFrame,
        risk_data: pd.DataFrame,
        aligned_signals: pd.DataFrame,
        initial_capital: float,
        top_n: int,
        vol_target: float,
        use_regime: bool,
        trade_threshold: float,
    ) -> Dict[str, Any]:
        asset_vols = prices.pct_change().rolling(60).std() * np.sqrt(252)
        strategy = RotationalStrategy(top_n=top_n, vol_target=vol_target, use_regime_filter=use_regime)
        weights = strategy.generate_weights(aligned_signals, risk_data, asset_vols)

        engine = VectorizedBacktester(initial_capital=initial_capital)
        return engine.run_backtest(prices, weights, trade_threshold=trade_threshold)

    def run_portfolio(
        self,
        strat_prices: pd.DataFrame,
        prices: pd.DataFrame,
        risk_data: pd.DataFrame,
        aligned_signals: pd.DataFrame,
        full_asset_vols: pd.DataFrame,
        benchmark_ticker: str,
        initial_capital: float,
        monthly_contribution: float,
        custom_cashflows: List[Dict[str, Any]],
        trade_threshold: float,
        top_n: int,
        vol_target: float,
        use_regime: bool,
        min_confidence: float,
        rebalance_freq: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        asset_vols = full_asset_vols.reindex(strat_prices.index).ffill()

        strategy = RotationalStrategy(
            top_n=top_n,
            vol_target=vol_target,
            use_regime_filter=use_regime,
            min_confidence=min_confidence,
            rebalance_freq=rebalance_freq,
        )
        weights = strategy.generate_weights(aligned_signals, risk_data, asset_vols)

        engine = VectorizedBacktester(initial_capital=initial_capital)
        results = engine.run_portfolio_simulation(
            strat_prices,
            weights,
            monthly_contribution=monthly_contribution,
            custom_cashflows=custom_cashflows,
            trade_threshold=trade_threshold,
        )

        bench_weights = pd.DataFrame(0.0, index=weights.index, columns=[benchmark_ticker])
        bench_weights[benchmark_ticker] = 1.0
        bench_prices = prices[[benchmark_ticker]]
        bench_results = engine.run_portfolio_simulation(
            bench_prices,
            bench_weights,
            monthly_contribution=monthly_contribution,
            custom_cashflows=custom_cashflows,
            trade_threshold=0.0,
        )

        return results, bench_results, weights
