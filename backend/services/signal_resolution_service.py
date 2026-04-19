from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from backtesting.signal_vectorizer import SignalVectorizer
from backend.domain.dto import ModelSourceSelection
from backend.services.backtest_orchestration import BacktestOrchestrationService


class SignalResolutionService:
    """Builds/normalizes/aligs signals for backtest and portfolio runs."""

    def __init__(
        self,
        orchestration: BacktestOrchestrationService,
        vectorizer: SignalVectorizer,
    ) -> None:
        self.orchestration = orchestration
        self.vectorizer = vectorizer

    def build_signal_universe(
        self,
        universe: Sequence[str],
        horizon: str,
        use_hybrid: bool,
    ) -> List[str]:
        return self.orchestration.build_signal_universe(universe, horizon=horizon, use_hybrid=use_hybrid)

    def generate_signals(
        self,
        signal_universe: Sequence[str],
        start_date: Optional[str],
        end_date: Optional[str],
        selection: ModelSourceSelection,
    ) -> Tuple[pd.DataFrame, List[Dict[str, str]]]:
        return self.orchestration.generate_signals(signal_universe, start_date, end_date, selection)

    def normalize_signals(
        self,
        raw_signals: pd.DataFrame,
        universe: Sequence[str],
        use_hybrid: bool,
        hybrid_profile: Optional[str],
    ) -> pd.DataFrame:
        if use_hybrid:
            weights = self.orchestration.get_hybrid_weights(hybrid_profile)
            return self.orchestration.build_hybrid_signals(raw_signals, universe, weights)
        return self.orchestration.rename_single_horizon_columns(raw_signals, universe)

    def align_for_backtest(self, raw_signals: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        return self.vectorizer.align_signals(raw_signals, prices)

    def align_for_portfolio(
        self,
        raw_signals: pd.DataFrame,
        strat_prices: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        aligned_signals = raw_signals.reindex(strat_prices.index).ffill()
        valid_dates = strat_prices.dropna().index.intersection(aligned_signals.dropna().index)
        return strat_prices.loc[valid_dates], aligned_signals.loc[valid_dates]
