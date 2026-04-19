import pandas as pd

from backend.domain.dto import ModelSourceSelection
from backend.services.signal_resolution_service import SignalResolutionService


class _Orchestration:
    def build_signal_universe(self, universe, horizon, use_hybrid):
        if use_hybrid:
            return [f"{asset}_1w" for asset in universe]
        return [f"{asset}_{horizon}" for asset in universe]

    def generate_signals(self, signal_universe, start_date, end_date, selection):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        cols = {signal_universe[0]: [0.6, 0.7]}
        return pd.DataFrame(cols, index=idx), [{"model": "x", "start": "2024-01-01", "end": "2024-01-02"}]

    def get_hybrid_weights(self, profile_id):
        return {"1w": 1.0, "1m": 0.0, "3m": 0.0}

    def build_hybrid_signals(self, raw_signals, universe, weights):
        out = pd.DataFrame(index=raw_signals.index)
        for asset in universe:
            col = f"{asset}_1w"
            if col in raw_signals.columns:
                out[asset] = raw_signals[col]
        return out

    def rename_single_horizon_columns(self, raw_signals, universe):
        rename_map = {}
        for asset in universe:
            candidate = f"{asset}_1m"
            if candidate in raw_signals.columns:
                rename_map[candidate] = asset
        return raw_signals.rename(columns=rename_map)


class _Vectorizer:
    def align_signals(self, raw_signals, prices):
        return raw_signals.reindex(prices.index).ffill().fillna(0.5)


def test_signal_resolution_service_builds_normalizes_and_aligns():
    svc = SignalResolutionService(orchestration=_Orchestration(), vectorizer=_Vectorizer())
    selection = ModelSourceSelection(source="live", category="dl", model_type="ensemble")

    universe = svc.build_signal_universe(["SP500"], horizon="1m", use_hybrid=False)
    assert universe == ["SP500_1m"]

    raw, timeline = svc.generate_signals(universe, "2024-01-01", "2024-01-02", selection)
    assert len(timeline) == 1

    normalized = svc.normalize_signals(raw, ["SP500"], use_hybrid=False, hybrid_profile=None)
    assert "SP500" in normalized.columns

    prices_index = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame({"SP500": [100, 101, 102, 103]}, index=prices_index)
    aligned = svc.align_for_backtest(normalized, prices)
    assert len(aligned) == 4

    strat_prices = prices.iloc[:3]
    strat_prices_out, aligned_portfolio = svc.align_for_portfolio(normalized, strat_prices)
    assert list(strat_prices_out.index) == list(aligned_portfolio.index)
