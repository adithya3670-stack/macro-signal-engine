import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.services.portfolio_simulation_service import PortfolioSimulationService


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "regression"


def _load_json(filename: str):
    with (FIXTURES_DIR / filename).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_deterministic_market_data():
    dates = pd.bdate_range("2024-01-01", periods=180)
    dates.name = "Date"
    x = np.arange(len(dates), dtype=float)

    prices = pd.DataFrame(
        {
            "SP500": 100 + 0.30 * x + 2.0 * np.sin(x / 7.0),
            "Gold": 80 + 0.18 * x + 1.0 * np.cos(x / 9.0),
        },
        index=dates,
    )
    risk = pd.DataFrame(
        {"VIX": 20.0, "DGS10": 3.5, "Liquidity_Impulse": 0.0},
        index=dates,
    )
    signals = pd.DataFrame(
        {
            "SP500_1m": 0.55 + 0.08 * np.sin(x / 5.0),
            "Gold_1m": 0.52 + 0.07 * np.cos(x / 6.0),
        },
        index=dates,
    )
    return prices, risk, signals


class _StubLoader:
    def __init__(self, prices: pd.DataFrame, risk: pd.DataFrame):
        self._prices = prices
        self._risk = risk

    def get_asset_prices(self, assets=None):
        if assets is None:
            return self._prices.copy()
        return self._prices[list(assets)].copy()

    def get_risk_factors(self):
        return self._risk.copy()


def _build_service_for_regression(prices: pd.DataFrame, risk: pd.DataFrame, signals: pd.DataFrame):
    service = PortfolioSimulationService(data_loader_factory=lambda: _StubLoader(prices, risk))
    service.signal_resolution.generate_signals = (
        lambda signal_universe, start_date, end_date, selection: (signals.copy(), [])
    )
    return service


def _assert_metric_block(actual: dict, expected: dict, atol: float = 1e-9):
    assert set(expected.keys()).issubset(set(actual.keys()))
    for key, expected_value in expected.items():
        assert actual[key] == pytest.approx(expected_value, abs=atol), f"Metric mismatch for '{key}'"


def test_run_backtest_v2_regression_parity_snapshot():
    payload = _load_json("backtest_payload.json")
    expected = _load_json("portfolio_backtest_expected.json")["backtest"]
    prices, risk, signals = _build_deterministic_market_data()
    service = _build_service_for_regression(prices, risk, signals)

    result = service.run_backtest_v2(payload)

    _assert_metric_block(result["metrics"], expected["metrics"])
    assert len(result["equity_curve"]) == expected["equity_len"]
    assert len(result["drawdown"]) == expected["drawdown_len"]
    assert len(result["weights"]) == expected["weights_len"]
    assert len(result["trades"]) == expected["trades_len"]
    assert result["equity_curve"][-1]["Equity"] == pytest.approx(expected["equity_last_equity"], abs=1e-6)
    assert result["equity_curve"][-1]["PctReturn"] == pytest.approx(expected["equity_last_pct_return"], abs=1e-6)
    assert result["benchmarks"]["SP500"][-1] == pytest.approx(expected["bench_sp500_last"], abs=1e-9)
    assert result["benchmarks"]["Gold"][-1] == pytest.approx(expected["bench_gold_last"], abs=1e-9)


def test_run_portfolio_regression_parity_snapshot():
    payload = _load_json("portfolio_payload.json")
    expected = _load_json("portfolio_backtest_expected.json")["portfolio"]
    prices, risk, signals = _build_deterministic_market_data()
    service = _build_service_for_regression(prices, risk, signals)

    result = service.run_portfolio(payload)

    assert result["benchmark_name"] == expected["benchmark_name"]
    _assert_metric_block(result["metrics"], expected["metrics"])
    _assert_metric_block(result["benchmark_metrics"], expected["benchmark_metrics"])
    assert len(result["equity_curve"]) == expected["equity_len"]
    assert len(result["benchmark_curve"]) == expected["benchmark_len"]
    assert len(result["trades"]) == expected["trades_len"]
    assert result["final_balance"] == pytest.approx(expected["final_balance"], abs=1e-2)
    assert result["equity_curve"][-1]["Equity"] == pytest.approx(expected["equity_last_equity"], abs=1e-6)
    assert result["benchmark_curve"][-1]["Equity"] == pytest.approx(expected["benchmark_last_equity"], abs=1e-6)
