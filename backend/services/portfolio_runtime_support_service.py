from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class PortfolioRuntimeSupportService:
    """Shared runtime helpers for portfolio/backtest data windows and metrics."""

    def filter_dates(
        self,
        prices: pd.DataFrame,
        risk_data: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if start_date:
            prices = prices[prices.index >= pd.to_datetime(start_date)]
            risk_data = risk_data[risk_data.index >= pd.to_datetime(start_date)]
        if end_date:
            prices = prices[prices.index <= pd.to_datetime(end_date)]
            risk_data = risk_data[risk_data.index <= pd.to_datetime(end_date)]
        return prices, risk_data

    def apply_recent_date_auto_correction(
        self,
        prices: pd.DataFrame,
        risk_data: pd.DataFrame,
        loader: Any,
        fetch_universe: Sequence[str],
        start_date: Optional[str],
        end_date: Optional[str],
        initial_capital: float,
        custom_cashflows: List[Dict[str, Any]],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, str, float, List[Dict[str, Any]], bool, str]:
        min_rows = 100
        original_start_date = str(start_date or "")
        auto_corrected = False

        if len(prices) >= min_rows or not original_start_date:
            return prices, risk_data, original_start_date, initial_capital, custom_cashflows, auto_corrected, original_start_date

        full_prices = loader.get_asset_prices(fetch_universe)
        latest_date = full_prices.index[-1]
        original_start_pd = pd.to_datetime(original_start_date)
        days_from_latest = (latest_date - original_start_pd).days

        if days_from_latest <= 180 and days_from_latest >= -30:
            corrected_start_date = (original_start_pd - pd.DateOffset(months=6)).strftime("%Y-%m-%d")
            contribution_entry = {"Date": original_start_date, "Amount": initial_capital}

            existing_dates = [cf.get("Date") for cf in custom_cashflows]
            if original_start_date not in existing_dates:
                custom_cashflows = [contribution_entry] + custom_cashflows

            initial_capital = 1.0
            auto_corrected = True

            prices = loader.get_asset_prices(fetch_universe)
            risk_data = loader.get_risk_factors()
            prices, risk_data = self.filter_dates(prices, risk_data, corrected_start_date, end_date)
            return (
                prices,
                risk_data,
                original_start_date,
                initial_capital,
                custom_cashflows,
                auto_corrected,
                corrected_start_date,
            )

        return prices, risk_data, original_start_date, initial_capital, custom_cashflows, auto_corrected, original_start_date

    def recalculate_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        if len(equity_curve) <= 1:
            return {
                "total_return": 0.0,
                "cagr": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0,
            }

        initial_balance = equity_curve.iloc[0]
        final_balance = equity_curve.iloc[-1]
        total_return = (final_balance - initial_balance) / initial_balance if initial_balance > 0 else 0.0

        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        cagr = ((final_balance / initial_balance) ** (1 / years) - 1) if years > 0 and initial_balance > 0 else 0.0

        returns = equity_curve.pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() > 0 else 0.0
        drawdown = (equity_curve - equity_curve.cummax()) / equity_curve.cummax()
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0

        return {
            "total_return": float(total_return),
            "cagr": float(cagr),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
        }
