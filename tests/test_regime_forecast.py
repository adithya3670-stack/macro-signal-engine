import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regime_engine import RegimeStateEngine
from analysis.regime_forecast import RegimeForecastEngine


class TestRegimeForecast(unittest.TestCase):
    def test_markov_transition_smoothing(self):
        engine = RegimeForecastEngine()
        model = engine._fit_markov(state_space=["A", "B"], from_states=["A"], to_states=["B"])
        dist_a, support_a = engine._markov_distribution(model, "A")
        dist_unknown, support_unknown = engine._markov_distribution(model, "C")

        self.assertEqual(support_a, 1)
        self.assertEqual(support_unknown, 0)
        self.assertAlmostEqual(sum(dist_a.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(dist_unknown.values()), 1.0, places=6)
        self.assertTrue(all(v > 0 for v in dist_a.values()))

    def test_logit_fallback_when_support_is_insufficient(self):
        engine = RegimeForecastEngine()
        x = pd.DataFrame({"f1": np.arange(10), "f2": np.arange(10) * 0.1})
        y = pd.Series(["S0"] * 5 + ["S1"] * 5)
        result = engine._fit_logit(X=x, y=y, state_space=["S0", "S1"])
        self.assertFalse(result["available"])
        self.assertIn("insufficient_rows", result["reason"])

    def test_build_forecasts_latest_top3_schema(self):
        engine = RegimeForecastEngine()
        dates = pd.bdate_range("2023-01-02", "2026-03-31")
        t = np.arange(len(dates), dtype=float)

        state_df = pd.DataFrame(index=dates)
        state_df["rule_inflation_high"] = ((t // 30) % 2).astype(int)
        state_df["rule_liquidity_expanding"] = ((t // 20) % 2).astype(int)
        state_df["rule_risk_off"] = ((t // 15) % 2).astype(int)
        state_df["rule_rates_positive"] = ((t // 25) % 2).astype(int)
        state_df["rule_code"] = (
            "I"
            + state_df["rule_inflation_high"].astype(str)
            + "L"
            + state_df["rule_liquidity_expanding"].astype(str)
            + "R"
            + state_df["rule_risk_off"].astype(str)
            + "T"
            + state_df["rule_rates_positive"].astype(str)
        )

        for horizon in ["3d", "1w", "1m"]:
            state_df[f"{horizon}_rule_id"] = state_df["rule_code"].map(lambda x: f"H={horizon}|R={x}")
            for group, assets in RegimeStateEngine.ASSET_GROUPS.items():
                latent = ((t // (10 + len(assets))) % 3).astype(int)
                state_df[f"{horizon}_{group}_latent_state"] = latent
                state_df[f"{horizon}_{group}_latent_conf"] = np.clip(0.6 + 0.3 * np.sin(t / 33.0), 0.0, 1.0)
                state_df[f"{horizon}_{group}_latent_id"] = [f"H={horizon}|G={group}|L={int(x)}" for x in latent]
                state_df[f"{horizon}_{group}_composite_id"] = [
                    f"H={horizon}|G={group}|R={rule}|L={int(lat)}"
                    for rule, lat in zip(state_df["rule_code"], latent)
                ]

        feature_tables = {}
        for horizon in ["3d", "1w", "1m"]:
            feat = pd.DataFrame(index=dates)
            feat["VIX"] = 18 + 3 * np.sin(t / 11.0)
            feat["Liquidity_Impulse"] = np.sin(t / 13.0)
            feat["Real_Yield"] = 0.5 * np.sin(t / 17.0)
            feat["CPI_YoY"] = 2.5 + 0.2 * np.sin(t / 21.0)
            feat["Curve_Steepening"] = np.sin(t / 15.0)
            for group_assets in RegimeStateEngine.ASSET_GROUPS.values():
                for asset in group_assets:
                    feat[f"{asset}_ret_1"] = 0.01 * np.sin(t / 9.0)
                    feat[f"{asset}_ret_3"] = 0.015 * np.sin(t / 13.0)
                    feat[f"{asset}_ret_5"] = 0.02 * np.sin(t / 17.0)
                    feat[f"{asset}_vol_5"] = 0.3 + 0.1 * np.sin(t / 8.0)
                    feat[f"{asset}_vol_10"] = 0.4 + 0.1 * np.sin(t / 12.0)
                    feat[f"{asset}_trend_20"] = np.sin(t / 19.0)
                    feat[f"{asset}_trend_60"] = np.sin(t / 27.0)
            feature_tables[horizon] = feat

        out = engine.build_forecasts(
            state_df=state_df,
            feature_tables=feature_tables,
            holdout_start="2025-01-01",
            holdout_end="2025-12-31",
            shadow_start="2026-01-01",
        )

        latest = out["latest"]["3d"]["equities"]
        self.assertIn("selected_key", latest)
        self.assertIn("confidence", latest)
        self.assertIn("top3", latest)
        self.assertLessEqual(len(latest["top3"]), 3)

        diag = out["horizons"]["3d"]["groups"]["equities"]["diagnostics"]["holdout"]
        self.assertIn("top1_accuracy", diag)
        self.assertIn("top3_recall", diag)
        self.assertIn("brier_score", diag)


if __name__ == "__main__":
    unittest.main()
