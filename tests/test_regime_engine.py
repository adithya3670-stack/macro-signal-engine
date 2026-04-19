import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regime_engine import RegimeStateEngine


class TestRegimeEngine(unittest.TestCase):
    def test_build_state_history_outputs_expected_columns(self):
        dates = pd.bdate_range("2024-01-01", "2025-04-30")
        t = np.arange(len(dates), dtype=float)
        master = pd.DataFrame(
            {
                "CPI_YoY": 2.5 + 0.2 * np.sin(t / 20.0),
                "Liquidity_Impulse": np.sin(t / 15.0),
                "VIX": 18 + 4 * np.sin(t / 10.0),
                "Real_Yield": 0.5 * np.sin(t / 25.0),
            },
            index=dates,
        )

        features = pd.DataFrame(
            {
                "SP500": 100 + 0.1 * t,
                "Nasdaq": 200 + 0.2 * t,
                "DJIA": 300 + 0.15 * t,
                "Russell2000": 50 + 0.05 * t,
                "Gold": 150 + 0.08 * t,
                "Silver": 20 + 0.02 * t,
                "Copper": 4 + 0.003 * t,
                "Oil": 70 + 0.04 * t,
                "VIX": master["VIX"].values,
                "Liquidity_Impulse": master["Liquidity_Impulse"].values,
                "Real_Yield": master["Real_Yield"].values,
                "CPI_YoY": master["CPI_YoY"].values,
            },
            index=dates,
        )
        feature_tables = {"3d": features.copy(), "1w": features.copy(), "1m": features.copy()}

        engine = RegimeStateEngine()
        states = engine.build_state_history(master_df=master, feature_tables=feature_tables, holdout_start="2025-01-01")

        self.assertIn("rule_code", states.columns)
        self.assertIn("3d_equities_composite_id", states.columns)
        self.assertIn("1w_precious_latent_id", states.columns)
        self.assertIn("1m_commodities_latent_conf", states.columns)
        self.assertFalse(states["rule_code"].isna().all())

    def test_stability_requires_dwell_and_confidence(self):
        labels = np.array([0, 1, 1, 1, 1], dtype=int)
        conf = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=float)
        stable = RegimeStateEngine._apply_stability(labels, conf, dwell_days=3, conf_floor=0.55)
        self.assertListEqual(stable.tolist(), [0, 0, 0, 1, 1])

        low_conf = np.array([0.9, 0.4, 0.4, 0.9, 0.9], dtype=float)
        stable_low = RegimeStateEngine._apply_stability(labels, low_conf, dwell_days=2, conf_floor=0.55)
        self.assertListEqual(stable_low.tolist(), [0, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
