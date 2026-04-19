import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.regime_predictability import RegimePredictabilityManager


class TestRegimePolicy(unittest.TestCase):
    def _build_inputs(self, rows, regime_conf, improvement_abs, improvement_rel):
        idx = pd.bdate_range("2026-01-15", periods=1)
        state_df = pd.DataFrame(
            {
                "3d_rule_id": ["H=3d|R=I1L1R0T1"],
                "3d_equities_composite_id": ["H=3d|G=equities|R=I1L1R0T1|L=1"],
                "3d_equities_latent_id": ["H=3d|G=equities|L=1"],
                "3d_equities_latent_conf": [regime_conf],
            },
            index=idx,
        )
        horizon_outputs = {
            "3d": {
                "predictability_rows": [
                    {
                        "horizon": "3d",
                        "asset": "SP500",
                        "group": "equities",
                        "key_type": "composite",
                        "key": "H=3d|G=equities|R=I1L1R0T1|L=1",
                        "rows": rows,
                        "improvement_abs": improvement_abs,
                        "improvement_rel": improvement_rel,
                        "mape_champion": 1.0,
                        "mape_naive": 1.2,
                        "predictability_confidence": 0.8,
                    }
                ],
                "bundle": {
                    "assets": {
                        "SP500": {
                            "selected_model": "patchtst",
                            "baseline_2025_mape": 1.2,
                            "selected_holdout_mape": 1.0,
                        }
                    }
                },
                "latest_predictions": {},
            }
        }
        return state_df, horizon_outputs

    def test_policy_reason_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RegimePredictabilityManager(regime_dir=tmpdir)

            state_df, horizon_outputs = self._build_inputs(rows=10, regime_conf=0.9, improvement_abs=0.5, improvement_rel=0.3)
            policy = mgr._build_policy(state_df, horizon_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "LOW_SUPPORT")

            state_df, horizon_outputs = self._build_inputs(rows=80, regime_conf=0.2, improvement_abs=0.5, improvement_rel=0.3)
            policy = mgr._build_policy(state_df, horizon_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "LOW_CONFIDENCE")

            state_df, horizon_outputs = self._build_inputs(rows=80, regime_conf=0.9, improvement_abs=0.01, improvement_rel=0.005)
            policy = mgr._build_policy(state_df, horizon_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "NO_EDGE")

            state_df, horizon_outputs = self._build_inputs(rows=80, regime_conf=0.9, improvement_abs=0.05, improvement_rel=0.05)
            policy = mgr._build_policy(state_df, horizon_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "USE_CHAMPION")

            # Fallback hierarchy should land on global when regime-specific keys are absent.
            horizon_outputs["3d"]["predictability_rows"] = [
                {
                    "horizon": "3d",
                    "asset": "SP500",
                    "group": "equities",
                    "key_type": "global",
                    "key": "H=3d|GLOBAL",
                    "rows": 90,
                    "improvement_abs": 0.06,
                    "improvement_rel": 0.06,
                    "mape_champion": 1.0,
                    "mape_naive": 1.2,
                    "predictability_confidence": 0.9,
                }
            ]
            policy = mgr._build_policy(state_df, horizon_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["fallback_level"], "global")

    def test_forecast_first_reason_codes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = RegimePredictabilityManager(regime_dir=tmpdir)
            state_df, horizon_outputs = self._build_inputs(rows=90, regime_conf=0.9, improvement_abs=0.06, improvement_rel=0.06)

            forecast_outputs = {
                "latest": {
                    "3d": {
                        "equities": {
                            "selected_key": "H=3d|G=equities|R=I1L1R0T1|L=1",
                            "confidence": 0.8,
                            "transition_support": 80,
                            "top3": [{"key": "H=3d|G=equities|R=I1L1R0T1|L=1", "prob": 0.8}],
                        }
                    }
                }
            }
            policy = mgr._build_policy(state_df, horizon_outputs, forecast_outputs=forecast_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "USE_CHAMPION_FORECAST")
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["policy_key_source"], "forecast")

            forecast_outputs["latest"]["3d"]["equities"]["confidence"] = 0.2
            policy = mgr._build_policy(state_df, horizon_outputs, forecast_outputs=forecast_outputs)
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["reason_code"], "FALLBACK_OBSERVED")
            self.assertEqual(policy["horizons"]["3d"]["SP500"]["policy_key_source"], "observed")


if __name__ == "__main__":
    unittest.main()
