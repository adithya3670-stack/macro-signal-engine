import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_1m_features import Price1MFeatureBuilder
from analysis.price_1w_features import Price1WFeatureBuilder
from analysis.price_3d_features import Price3DFeatureBuilder
from analysis.regime_predictability import RegimePredictabilityManager
from app import app


class TestRegimeApiWorkflow(unittest.TestCase):
    def _build_raw_master(self):
        dates = pd.bdate_range("2024-01-01", "2026-03-31")
        t = np.arange(len(dates), dtype=float)
        return pd.DataFrame(
            {
                "SP500": 5000 + 2.0 * t + 30 * np.sin(t / 25.0),
                "Nasdaq": 15000 + 4.0 * t + 60 * np.sin(t / 23.0),
                "DJIA": 35000 + 1.6 * t + 50 * np.sin(t / 27.0),
                "Russell2000": 2000 + 0.6 * t + 12 * np.sin(t / 21.0),
                "Gold": 1800 + 0.8 * t + 8 * np.sin(t / 18.0),
                "Silver": 22 + 0.03 * t + 0.5 * np.sin(t / 16.0),
                "Copper": 4 + 0.004 * t + 0.08 * np.sin(t / 19.0),
                "Oil": 70 + 0.05 * t + 2.0 * np.sin(t / 17.0),
                "VIX": 18 + 3 * np.sin(t / 11.0),
                "Liquidity_Impulse": np.sin(t / 13.0),
                "Real_Yield": 0.5 * np.sin(t / 35.0),
                "CPI_YoY": 2.7 + 0.3 * np.sin(t / 40.0),
            },
            index=dates,
        )

    def _write_standard_artifacts(self, folder: str, metrics_name: str, champions_name: str):
        os.makedirs(folder, exist_ok=True)
        metrics = [
            {
                "asset": "SP500",
                "model_type": "nlinear",
                "discovery_summary": {"std_mape": 0.2},
                "holdout_metrics": {"mape": 1.1},
                "shadow_metrics": {"mape": 1.2},
                "baseline_2025": {"mape": 1.3},
            }
        ]
        champions = {
            "holdout_year": 2025,
            "assets": {
                "SP500": {
                    "selected_model": "nlinear",
                    "deployment_strategy": "model",
                    "baseline_2025_mape": 1.3,
                    "selected_holdout_mape": 1.1,
                    "improvement_relative": 0.15,
                    "improvement_absolute": 0.2,
                    "artifact_prefix": "SP500_nlinear",
                }
            },
        }
        with open(os.path.join(folder, metrics_name), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle)
        with open(os.path.join(folder, champions_name), "w", encoding="utf-8") as handle:
            json.dump(champions, handle)

    def test_regime_endpoints_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = self._build_raw_master()
            master_path = os.path.join(tmpdir, "master.csv")
            raw.to_csv(master_path)

            f3d = os.path.join(tmpdir, "price_3d_features.csv")
            f1w = os.path.join(tmpdir, "price_1w_features.csv")
            f1m = os.path.join(tmpdir, "price_1m_features.csv")
            Price3DFeatureBuilder(assets=["SP500"]).build_from_dataframe(raw).to_csv(f3d)
            Price1WFeatureBuilder(assets=["SP500"]).build_from_dataframe(raw).to_csv(f1w)
            Price1MFeatureBuilder(assets=["SP500"]).build_from_dataframe(raw).to_csv(f1m)

            prod3d = os.path.join(tmpdir, "models_price_3d")
            prod1w = os.path.join(tmpdir, "models_price_1w")
            prod1m = os.path.join(tmpdir, "models_price_1m")
            self._write_standard_artifacts(prod3d, "price3d_metrics.json", "price3d_champions.json")
            self._write_standard_artifacts(prod1w, "price1w_metrics.json", "price1w_champions.json")
            self._write_standard_artifacts(prod1m, "price1m_metrics.json", "price1m_champions.json")

            regime_dir = os.path.join(tmpdir, "models_regime")
            mgr = RegimePredictabilityManager(
                regime_dir=regime_dir,
                master_data_path=master_path,
                feature_paths={"3d": f3d, "1w": f1w, "1m": f1m},
                production_dirs={"3d": prod3d, "1w": prod1w, "1m": prod1m},
                holdout_dirs={"3d": os.path.join(tmpdir, "h3d"), "1w": os.path.join(tmpdir, "h1w"), "1m": os.path.join(tmpdir, "h1m")},
            )

            app.testing = True
            client = app.test_client()
            with patch("routes.analysis.RegimePredictabilityManager", return_value=mgr):
                rebuild = client.post("/api/regime/rebuild?source=production&year=2025")
                self.assertEqual(rebuild.status_code, 200)
                self.assertEqual(rebuild.get_json()["status"], "success")

                state_latest = client.get("/api/regime/state_latest")
                self.assertEqual(state_latest.status_code, 200)
                self.assertEqual(state_latest.get_json()["status"], "success")

                predictability = client.get("/api/regime/predictability?horizon=3d")
                self.assertEqual(predictability.status_code, 200)
                self.assertEqual(predictability.get_json()["status"], "success")
                self.assertIn("forecast_diagnostics", predictability.get_json())

                forecast_latest = client.get("/api/regime/forecast_latest")
                self.assertEqual(forecast_latest.status_code, 200)
                self.assertEqual(forecast_latest.get_json()["status"], "success")

                policy = client.get("/api/regime/policy_latest")
                self.assertEqual(policy.status_code, 200)
                self.assertEqual(policy.get_json()["status"], "success")
                policy_json = policy.get_json()
                sp500_policy = policy_json.get("horizons", {}).get("3d", {}).get("SP500", {})
                self.assertIn("forecast_key", sp500_policy)
                self.assertIn("forecast_confidence", sp500_policy)
                self.assertIn("policy_key_source", sp500_policy)

                predict_latest = client.get("/api/regime/predict_latest")
                self.assertEqual(predict_latest.status_code, 200)
                body = predict_latest.get_json()
                self.assertEqual(body["status"], "success")
                self.assertTrue(len(body["predictions"]) >= 1)
                self.assertIn("forecast_key", body["predictions"][0])
                self.assertIn("forecast_confidence", body["predictions"][0])
                self.assertIn("policy_key_source", body["predictions"][0])

            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_state_history.parquet")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_predictability_3d.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_predictability_1w.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_predictability_1m.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_forecast_3d.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_forecast_1w.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_forecast_1m.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_forecast_latest.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_policy.json")))
            self.assertTrue(os.path.exists(os.path.join(regime_dir, "regime_latest.json")))


if __name__ == "__main__":
    unittest.main()
