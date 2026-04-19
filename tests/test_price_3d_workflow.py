import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_3d_features import Price3DFeatureBuilder
from analysis.price_3d_regression import Price3DRegressionManager
from app import app


class TestPrice3DWorkflow(unittest.TestCase):
    def _write_synthetic_features(self, path: str):
        dates = pd.bdate_range("2020-01-01", "2026-03-31")
        t = np.arange(len(dates), dtype=float)
        raw = pd.DataFrame(
            {
                "SP500": 100 + (0.08 * t) + (2.5 * np.sin(t / 18.0)),
                "VIX": 20 + (1.2 * np.sin(t / 13.0)),
            },
            index=dates,
        )
        builder = Price3DFeatureBuilder(assets=["SP500"])
        features = builder.build_from_dataframe(raw)
        features.to_csv(path)

    def test_api_workflow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "price_3d_features.csv")
            model_dir = os.path.join(tmpdir, "models_price_3d")
            holdout_root = os.path.join(tmpdir, "holdout_price_3d")
            self._write_synthetic_features(data_path)

            mgr = Price3DRegressionManager(
                data_path=data_path,
                model_dir=model_dir,
                holdout_root=holdout_root,
                assets=["SP500"],
            )

            app.testing = True
            client = app.test_client()

            with patch("routes.training.Price3DRegressionManager", return_value=mgr), patch(
                "routes.analysis.Price3DRegressionManager", return_value=mgr
            ):
                response = client.get("/api/train/price_3d_stream?epochs=1&assets=SP500&year=2025")
                payload = response.get_data(as_text=True)
                self.assertIn("DONE", payload)

                metrics = client.get("/api/price_3d/metrics?year=2025")
                self.assertEqual(metrics.status_code, 200)
                self.assertIn("metrics", metrics.get_json())

                promote = client.post("/api/price_3d/promote_champions", json={"year": 2025})
                self.assertEqual(promote.status_code, 200)
                self.assertEqual(promote.get_json()["status"], "success")

                predict = client.get("/api/price_3d/predict_latest?source=production")
                body = predict.get_json()
                self.assertEqual(predict.status_code, 200)
                self.assertEqual(body["status"], "success")
                self.assertTrue(any(item["asset"] == "SP500" for item in body["predictions"]))

            self.assertTrue(os.path.exists(os.path.join(holdout_root, "2025", "price3d_metrics.json")))
            self.assertTrue(os.path.exists(os.path.join(holdout_root, "2025", "price3d_champions.json")))
            self.assertTrue(os.path.exists(os.path.join(model_dir, "price3d_champions.json")))


if __name__ == "__main__":
    unittest.main()
