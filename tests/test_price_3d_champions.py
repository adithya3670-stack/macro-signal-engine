import os
import sys
import tempfile
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_3d_regression import Price3DRegressionManager


class TestPrice3DChampions(unittest.TestCase):
    def test_promotion_threshold_blocks_weak_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = Price3DRegressionManager(
                data_path=os.path.join(tmpdir, "features.csv"),
                model_dir=os.path.join(tmpdir, "models"),
                holdout_root=os.path.join(tmpdir, "holdout"),
                assets=["SP500"],
            )
            metrics = [
                {
                    "asset": "SP500",
                    "model_type": "nlinear",
                    "discovery_summary": {"median_mape": 0.9, "std_mape": 0.1},
                    "holdout_metrics": {"mape": 0.98, "latency_ms": 0.5},
                    "baseline_2025": {"mape": 1.0},
                }
            ]
            champions = mgr.select_champions_from_metrics(metrics, holdout_year=2025)
            self.assertEqual(champions["assets"]["SP500"]["deployment_strategy"], "naive_last")

    def test_promotion_threshold_allows_clear_winner(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = Price3DRegressionManager(
                data_path=os.path.join(tmpdir, "features.csv"),
                model_dir=os.path.join(tmpdir, "models"),
                holdout_root=os.path.join(tmpdir, "holdout"),
                assets=["SP500"],
            )
            metrics = [
                {
                    "asset": "SP500",
                    "model_type": "nlinear",
                    "discovery_summary": {"median_mape": 0.8, "std_mape": 0.1},
                    "holdout_metrics": {"mape": 0.9, "latency_ms": 0.5},
                    "baseline_2025": {"mape": 1.0},
                }
            ]
            champions = mgr.select_champions_from_metrics(metrics, holdout_year=2025)
            self.assertEqual(champions["assets"]["SP500"]["deployment_strategy"], "model")


if __name__ == "__main__":
    unittest.main()
