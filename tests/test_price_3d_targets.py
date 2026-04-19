import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_3d_features import Price3DFeatureBuilder
from analysis.price_3d_regression import Price3DRegressionManager


class TestPrice3DTargets(unittest.TestCase):
    def test_target_math_and_mape_mask(self):
        dates = pd.bdate_range("2025-01-01", periods=6)
        raw = pd.DataFrame(
            {
                "SP500": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
                "Oil": [10.0, 9.0, 0.5, -1.0, 2.0, 3.0],
            },
            index=dates,
        )

        builder = Price3DFeatureBuilder(assets=["SP500", "Oil"])
        features = builder.build_from_dataframe(raw)

        self.assertAlmostEqual(features.loc[dates[0], "FuturePrice_SP500_3d"], 103.0)
        self.assertAlmostEqual(features.loc[dates[0], "NormPrice_SP500_3d"], 1.03)
        self.assertAlmostEqual(features.loc[dates[0], "CenteredNormPrice_SP500_3d"], 0.03)
        self.assertEqual(int(features.loc[dates[0], "MAPEValid_Oil_3d"]), 0)
        self.assertEqual(int(features.loc[dates[1], "MAPEValid_Oil_3d"]), 1)

    def test_target_inversion_yields_zero_mape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = Price3DRegressionManager(
                data_path=os.path.join(tmpdir, "features.csv"),
                model_dir=os.path.join(tmpdir, "models"),
                holdout_root=os.path.join(tmpdir, "holdout"),
                assets=["SP500"],
            )
            metrics = mgr._evaluate_predictions(
                current_prices=[100.0],
                actual_future=[103.0],
                pred_centered=[0.03],
                mape_valid=[True],
            )
            self.assertAlmostEqual(metrics["mape"], 0.0, places=8)
            self.assertAlmostEqual(metrics["smape"], 0.0, places=8)
            self.assertAlmostEqual(metrics["mae"], 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
