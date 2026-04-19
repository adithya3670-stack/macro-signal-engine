import os
import sys
import tempfile
import unittest

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_3d_regression import Price3DRegressionManager


class TestPrice3DLeakage(unittest.TestCase):
    def test_scaler_uses_train_only_rows(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]}, index=dates)
        train_mask = df.index <= dates[4]

        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = Price3DRegressionManager(
                data_path=os.path.join(tmpdir, "features.csv"),
                model_dir=os.path.join(tmpdir, "models"),
                holdout_root=os.path.join(tmpdir, "holdout"),
                assets=["SP500"],
            )
            scaler = mgr.fit_scaler(df, ["feature"], train_mask)
            self.assertAlmostEqual(float(scaler.center_[0]), 3.0)


if __name__ == "__main__":
    unittest.main()
