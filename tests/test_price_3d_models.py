import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.price_3d_models import build_price_model
from analysis.price_3d_regression import Price3DRegressionManager


class TestPrice3DModels(unittest.TestCase):
    def test_model_save_load_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = Price3DRegressionManager(
                data_path=os.path.join(tmpdir, "features.csv"),
                model_dir=os.path.join(tmpdir, "models"),
                holdout_root=os.path.join(tmpdir, "holdout"),
                assets=["SP500"],
            )
            input_size = 4

            for model_type, config in mgr.MODEL_CONFIGS.items():
                torch.manual_seed(7)
                model = build_price_model(model_type, input_size, config)
                model.eval()
                x = torch.randn(2, int(config["window_size"]), input_size)
                out1 = model(x).detach().numpy()
                self.assertEqual(out1.shape, (2, 1))

                path = os.path.join(tmpdir, f"{model_type}.pth")
                torch.save(model.state_dict(), path)

                reloaded = build_price_model(model_type, input_size, config)
                reloaded.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
                reloaded.eval()
                out2 = reloaded(x).detach().numpy()
                self.assertTrue(np.allclose(out1, out2, atol=1e-6), model_type)


if __name__ == "__main__":
    unittest.main()
