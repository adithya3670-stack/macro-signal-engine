import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.deep_learning_model import DLMacroModel

class TestDLDateHandling(unittest.TestCase):
    def setUp(self):
        self.model = DLMacroModel(model_dir="tests/temp_dl_models")
        
        # Mock load_and_preprocess
        dates = pd.date_range("2020-01-01", periods=100)
        data = {
            'SP500': np.random.rand(100),
            'Feature1': np.random.rand(100)
        }
        self.df = pd.DataFrame(data, index=dates)
        self.model.load_and_preprocess = MagicMock(return_value=self.df)

    def test_predict_range_none_date(self):
        print("\nTesting predict_range with None end_date...")
        # Should NOT raise TypeError
        try:
            # We expect empty DF because no models are trained/mocked fully, 
            # but we just want to ensure it passes the mask creation line.
            self.model.predict_range(start_date="2020-01-01", end_date=None)
            print("Passed: No TypeError with end_date=None")
        except TypeError as e:
            self.fail(f"Raised TypeError: {e}")
        except Exception:
            # Other errors (like missing scaler) are expected and acceptable for this smoke test
            pass

if __name__ == '__main__':
    unittest.main()
