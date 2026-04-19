import unittest
import os
import sys
import pandas as pd
import numpy as np
import io
from unittest.mock import MagicMock

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.deep_learning_model import DLMacroModel

class TestDLCrash(unittest.TestCase):
    def setUp(self):
        self.model = DLMacroModel(model_dir="tests/temp_dl_models")
        os.makedirs("tests/temp_dl_models", exist_ok=True)
        
        # Mock load_and_preprocess to return dummy data
        dates = pd.date_range("2020-01-01", periods=200)
        data = {
            'SP500': np.random.rand(200),
            'Feature1': np.random.rand(200),
            'Feature2': np.random.rand(200),
            'Target_SP500_1w': np.where(np.random.rand(200) > 0.5, 1, 0) # Binary target
        }
        self.df = pd.DataFrame(data, index=dates)
        self.model.load_and_preprocess = MagicMock(return_value=self.df)
        
        # Mock assets/horizons to limit scope
        self.model.assets = ['SP500']
        self.model.horizons = ['1w']

    def test_optimize_models_crash(self):
        print("\nTesting optimize_models for crash...")
        # Should run without error (workers=0 patch check)
        # Using 1 iteration, 1 epoch
        self.model.optimize_models(model_type='lstm', iterations=1, epochs=1, batch_size=8)
        print("optimize_models completed successfully.")

    def test_train_all_models_crash(self):
        print("\nTesting train_all_models for crash...")
        self.model.train_all_models(model_type='lstm', epochs=1, batch_size=8)
        print("train_all_models completed successfully.")
        
    def tearDown(self):
        import shutil
        if os.path.exists("tests/temp_dl_models"):
            shutil.rmtree("tests/temp_dl_models")

if __name__ == '__main__':
    unittest.main()
