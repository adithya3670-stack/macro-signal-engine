import unittest
import os
import sys
import pandas as pd
import numpy as np
import torch

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.deep_learning_model import DLMacroModel

class TestDLMethods(unittest.TestCase):
    def test_methods_exist(self):
        print("\nTesting DLMacroModel methods...")
        model = DLMacroModel()
        
        # Test create_sequences
        data = np.random.rand(100, 5) # 100 timesteps, 5 features
        targets = np.random.randint(0, 2, 100) # 100 targets
        seq_len = 10
        
        X, y = model.create_sequences(data, seq_len, targets)
        print(f"Sequences Created: X={X.shape}, y={y.shape}")
        
        expected_len = 100 - seq_len
        self.assertEqual(len(X), expected_len)
        self.assertEqual(len(y), expected_len)
        self.assertEqual(X.shape[1], seq_len)
        self.assertEqual(X.shape[2], 5)
        
        # Test load_and_preprocess existence (mocking file existence might be hard, just check callable)
        self.assertTrue(callable(model.load_and_preprocess))
        print("load_and_preprocess is callable.")

if __name__ == '__main__':
    unittest.main()
