import unittest
import os
import sys
import pandas as pd
import warnings

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from services import data_service
from app import app
from analysis.base_model import BaseModel
# from analysis.model_builder import MacroModelBuilder # DELETED
from analysis.deep_learning_model import DLMacroModel

warnings.filterwarnings("ignore")

class TestRefactor(unittest.TestCase):
    
    def test_01_configuration(self):
        """Verify Configuration Centralization"""
        print("\n[Test 01] Verifying Configuration...")
        self.assertTrue(os.path.exists(settings.BASE_DIR), "BASE_DIR does not exist")
        self.assertTrue(os.path.exists(settings.DATA_DIR), "DATA_DIR does not exist")
        print(f"  BASE_DIR: {settings.BASE_DIR}")
        print(f"  DATA_DIR: {settings.DATA_DIR}")
        
    def test_02_data_service(self):
        """Verify Data Service Extraction"""
        print("\n[Test 02] Verifying Data Service...")
        # Should initiate completely empty
        self.assertIsNone(data_service.CACHE['data'])
        
        # Load Data
        success = data_service.load_from_cache_or_csv()
        self.assertTrue(success, "Failed to load data from CSV/Cache")
        self.assertIsNotNone(data_service.CACHE['data'], "Cache 'data' is still None")
        self.assertIsNotNone(data_service.CACHE['correlations'], "Cache 'correlations' is None")
        
        df = data_service.CACHE['data']
        print(f"  Data Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
    def test_03_routes(self):
        """Verify Route Extraction (Flask Blueprints)"""
        print("\n[Test 03] Verifying Routes...")
        tester = app.test_client(self)
        
        # 1. Main Index
        response = tester.get('/')
        self.assertEqual(response.status_code, 200, "Main route '/' failed")
        
        # 2. Dashboard API (analysis_bp)
        response = tester.get('/api/dashboard')
        self.assertEqual(response.status_code, 200, "Route '/api/dashboard' failed")
        json_data = response.get_json()
        self.assertIn('chart_data', json_data)
        
        # 3. Model Info (analysis_bp)
        response = tester.get('/api/model_info')
        self.assertEqual(response.status_code, 200, "Route '/api/model_info' failed")
        
        # 4. Automation Logic (automation_bp)
        response = tester.get('/api/automation/config')
        self.assertEqual(response.status_code, 200, "Route '/api/automation/config' failed")
        
    def test_04_model_standardization(self):
        """Verify Phase 4: Model Interface"""
        print("\n[Test 04] Verifying Model Standardization...")
        
        # 1. MacroModelBuilder (XGB) - DELETED
        print("  XGB/ML tests skipped (Artifacts removed)")

        # 2. DLMacroModel (PyTorch)
        dl_model = DLMacroModel()
        self.assertIsInstance(dl_model, BaseModel)
        self.assertTrue(hasattr(dl_model, 'train_all_models'))
        self.assertTrue(hasattr(dl_model, 'predict_latest'))
        
        try:
            dl_preds = dl_model.predict_latest()
            print(f"  DL Predictions: {len(dl_preds)} items")
        except Exception as e:
            print(f"  DL Predict Warning (Expected if no models): {e}")

if __name__ == '__main__':
    unittest.main()
