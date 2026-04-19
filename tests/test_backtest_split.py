import unittest
import json
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from config.settings import HOLDOUT_DIR

class TestBacktestSplit(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_prepare_split(self):
        print("\nTesting /api/backtest/prepare_split...")
        payload = {'cutoff_year': 2022}
        
        response = self.app.post('/api/backtest/prepare_split', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        
        print(f"Status Code: {response.status_code}")
        data = json.loads(response.data)
        print(f"Response: {data}")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('train_rows', data)
        self.assertIn('test_rows', data)
        self.assertEqual(data['cutoff_year'], 2022)
        
        # Verify Manifest
        version_dir = os.path.join(HOLDOUT_DIR, '2022')
        manifest_path = os.path.join(version_dir, 'manifest.json')
        self.assertTrue(os.path.exists(manifest_path), "Manifest file not created")

if __name__ == '__main__':
    unittest.main()
