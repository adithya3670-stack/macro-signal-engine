import unittest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.etl import load_and_merge_data

class TestETLPatch(unittest.TestCase):
    @patch('data.etl.get_market_data')
    @patch('data.etl.get_macro_data')
    @patch('data.etl.get_sentiment_data')
    @patch('data.etl.get_indicators_data')
    @patch('data.etl.get_commodities_data')
    def test_load_no_save(self, mock_comm, mock_ind, mock_sent, mock_macro, mock_market):
        print("\nTesting load_and_merge_data(save_to_disk=False)...")
        
        # Mock DataFrames
        dates = pd.date_range('2023-01-01', periods=5)
        mock_df = pd.DataFrame({'Close': [100]*5}, index=dates)
        mock_market.return_value = pd.DataFrame({'SP500': [100]*5, 'Nasdaq': [100]*5}, index=dates)
        mock_macro.return_value = mock_df
        mock_sent.return_value = mock_df
        mock_ind.return_value = mock_df
        mock_comm.return_value = mock_df
        
        # Mock to_csv to ensure it's NOT called
        with patch('pandas.DataFrame.to_csv') as mock_to_csv:
            df = load_and_merge_data('2023-01-01', '2023-01-05', save_to_disk=False)
            
            mock_to_csv.assert_not_called()
            print("Verified: to_csv was NOT called.")
            
            self.assertFalse(df.empty)
            print(f"Verified: Data returned with shape {df.shape}")

if __name__ == '__main__':
    unittest.main()
