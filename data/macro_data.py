import pandas_datareader.data as web
import pandas as pd
from datetime import datetime, timedelta

def get_macro_data(start_date, end_date):
    """
    Fetches Macro Economic data from FRED:
    - FEDFUNDS: Effective Federal Funds Rate (Monthly, but we can resample)
    - CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items (Monthly)
    """
    # FRED Series IDs
    # Existing: FEDFUNDS (Rates), CPIAUCSL (Inflation)
    # New: PPIACO (PPI), UNRATE (Unemployment), PAYEMS (NFP), M2SL (M2), 
    # T10Y3M (Yield Curve), UMCSENT (Sentiment), WALCL (Fed Balance Sheet),
    # DGS10 (10Y Yield), A191RL1Q225SBEA (Real GDP % Change)
    series_ids = [
        'FEDFUNDS', 'CPIAUCSL', 
        'PPIACO', 'UNRATE', 'PAYEMS', 'M2SL', 
        'T10Y3M', 'UMCSENT', 'WALCL', 'DGS10', 
        'A191RL1Q225SBEA'
    ]
    
    try:
        data = web.DataReader(series_ids, 'fred', start_date, end_date)
        
        # Forward fill to handle monthly data on a daily basis if needed later, 
        # but here we return raw and let the aggregator handle alignment.
        return data
    except Exception as e:
        print(f"Error fetching FRED data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = get_macro_data(start, end)
    print(df.head())
    print(df.tail())
