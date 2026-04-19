import yfinance as yf
import pandas as pd

def get_commodities_data(start_date, end_date):
    """
    Fetches Commodities data:
    - Gold: GC=F
    - Silver: SI=F
    - Crude Oil: CL=F
    - Copper: HG=F
    """
    tickers = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Oil': 'CL=F',
        'Copper': 'HG=F',
        'Bitcoin': 'BTC-USD'
    }
    
    data_frames = []
    
    for name, ticker in tickers.items():
        try:
            print(f"Fetching {name} ({ticker})...")
            # auto_adjust=True to handle splits/dividends (though less relevant for futures/crypto, good practice)
            df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            # Keep only Close (or 'Close' column if auto_adjust=False, but with auto_adjust=True it's usually just price)
            # yfinance structure varies by version. Assuming standard 'Close'.
            if 'Close' in df.columns:
                series = df['Close']
            else:
                # Fallback if structure is different
                series = df.iloc[:, 0]
                
            series.name = name
            data_frames.append(series)
            
        except Exception as e:
            print(f"Error fetching {name}: {e}")
            
    if data_frames:
        # Merge all series into a single DataFrame
        combined_df = pd.concat(data_frames, axis=1)
        
        # Explicitly rename columns to match keys
        # The series.name assignment above might be lost during concat depending on pandas version/methods
        # So we force rename here based on the order or keys
        combined_df.columns = [name for name in tickers.keys() if name in [s.name for s in data_frames]]
        
        # ACTUALLY, safer way:
        # Re-assign columns based on the frame list order which we controlled
        combined_df.columns = [s.name for s in data_frames]
        
        # Forward fill to handle different trading hours/holidays (Crypto is 24/7, Futures have breaks)
        combined_df = combined_df.ffill()
        return combined_df
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=365)
    df = get_commodities_data(start, end)
    print(df.tail())
