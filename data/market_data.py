import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_market_data(start_date, end_date):
    """
    Fetches daily OHLCV data for:
    - S&P 500 (^GSPC)
    - Nasdaq (^IXIC)
    - Dow Jones (^DJI)
    - Russell 2000 (^RUT)
    Returns a DataFrame with Close prices.
    """
    tickers = ['^GSPC', '^IXIC', '^DJI', '^RUT']
    data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # Rename columns for clarity
    data = data.rename(columns={
        '^GSPC': 'SP500', 
        '^IXIC': 'Nasdaq',
        '^DJI': 'DJIA',
        '^RUT': 'Russell2000'
    })
    
    return data

if __name__ == "__main__":
    # Test
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = get_market_data(start, end)
    print(df.head())
    print(df.tail())
