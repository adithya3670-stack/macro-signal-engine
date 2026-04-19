import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_sentiment_data(start_date, end_date):
    """
    Fetches VIX (^VIX) data as a proxy for Market Fear/Sentiment.
    High VIX = High Fear, Low VIX = Low Fear (Greed).
    """
    ticker = '^VIX'
    data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # Check if data is a Series or DataFrame (yfinance output changes based on auto_adjust/version)
    if isinstance(data, pd.DataFrame):
         # If it's a dataframe with single column, rename it. If multi-index, handle it.
         if ticker in data.columns:
             data = data[[ticker]]
         data.columns = ['VIX']
    elif isinstance(data, pd.Series):
        data = data.to_frame(name='VIX')
        
    return data

if __name__ == "__main__":
    # Test
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = get_sentiment_data(start, end)
    print(df.head())
    print(df.tail())
