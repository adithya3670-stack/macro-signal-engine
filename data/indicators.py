import yfinance as yf
import pandas as pd
import numpy as np

def get_indicators_data(start_date, end_date):
    """
    Fetches data and calculates proxies for the 7 Fear & Greed Indicators.
    Returns a DataFrame with the 7 indicator values.
    """
    # Tickers needed: 
    # ^GSPC (S&P 500), ^VIX (Volatility), HYG (High Yield Bonds), 
    # IEF (7-10 Year Treasury), SPY (S&P 500 ETF), TLT (20+ Year Treasury)
    tickers = ['^GSPC', '^VIX', 'HYG', 'IEF', 'SPY', 'TLT']
    
    print(f"Fetching indicator data for: {tickers}")
    raw_data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
    
    # 1. Market Momentum: S&P 500 vs 125-day MA
    # Value > 0 means S&P 500 is above MA (Greed), < 0 means below (Fear)
    ma_125 = raw_data['^GSPC'].rolling(window=125).mean()
    momentum = raw_data['^GSPC'] - ma_125
    
    # 2. Stock Price Strength: RSI of S&P 500
    # RSI > 70 (Greed), RSI < 30 (Fear)
    delta = raw_data['^GSPC'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 3. Stock Price Breadth: Volume Trend (Proxy using ^GSPC Volume)
    # Ideally we need adv/decl volume, but index volume is a decent proxy for activity
    # Using OBV (On-Balance Volume) logic on Index can show accumulation/distribution
    # We need Volume for this.
    try:
        vol_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False, auto_adjust=True)['Volume']
        # Simple proxy: Rolling 30d Avg Volume. Rising volume in uptrend = Greed.
        # Let's just use Volume relative to 30d MA for now as a "Breadth/Activity" proxy.
        breadth = vol_data.rolling(window=30).mean()
    except Exception as e:
        print(f"Error fetching Volume: {e}")
        breadth = pd.Series(0, index=raw_data.index)

    # 4. Put and Call Options: Using VIX as proxy
    # High VIX = Fear, Low VIX = Greed. Invert it so High = Greed logic? 
    # Usually F&G index uses Put/Call ratio. VIX is a good enough inverse proxy.
    options_proxy = raw_data['^VIX']
    
    # 5. Junk Bond Demand: HYG vs IEF
    # High Ratio = Investors want risk (Junk) = Greed. Low Ratio = Safe Gov Bonds = Fear.
    junk_bond_demand = raw_data['HYG'] / raw_data['IEF']
    
    # 6. Market Volatility: VIX vs 50-day MA
    # VIX > 50d MA = Rising Fear. 
    vix_ma_50 = raw_data['^VIX'].rolling(window=50).mean()
    volatility_score = raw_data['^VIX'] - vix_ma_50
    
    # 7. Safe Haven Demand: SPY vs TLT
    # Stocks outperforming Bonds = Greed.
    # We look at 20-day return difference? Or just Price Ratio.
    # Price ratio is simpler for trend.
    safe_haven_demand = raw_data['SPY'] / raw_data['TLT']
    
    # Ensure 1D Series for DataFrame construction
    # yfinance sometimes returns (N, 1) DataFrames for single tickers in recent versions
    def squeeze_if_needed(data):
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]
        return data

    momentum = squeeze_if_needed(momentum)
    rsi = squeeze_if_needed(rsi)
    breadth = squeeze_if_needed(breadth)
    options_proxy = squeeze_if_needed(options_proxy)
    junk_bond_demand = squeeze_if_needed(junk_bond_demand)
    volatility_score = squeeze_if_needed(volatility_score)
    safe_haven_demand = squeeze_if_needed(safe_haven_demand)
    
    # Combine into DataFrame
    indicators = pd.DataFrame({
        'Momentum': momentum,
        'Strength_RSI': rsi,
        'Breadth_Vol': breadth,
        'Options_VIX': options_proxy,
        'Junk_Bond_Demand': junk_bond_demand,
        'Volatility_Spread': volatility_score,
        'Safe_Haven_Demand': safe_haven_demand
    })
    
    return indicators

if __name__ == "__main__":
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=365*2)
    df = get_indicators_data(start, end)
    print(df.tail())
