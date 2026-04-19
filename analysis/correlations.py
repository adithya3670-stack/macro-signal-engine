import pandas as pd

def calculate_correlations(df):
    """
    Calculates Short, Medium, and Long term correlations between Factors and Markets.
    """
    factors = [
        'FEDFUNDS', 'CPIAUCSL', 'VIX',
        'Momentum', 'Strength_RSI', 'Breadth_Vol', 
        'Options_VIX', 'Junk_Bond_Demand', 'Volatility_Spread', 'Safe_Haven_Demand'
    ]
    markets = ['SP500', 'Nasdaq']
    
    results = {}
    
    # Define Windows
    windows = {
        'Short Term (30d)': 30,
        'Medium Term (90d)': 90,
        'Long Term (1y)': 252
    }
    
    for term, window in windows.items():
        term_correlations = pd.DataFrame()
        
        for market in markets:
            for factor in factors:
                # Calculate rolling correlation
                # We rename the series to something readable like 'SP500 vs VIX'
                col_name = f"{market} vs {factor}"
                term_correlations[col_name] = df[market].rolling(window=window).corr(df[factor])
        
        results[term] = term_correlations
        
    return results

def get_latest_drivers(df, correlations):
    """
    Identifies what is currently driving the market based on the latest correlation values.
    Returns a summary text or structure.
    """
    latest_drivers = {}
    
    for term, corr_df in correlations.items():
        latest = corr_df.iloc[-1] # Get latest day stats
        
        # Find the factor with the highest absolute correlation for SP500
        sp500_drivers = latest[latest.index.str.contains('SP500')]
        strongest_driver_name = sp500_drivers.abs().idxmax()
        strongest_driver_val = sp500_drivers[strongest_driver_name]
        
        latest_drivers[term] = {
            'driver': strongest_driver_name,
            'correlation': strongest_driver_val
        }
        
    return latest_drivers
