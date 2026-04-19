
import pandas as pd
from backtesting.data_loader import DataLoader

def check_oil_prices():
    loader = DataLoader()
    prices = loader.get_asset_prices(['Oil'])
    
    start_date = '2020-04-15'
    end_date = '2020-04-25'
    
    mask = (prices.index >= start_date) & (prices.index <= end_date)
    oil_prices = prices.loc[mask]
    
    print("\nOil Prices (April 2020):")
    print(oil_prices)
    
    negative_prices = prices[prices['Oil'] < 0]
    if not negative_prices.empty:
        print("\nWARNING: Negative Prices Found:")
        print(negative_prices)
    else:
        print("\nNo negative prices found in the dataset.")

if __name__ == "__main__":
    check_oil_prices()
