import pandas as pd
import numpy as np
import os
from config.settings import ENGINEERED_FEATURES_FILE

class DataLoader:
    """
    Standardized Data Loader for the Backtesting Engine.
    Reads 'engineered_features.csv' and prepares separate DataFrames for:
    1. Prices (Close)
    2. Features (for Signals)
    3. Market Data (VIX, Yields - for Risk Management)
    """
    def __init__(self, data_path=ENGINEERED_FEATURES_FILE):
        self.data_path = data_path
        self.raw_df = None
        self.prices = None
        
    def load_data(self):
        """Loads and cleans the main dataset."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found at {self.data_path}")
            
        print(f"Loading data from {self.data_path}...")
        df = pd.read_csv(self.data_path)
        
        # Handle Date Index
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Sort and clean
        df.sort_index(inplace=True)
        
        # We need to fill NaNs for vectorized ops, BUT we should be careful about look-ahead.
        # ffill is generally safe for macro data (assumes last known value).
        df.ffill(inplace=True)
        
        self.raw_df = df
        return df

    def get_asset_prices(self, assets=None):
        """
        Returns a DataFrame of Close prices for the specified assets.
        If assets is None, attempts to detect all known assets.
        """
        if self.raw_df is None:
            self.load_data()
            
        if assets is None:
            # Auto-detect standard assets
            assets = ['SP500', 'Nasdaq', 'Gold', 'Silver', 'Oil']
            # Filter to ones actually in columns
            assets = [a for a in assets if a in self.raw_df.columns]
            
        price_data = self.raw_df[assets].copy()
        return price_data

    def get_risk_factors(self):
        """
        Returns a DataFrame of Risk Factors (VIX, Yields, Volatility).
        Used for regime filters and position sizing.
        """
        if self.raw_df is None:
            self.load_data()
            
        risk_cols = ['VIX', 'DGS10', 'Liquidity_Impulse']
        # Add Volatility_Spread if available
        if 'Volatility_Spread' in self.raw_df.columns:
            risk_cols.append('Volatility_Spread')
            
        # Filter to existing
        risk_cols = [c for c in risk_cols if c in self.raw_df.columns]
        
        return self.raw_df[risk_cols].copy()

if __name__ == "__main__":
    # Test
    loader = DataLoader()
    try:
        df = loader.load_data()
        prices = loader.get_asset_prices()
        print("Prices Shape:", prices.shape)
        print("Assets:", prices.columns.tolist())
        print("Last Row:\n", prices.iloc[-1])
    except Exception as e:
        print(f"Error loading data: {e}")
