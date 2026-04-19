import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.raw_df = df.copy()
        self.df = df.copy()
        
    def generate_features(self):
        """Main pipeline to execute all feature transformations."""
        print("Starting Feature Engineering...")
        print("Available Columns:", self.df.columns.tolist())
        
        self._add_macro_derivatives()
        self._add_market_structure()
        self._add_cross_asset_signals()
        self._add_technical_transforms()
        self._add_regime_labels() # New Categorical Regimes
        self._add_targets() # Add forward looking targets
        
        # Handling NaNs:
        # 1. Rolling windows (e.g. 252d Z-Score) create NaNs at the START of the DF.
        # 2. Forward targets (e.g. Fut_60d) create NaNs at the END of the DF.
        # We want to keep the END for live trading signals (where Target is NaN but Feature is valid).
        # We can drop the START (warmup period).
        
        # Drop rows where Features are NaN (roughly first 252 rows)
        # We assume 'Liquidity_Impulse' is a representative lagging feature
        if 'Liquidity_Impulse' in self.df.columns:
             self.df = self.df[self.df['Liquidity_Impulse'].notna()]
        else:
             self.df.dropna(inplace=True) # Fallback if specific col missing
        
        print(f"Feature Engineering Complete. Shape: {self.df.shape}")
        return self.df

    def _add_macro_derivatives(self):
        print("-> Adding Macro Derivatives...")
        
        # Inflation Delta (YoY - 12 months)
        # Macro data is already ffilled daily, so we take 252 days ago roughly
        self.df['CPI_YoY'] = self.df['CPIAUCSL'].pct_change(252) * 100
        self.df['PPI_YoY'] = self.df['PPIACO'].pct_change(252) * 100
        
        # Real Yields: 10Y Yield - CPI YoY
        # Note: Yield is in %, CPI_YoY is in %
        self.df['Real_Yield'] = self.df['DGS10'] - self.df['CPI_YoY']
        
        # Liquidity Impulse: Average YoY growth of M2 and Fed Balance Sheet
        m2_growth = self.df['M2SL'].pct_change(252) * 100
        bs_growth = self.df['WALCL'].pct_change(252) * 100
        self.df['Liquidity_Impulse'] = (m2_growth + bs_growth) / 2
        
        # Curve Steepening (Rate of change of the spread)
        self.df['Curve_Steepening'] = self.df['T10Y3M'].diff(20) # 1 month change in curve

    def _add_market_structure(self):
        print("-> Adding Market Structure...")
        
        # Bond-Stock Correlation Regime (Rolling 60-day correlation)
        # High Positive = Inflation Regime (Bonds & Stocks die together)
        # Negative = Normal/Growth Regime (Bonds hedge Stocks)
        # Using TLT (Long Term Treasuries) as proxy since IEF might be missing
        if 'IEF' in self.df.columns:
            bond_proxy = self.df['IEF']
        elif 'TLT' in self.df.columns:
            bond_proxy = self.df['TLT']
        else:
            # Fallback to inverse of Yield if no price ETF available
            bond_proxy = -1 * self.df['DGS10']
            
        self.df['Bond_Stock_Corr'] = self.df['SP500'].rolling(60).corr(bond_proxy)
        
        # Volatility Regime (VIX vs its 50d MA)
        self.df['VIX_Regime'] = self.df['VIX'] / self.df['VIX'].rolling(50).mean()
        
        # Risk Rotation: Nasdaq (Tech) vs SP500 (Broad)
        self.df['Tech_vs_Broad'] = self.df['Nasdaq'] / self.df['SP500']

    def _add_cross_asset_signals(self):
        print("-> Adding Cross-Asset Signals...")
        
        # Silver/Gold Ratio (Industrial vs Monetary)
        # Prevent division by zero just in case
        self.df['Silver_Gold_Ratio'] = self.df['Silver'] / self.df['Gold'].replace(0, np.nan)
        
        # Oil/Gold Ratio (Energy Cost vs Currency Protection)
        self.df['Oil_Gold_Ratio'] = self.df['Oil'] / self.df['Gold'].replace(0, np.nan)
        
        # Flight to Safety: TLT (Long Bonds) vs SP500
        if 'TLT' in self.df.columns:
            safety = self.df['TLT']
        else:
            # Fallback to Safe_Haven_Demand (which is TLT price usually)
            safety = self.df.get('Safe_Haven_Demand', pd.Series(0, index=self.df.index))
            
        self.df['Safety_vs_Risk'] = safety / self.df['SP500']
        
    def _add_targets(self):
        print("-> Adding Target Variables (Forward Returns)...")
        assets = ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil']
        horizons = {
            '1d': 1,
            '1w': 5,
            '1m': 20,
            '3m': 60
        }
        
        for asset in assets:
            if asset in self.df.columns:
                for name, days in horizons.items():
                    # Calculate Forward Return: (Price[t+n] - Price[t]) / Price[t]
                    # We use shift(-days) to bring future value to current row
                    target_col = f'Target_{asset}_{name}'
                    self.df[target_col] = self.df[asset].pct_change(days).shift(-days) * 100
                    
    def _add_technical_transforms(self):
        print("-> Adding Technical Transforms (Z-Scores)...")
        
        # Z-Score (Distance from mean in Std Devs) - Rolling 1 Year (252 days)
        # Takes all numeric columns
        for col in self.df.columns:
            # Skip if it's already a derived feature or categorical
            if self.df[col].dtype in [np.float64, np.int64]:
                # 252 Day Z-Score
                roll_mean = self.df[col].rolling(252).mean()
                roll_std = self.df[col].rolling(252).std()
                
                # Careful not to overwrite original if we want to keep it?
                # User said "transform these features". Usually we KEEP raw + Add features.
                # But to keep dimensionality sane, maybe we only Z-Score specific important ones.
                # Let's add Z-Score separate columns for the critical assets
                if col in ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil', 'VIX', 'DGS10', 'Bitcoin']:
                    self.df[f'{col}_ZScore'] = (self.df[col] - roll_mean) / roll_std
                    
        # Distance from 200 DMA (Trend)
        for col in ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil', 'Bitcoin']:
            if col in self.df.columns:
                ma_200 = self.df[col].rolling(200).mean()
                self.df[f'{col}_Trend_Dist'] = (self.df[col] / ma_200) - 1

    def _add_regime_labels(self):
        print("-> Adding Categorical Regimes...")
        
        # 1. Inflation Regime (Threshold: 3.0% CPI)
        if 'CPI_YoY' in self.df.columns:
            self.df['Regime_Inflation'] = np.where(self.df['CPI_YoY'] > 3.0, 'High Inflation (>3%)', 'Low Inflation (<3%)')
            
        # 2. Liquidity Regime (Threshold: 0)
        if 'Liquidity_Impulse' in self.df.columns:
            self.df['Regime_Liquidity'] = np.where(self.df['Liquidity_Impulse'] > 0, 'Liquidity Expanding', 'Liquidity Contracting')
            
        # 3. Risk Regime (VIX Threshold: 20)
        if 'VIX' in self.df.columns:
            self.df['Regime_Risk'] = np.where(self.df['VIX'] > 20, 'Risk Off (High Vol)', 'Risk On (Low Vol)')
            
        # 4. Real Yield Regime (Positive vs Negative)
        if 'Real_Yield' in self.df.columns:
            self.df['Regime_Rates'] = np.where(self.df['Real_Yield'] > 0, 'Positive Real Rates', 'Negative Real Rates')

if __name__ == "__main__":
    # Test run
    try:
        master_path = 'master_data/master_dataset.csv'
        from config.settings import ENGINEERED_FEATURES_FILE
        output_path = ENGINEERED_FEATURES_FILE
        
        print(f"Loading {master_path}...")
        df = pd.read_csv(master_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        # Handle ffill if not done in master (it should be, but safety first for rolling calc)
        df = df.replace(0, np.nan).ffill().fillna(0) 
        
        engineer = FeatureEngineer(df)
        features_df = engineer.generate_features()
        
        features_df.to_csv(output_path)
        print(f"Derived {features_df.shape[1]} features.")
        print(f"Saved to {output_path}")
        print("Columns:", features_df.columns.tolist())
        
    except Exception as e:
        print(f"Error: {e}")
