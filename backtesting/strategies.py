import pandas as pd
import numpy as np

class RotationalStrategy:
    """
    Implements the 'Max Profit / Min Drawdown' Rotational Logic.
    """
    def __init__(self, top_n=2, vol_target=0.15, use_regime_filter=True, min_confidence=0.5, rebalance_freq='daily'):
        self.top_n = top_n
        self.vol_target = vol_target
        self.use_regime_filter = use_regime_filter
        self.min_confidence = min_confidence
        self.rebalance_freq = rebalance_freq  # 'daily', 'weekly', 'monthly'

    def generate_weights(self, signal_df, risk_df, asset_vols):
        """
        Calculates Portfolio Weights based on Signals and Risk.
        
        Args:
            signal_df (pd.DataFrame): Aligned signals (0-1 probs or scores).
            risk_df (pd.DataFrame): Risk factors (VIX, etc).
            asset_vols (pd.DataFrame): Rolling volatility of assets.
            
        Returns:
            pd.DataFrame: Asset Weights (Sum <= 1.0)
        """
        # 1. Score Assets: Signal / Volatility (Risk-Adjusted Confidence)
        # Avoid div by zero
        scores = signal_df / asset_vols.replace(0, np.nan)
        
        # 2. Rank Assets row-wise
        # rank(ascending=False) means 1 is highest score
        ranks = scores.rank(axis=1, ascending=False)
        
        # 3. Select Top N
        # Mask: 1 if in Top N, 0 otherwise
        selection_mask = (ranks <= self.top_n).astype(int)
        
        # 4. Filter: Minimum Confidence Check
        # If signal < min_confidence, don't buy even if it's rank #1
        confidence_mask = (signal_df > self.min_confidence).astype(int)
        
        # FORCE INITIAL ENTRY:
        # User wants to be invested immediately on Day 1 regardless of signal strength.
        if len(confidence_mask) > 0:
            confidence_mask.iloc[0] = 1.0 # Force all assets to be "Buyable" on Day 1
            
        final_selection = selection_mask * confidence_mask
        
        # 5. Position Sizing (Volatility Targeting)
        # Target Weight = Target Vol / Asset Vol
        # e.g. Target 15%, Asset Vol 30% -> Weight 0.5
        raw_weights = (self.vol_target / asset_vols).replace([np.inf, -np.inf], 0)
        
        # Apply selection
        allocations = raw_weights * final_selection
        
        # 6. Apply Regime Filter (Macro Safety)
        if self.use_regime_filter and 'VIX' in risk_df.columns:
            # If VIX > 30, reduce exposure by 50%
            # If VIX > 45, go to cash (0%)
            vix = risk_df['VIX']
            risk_scalar = pd.Series(1.0, index=vix.index)
            risk_scalar[vix > 30] = 0.5
            risk_scalar[vix > 45] = 0.0
            
            # Broadcast scalar across columns
            allocations = allocations.multiply(risk_scalar, axis=0)
            
        # 7. Normalize Leverage (Optional)
        # Currently we allow leverage < 1 (Cash) but usually cap at 1.0 or 1.5
        # For "Safety", let's cap total leverage at 1.0
        total_leverage = allocations.sum(axis=1)
        # If sum > 1, scale down
        scale_factor = total_leverage.map(lambda x: 1.0 / x if x > 1.0 else 1.0)
        allocations = allocations.multiply(scale_factor, axis=0)
        
        # Apply Rebalancing Frequency
        allocations = self.apply_rebalance_frequency(allocations)
        
        return allocations.fillna(0)
    
    def apply_rebalance_frequency(self, weights_df):
        """
        Resamples weights based on rebalance_freq (daily, weekly, monthly).
        For non-daily frequencies, weights are held constant until the next rebalance date.
        """
        if self.rebalance_freq == 'daily':
            return weights_df
        
        # Determine resampling rule
        if self.rebalance_freq == 'weekly':
            rule = 'W-FRI'  # Rebalance on Fridays
        elif self.rebalance_freq == 'monthly':
            rule = 'M'  # Month End (Using 'M' for compatibility with older pandas)
        else:
            return weights_df  # Default to daily
        
        # Get rebalance dates (last trading day of each period)
        rebalance_dates = weights_df.resample(rule).last().index
        
        # Create a new weights dataframe with same index but only change on rebalance dates
        result = weights_df.copy()
        
        # Forward-fill from rebalance dates only
        # First, set all non-rebalance dates to NaN
        mask = ~result.index.isin(rebalance_dates)
        result.loc[mask] = np.nan
        
        # Forward fill to carry weights forward until next rebalance
        result = result.ffill()
        
        # Handle initial NaN (before first rebalance date)
        result = result.bfill()
        
        return result

if __name__ == "__main__":
    # Test
    dates = pd.date_range('2023-01-01', periods=5)
    sigs = pd.DataFrame({'A': [0.6, 0.8, 0.4, 0.9, 0.2], 'B': [0.55, 0.3, 0.9, 0.1, 0.1]}, index=dates)
    vols = pd.DataFrame({'A': [0.2, 0.2, 0.2, 0.2, 0.2], 'B': [0.1, 0.1, 0.1, 0.1, 0.1]}, index=dates)
    risk = pd.DataFrame({'VIX': [20, 20, 35, 20, 50]}, index=dates)
    
    strat = RotationalStrategy(top_n=1)
    w = strat.generate_weights(sigs, risk, vols)
    print("Weights:\n", w)
