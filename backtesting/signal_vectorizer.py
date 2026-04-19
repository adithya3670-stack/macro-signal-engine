import pandas as pd
import numpy as np

class SignalVectorizer:
    """
    Prepares and aligns signals for vectorized backtesting.
    Ensures all assets have aligned timestamps and handles missing data logic.
    """
    def __init__(self):
        pass

    def align_signals(self, signal_df, price_df, method='ffill'):
        """
        Aligns signals to the price data index.
        Crucial because Deep Learning models might have gaps (e.g. removed weekends) 
        different from the main price feed.
        """
        # Reindex signals to match prices
        aligned_signals = signal_df.reindex(price_df.index)
        
        # Fill missing signals
        if method == 'ffill':
            aligned_signals.ffill(inplace=True)
            
        # Fill remaining NaNs (e.g. start of history) with 0.5 (Neutral) or 0
        aligned_signals.fillna(0.5, inplace=True)
        
        return aligned_signals

    def apply_smoothing(self, signal_df, window=3):
        """
        Applies a moving average to the probability signals to reduce churn.
        """
        return signal_df.rolling(window).mean()

    def normalize_signals(self, signal_df):
        """
        Scales signals to be centered around 0 (-1 to 1) instead of 0 to 1.
        Useful for Long/Short strategies.
        """
        return (signal_df - 0.5) * 2

if __name__ == "__main__":
    # Test
    dates = pd.date_range('2023-01-01', periods=10)
    prices = pd.DataFrame(np.random.randn(10, 2), index=dates, columns=['A', 'B'])
    
    sigs = pd.DataFrame(np.random.rand(5, 2), index=dates[::2], columns=['A', 'B']) # Gappy signals
    
    vec = SignalVectorizer()
    aligned = vec.align_signals(sigs, prices)
    print("Aligned Signals:\n", aligned)
