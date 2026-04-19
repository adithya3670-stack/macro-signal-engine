import os
import sys
import pandas as pd
from backtesting.signal_generator import SignalGenerator

# Mock class to simulate the API call details
def test_horizon_generation(horizon='3m'):
    print(f"\n--- Testing Horizon: {horizon} ---")
    
    # 1. Simulate API Universe Construction
    universe = ['SP500', 'Bitcoin']
    signal_universe = []
    for asset in universe:
        if '_' in asset:
            signal_universe.append(asset)
        else:
             signal_universe.append(f"{asset}_{horizon}")
    
    print(f"Signal Universe: {signal_universe}")
    
    # 2. Initialize Gen
    sig_gen = SignalGenerator()
    
    # 3. Generate
    try:
        signals = sig_gen.generate_signals(signal_universe, force_refresh=True)
        print("Signal Generation Success!")
        print(signals.head())
        print(f"Columns: {signals.columns.tolist()}")
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test valid 1m first (baseline)
    test_horizon_generation('1m')
    # Test 3m (New feature)
    test_horizon_generation('3m')
    # Test 1w (New feature)
    test_horizon_generation('1w')
