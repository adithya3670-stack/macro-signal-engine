import pandas as pd
import numpy as np
import os
from backtesting.data_loader import DataLoader
from backtesting.signal_generator import SignalGenerator
from backtesting.signal_vectorizer import SignalVectorizer
from backtesting.strategies import RotationalStrategy
from backtesting.engine import VectorizedBacktester

def test_api_logic():
    print("--- Starting API Logic Test ---")
    
    # Mock Inputs
    start_date = '2015-01-01'
    end_date = None
    initial_capital = 10000.0
    # Full Default Universe
    universe = ['SP500', 'Nasdaq', 'Gold', 'Silver', 'Bitcoin', 'Oil']
    horizon = '3m'
    top_n = 2
    vol_target = 0.15
    use_regime = True
    trade_threshold = 0.005
    
    # 1. Load Data
    print("1. Loading Data...")
    data_loader = DataLoader()
    prices = data_loader.get_asset_prices(universe)
    risk_data = data_loader.get_risk_factors()
    
    # Filter Dates
    prices = prices[prices.index >= pd.to_datetime(start_date)]
    risk_data = risk_data[risk_data.index >= pd.to_datetime(start_date)]
    
    # 2. Generate Signals
    print(f"2. Generating Signals for Horizon {horizon}...")
    signal_gen = SignalGenerator()
    
    signal_universe = []
    for asset in universe:
        if '_' in asset:
            signal_universe.append(asset)
        else:
             signal_universe.append(f"{asset}_{horizon}")
    
    print(f"   Signal Universe: {signal_universe}")
    raw_signals = signal_gen.generate_signals(signal_universe, start_date, end_date, force_refresh=True)
    
    if raw_signals.empty:
        print("ERROR: Signals empty")
        return

    # 3. Rename Logic (CRITICAL CHECK)
    print("3. Renaming Columns...")
    print(f"   Before: {raw_signals.columns.tolist()}")
    rename_map = {}
    for col in raw_signals.columns:
        for original_asset in universe:
            if col.startswith(original_asset):
                rename_map[col] = original_asset
                break
    raw_signals.rename(columns=rename_map, inplace=True)
    print(f"   After: {raw_signals.columns.tolist()}")
    
    # 4. Vectorize
    print("4. Vectorizing...")
    vectorizer = SignalVectorizer()
    aligned_signals = vectorizer.align_signals(raw_signals, prices)
    asset_vols = prices.pct_change().rolling(60).std() * np.sqrt(252)
    
    # 5. Strategy
    print("5. Strategy...")
    strat = RotationalStrategy(top_n=top_n, vol_target=vol_target, use_regime_filter=use_regime)
    weights = strat.generate_weights(aligned_signals, risk_data, asset_vols)
    
    # 6. Engine
    print("6. Engine...")
    engine = VectorizedBacktester(initial_capital=initial_capital)
    results = engine.run_backtest(prices, weights, trade_threshold=trade_threshold)
    
    print("--- SUCCESS ---")
    print(f"Final Equity: {results['equity_curve'].iloc[-1]}")
    print(f"Trades: {len(results['trades'])}")

if __name__ == "__main__":
    test_api_logic()
