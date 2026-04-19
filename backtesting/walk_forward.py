import pandas as pd
import numpy as np
from backtesting.strategies import RotationalStrategy
from backtesting.engine import VectorizedBacktester

class WalkForwardValidator:
    """
    Orchestrates Walk-Forward Analysis to prevent overfitting.
    """
    def __init__(self, data_loader, signal_gen, vectorizer):
        self.loader = data_loader
        self.signal_gen = signal_gen
        self.vectorizer = vectorizer
        
    def run_validation(self, universe, strategy_config, start_year=2015, step_years=1):
        """
        Runs the strategy year-by-year (Walking Forward).
        In a real ML context, we would Retrain models here.
        Since our models are pre-trained, we simulate 'parameter stability' 
        or simply validate the strategy logic on out-of-sample blocks.
        
        Args:
            universe (list): Assets
            strategy_config (dict): config for RotationalStrategy
            start_year (int): Year to start the first Out-of-Sample block
            
        Returns:
            dict: Combined results
        """
        print(f"Starting Walk-Forward Validation from {start_year}...")
        
        # 1. Load All Data
        prices = self.loader.get_asset_prices(universe)
        risk_data = self.loader.get_risk_factors()
        
        # 2. Generate All Signals (Pre-compute for speed)
        # In a strict WF, we'd generate year by year, but assuming no lookahead in model *training*
        # (which we can't control here easily without retraining), we use the pre-computed inference.
        raw_signals = self.signal_gen.generate_signals(universe)
        aligned_signals = self.vectorizer.align_signals(raw_signals, prices)
        
        asset_vols = prices.pct_change().rolling(60).std() * np.sqrt(252)
        
        # 3. Stitching Loop
        equity_segments = []
        full_drawdown = []
        
        # Determine years
        max_year = prices.index.year.max()
        current_year = start_year
        
        combined_curve = pd.Series(dtype=float)
        current_capital = 10000.0
        
        while current_year <= max_year:
            test_start = f"{current_year}-01-01"
            test_end = f"{current_year}-12-31"
            
            # Slice Data for this Test Block
            block_prices = prices.loc[test_start:test_end]
            if block_prices.empty:
                break
                
            block_signals = aligned_signals.loc[test_start:test_end]
            block_risk = risk_data.loc[test_start:test_end]
            block_vols = asset_vols.loc[test_start:test_end]
            
            # Run Strategy Logic
            strat = RotationalStrategy(**strategy_config)
            # Note: In a true optimization WF, we would optimize 'strat' params 
            # on [2010 -> current_year-1] here before running this block.
            # For now, we validate the *fixed* strategy out-of-sample.
            
            weights = strat.generate_weights(block_signals, block_risk, block_vols)
            
            # Run Engine
            engine = VectorizedBacktester(initial_capital=current_capital)
            res = engine.run_backtest(block_prices, weights)
            
            # Store Result
            block_equity = res['equity_curve']
            
            if not block_equity.empty:
                current_capital = block_equity.iloc[-1] # Carry over capital
                combined_curve = pd.concat([combined_curve, block_equity])
            
            current_year += step_years
            
        # 4. Recalculate Global Metrics on Stitched Curve
        # Re-derive returns from the stitched equity curve to correct for gaps/resets
        stitched_returns = combined_curve.pct_change().fillna(0)
         
        # Drawdown
        running_max = combined_curve.cummax()
        drawdown = (combined_curve - running_max) / running_max
        
        engine = VectorizedBacktester() # Helper for metrics
        metrics = engine._calculate_metrics(stitched_returns, drawdown)
        
        return {
            'equity_curve': combined_curve,
            'drawdown': drawdown,
            'metrics': metrics
        }
