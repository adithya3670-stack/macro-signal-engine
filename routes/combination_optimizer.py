"""
Combination Optimizer API
Grid-search across all model/asset/horizon combinations to find optimal configs.
"""
from flask import Blueprint, jsonify, request, Response, stream_with_context
import json
import itertools
import random
import os
import pandas as pd
import numpy as np
from datetime import datetime

combo_bp = Blueprint('combo', __name__)

# Fixed Parameters
FIXED_CONFIG = {
    'initial_capital': 10000,
    'volatility_target': 1.0,  # 100%
    'use_regime_filter': False,
    'rebalance_freq': 'weekly',
    'trade_threshold': 0.15,  # 15%
    'min_confidence': 0.50
}

# Variable Parameters
ALL_ASSETS = ['SP500', 'Nasdaq', 'Gold', 'Silver', 'Copper', 'Russell2000', 'Oil', 'DJIA']
ALLOCATIONS = [2, 3]  # Top 2 or Top 3
HORIZONS = ['1w', '1m', '3m', 'ensemble']
HYBRID_WEIGHT_PROFILES = {
    "1":  {"name": "short_max",     "weights": {"1w": 0.70, "1m": 0.20, "3m": 0.10}},
    "2":  {"name": "short_tilt",    "weights": {"1w": 0.60, "1m": 0.30, "3m": 0.10}},
    "3":  {"name": "short_heavy",   "weights": {"1w": 0.60, "1m": 0.20, "3m": 0.20}},
    "4":  {"name": "mid_tilt",      "weights": {"1w": 0.50, "1m": 0.30, "3m": 0.20}},
    "5":  {"name": "mid_blend",     "weights": {"1w": 0.40, "1m": 0.40, "3m": 0.20}},
    "6":  {"name": "mid_heavy",     "weights": {"1w": 0.40, "1m": 0.30, "3m": 0.30}},
    "7":  {"name": "balanced",      "weights": {"1w": 1/3,  "1m": 1/3,  "3m": 1/3}},
    "8":  {"name": "barbell",       "weights": {"1w": 0.30, "1m": 0.40, "3m": 0.30}},
    "9":  {"name": "defensive",     "weights": {"1w": 0.30, "1m": 0.30, "3m": 0.40}},
    "10": {"name": "long_tilt",     "weights": {"1w": 0.20, "1m": 0.40, "3m": 0.40}},
    "11": {"name": "long_heavy",    "weights": {"1w": 0.20, "1m": 0.30, "3m": 0.50}},
    "12": {"name": "long_max",      "weights": {"1w": 0.10, "1m": 0.30, "3m": 0.60}},
    "13": {"name": "mid_core",      "weights": {"1w": 0.20, "1m": 0.60, "3m": 0.20}},
    "14": {"name": "mid_anchor",    "weights": {"1w": 0.15, "1m": 0.60, "3m": 0.25}}
}


def get_available_folders():
    """Returns list of available holdout folders."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    holdout_dir = os.path.join(base_path, 'models', 'holdout_dl')
    
    if not os.path.exists(holdout_dir):
        return []
    
    folders = []
    for item in os.listdir(holdout_dir):
        full_path = os.path.join(holdout_dir, item)
        # Check for numeric folders (years) and ensure they are directories
        if os.path.isdir(full_path) and len(item) > 0 and item[0].isdigit():
            folders.append(item)
    
    return sorted(folders)


def get_architectures_for_folder(folder_name):
    """Returns list of available architectures in a folder."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder_path = os.path.join(base_path, 'models', 'holdout_dl', folder_name)
    
    architectures = set()
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.endswith('.pth'):
                # Format: Asset_Horizon_Architecture[_foldX].pth
                try:
                    name_no_ext = f.replace('.pth', '')
                    parts = name_no_ext.split('_')
                    
                    # Last part roughly
                    arch = parts[-1]
                    
                    # Handle Fold Suffix (e.g. Copper_1m_lstm_fold1)
                    if arch.startswith('fold'):
                        if len(parts) >= 2:
                            arch = parts[-2]
                    
                    # Validate arch (basic heuristic: not a horizon, not fold)
                    if arch not in ['1w', '1m', '3m'] and not arch.startswith('fold'):
                        architectures.add(arch)
                except:
                    continue
            
            # Check for Winner Ensemble JSONs
            elif f.startswith('winner_ensemble_') and f.endswith('.json'):
                # extracting generic name e.g. winner_ensemble_accuracy
                # Format: winner_ensemble_{metric}.json
                base_name = f.replace('.json', '')
                architectures.add(base_name)

    archs = list(architectures)
    if archs and 'ensemble' not in archs:
        archs.append('ensemble')  # Always add ensemble option
    return sorted(list(set(archs)))


def generate_asset_combinations(min_assets=3):
    """Generates all asset combinations with minimum size."""
    combos = []
    for r in range(min_assets, len(ALL_ASSETS) + 1):
        for combo in itertools.combinations(ALL_ASSETS, r):
            combos.append(list(combo))
    return combos


def generate_all_configs(quick_mode=False):
    """Generates all configuration permutations."""
    folders = get_available_folders()
    asset_combos = generate_asset_combinations(min_assets=3)
    
    all_configs = []
    
    for folder in folders:
        architectures = get_architectures_for_folder(folder)
        
        for arch in architectures:
            for assets in asset_combos:
                for alloc in ALLOCATIONS:
                    for horizon in HORIZONS:
                        # Standard horizons
                        config = {
                            'folder': folder,
                            'architecture': arch,
                            'assets': assets,
                            'allocation': alloc,
                            'horizon': horizon,
                            **FIXED_CONFIG
                        }
                        all_configs.append(config)

                    # Hybrid horizons: loop over all profiles
                    for profile_id in HYBRID_WEIGHT_PROFILES.keys():
                        config = {
                            'folder': folder,
                            'architecture': arch,
                            'assets': assets,
                            'allocation': alloc,
                            'horizon': 'hybrid',
                            'hybrid_profile': profile_id,
                            **FIXED_CONFIG
                        }
                        all_configs.append(config)
    
    if quick_mode and len(all_configs) > 800:
        # Sample representative subset
        random.seed(42)  # Reproducible
        all_configs = random.sample(all_configs, 800)
    
    return all_configs


def run_single_simulation(config, start_date):
    """Runs a single backtest simulation and returns metrics."""
    from backtesting.engine import VectorizedBacktester
    from backtesting.signal_generator import SignalGenerator
    from backtesting.data_loader import DataLoader
    from backtesting.strategies import RotationalStrategy
    import os
    
    try:
        # Load data
        loader = DataLoader()
        prices = loader.get_asset_prices(config['assets'])
        
        if prices.empty:
            return None
        
        # Filter by start date
        prices = prices[prices.index >= pd.to_datetime(start_date)]
        
        if len(prices) < 50:
            return None
        
        # Build model directory path
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_path, 'models', 'holdout_dl', config['folder'])
        
        # Generate signals using SignalGenerator
        signal_gen = SignalGenerator(model_dir=model_dir)
        
        horizon = config['horizon']

        # Build asset list with horizon for signal generation
        if horizon == 'hybrid':
            assets_with_horizon = []
            for asset in config['assets']:
                assets_with_horizon.extend([f"{asset}_1w", f"{asset}_1m", f"{asset}_3m"])
        else:
            hz = horizon if horizon != 'ensemble' else '1m'  # Default for ensemble
            assets_with_horizon = [f"{asset}_{hz}" for asset in config['assets']]
        
        signals = signal_gen.generate_signals(
            assets_with_horizon,
            start_date=start_date,
            model_type=config['architecture'] if config['architecture'] != 'ensemble' else 'ensemble'
        )
        
        if signals.empty:
            return None
        
        # Simplify / blend signals
        if horizon == 'hybrid':
            # Weighted blend per asset
            profile_id = str(config.get('hybrid_profile', '7'))
            profile = HYBRID_WEIGHT_PROFILES.get(profile_id, HYBRID_WEIGHT_PROFILES['7'])
            config['hybrid_name'] = profile['name']
            weights = profile['weights']
            blended = pd.DataFrame(index=signals.index)
            for asset in config['assets']:
                cols = []
                wts = []
                for h in ['1w', '1m', '3m']:
                    cands = [c for c in signals.columns if c.startswith(f"{asset}_{h}")]
                    if cands:
                        cols.append(cands[0])
                        wts.append(weights.get(h, 0.0))
                if cols:
                    ws = np.array(wts, dtype=float)
                    if ws.sum() > 0:
                        ws = ws / ws.sum()
                        blended[asset] = (signals[cols] * ws).sum(axis=1)
                    else:
                        blended[asset] = signals[cols].mean(axis=1)
            signals = blended
        else:
            # Simplify signal columns to match price columns
            signal_cols = {}
            for col in signals.columns:
                for asset in config['assets']:
                    if col.startswith(asset):
                        signal_cols[col] = asset
                        break
            
            signals = signals.rename(columns=signal_cols)
            signals = signals[[c for c in signals.columns if c in config['assets']]]
        
        if signals.empty:
            return None
        
        # Align signals to prices
        signals = signals.reindex(prices.index).ffill().fillna(0.5)
        
        # Compute asset volatilities (20-day rolling std of returns, annualized)
        returns = prices.pct_change().fillna(0)
        asset_vols = returns.rolling(window=20, min_periods=5).std() * np.sqrt(252)
        asset_vols = asset_vols.fillna(0.2)  # Default 20% vol for early periods
        
        # Create risk dataframe (VIX if available, else dummy)
        risk_df = pd.DataFrame(index=prices.index)
        risk_df['VIX'] = 20.0  # Use neutral VIX since regime filter is OFF
        
        # Apply strategy
        strategy = RotationalStrategy(
            top_n=config['allocation'],
            rebalance_freq=config['rebalance_freq'],
            vol_target=config['volatility_target'],
            use_regime_filter=config['use_regime_filter'],
            min_confidence=config['min_confidence']
        )
        
        weights = strategy.generate_weights(signals, risk_df, asset_vols)
        
        # Run backtest
        backtester = VectorizedBacktester(
            initial_capital=config['initial_capital'],
            transaction_cost_bps=10
        )
        
        results = backtester.run_backtest(prices, weights, trade_threshold=config['trade_threshold'])
        
        return {
            'total_return': results['metrics'].get('total_return', 0),
            'max_drawdown': results['metrics'].get('max_drawdown', 0),
            'sharpe': results['metrics'].get('sharpe', 0),
            'trades': len(results.get('trades', []))
        }
        
    except Exception as e:
        print(f"Simulation error: {e}")
        return None


def rank_by_objective(results, objective):
    """Ranks results by objective and returns top 3."""
    if objective == 'max_return':
        # Sort by total_return descending
        sorted_results = sorted(results, key=lambda x: x['metrics']['total_return'], reverse=True)
    elif objective == 'min_drawdown':
        # Sort by max_drawdown ascending (less negative is better)
        sorted_results = sorted(results, key=lambda x: x['metrics']['max_drawdown'], reverse=True)
    else:  # balanced
        # Sort by return/drawdown ratio (higher is better)
        def balanced_score(r):
            dd = abs(r['metrics']['max_drawdown'])
            ret = r['metrics']['total_return']
            if dd < 0.01:
                dd = 0.01  # Avoid division by zero
            return ret / dd
        sorted_results = sorted(results, key=balanced_score, reverse=True)
    
    return sorted_results[:3]


@combo_bp.route('/api/combo/run')
def api_combo_run():
    """Streams combination grid search progress."""
    quick_mode = request.args.get('quick', 'false').lower() == 'true'
    start_date = request.args.get('start_date', '2019-01-01')
    
    def generate():
        configs = generate_all_configs(quick_mode=quick_mode)
        total = len(configs)
        
        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"
        
        all_results = []
        
        for i, config in enumerate(configs):
            hp = config.get('hybrid_profile')
            hp_str = f"hybrid-{hp}" if config['horizon'] == 'hybrid' else config['horizon']
            config_str = f"{config['folder']}/{config['architecture']}/{'+'.join(config['assets'][:2])}+.../Top{config['allocation']}/{hp_str}"
            
            metrics = run_single_simulation(config, start_date)
            
            if metrics:
                all_results.append({
                    'config': config,
                    'metrics': metrics
                })
            
            progress = int((i + 1) / total * 100)
            
            yield f"data: {json.dumps({'type': 'progress', 'progress': progress, 'current': config_str, 'completed': i+1, 'total': total})}\n\n"
        
        # Rank by objectives
        final_results = {
            'max_return': rank_by_objective(all_results, 'max_return'),
            'min_drawdown': rank_by_objective(all_results, 'min_drawdown'),
            'balanced': rank_by_objective(all_results, 'balanced')
        }
        
        # Save results
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_path = os.path.join(base_path, 'models', 'combo_results.json')
        
        with open(results_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'start_date': start_date,
                'quick_mode': quick_mode,
                'total_tested': len(all_results),
                'results': final_results
            }, f, indent=2, default=str)
        
        # === SAVE TO EXCEL IN comboSim FOLDER ===
        try:
            combo_sim_dir = os.path.join(base_path, 'comboSim')
            os.makedirs(combo_sim_dir, exist_ok=True)
            
            # Create timestamped subfolder
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            run_folder = os.path.join(combo_sim_dir, timestamp)
            os.makedirs(run_folder, exist_ok=True)
            
            # Build Excel data from all_results
            excel_rows = []
            for result in all_results:
                cfg = result['config']
                mtx = result['metrics']
                row = {
                    'Folder': cfg.get('folder', ''),
                    'Architecture': cfg.get('architecture', ''),
                    'Assets': ', '.join(cfg.get('assets', [])),
                    'Num_Assets': len(cfg.get('assets', [])),
                    'Allocation': cfg.get('allocation', ''),
                    'Horizon': cfg.get('horizon', ''),
                    'Hybrid_Profile': cfg.get('hybrid_profile', ''),
                    'Total_Return_Pct': round(mtx.get('total_return', 0) * 100, 2),
                    'Max_Drawdown_Pct': round(mtx.get('max_drawdown', 0) * 100, 2),
                    'Sharpe': round(mtx.get('sharpe', 0), 3),
                    'Trades': mtx.get('trades', 0),
                    'Initial_Capital': cfg.get('initial_capital', 10000),
                    'Vol_Target': cfg.get('volatility_target', 1.0),
                    'Trade_Threshold': cfg.get('trade_threshold', 0.15),
                    'Min_Confidence': cfg.get('min_confidence', 0.5),
                    'Regime_Filter': cfg.get('use_regime_filter', False)
                }
                excel_rows.append(row)
            
            if excel_rows:
                df_excel = pd.DataFrame(excel_rows)
                # Sort by Total Return descending
                df_excel = df_excel.sort_values('Total_Return_Pct', ascending=False)
                
                excel_path = os.path.join(run_folder, f'combo_results_{timestamp}.xlsx')
                df_excel.to_excel(excel_path, index=False, sheet_name='Results')
                print(f"[ComboSim] Results saved to: {excel_path}")
        except Exception as e:
            print(f"[ComboSim] Excel save error: {e}")
        
        yield f"data: {json.dumps({'type': 'done', 'results': final_results})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@combo_bp.route('/api/combo/results')
def api_combo_results():
    """Returns last saved combo results."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = os.path.join(base_path, 'models', 'combo_results.json')
    
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return jsonify(json.load(f))
    
    return jsonify({'error': 'No results found. Run a combination search first.'})


@combo_bp.route('/api/combo/folders')
def api_combo_folders():
    """Returns available holdout folders."""
    return jsonify({'folders': get_available_folders()})
