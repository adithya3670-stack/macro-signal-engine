"""
Model Ensembling Routes
API endpoints for loading holdout metrics and generating winner ensemble configurations.
"""

from flask import Blueprint, jsonify, request
import os
import json
from datetime import datetime

ensembling_bp = Blueprint('ensembling', __name__)

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_DL_DIR = os.path.join(BASE_DIR, 'models', 'holdout_dl')

# Assets and horizons
ASSETS = ['SP500', 'Nasdaq', 'DJIA', 'Russell2000', 'Gold', 'Silver', 'Copper', 'Oil']
HORIZONS = ['1w', '1m', '3m']
MODEL_TYPES = ['lstm', 'transformer', 'nbeats']


@ensembling_bp.route('/api/ensembling/list_holdout_folders')
def api_list_holdout_folders():
    """Returns list of available holdout folders (years)."""
    try:
        if not os.path.exists(HOLDOUT_DL_DIR):
            return jsonify({'status': 'error', 'message': 'Holdout directory not found'}), 404
        
        folders = []
        for item in os.listdir(HOLDOUT_DL_DIR):
            item_path = os.path.join(HOLDOUT_DL_DIR, item)
            if os.path.isdir(item_path):
                # Check if it contains metric files
                has_metrics = any(
                    os.path.exists(os.path.join(item_path, f'dl_metrics_{mt}.json'))
                    for mt in MODEL_TYPES
                )
                folders.append({
                    'name': item,
                    'path': item_path,
                    'has_metrics': has_metrics
                })
        
        # Sort by name (year)
        folders.sort(key=lambda x: x['name'], reverse=True)
        
        return jsonify({
            'status': 'success',
            'folders': folders
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ensembling_bp.route('/api/ensembling/list_winner_configs')
def api_list_winner_configs():
    """Returns list of available winner ensemble configs for portfolio management."""
    try:
        if not os.path.exists(HOLDOUT_DL_DIR):
            return jsonify({'status': 'success', 'configs': []})
        
        configs = []
        for item in os.listdir(HOLDOUT_DL_DIR):
            item_path = os.path.join(HOLDOUT_DL_DIR, item)
            
            if os.path.isdir(item_path):
                # Scan for all winner_ensemble_*.json files
                for f in os.listdir(item_path):
                    if f.startswith('winner_ensemble_') and f.endswith('.json'):
                        config_path = os.path.join(item_path, f)
                        
                        try:
                            # Load config to get metadata
                            with open(config_path, 'r') as cf:
                                config_data = json.load(cf)
                            
                            configs.append({
                                'folder': item,
                                'filename': f,
                                'path': config_path,
                                'metric_used': config_data.get('metric_used', 'unknown'),
                                'generated_at': config_data.get('generated_at', 'unknown'),
                                'winner_count': len(config_data.get('winners', {}))
                            })
                        except Exception as e:
                            print(f"Error loading config {config_path}: {e}")
        
        # Sort by folder name (year) descending, then metric
        configs.sort(key=lambda x: (x['folder'], x['metric_used']), reverse=True)
        
        return jsonify({
            'status': 'success',
            'configs': configs
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ensembling_bp.route('/api/ensembling/scan_folder_architectures')
def api_scan_folder_architectures():
    """
    Scans a holdout folder and returns available model architectures.
    Returns: ensemble, winner_ensemble variants, lstm, transformer, nbeats options.
    """
    folder = request.args.get('folder', None)
    
    if not folder:
        return jsonify({'status': 'error', 'message': 'Folder parameter required'}), 400
    
    # NEW: Handle 'rolling_master' virtual folder
    if folder == 'rolling_master':
        # Auto-detect latest MasterDl folder
        master_dl_dir = os.path.join(BASE_DIR, 'MasterDl')
        if not os.path.exists(master_dl_dir):
             return jsonify({'status': 'error', 'message': 'MasterDl directory not found'}), 404
             
        # Find latest timestamped folder
        subdirs = [os.path.join(master_dl_dir, d) for d in os.listdir(master_dl_dir) 
                  if os.path.isdir(os.path.join(master_dl_dir, d))]
        
        if not subdirs:
            return jsonify({'status': 'error', 'message': 'No MasterDl models found'}), 404
            
        # Sort by creation time or name (name is ISO-like YYYY-MM-DD...)
        latest_folder = max(subdirs, key=os.path.getmtime)
        
        # Point existing logic to this physical path
        folder_path = latest_folder
        
    else:
        # Standard Holdout Logic
        folder_path = os.path.join(HOLDOUT_DL_DIR, folder)
    
    if not os.path.exists(folder_path):
        return jsonify({'status': 'error', 'message': f'Folder not found: {folder}'}), 404
    
    try:
        architectures = []
        
        # Check for each model type - files are named like SP500_1m_lstm.pth
        files_in_folder = os.listdir(folder_path)
        # Check for both standard and folded models
        has_lstm = any(('_lstm.pth' in f or '_lstm_fold' in f) for f in files_in_folder)
        has_transformer = any(('_transformer.pth' in f or '_transformer_fold' in f) for f in files_in_folder)
        has_nbeats = any(('_nbeats.pth' in f or '_nbeats_fold' in f) for f in files_in_folder)
        
        # Build options list
        if has_lstm or has_transformer or has_nbeats:
            architectures.append({
                'value': 'ensemble',
                'label': 'Ensemble (All Models)',
                'available': True
            })
        
        # Check for winner ensemble configs (multiple)
        found_new_winner_configs = False
        for f in files_in_folder:
            if f.startswith('winner_ensemble_') and f.endswith('.json'):
                found_new_winner_configs = True
                try:
                    with open(os.path.join(folder_path, f), 'r') as cf:
                        config = json.load(cf)
                    metric = config.get('metric_used', 'unknown')
                    architectures.append({
                        'value': f'winner_ensemble_{metric}', # Unique value
                        'label': f'🏆 Winner Ensemble ({metric})',
                        'available': True,
                        'metric': metric,
                        'filename': f
                    })
                except:
                    pass
        
        # Also check for legacy 'winner_ensemble_config.json' if no new ones found?
        # Or just support it as valid
        legacy_path = os.path.join(folder_path, 'winner_ensemble_config.json')
        if os.path.exists(legacy_path) and not found_new_winner_configs:
             architectures.append({
                'value': 'winner_ensemble',
                'label': '🏆 Winner Ensemble (Legacy)',
                'available': True,
                'metric': 'unknown',
                'filename': 'winner_ensemble_config.json'
            })
        
        if has_lstm:
            architectures.append({
                'value': 'lstm',
                'label': 'LSTM Only',
                'available': True
            })
        
        if has_transformer:
            architectures.append({
                'value': 'transformer',
                'label': 'Transformer Only',
                'available': True
            })
        
        if has_nbeats:
            architectures.append({
                'value': 'nbeats',
                'label': 'N-BEATS Only',
                'available': True
            })
        
        return jsonify({
            'status': 'success',
            'folder': folder,
            'architectures': architectures
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ensembling_bp.route('/api/ensembling/load_metrics')
def api_load_metrics():
    """Loads dl_metrics_*.json files from the specified holdout folder."""
    folder = request.args.get('folder', None)
    
    if not folder:
        return jsonify({'status': 'error', 'message': 'Folder parameter required'}), 400
    
    folder_path = os.path.join(HOLDOUT_DL_DIR, folder)
    
    if not os.path.exists(folder_path):
        return jsonify({'status': 'error', 'message': f'Folder not found: {folder}'}), 404
    
    try:
        all_metrics = {}
        
        for model_type in MODEL_TYPES:
            metrics_file = os.path.join(folder_path, f'dl_metrics_{model_type}.json')
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Convert list to dict keyed by asset_horizon
                for entry in metrics_data:
                    asset = entry['asset']
                    horizon = entry['horizon']
                    key = f"{asset}_{horizon}"
                    
                    if key not in all_metrics:
                        all_metrics[key] = {
                            'asset': asset,
                            'horizon': horizon
                        }
                    
                    all_metrics[key][model_type] = entry['metrics']
        
        # Convert to list for frontend
        metrics_list = list(all_metrics.values())
        
        # Sort by asset then horizon
        horizon_order = {'1w': 0, '1m': 1, '3m': 2}
        metrics_list.sort(key=lambda x: (x['asset'], horizon_order.get(x['horizon'], 99)))
        
        return jsonify({
            'status': 'success',
            'folder': folder,
            'metrics': metrics_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@ensembling_bp.route('/api/ensembling/generate_winner_ensemble', methods=['POST'])
def api_generate_winner_ensemble():
    """
    Generates winner ensemble config based on selected metric.
    Saves to folder/winner_ensemble_{metric}.json.
    """
    data = request.get_json() or {}
    folder = data.get('folder')
    metric = data.get('metric', 'accuracy')  # accuracy, precision, robustness
    
    if not folder:
        return jsonify({'status': 'error', 'message': 'Folder parameter required'}), 400
    
    folder_path = os.path.join(HOLDOUT_DL_DIR, folder)
    
    if not os.path.exists(folder_path):
        return jsonify({'status': 'error', 'message': f'Folder not found: {folder}'}), 404
    
    try:
        # Load all metrics
        all_model_metrics = {}
        
        for model_type in MODEL_TYPES:
            metrics_file = os.path.join(folder_path, f'dl_metrics_{model_type}.json')
            
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                for entry in metrics_data:
                    asset = entry['asset']
                    horizon = entry['horizon']
                    key = f"{asset}_{horizon}"
                    
                    if key not in all_model_metrics:
                        all_model_metrics[key] = {}
                    
                    all_model_metrics[key][model_type] = entry['metrics']
        
        # Calculate scores and pick winners
        winners = {}
        
        for key, models in all_model_metrics.items():
            best_model = None
            best_score = -1
            
            for model_type, metrics in models.items():
                if metric == 'accuracy':
                    score = metrics.get('accuracy', 0)
                elif metric == 'precision':
                    score = metrics.get('precision', 0)
                elif metric == 'robustness':
                    # Composite: accuracy * 0.4 + precision * 0.3 + auc * 0.3
                    acc = metrics.get('accuracy', 0)
                    prec = metrics.get('precision', 0)
                    auc = metrics.get('auc', 0)
                    score = (acc * 0.4) + (prec * 0.3) + (auc * 0.3)
                else:
                    score = metrics.get('accuracy', 0)
                
                if score > best_score:
                    best_score = score
                    best_model = model_type
            
            winners[key] = {
                'model': best_model,
                'score': round(best_score, 4),
                'metric_used': metric
            }
        
        # Create config
        config = {
            'generated_at': datetime.now().isoformat(),
            'folder': folder,
            'metric_used': metric,
            'winners': winners
        }
        
        # Save to file with metric component
        filename = f'winner_ensemble_{metric}.json'
        config_path = os.path.join(folder_path, filename)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[ENSEMBLING] Saved winner config to: {config_path}")
        
        return jsonify({
            'status': 'success',
            'config_path': config_path,
            'config': config
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
