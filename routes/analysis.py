from flask import Blueprint, jsonify, request
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from services.data_service import (
    CACHE, load_from_cache_or_csv, refresh_data, clean_for_json
)
from config.settings import (
    EXTRACTED_DATA_FILE, ENGINEERED_FEATURES_FILE, MASTER_DATA_FILE, METRICS_FILE
)
from analysis.correlations import calculate_correlations, get_latest_drivers
from analysis.feature_engine import FeatureEngineer
from analysis.price_3d_regression import Price3DRegressionManager
from analysis.minor_3d_signal_manager import Minor3DSignalManager
from analysis.regime_predictability import RegimePredictabilityManager
from analysis.dl.inference import DeepLearningInferenceService
from backend.domain.dto import SnapshotCreateRequest
from backend.services.price_pipeline_registry import PricePipelineRegistry
from backend.services.model_snapshot_service import ModelSnapshotService
from backend.shared.http import error_payload, error_status
from data.etl import load_and_merge_data

analysis_bp = Blueprint('analysis', __name__)
snapshot_service = ModelSnapshotService(cache_ref=CACHE)
price_pipelines = PricePipelineRegistry()


def _get_price_3d_pipeline():
    """
    Returns the shared 3D adapter while preserving legacy route patch tests.
    """
    pipeline = price_pipelines.get("3d")
    if hasattr(pipeline, "manager"):
        try:
            pipeline.manager = Price3DRegressionManager()
        except Exception:
            pass
    return pipeline

@analysis_bp.route('/api/dashboard')
def api_dashboard():
    try:
        # Try to load from memory or disk
        data_available = load_from_cache_or_csv()
        
        if not data_available:
            return jsonify({
                'chart_data': [],
                'latest_drivers': {'Short Term (30d)': {'driver': 'No Data', 'correlation': 0},
                                   'Medium Term (90d)': {'driver': 'No Data', 'correlation': 0},
                                   'Long Term (1y)': {'driver': 'No Data', 'correlation': 0}},
                'correlations': {'Short Term (30d)': {}, 'Medium Term (90d)': {}, 'Long Term (1y)': {}}
            })
            
        df = CACHE['data']
        correlations = CACHE['correlations']
        drivers = CACHE['drivers']
        
        # Prepare data for JSON
        chart_df = df.reset_index().rename(columns={'index': 'Date'})
        if 'Date' not in chart_df.columns:
             chart_df['Date'] = chart_df.iloc[:, 0]
             
        chart_df['Date'] = chart_df['Date'].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
        chart_data = chart_df.to_dict(orient='records')
        
        corr_json = {}
        for term, corr_df in correlations.items():
            if isinstance(corr_df, pd.DataFrame):
                if not corr_df.empty:
                    corr_json[term] = corr_df.iloc[-1].to_dict()
                else:
                    corr_json[term] = {}
            else:
                 corr_json[term] = {}
            
        raw_response = {
            'chart_data': chart_data,
            'latest_drivers': drivers,
            'correlations': corr_json
        }
        return jsonify(clean_for_json(raw_response))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@analysis_bp.route('/api/refresh', methods=['POST'])
def api_refresh():
    data = request.get_json()
    start_str = data.get('start_date')
    end_str = data.get('end_date')
    use_latest = data.get('use_latest', False)
    
    start_date = datetime.strptime(start_str, '%Y-%m-%d') if start_str else None
    
    end_date = None
    if use_latest:
        end_date = datetime.now()
    elif end_str:
        end_date = datetime.strptime(end_str, '%Y-%m-%d')
        
    refresh_data(start_date, end_date)
    return jsonify({'status': 'success'})

@analysis_bp.route('/api/master_update', methods=['POST'])
def api_master_update():
    print("Starting Master Update (2008-Present)...")
    
    start_date = datetime(2008, 1, 1)
    end_date = datetime.now()
    
    if not os.path.exists('master_data'):
        os.makedirs('master_data')
        
    df = load_and_merge_data(start_date, end_date)
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    df.to_csv(MASTER_DATA_FILE)
    print(f"Master Data saved to {MASTER_DATA_FILE}")
    
    # Reload Cache
    correlations = calculate_correlations(df)
    drivers = get_latest_drivers(df, correlations)
    
    # Clean correlations
    for term, corr_df in correlations.items():
        correlations[term] = corr_df.replace([np.inf, -np.inf], 0).fillna(0)

    CACHE['data'] = df
    CACHE['correlations'] = correlations
    CACHE['drivers'] = drivers
    
    df.to_csv(EXTRACTED_DATA_FILE)
    
    return jsonify({'status': 'success', 'message': f'Master data saved to {MASTER_DATA_FILE}'})

@analysis_bp.route('/api/update_features', methods=['POST'])
def api_update_features():
    if not os.path.exists(MASTER_DATA_FILE):
        return jsonify({'status': 'error', 'message': 'Master dataset not found. Run Master Update first.'}), 404
        
    try:
        print(f"Loading Master Data from {MASTER_DATA_FILE}...")
        df = pd.read_csv(MASTER_DATA_FILE)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        # Clean
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
        
        print("Running Feature Engineer...")
        engineer = FeatureEngineer(df)
        features_df = engineer.generate_features()
        
        features_df.to_csv(ENGINEERED_FEATURES_FILE)
        print(f"Features saved to {ENGINEERED_FEATURES_FILE}")
        
        return jsonify({
            'status': 'success', 
            'message': 'Features generated successfully',
            'columns': features_df.columns.tolist()
        })
    except Exception as e:
        print(f"Feature Generation Failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/feature_data')
def api_feature_data():
    feature_path = ENGINEERED_FEATURES_FILE
    if not os.path.exists(feature_path):
        return jsonify({'status': 'error', 'message': 'Feature data not found.'}), 404
        
    try:
        df = pd.read_csv(feature_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        df = df.replace([np.inf, -np.inf], 0).fillna(0)
        
        assets = {}
        for target in ['SP500', 'Nasdaq', 'Gold', 'Silver', 'Oil']:
            if target in df.columns:
                assets[target] = df[target].tolist()
                
        response_data = {
            'dates': df.index.strftime('%Y-%m-%d').tolist(),
            'assets': assets,
            'all_columns': df.columns.tolist(),
            'data': df.to_dict(orient='list')
        }
        return jsonify(clean_for_json(response_data))
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/analysis/predictive_correlations')
def api_predictive_correlations():
    feature_path = ENGINEERED_FEATURES_FILE
    if not os.path.exists(feature_path):
         return jsonify({'status': 'error', 'message': 'Run Feature Engine first.'}), 404
         
    try:
        df = pd.read_csv(feature_path)
        target_cols = [c for c in df.columns if c.startswith('Target_')]
        
        exclude_keywords = ['Target_', 'Date', 'Unnamed', 'Open', 'High', 'Low', 'Close', 'Volume']
        feature_cols = [c for c in df.columns if not any(k in c for k in exclude_keywords)]
        
        priority_features = [
            'Liquidity_Impulse', 'Real_Yield', 'Curve_Steepening', 'CPI_YoY', 'PPI_YoY',
            'Bond_Stock_Corr', 'VIX_Regime', 'Tech_vs_Broad', 'Silver_Gold_Ratio', 
            'SP500_ZScore', 'Gold_ZScore', 'VIX_ZScore', 'DGS10_ZScore'
        ]
        selected_features = [f for f in priority_features if f in df.columns]
        if not selected_features: selected_features = feature_cols[:20]
            
        correlation_matrix = df[selected_features + target_cols].corr()
        
        heatmap_data = []
        for feature in selected_features:
            row_data = {'feature': feature}
            for target in target_cols:
                val = correlation_matrix.loc[feature, target]
                row_data[target] = 0 if np.isnan(val) else val
            heatmap_data.append(row_data)
            
        return jsonify({
            'status': 'success',
            'targets': target_cols,
            'features': selected_features,
            'matrix': heatmap_data
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/analysis/regime_performance')
def api_regime_performance():
    feature_path = ENGINEERED_FEATURES_FILE
    if not os.path.exists(feature_path):
        return jsonify({'status': 'error', 'message': 'Run Feature Engine first.'}), 404
        
    try:
        df = pd.read_csv(feature_path)
        regime_cols = ['Regime_Inflation', 'Regime_Liquidity', 'Regime_Risk', 'Regime_Rates']
        targets = {
            'S&P 500': 'Target_SP500_1m',
            'Nasdaq': 'Target_Nasdaq_1m',
            'Gold': 'Target_Gold_1m',
        }
        
        results = {}
        for regime in regime_cols:
            if regime not in df.columns: continue
            dataset = []
            labels = df[regime].dropna().unique().tolist()
            for label in labels:
                subset = df[df[regime] == label]
                data_point = {'category': label}
                for asset_name, target_col in targets.items():
                    if target_col in df.columns:
                        avg_ret = subset[target_col].mean()
                        data_point[asset_name] = round(avg_ret, 2) if pd.notnull(avg_ret) else 0
                dataset.append(data_point)
            results[regime] = dataset
            
        return jsonify({'status': 'success', 'regimes': results})
    except Exception as e:
         return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/analysis/rolling_correlations')
def api_rolling_correlations():
    feature_path = ENGINEERED_FEATURES_FILE
    asset = request.args.get('asset', 'SP500')
    feature = request.args.get('feature', 'Liquidity_Impulse')
    window = int(request.args.get('window', 90))
    
    if not os.path.exists(feature_path):
        return jsonify({'status': 'error', 'message': 'Run Feature Engine first.'}), 404
        
    try:
        df = pd.read_csv(feature_path)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        if asset not in df.columns or feature not in df.columns:
             return jsonify({'status': 'error', 'message': 'Columns not found'}), 400
             
        rolling_corr = df[asset].rolling(window).corr(df[feature])
        valid_data = rolling_corr.dropna()
        
        return jsonify({
            'status': 'success',
            'dates': valid_data.index.strftime('%Y-%m-%d').tolist(),
            'values': valid_data.values.tolist(),
            'asset': asset,
            'feature': feature,
            'window': window
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/analysis/forecast')
def api_forecast():
    # Replaced ML Builder with DL Model for default forecast
    from analysis.deep_learning_model import DLMacroModel
    try:
        dl = DLMacroModel()
        predictions = dl.predict_latest()
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/update_latest', methods=['POST'])
def api_update_latest():
    print("Updating Master Data to Latest...")
    
    if not os.path.exists(MASTER_DATA_FILE):
        return jsonify({'status': 'error', 'message': 'Master dataset not found. Run Master Update first.'}), 404
        
    try:
        master_df = pd.read_csv(MASTER_DATA_FILE)
        if 'Date' in master_df.columns:
            try:
                master_df['Date'] = pd.to_datetime(master_df['Date'])
            except Exception:
                master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
            master_df.set_index('Date', inplace=True)
        elif 'Unnamed: 0' in master_df.columns:
            master_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
            try:
                master_df['Date'] = pd.to_datetime(master_df['Date'])
            except Exception:
                master_df['Date'] = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce')
            master_df.set_index('Date', inplace=True)
            
        master_df = master_df[master_df.index.notnull()]
            
        last_date = master_df.index.max()
        if pd.isna(last_date):
             return jsonify({'status': 'error', 'message': 'Master dataset has no valid dates.'}), 500
             
        now = datetime.now()
        if last_date.date() >= now.date():
            return jsonify({'status': 'success', 'message': 'Data is already up to date.'})
            
        fetch_start = last_date - timedelta(days=60)
        print(f"Fetching updates from {fetch_start} to {now}...")
        
        new_df = load_and_merge_data(fetch_start, now)
        
        cutoff_mask = master_df.index < fetch_start
        trimmed_master = master_df[cutoff_mask]
        
        updated_df = pd.concat([trimmed_master, new_df])
        updated_df = updated_df.ffill()
        updated_df = updated_df.replace([np.inf, -np.inf], 0).fillna(0)
        updated_df = updated_df.sort_index()
        
        updated_df.to_csv(MASTER_DATA_FILE)
        print(f"Updated Master Data saved to {MASTER_DATA_FILE}")
        
        correlations = calculate_correlations(updated_df)
        drivers = get_latest_drivers(updated_df, correlations)
        
        for term, corr_df in correlations.items():
            correlations[term] = corr_df.replace([np.inf, -np.inf], 0).fillna(0)
            
        CACHE['data'] = updated_df
        CACHE['correlations'] = correlations
        CACHE['drivers'] = drivers
        
        return jsonify({'status': 'success', 'message': f'Updated data to {now.strftime("%Y-%m-%d")}'})
        
    except Exception as e:
        print(f"Update failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@analysis_bp.route('/api/data/range', methods=['GET'])
def api_data_range():
    """Returns the start and end date of the master dataset."""
    if not load_from_cache_or_csv():
        return jsonify({'error': 'Data not available'}), 404
    
    df = CACHE['data']
    if df is None or df.empty:
        return jsonify({'error': 'Dataset empty'}), 404
        
    return jsonify({
        'min_date': df.index.min().strftime('%Y-%m-%d'),
        'max_date': df.index.max().strftime('%Y-%m-%d')
    })

@analysis_bp.route('/api/model_info')
def api_model_info():
    import json as json_lib
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, 'r') as f:
            data = json_lib.load(f)
            if isinstance(data, dict) and 'train_end_date' in data:
                return jsonify({
                    'train_end_date': data['train_end_date'],
                    'model_count': len(data.get('models', []))
                })
            else:
                return jsonify({
                    'train_end_date': 'Unknown (retrain models)',
                    'model_count': len(data) if isinstance(data, list) else 0
                })
    return jsonify({'train_end_date': 'No models trained', 'model_count': 0})

@analysis_bp.route('/api/deep_forecast')
def api_deep_forecast():
    try:
        model_type = request.args.get('model', 'lstm')
        inference_service = DeepLearningInferenceService()
        predictions = inference_service.predict_latest(model_type=model_type)
        return jsonify({'status': 'success', 'predictions': predictions})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


def _snapshot_create_payload():
    data = request.get_json(silent=True) or {}
    tag = data.get('tag') or data.get('name') or 'Snapshot'
    description = data.get('description', '')
    return SnapshotCreateRequest(tag=tag, name=tag, description=description)


def _create_snapshot_response():
    created = snapshot_service.create_snapshot(_snapshot_create_payload())
    return jsonify({'status': 'success', **created})


def _list_snapshots_response():
    snapshots = snapshot_service.list_snapshots()
    return jsonify(snapshots)


def _restore_snapshot_response():
    data = request.get_json(silent=True) or {}
    snapshot_id = data.get('id')
    if not snapshot_id:
        return jsonify({'error': 'Snapshot id is required.'}), 400
    restored = snapshot_service.restore_snapshot(str(snapshot_id))
    return jsonify({'status': 'success', 'id': restored.get('id'), 'snapshot': restored})


@analysis_bp.route('/api/create_snapshot', methods=['POST'])
def api_create_snapshot():
    try:
        return _create_snapshot_response()
    except Exception as e:
        return jsonify(error_payload(e)), error_status(e)


@analysis_bp.route('/api/list_snapshots')
def api_list_snapshots():
    try:
        return _list_snapshots_response()
    except Exception as e:
        return jsonify(error_payload(e)), error_status(e)


@analysis_bp.route('/api/models/snapshot', methods=['POST'])
def api_models_snapshot_alias():
    try:
        return _create_snapshot_response()
    except Exception as e:
        return jsonify(error_payload(e)), error_status(e)


@analysis_bp.route('/api/models/snapshots', methods=['GET'])
def api_models_snapshots_alias():
    try:
        return _list_snapshots_response()
    except Exception as e:
        return jsonify(error_payload(e)), error_status(e)


@analysis_bp.route('/api/models/restore', methods=['POST'])
def api_models_restore_alias():
    try:
        return _restore_snapshot_response()
    except Exception as e:
        return jsonify(error_payload(e)), error_status(e)


@analysis_bp.route('/api/update_price_3d_features', methods=['POST'])
def api_update_price_3d_features():
    try:
        pipeline = _get_price_3d_pipeline()
        result = pipeline.refresh_feature_cache()
        return jsonify({'status': 'success', 'result': result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/price_3d/metrics')
def api_price_3d_metrics():
    try:
        pipeline = _get_price_3d_pipeline()
        source = request.args.get('source', 'holdout').lower()
        year = request.args.get('year', None)
        holdout_year = int(year) if year and year.isdigit() else None
        model_dir = None
        if source == 'production':
            model_dir = getattr(getattr(pipeline, 'manager', None), 'model_dir', None)
        payload = pipeline.load_metrics(
            holdout_year=holdout_year if source != 'production' else None,
            model_dir=model_dir
        )
        return jsonify({'status': 'success', **clean_for_json(payload)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/price_3d/promote_champions', methods=['POST'])
def api_price_3d_promote_champions():
    try:
        pipeline = _get_price_3d_pipeline()
        mgr = getattr(pipeline, 'manager', Price3DRegressionManager())
        payload = request.get_json(silent=True) or {}
        year = payload.get('year', request.args.get('year'))
        holdout_year = int(year) if year and str(year).isdigit() else None
        result = mgr.promote_champions(holdout_year=holdout_year)
        return jsonify({'status': 'success', **clean_for_json(result)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/price_3d/predict_latest')
def api_price_3d_predict_latest():
    try:
        pipeline = _get_price_3d_pipeline()
        source = request.args.get('source', 'production').lower()
        year = request.args.get('year', None)
        holdout_year = int(year) if year and year.isdigit() else None
        if source == 'holdout':
            predictions = pipeline.predict_latest(year=holdout_year)
        else:
            predictions = pipeline.predict_latest()
        return jsonify({'status': 'success', 'predictions': clean_for_json(predictions)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/rebuild', methods=['POST'])
def api_regime_rebuild():
    def as_bool(value, default=False):
        if value is None:
            return default
        return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}

    try:
        mgr = RegimePredictabilityManager()
        payload = request.get_json(silent=True) or {}
        year = payload.get('year', request.args.get('year'))
        source = payload.get('source', request.args.get('source', 'production'))
        refresh_arg = payload.get('refresh_features', request.args.get('refresh_features', False))
        result = mgr.rebuild(
            year=int(year) if year and str(year).isdigit() else None,
            source=source,
            refresh_features=as_bool(refresh_arg, default=False),
        )
        return jsonify({'status': 'success', **clean_for_json(result)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/state_latest')
def api_regime_state_latest():
    try:
        mgr = RegimePredictabilityManager()
        result = mgr.load_state_latest()
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/predictability')
def api_regime_predictability():
    try:
        mgr = RegimePredictabilityManager()
        horizon = request.args.get('horizon', None)
        asset = request.args.get('asset', None)
        group = request.args.get('group', None)
        result = mgr.load_predictability(horizon=horizon, asset=asset, group=group)
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/forecast_latest')
def api_regime_forecast_latest():
    try:
        mgr = RegimePredictabilityManager()
        result = mgr.load_forecast_latest()
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/policy_latest')
def api_regime_policy_latest():
    try:
        mgr = RegimePredictabilityManager()
        result = mgr.load_policy_latest()
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/regime/predict_latest')
def api_regime_predict_latest():
    def as_bool(value, default=False):
        if value is None:
            return default
        return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}

    try:
        mgr = RegimePredictabilityManager()
        force_rebuild = as_bool(request.args.get('force_rebuild', False), default=False)
        source = request.args.get('source', 'production')
        year = request.args.get('year', None)
        result = mgr.predict_latest(
            force_rebuild=force_rebuild,
            source=source,
            year=int(year) if year and str(year).isdigit() else None,
        )
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/minor_3d/rebuild', methods=['POST'])
def api_minor_3d_rebuild():
    try:
        mgr = Minor3DSignalManager()
        payload = request.get_json(silent=True) or {}
        precision_floor = payload.get('precision_floor', request.args.get('precision_floor'))
        cooldown = payload.get('cooldown', request.args.get('cooldown'))
        seed = payload.get('seed', request.args.get('seed'))
        warmup = payload.get('warmup', request.args.get('warmup'))
        rule_search_samples = payload.get('rule_search_samples', request.args.get('rule_search_samples'))

        result = mgr.rebuild(
            precision_floor=float(precision_floor) if precision_floor is not None else None,
            cooldown=int(cooldown) if cooldown is not None else None,
            seed=int(seed) if seed is not None else None,
            warmup=int(warmup) if warmup is not None else None,
            rule_search_samples=int(rule_search_samples) if rule_search_samples is not None else None,
        )
        return jsonify({'status': 'success', **clean_for_json(result)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/minor_3d/policy_latest')
def api_minor_3d_policy_latest():
    def as_bool(value, default=False):
        if value is None:
            return default
        return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}

    try:
        mgr = Minor3DSignalManager()
        auto_refresh = as_bool(request.args.get('auto_refresh', True), default=True)
        precision_floor = request.args.get('precision_floor', None)
        result = mgr.load_policy(
            auto_refresh=auto_refresh,
            precision_floor=float(precision_floor) if precision_floor is not None else None,
        )
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@analysis_bp.route('/api/minor_3d/status_latest')
def api_minor_3d_status_latest():
    def as_bool(value, default=False):
        if value is None:
            return default
        return str(value).lower() in {'1', 'true', 'yes', 'y', 'on'}

    try:
        mgr = Minor3DSignalManager()
        auto_refresh = as_bool(request.args.get('auto_refresh', True), default=True)
        precision_floor = request.args.get('precision_floor', None)
        result = mgr.load_latest_status(
            auto_refresh=auto_refresh,
            precision_floor=float(precision_floor) if precision_floor is not None else None,
        )
        return jsonify(clean_for_json(result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500
