from flask import Blueprint, jsonify, request, Response, stream_with_context, copy_current_request_context
import json
import threading
import queue
import os
import pandas as pd
from analysis.deep_learning_model import DLMacroModel
from analysis.price_3d_regression import Price3DRegressionManager
from analysis.dl.hyperparameter_search import DeepLearningHyperparameterSearch
from analysis.dl.training_orchestrator import DeepLearningTrainingOrchestrator
from backend.services.price_pipeline_registry import PricePipelineRegistry

training_bp = Blueprint('training', __name__)
price_pipeline_registry = PricePipelineRegistry()

# Store the prepared holdout folder path (used by training endpoint)
_prepared_holdout = {'folder': None, 'year': None}


def _get_price_3d_pipeline():
    """
    Returns the shared 3D pipeline adapter while preserving legacy test patchability.
    Tests patch routes.training.Price3DRegressionManager and expect route usage.
    """
    pipeline = price_pipeline_registry.get("3d")
    if hasattr(pipeline, "manager"):
        try:
            pipeline.manager = Price3DRegressionManager()
        except Exception:
            pass
    return pipeline


@training_bp.route('/api/prepare_holdout')
def api_prepare_holdout():
    """Step 1: Create holdout folder and save split info BEFORE training starts"""
    global _prepared_holdout
    
    year = request.args.get('year', None)
    
    if not year or not year.isdigit():
        return jsonify({'error': 'Valid year required (e.g., 2023)'}), 400
    
    try:
        # Create folder
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        holdout_dir = os.path.join(base_path, 'models', 'holdout_dl', year)
        os.makedirs(holdout_dir, exist_ok=True)
        print(f"[PREPARE] Created holdout folder: {holdout_dir}")
        cutoff_date = f"{year}-12-31"

        # Default split payload so endpoint stays usable even when data loaders fail.
        split_info = {
            'cutoff_year': year,
            'cutoff_date': cutoff_date,
            'train_start': 'N/A',
            'train_end': 'N/A',
            'train_rows': 0,
            'test_start': 'N/A',
            'test_end': 'N/A',
            'test_rows': 0
        }
        warning = None

        try:
            dl = DLMacroModel()
            df = dl.load_and_preprocess()

            train_df = df.loc[:cutoff_date]
            test_df = df.loc[cutoff_date:]

            split_info.update({
                'train_start': str(train_df.index.min().date()) if len(train_df) > 0 else 'N/A',
                'train_end': str(train_df.index.max().date()) if len(train_df) > 0 else 'N/A',
                'train_rows': len(train_df),
                'test_start': str(test_df.index.min().date()) if len(test_df) > 0 else 'N/A',
                'test_end': str(test_df.index.max().date()) if len(test_df) > 0 else 'N/A',
                'test_rows': len(test_df)
            })
        except Exception as data_err:
            warning = f"Holdout folder prepared, but split analysis failed: {data_err}"
            print(f"[PREPARE][WARN] {warning}")
        
        # Save split metadata (Excel when available; CSV fallback keeps endpoint usable)
        split_df = pd.DataFrame([split_info])
        excel_path = os.path.join(holdout_dir, 'train_test_split.xlsx')
        csv_path = os.path.join(holdout_dir, 'train_test_split.csv')

        split_file = csv_path
        try:
            split_df.to_excel(excel_path, index=False)
            split_file = excel_path
            print(f"[PREPARE] Saved split info: {excel_path}")
        except Exception as e:
            split_df.to_csv(csv_path, index=False)
            split_file = csv_path
            print(f"[PREPARE] Excel export unavailable ({e}). Saved CSV split info instead: {csv_path}")
        
        # Store for training endpoint
        _prepared_holdout['folder'] = holdout_dir
        _prepared_holdout['year'] = year
        
        payload = {
            'success': True,
            'folder': holdout_dir,
            'split_file': split_file,
            'split_info': split_info
        }
        if warning:
            payload['warning'] = warning
        return jsonify(payload)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@training_bp.route('/api/train/deep_learning_stream')
def api_train_deep_learning_stream():
    """Streams Deep Learning training progress using Server-Sent Events"""
    model_type = request.args.get('model', 'lstm')
    
    def generate():
        trainer = DeepLearningTrainingOrchestrator()
        optimizer = DeepLearningHyperparameterSearch()
        q = queue.Queue()
        
        # Worker thread
        def worker():
             try:
                 def q_callback(pct, msg):
                     q.put({"type": "data", "pct": pct, "msg": msg})
                 
                 # Parse params
                 epochs = int(request.args.get('epochs', 50))
                 
                 # Check if holdout was prepared (step-by-step workflow)
                 global _prepared_holdout
                 if _prepared_holdout['folder'] and os.path.exists(_prepared_holdout['folder']):
                     cutoff = f"{_prepared_holdout['year']}-12-31"
                     print(f"[TRAIN] Using prepared holdout folder: {_prepared_holdout['folder']}")
                     print(f"[TRAIN] Cutoff date: {cutoff}")
                 else:
                     # Fallback: Try 'year' parameter directly
                     year_param = request.args.get('year', None)
                     cutoff = request.args.get('cutoff_date', None)
                     
                     if year_param and year_param.isdigit():
                         cutoff = f"{year_param}-12-31"
                         print(f"[TRAIN] Using 'year' param -> cutoff = {cutoff}")
                     elif cutoff == "" or cutoff == "None" or cutoff is None:
                         cutoff = None
                         print(f"[TRAIN] No cutoff provided, using models_dl")
                     else:
                         print(f"[TRAIN] cutoff_date from request = '{cutoff}'")
                 
                 # 1. OPTIMIZATION (30%)
                 # ---------------------------------------------------------
                 def opt_cb(pct, msg):
                     # Map 0-100 -> 0-30
                     overall = pct * 0.3
                     q_callback(overall, f"[OPT] {msg}")
                
                 q_callback(0, f"Starting {model_type.upper()} Optimization...")
                 # Iterate 10 times to find best hyperparameters (Using Cutoff if provided)
                 optimizer.run(model_type=model_type, iterations=10, progress_callback=opt_cb, train_cutoff_date=cutoff)

                 # 2. DISCOVERY (Ensemble Validation) (40%)
                 # ---------------------------------------------------------
                 def ens_cb(pct, msg):
                     # Map 0-100 -> 30-70
                     overall = 30 + (pct * 0.4)
                     q_callback(overall, f"[DISC] {msg}")

                 q_callback(30, f"Training Discovery Ensemble (Findings Optimal Epochs)...")
                 # Train 1 Smart Fold per asset (Discovery Mode)
                 discovery_results = trainer.train_all(
                     model_type=model_type, 
                     epochs=epochs, 
                     train_cutoff_date=cutoff, 
                     use_bagging_ensemble=True,
                     n_folds=1, 
                     progress_callback=ens_cb
                 )
                 
                 # Extract Optimal Epochs from Discovery
                 manual_epochs = {}
                 for res in discovery_results:
                     key = f"{res['asset']}_{res['horizon']}_{model_type}"
                     found_epochs = res['metrics'].get('optimal_epochs', epochs)
                     manual_epochs[key] = found_epochs

                 # 3. PRODUCTION (Full Ensemble) (30%)
                 # ---------------------------------------------------------
                 def prod_cb(pct, msg):
                     # Map 0-100 -> 70-100
                     overall = 70 + (pct * 0.3)
                     q_callback(overall, f"[PROD] {msg}")
                
                 q_callback(70, f"Final Production Training (Gap Bagging Ensemble)...")
                 trainer.train_all(
                     model_type=model_type, 
                     epochs=epochs, 
                     train_cutoff_date=cutoff, 
                     use_bagging_ensemble=True,  # <--- Changed to True (3 Models)
                     n_folds=3,                  # Ensure 3 folds
                     force_full_training=True,   # Enforce Fixed Epochs (no early stop)
                     manual_epochs_dict=manual_epochs,
                     progress_callback=prod_cb
                 )
                 
                 q.put({"type": "done"})
             except Exception as e:
                 import traceback
                 traceback.print_exc()
                 q.put({"type": "error", "error": str(e)})

        # Start thread
        t = threading.Thread(target=copy_current_request_context(worker), daemon=True)
        t.start()
        
        # Main Loop (Generator)
        while True:
            try:
                # Wait 1s for data, else Ping
                item = q.get(timeout=1.0)
                
                if item['type'] == 'data':
                    data = json.dumps({'progress': item['pct'], 'message': item['msg']})
                    yield f"data: {data}\n\n"
                elif item['type'] == 'done':
                    yield f"data: {json.dumps({'progress': 100, 'message': 'Training Complete!'})}\n\n"
                    yield "data: DONE\n\n"
                    break
                elif item['type'] == 'error':
                    yield f"data: {json.dumps({'error': item['error']})}\n\n"
                    break
            except queue.Empty:
                # Heartbeat
                yield ": ping\n\n"
            except Exception as e:
                print(f"Stream Error: {e}")
                
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@training_bp.route('/api/optimize/deep_learning_stream')
def api_optimize_deep_learning_stream():
    """Streams Deep Learning OPTIMIZATION using Server-Sent Events"""
    model_type = request.args.get('model', 'lstm')
    
    def generate():
        optimizer = DeepLearningHyperparameterSearch()
        q = queue.Queue()
        
        def worker():
             try:
                 def q_callback(pct, msg):
                     q.put({"type": "data", "pct": pct, "msg": msg})
                 
                 # Optimization Loop (Slow)
                 optimizer.run(model_type=model_type, iterations=10, progress_callback=q_callback)
                 q.put({"type": "done"})
             except Exception as e:
                 import traceback
                 traceback.print_exc()
                 q.put({"type": "error", "error": str(e)})

        t = threading.Thread(target=copy_current_request_context(worker), daemon=True)
        t.start()
        
        while True:
            try:
                item = q.get(timeout=1.0)
                if item['type'] == 'data':
                    yield f"data: {json.dumps({'progress': item['pct'], 'message': item['msg']})}\n\n"
                elif item['type'] == 'done':
                    yield f"data: {json.dumps({'progress': 100, 'message': 'Optimization Complete!'})}\n\n"
                    yield "data: DONE\n\n"
                    break
                elif item['type'] == 'error':
                    yield f"data: {json.dumps({'error': item['error']})}\n\n"
                    break
            except queue.Empty:
                yield ": ping\n\n"
            except Exception as e:
                print(f"Stream Error: {e}")

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


# --- Master DL (20 epoch, no cutoff, full master data) -----------------------
@training_bp.route('/api/train/master_dl', methods=['POST'])
def api_train_master_dl():
    """
    Trains all DL model types (lstm/transformer/nbeats) for 20 epochs on the latest
    full master dataset (no cutoff) and saves outputs under /MasterDl.
    """
    def worker():
        try:
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            # Create timestamped subfolder
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            master_dir = os.path.join(base_path, 'MasterDl', timestamp)
            
            os.makedirs(master_dir, exist_ok=True)

            dl = DLMacroModel(model_dir=master_dir)
            for mtype in ['lstm', 'transformer', 'nbeats']:
                dl.train_all_models(model_type=mtype, epochs=20, train_cutoff_date=None)
            print("[MasterDl] Training complete.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[MasterDl] Training failed: {e}")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({'status': 'started', 'message': 'Master DL training launched (20 epochs, full data, saved to /MasterDl).'})


@training_bp.route('/api/train/master_dl_stream')
def api_train_master_dl_stream():
    """
    SSE stream for Master DL training with progress updates.
    Mirrors the standard Deep Learning Auto-Pilot pipeline (train_all_models)
    but writes outputs to /MasterDl and uses full data (no holdout/cutoff).
    """
    def generate():
        q = queue.Queue()
        model_types = ['lstm', 'transformer', 'nbeats']
        total_types = len(model_types)
        # epochs matches “Run Pipeline” depth; default 20 if not provided
        try:
            epochs = int(request.args.get('epochs', 50))
        except Exception:
            epochs = 50

        def worker():
            try:
                base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                
                # Create timestamped subfolder
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                master_dir = os.path.join(base_path, 'MasterDl', timestamp)
                
                os.makedirs(master_dir, exist_ok=True)
                dl = DLMacroModel(model_dir=master_dir)

                model_types = ['lstm'] # User requested LSTM only for now
                total_types = len(model_types)
                slot_size = 100.0 / total_types

                for idx, mtype in enumerate(model_types):
                    base_offset = idx * slot_size

                    # OPTIMIZATION STEP (30% of slot)
                    # -----------------------------------------------------------------
                    def opt_cb(pct, msg):
                        # Map 0-100 (Opt) -> 0-30 of slot
                        rel_pct = pct * 0.3
                        overall = base_offset + rel_pct # percent of total 100
                        q.put({'type': 'data', 'progress': overall, 'message': f"[{mtype.upper()} OPT] {msg}"})

                    q.put({'type': 'data', 'progress': base_offset, 'message': f"Starting {mtype.upper()} Optimization..."})
                    dl.optimize_models(model_type=mtype, iterations=10, progress_callback=opt_cb)

                    # DISCOVERY STEP (20% of slot)
                    # -----------------------------------------------------------------
                    def disc_cb(pct, msg):
                        # Map 0-100 (Disc) -> 30-50 of slot
                        rel_pct = 30 + (pct * 0.2)
                        overall = base_offset + rel_pct 
                        q.put({'type': 'data', 'progress': overall, 'message': f"[{mtype.upper()} DISC] {msg}"})
                        
                    q.put({'type': 'data', 'progress': base_offset + 30, 'message': f"Finding Optimal Epochs {mtype.upper()}..."})
                    # Use 'epochs' as max cap for discovery, enable Bagging for robust validation
                    discovery_results = dl.train_all_models(model_type=mtype, epochs=epochs, force_full_training=False, use_bagging_ensemble=True, progress_callback=disc_cb)
                    
                    # Extract Optimal Epochs
                    manual_epochs = {}
                    for res in discovery_results:
                        key = f"{res['asset']}_{res['horizon']}_{mtype}"
                        found_epochs = res['metrics'].get('optimal_epochs', epochs) # Fallback to max
                        manual_epochs[key] = found_epochs
                        # q.put({'type': 'log', 'message': f"Optimal Epochs for {key}: {found_epochs}"}) # verbose log?

                    # PRODUCTION STEP (50% of slot)
                    # -----------------------------------------------------------------
                    def prod_cb(pct, msg):
                        # Map 0-100 (Prod) -> 50-100 of slot
                        rel_pct = 50 + (pct * 0.5)
                        overall = base_offset + rel_pct
                        q.put({'type': 'data', 'progress': overall, 'message': f"[{mtype.upper()} FULL] {msg}"})
                        
                    q.put({'type': 'data', 'progress': base_offset + 50, 'message': f"Production Training (Gap Bagging Ensemble)..."})
                    dl.train_all_models(
                        model_type=mtype, 
                        epochs=epochs, 
                        force_full_training=True, 
                        use_bagging_ensemble=True, # <--- Changed to True
                        n_folds=3,
                        manual_epochs_dict=manual_epochs, 
                        progress_callback=prod_cb
                    )
                    
                    # Mark slot complete
                    q.put({'type': 'data', 'progress': base_offset + slot_size, 'message': f"{mtype.upper()} Pipeline Complete"})

                q.put({'type': 'done', 'message': 'Master DL Pipeline (Optimization + Training) Finished'})
            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put({'type': 'error', 'message': str(e)})

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item['type'] == 'data':
                yield f"data: {json.dumps({'progress': item.get('progress', 0), 'message': item.get('message', '')})}\n\n"
            elif item['type'] == 'done':
                yield f"data: {json.dumps({'progress': 100, 'message': item.get('message', 'done')})}\n\n"
                yield "data: DONE\n\n"
                break
            elif item['type'] == 'error':
                yield f"data: {json.dumps({'error': item.get('message', 'unknown error')})}\n\n"
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@training_bp.route('/api/train/master_dl_status', methods=['GET'])
def api_get_master_dl_status():
    """
    Returns the date of the last successful Master DL training.
    """
    try:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        master_dir = os.path.join(base_path, 'MasterDl')
        
        if not os.path.exists(master_dir):
             return jsonify({'status': 'none', 'last_trained': None})
             
        # List all subfolders
        subfolders = [f for f in os.listdir(master_dir) if os.path.isdir(os.path.join(master_dir, f))]
        
        if not subfolders:
            return jsonify({'status': 'none', 'last_trained': None})
            
        # Sort by name (timestamp YYYY-MM-DD_HH-MM-SS sorts correctly)
        subfolders.sort(reverse=True)
        latest = subfolders[0]
        
        # Parse timestamp
        try:
             # Format: 2025-01-24_14-30-00
             from datetime import datetime
             dt = datetime.strptime(latest, "%Y-%m-%d_%H-%M-%S")
             last_trained_iso = dt.isoformat()
             display_date = dt.strftime("%Y-%m-%d")
        except:
             last_trained_iso = None
             display_date = latest
             
        # Calculate health
        status = 'fresh'
        days_since = 0
        if last_trained_iso:
             from datetime import datetime
             delta = datetime.now() - datetime.fromisoformat(last_trained_iso)
             days_since = delta.days
             
             if days_since > 90:
                 status = 'obsolete'
             elif days_since > 60:
                 status = 'aging'
        
        return jsonify({
            'status': status,
            'last_trained': display_date,
            'last_trained_full': last_trained_iso,
            'days_since': days_since,
            'folder': latest
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@training_bp.route('/api/train/price_3d_stream')
def api_train_price_3d_stream():
    """Streams 3-day price-regression training progress using Server-Sent Events."""
    def generate():
        pipeline = _get_price_3d_pipeline()
        q = queue.Queue()

        def worker():
            try:
                def q_callback(pct, msg):
                    q.put({"type": "data", "pct": pct, "msg": msg})

                try:
                    epochs = int(request.args.get('epochs', 20))
                except Exception:
                    epochs = 20

                refresh_features = str(request.args.get('refresh_features', 'false')).lower() in {'1', 'true', 'yes', 'y'}
                year = request.args.get('year', None)
                holdout_year = int(year) if year and year.isdigit() else 2025
                assets_arg = request.args.get('assets', '')
                asset_subset = [a.strip() for a in assets_arg.split(',') if a.strip()] or None

                q_callback(0, "Starting 3D price-MAPE pipeline...")
                result = pipeline.train_holdout_pipeline(
                    holdout_year=holdout_year,
                    progress_callback=q_callback,
                    epochs=epochs,
                    asset_subset=asset_subset,
                    force_refresh_features=refresh_features
                )
                q.put({"type": "done", "payload": result})
            except Exception as e:
                import traceback
                traceback.print_exc()
                q.put({"type": "error", "error": str(e)})

        t = threading.Thread(target=copy_current_request_context(worker), daemon=True)
        t.start()

        while True:
            try:
                item = q.get(timeout=1.0)
                if item['type'] == 'data':
                    yield f"data: {json.dumps({'progress': item['pct'], 'message': item['msg']})}\n\n"
                elif item['type'] == 'done':
                    payload = {
                        'progress': 100,
                        'message': '3D price pipeline complete',
                        'holdout_dir': item['payload']['holdout_dir'],
                        'metrics_path': item['payload']['metrics_path'],
                        'champions_path': item['payload']['champions_path']
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    yield "data: DONE\n\n"
                    break
                elif item['type'] == 'error':
                    yield f"data: {json.dumps({'error': item['error']})}\n\n"
                    break
            except queue.Empty:
                yield ": ping\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@training_bp.route('/api/train/forecast_stream')
def api_train_forecast_stream_alias():
    """
    Compatibility alias expected by frontend clients.
    Maps to the canonical deep-learning SSE training pipeline.
    """
    return api_train_deep_learning_stream()
